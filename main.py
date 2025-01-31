import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from types import SimpleNamespace
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from collections import OrderedDict
import argparse
import neptune  # Added import for Neptune
import os

from distributed import wrap_in_fsdp, get_dataloader
from scheduler import CosineScheduler


def initialize_distributed(device):
    # Get rank and local rank from environment variables
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    dist.init_process_group(backend="nccl", init_method="env://", rank=global_rank)
    world_size = dist.get_world_size()

    if device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return device, global_rank, local_rank, world_size


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings


class AttentionMechanism(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        is_causal: bool,
    ):
        return F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            is_causal=is_causal,
        )


class AttentionLayer(nn.Module):
    def __init__(
        self,
        dmodel,
        heads,
    ):
        super(AttentionLayer, self).__init__()

        self.ln = nn.LayerNorm(dmodel)
        self.heads = heads
        self.input_projection = nn.Linear(dmodel, 3 * dmodel, bias=False)
        self.output_projection = nn.Linear(dmodel, dmodel, bias=False)
        self.attention_mechanism = AttentionMechanism()

    def forward(self, x, attention_mask):
        x = self.ln(x)

        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        key = k_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        value = v_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)

        with torch.nn.attention.sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            attention_output = self.attention_mechanism(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                is_causal=True,
            )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output


def FeedForward(
    dmodel,
):
    return nn.Sequential(
        OrderedDict(
            [
                ("ff_layernorm", nn.LayerNorm(dmodel)),
                (
                    "pre_relu",
                    nn.Linear(
                        dmodel,
                        4 * dmodel,
                        bias=True,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "post_relu",
                    nn.Linear(
                        4 * dmodel,
                        dmodel,
                        bias=True,
                    ),
                ),
            ]
        )
    )


class Block(nn.Module):
    def __init__(
        self,
        dmodel,
        heads,
    ):
        super().__init__()
        self.attention_layer = AttentionLayer(dmodel, heads)
        self.feed_forward_layer = FeedForward(dmodel)

    def forward(self, x, attention_mask):
        out_attention = self.attention_layer(x, attention_mask)
        x = x + out_attention

        out_feed_forward = self.feed_forward_layer(x)
        x = x + out_feed_forward
        return x


class Head(nn.Module):
    def __init__(self, dmodel, vocab_size):
        super().__init__()
        self.head = nn.Linear(dmodel, vocab_size, bias=False)

    def forward(self, x):
        return self.head(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = EmbeddingLayer(
            config.vocab_size, config.dmodel, config.seq_len
        )
        self.blocks = nn.ModuleList(
            [Block(config.dmodel, config.n_heads) for _ in range(config.n_layers)]
        )

        self.head = Head(config.dmodel, config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        output = self.embedding_layer(input_ids)

        for block in self.blocks:
            output = block(output, attention_mask)

        output = self.head(output)
        return output


def calculate_valid_loss(model, valid_dataloader, device, validation_steps):
    valid_losses = []
    model.eval()
    for _, batch in zip(range(validation_steps), valid_dataloader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"]
            outputs = model(input_ids)
            mask_loss = F.cross_entropy(
                outputs.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
            loss = mask_loss.mean().item()
            valid_losses.append(loss)
    mean_valid_loss = sum(valid_losses) / validation_steps
    return mean_valid_loss


def print_grad(model):
    last_ff = model.blocks[-1].feed_forward_layer  # Access the last feed-forward layer

    # Collect gradients
    grads = [p.grad for p in last_ff.parameters() if p.grad is not None]

    # Compute mean and std of gradients
    mean_grad = torch.cat([g.view(-1) for g in grads]).mean().item() if grads else None
    std_grad = torch.cat([g.view(-1) for g in grads]).std().item() if grads else None

    print(f"Last FF Layer Gradients - Mean: {mean_grad}, Std: {std_grad}")


def train_model(config, device, run):  # Added 'run' parameter
    if config.use_fsdp:
        data_seed = config.global_rank + 42
        world_size = dist.get_world_size()
    else:
        data_seed = 42
        world_size = 1
    dataloader = get_dataloader(
        config.batch_size_per_gpu,
        config.seq_len,
        data_path=config.dataset_path,
        seed=data_seed,
    )
    valid_dataloader = get_dataloader(
        config.batch_size_per_gpu,
        config.seq_len,
        split="validation",
        data_path=config.dataset_path,
        seed=data_seed,
    )
    validation_steps = int(
        1e06 // (config.batch_size_per_gpu * config.seq_len)
    )  # we want to evaluate on 1M tokens
    model = Transformer(config)
    model.to(device)

    if config.use_fsdp:
        model = wrap_in_fsdp(
            module=model,
            local_rank=config.local_rank,
            mixed_precision_dtype=config.mixed_precision_dtype,
            modules_to_wrap=[
                EmbeddingLayer,
                Block,
                Head,
            ],
            mixed_precision_ignored_classes=config.high_precision_modules,
        )

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineScheduler(
        n_steps=config.n_training_steps,
        lr=config.learning_rate,
        lr_warmup_fraction=args.lr_warmup_fraction,
        final_lr_fraction=args.final_lr_fraction,
    )

    model.train()

    for i, batch in zip(range(config.n_training_steps), dataloader):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if i < 2:
            print(
                f'i: {i}t\trank, seed: {config.global_rank}, {data_seed}\tids: {batch["input_ids"][:5]}'
            )

        optimizer.zero_grad()
        outputs = model(input_ids)

        mask_loss = F.cross_entropy(
            outputs.flatten(0, -2),
            target_ids.reshape(-1).long(),
            reduction="none",
        )
        mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
        loss = mask_loss.mean()

        with torch.no_grad():
            loss_for_logging = torch.tensor(loss.item(), device=device)
            if config.use_fsdp:
                dist.reduce(loss_for_logging, dst=0, op=dist.ReduceOp.SUM)

            if i % config.log_train_loss_freq == 0 and config.global_rank == 0:
                loss_for_logging /= world_size
                print(f"Step:{i}, Train Loss:{loss_for_logging}")
                run["train/loss"].log(
                    value=loss_for_logging.item(), step=i
                )  # Log training loss to Neptune
                lr_for_logging = scheduler.get_lr(i)
                run["learning_rate"].log(
                    value=lr_for_logging, step=i
                )  # Log training loss to Neptune

            if i % config.log_valid_loss_freq == 0:
                valid_loss = calculate_valid_loss(
                    model, valid_dataloader, device, validation_steps
                )
                valid_loss = torch.tensor(valid_loss, device=device)
                if config.use_fsdp:
                    dist.reduce(valid_loss, dst=0, op=dist.ReduceOp.SUM)
                if config.global_rank == 0:
                    valid_loss /= world_size
                    print(f"Valid loss:{valid_loss}")
                    run["validation/loss"].log(
                        value=valid_loss, step=i
                    )  # Log validation loss to Neptune

        if i % 100 == 0:
            print(f"rank: {config.global_rank}\tloss: {loss.item()}")
        loss.backward()
        if i % 100 == 0:
            print_grad(model)
        scheduler.set_lr(step=i, optimizer=optimizer)
        optimizer.step()

    final_valid_loss = calculate_valid_loss(
        model, valid_dataloader, device, validation_steps
    )
    print(f"Final valid loss:{final_valid_loss}")
    if config.global_rank == 0:
        run["validation/final_loss"].log(
            final_valid_loss
        )  # Log final validation loss to Neptune


def init_neptune_run(rank):
    if rank == 0:
        # Initialize Neptune
        neptune_project = os.getenv("NEPTUNE_PROJECT")
        neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
        if not neptune_project or not neptune_api_token:
            print(f"neptune_project: {neptune_project}")
            print(f"neptune_api_token: {neptune_api_token}")
            raise ValueError(
                "Neptune project or API token not set in environment variables."
            )

        run = neptune.init_run(
            project=neptune_project,  # Replace with your Neptune project
            api_token=neptune_api_token,  # Replace with your Neptune API token
            tags=[
                f"{key}={value}" for key, value in vars(args).items()
            ],  # Log arguments as tags
        )
        return run
    else:
        return None


def main(args):
    if args.use_fsdp == "true":
        device, global_rank, local_rank, world_size = initialize_distributed(
            args.device
        )
        print(f"global_rank: {global_rank}")
        print(f"local_rank: {local_rank}")
        assert args.batch_size % world_size == 0
        use_fsdp = True
        batch_size_per_gpu = args.batch_size // world_size
        if args.mixed_precision_dtype == "bfloat16":
            mixed_precision_dtype = torch.bfloat16
        else:
            mixed_precision_dtype = torch.float32
    else:
        device = torch.device(args.device)
        global_rank = local_rank = 0
        mixed_precision_dtype = None
        use_fsdp = False
        batch_size_per_gpu = args.batch_size

    if args.use_high_precision_modules == "true":
        high_precision_modules = [AttentionMechanism]
    else:
        high_precision_modules = None

    run = init_neptune_run(global_rank)

    if global_rank == 0:
        args_dict = vars(args)
        run["args"] = args_dict

    config = SimpleNamespace(
        vocab_size=50257,
        dmodel=args.dmodel,
        n_heads=4,
        n_layers=args.n_layers,
        learning_rate=args.learning_rate,
        n_training_steps=args.n_training_steps,
        lr_warmup_fraction=args.lr_warmup_fraction,
        final_lr_fraction=args.final_lr_fraction,
        dropout=0.0,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        batch_size_per_gpu=batch_size_per_gpu,
        log_train_loss_freq=args.log_train_loss_freq,
        log_valid_loss_freq=args.log_valid_loss_freq,
        dataset_path=args.dataset_path,
        local_rank=local_rank,
        global_rank=global_rank,
        use_fsdp=use_fsdp,
        high_precision_modules=high_precision_modules,
        mixed_precision_dtype=mixed_precision_dtype,
    )
    if device.type == "cpu":
        print(f"Device type is: {device}. Remember to train on GPU.")
    train_model(config, device, run)  # Pass 'run' to train_model
    if global_rank == 0:
        run.stop()  # Stop Neptune run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Model Training")
    parser.add_argument(
        "--n_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument("--dmodel", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--n_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/net/tscratch/people/plgkciebiera/datasets/c4/",
    )
    parser.add_argument(
        "--use_fsdp",
        type=str,
        default="false",
        help='use FSDP iff equal to string "true"',
    )
    parser.add_argument("--mixed_precision_dtype", type=str, default="bfloat16")
    parser.add_argument("--use_high_precision_modules", type=str, default="true")

    parser.add_argument("--n_training_steps", type=int, default=1001)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_fraction", type=float, default=0.01)
    parser.add_argument("--final_lr_fraction", type=float, default=0.1)

    parser.add_argument("--log_train_loss_freq", type=int, default=100)
    parser.add_argument("--log_valid_loss_freq", type=int, default=200)

    args = parser.parse_args()

    main(args)
