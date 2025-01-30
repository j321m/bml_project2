from datasets import load_dataset, load_from_disk
from transformers import GPT2TokenizerFast
from torch.utils.data import DataLoader
from functools import partial

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy


def collate_tokenize(tokenizer, sequence_length, data):
    text_batch = [element["text"] for element in data]
    tokenized = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=sequence_length + 1,
    )
    input_ids = tokenized["input_ids"]
    tokenized["input_ids"] = input_ids[:, :-1]
    tokenized["target_ids"] = input_ids[:, 1:]
    tokenized["attention_mask"] = tokenized["attention_mask"][:, :-1]
    return tokenized


def get_dataloader(
    batch_size,
    sequence_length,
    split="train",
    buffer_size=10000,
    seed=42,
    num_workers=2,
    data_path="/net/tscratch/people/plgkciebiera/datasets/c4/",
):
    if split == "train":
        hf_dataset = load_from_disk(f"{data_path}train")
    else:
        hf_dataset = load_from_disk(f"{data_path}validation")
    hf_dataset = hf_dataset.to_iterable_dataset(num_shards=64)
    hf_dataset = hf_dataset.shuffle(buffer_size=buffer_size, seed=seed)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_tokenize, tokenizer, sequence_length),
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return dataloader


def wrap_in_fsdp(
    module,
    local_rank,
    mixed_precision_dtype,
    min_num_params,
    mixed_precision_ignored_classes,
):

    wrap_policy = (
        partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        if min_num_params is not None
        else size_based_auto_wrap_policy
    )

    mixed_precision = MixedPrecision(
        param_dtype=mixed_precision_dtype,
        reduce_dtype=mixed_precision_dtype,
        cast_forward_inputs=True,
        _module_classes_to_ignore=mixed_precision_ignored_classes,
    )

    wrapped = FSDP(
        module,
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        auto_wrap_policy=wrap_policy,
    )

    print("------- MODEL AFTER WRAPPING IN FSDP -------")
    print(wrapped)
    print("--------------------------------------------")

    return wrapped
