from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


def wrap_in_fsdp(
    module,
    local_rank,
    mixed_precision_dtype,
    classes_to_wrap,
    mixed_precision_ignored_classes,
):

    wrap_policy = ModuleWrapPolicy(classes_to_wrap)

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
