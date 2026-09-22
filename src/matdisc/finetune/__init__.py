"""Machine-learning interatomic potential fine-tuning on DFT labels.

DFT relaxation output becomes a DeePMD-kit dataset, a training configuration is built from it, and
the DeePMD-kit training command fine-tunes a pretrained DPA-3 checkpoint on that data.
"""

from matdisc.finetune.dataset import (
    ConversionResult,
    convert_dft_to_deepmd,
    convert_outcar,
    find_system_outcars,
    get_type_map,
    split_train_val,
)
from matdisc.finetune.train import (
    FinetuneResult,
    build_dpa3_config,
    check_training_progress,
    find_latest_checkpoint,
    finetune,
    resolve_pretrained_model,
    write_dpa3_config,
    write_finetune_slurm_script,
)

__all__ = [
    "ConversionResult",
    "FinetuneResult",
    "build_dpa3_config",
    "check_training_progress",
    "convert_dft_to_deepmd",
    "convert_outcar",
    "find_latest_checkpoint",
    "find_system_outcars",
    "finetune",
    "get_type_map",
    "resolve_pretrained_model",
    "split_train_val",
    "write_dpa3_config",
    "write_finetune_slurm_script",
]
