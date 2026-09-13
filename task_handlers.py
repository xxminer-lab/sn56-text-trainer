"""Task-type handlers for the G.O.D text tournaments that are NOT plain SFT.

DPO, GRPO and continuous-SFT (chat) tasks run through axolotl's training CLI
(the framework the evaluator itself is built on) using a config built from the
task arguments; the plain-InstructTextTask path stays in train_text.py.

Nothing here invents a training loop: `axolotl.cli.train` (DPO/SFT) and
axolotl's TRL GRPO trainer do the optimization; this module only assembles the
config from the standardized CLI arguments the validator passes
(docs/miner.md runtime arguments) and keeps the output contract
(/app/checkpoints/{task_id}/{expected_repo_name}).
"""
import json
import os
import subprocess

from god_core.dataset_models import DpoDatasetType
from god_core.dataset_models import GrpoDatasetType
from god_core.dataset_models import ChatTemplateDatasetType

CACHE = os.environ.get("GOD_CACHE", "/cache")
CHECKPOINTS = os.environ.get("GOD_CHECKPOINTS", "/app/checkpoints")

# Base hyperparameters for the framework paths. Deliberately conservative:
# these task types are decided per-sample with a dead zone, so a stable loss
# curve that finishes inside the wall clock beats an aggressive one that dies.
DPO_BASE = dict(
    adapter="lora", lora_r=64, lora_alpha=128, lora_dropout=0.05, lora_target_linear=True,
    sequence_len=4096, sample_packing=False, pad_to_sequence_len=True,
    gradient_accumulation_steps=8, micro_batch_size=1, num_epochs=1,
    optimizer="adamw_bnb_8bit", lr_scheduler="cosine", learning_rate=5.0e-6,
    warmup_steps=20, bf16="auto", flash_attention=True, train_on_inputs=False,
    saves_per_epoch=4, output_dir=None, hub_strategy="every_save",
)
GRPO_BASE = dict(
    sequence_len=4096, adapter="lora", lora_r=64, lora_alpha=128, lora_target_linear=True,
    gradient_accumulation_steps=8, micro_batch_size=1, optimizer="adamw_bnb_8bit",
    lr_scheduler="cosine", learning_rate=1.0e-6, warmup_steps=10, bf16="auto",
    num_epochs=1, flash_attention=True,
)


def dataset_task_path(task_id: str, dataset: str, file_format: str) -> str:
    if file_format == "s3":
        return os.path.join(CACHE, "datasets", f"{task_id}_train_data.json")
    return os.path.join(CACHE, "datasets", dataset.replace("/", "--"))


def model_cache_path(model_id: str) -> str:
    return os.path.join(CACHE, "models", model_id.replace("/", "--"))


def _config_path(task_id: str) -> str:
    d = os.path.join(CACHE, "run", task_id, "configs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{task_id}.yml")


def _write_config(cfg: dict, task_id: str) -> str:
    import yaml

    p = _config_path(task_id)
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return p


def _stage_rows(dataset_path: str, workdir: str) -> str:
    """axolotl's JSON loader wants a directory that holds the rows."""
    import shutil

    data_dir = os.path.join(workdir, "data")
    os.makedirs(data_dir, exist_ok=True)
    dst = os.path.join(data_dir, "train.json")
    if not os.path.exists(dst):
        shutil.copyfile(dataset_path, dst)
    return data_dir


def _clean_upload_keys(cfg: dict) -> dict:
    for k in [k for k in list(cfg) if k.startswith("wandb") or k.startswith("hub")]:
        cfg.pop(k)
    return cfg


def run_dpo(task_id: str, model: str, dataset: str, dataset_type_dict: dict, file_format: str,
            expected_repo_name: str) -> str:
    """DPO via axolotl: prompt/chosen/rejected columns from the task payload."""
    dt = DpoDatasetType(**dataset_type_dict)
    workdir = os.path.join(CACHE, "run", task_id)
    os.makedirs(workdir, exist_ok=True)
    data_dir = _stage_rows(dataset_task_path(task_id, dataset, file_format), workdir)
    out_dir = os.path.join(CHECKPOINTS, task_id, expected_repo_name)

    cfg = dict(DPO_BASE)
    cfg["base_model"] = model_cache_path(model)
    cfg["rl"] = "dpo"
    cfg["datasets"] = [{
        "path": data_dir, "ds_type": "json", "data_files": ["train.json"],
        "field_prompt": dt.field_prompt,
        "field_chosen": dt.field_chosen,
        "field_rejected": dt.field_rejected,
        **({"field_system": dt.field_system} if getattr(dt, "field_system", None) else {}),
    }]
    cfg["output_dir"] = out_dir
    cfg = _clean_upload_keys(cfg)
    p = _write_config(cfg, task_id)
    subprocess.run(["accelerate", "launch", "-m", "axolotl.cli.train", p], check=True)
    return out_dir


def run_grpo(task_id: str, model: str, dataset: str, dataset_type_dict: dict, file_format: str,
             expected_repo_name: str, hours_to_complete: float) -> str:
    """GRPO via axolotl's TRL path: reward functions from the task payload are
    written to a module axolotl imports (same contract as the reference repo)."""
    dt = GrpoDatasetType(**dataset_type_dict)
    workdir = os.path.join(CACHE, "run", task_id)
    src_dir = os.path.join(workdir, "src")
    os.makedirs(src_dir, exist_ok=True)
    data_dir = _stage_rows(dataset_task_path(task_id, dataset, file_format), workdir)
    out_dir = os.path.join(CHECKPOINTS, task_id, expected_repo_name)

    names = []
    for idx, rf in enumerate(dt.reward_functions or []):
        mod = f"reward_{task_id}_{idx}".replace("-", "_")
        with open(os.path.join(src_dir, f"{mod}.py"), "w") as f:
            f.write(str(rf.reward_func).strip() + "\n")
    names = [f"reward_{task_id}_{i}".replace("-", "_") for i in range(len(dt.reward_functions or []))]
    sys.path.insert(0, src_dir)

    cfg = dict(GRPO_BASE)
    cfg["base_model"] = model_cache_path(model)
    cfg["rl"] = "grpo"
    cfg["datasets"] = [{
        "path": data_dir, "ds_type": "json", "data_files": ["train.json"], "field": dt.field_prompt,
    }]
    cfg["trl"] = {
        "reward_funcs": names,
        "reward_weights": [rf.reward_weight for rf in (dt.reward_functions or [])],
    }
    import math

    # GRPO rolls generations; keep the wall clock same order as the training hour.
    cfg["max_steps"] = max(16, int((hours_to_complete * 3600 * 0.7) / (20 * 8)))  # ~20s per roll-batch
    cfg["output_dir"] = out_dir
    cfg = _clean_upload_keys(cfg)
    p = _write_config(cfg, task_id)
    subprocess.run(["accelerate", "launch", "-m", "axolotl.cli.train", p], check=True)
    return out_dir


def run_chat(task_id: str, model: str, dataset: str, dataset_type_dict: dict, file_format: str,
             expected_repo_name: str, hours_to_complete: float) -> str:
    """Continuous-SFT chat task = plain SFT over the conversation column,
    using axolotl's chat_template dataset type."""
    dt = ChatTemplateDatasetType(**dataset_type_dict)
    workdir = os.path.join(CACHE, "run", task_id)
    os.makedirs(workdir, exist_ok=True)
    data_dir = _stage_rows(dataset_task_path(task_id, dataset, file_format), workdir)
    out_dir = os.path.join(CHECKPOINTS, task_id, expected_repo_name)

    cfg = dict(DPO_BASE)
    cfg["base_model"] = model_cache_path(model)
    cfg["datasets"] = [{
        "path": data_dir, "ds_type": "json", "data_files": ["train.json"],
        "type": "chat_template",
        "chat_template": dt.chat_template,
        "field_messages": dt.chat_column,
        "message_field_role": dt.chat_role_field,
        "message_field_content": dt.chat_content_field,
        "roles": {"assistant": [dt.chat_assistant_reference], "user": [dt.chat_user_reference]},
    }]
    cfg["output_dir"] = out_dir
    cfg = _clean_upload_keys(cfg)
    p = _write_config(cfg, task_id)
    subprocess.run(["accelerate", "launch", "-m", "axolotl.cli.train", p], check=True)
    return out_dir
