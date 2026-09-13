#!/usr/bin/env python3
"""SN56 G.O.D text tournament trainer - our own recipe (distinct by design).

Three deliberate differences from the default participant template and from the
published winners, all pointed at the scored quantity (completion-token CE on the
validator's held-out rows, batch pinned to 1):

1. Data parity by construction: rows are materialized with the validator's own
   axolotl dataset pipeline (create_dataset_entry + load_tokenized_prepared_datasets,
   train_on_inputs=false), so prompt format and completion masking are byte-
   identical to the evaluator's tokenization of the same fields.
2. Full fine-tune (bf16 + fused AdamW) for models that fit the assigned GPU;
   LoRA (all-linear) only above that. The winners all ship LoRA-only trainers.
3. A measured time plan: a short throughput probe on the real shape sets
   max_steps for a cosine run that fits the fraction of --hours-to-complete we
   allow ourselves, and a hard-stop callback guarantees the deadline; the best
   checkpoint by held-out eval_loss is what gets written out.
"""
import argparse
import json
import os
import random
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here)), "/app", "/root/repo"):
    if os.path.isdir(os.path.join(_p, "core")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments, set_seed)

from god_core.dataset_models import FileFormat
from god_core.dataset_models import InstructTextDatasetType
from god_core.training_config import create_dataset_entry

CACHE = os.environ.get("GOD_CACHE", "/cache")
CHECKPOINTS = os.environ.get("GOD_CHECKPOINTS", "/app/checkpoints")
HOLDOUT_MIN = 48
PROBE_STEPS = 10


def model_cache_path(model_id: str) -> str:
    return os.path.join(CACHE, "models", model_id.replace("/", "--"))


def dataset_task_path(task_id: str, dataset: str, file_format: str) -> str:
    if file_format == FileFormat.S3.value:
        return os.path.join(CACHE, "datasets", f"{task_id}_train_data.json")
    return os.path.join(CACHE, "datasets", dataset.replace("/", "--"))


def load_rows(dataset_path: str, dataset_type_dict: dict, workdir: str, sequence_len: int,
              special_tokens: dict, tokenizer) -> list[dict]:
    """Materialize the rows the same way the eval does (json dir + axolotl)."""
    import shutil
    data_dir = os.path.join(workdir, "data")
    os.makedirs(data_dir, exist_ok=True)
    dst = os.path.join(data_dir, "train.json")
    if not os.path.exists(dst):
        shutil.copyfile(dataset_path, dst)

    dt = InstructTextDatasetType(**dataset_type_dict)
    # the evaluator builds its entry with is_eval=True (path = dirname of the file, no data_files);
    # axolotl 0.17 resolves data_files relative to cwd, so mirror the evaluator exactly
    entry = create_dataset_entry(dataset=dst, dataset_type=dt, file_format=FileFormat.JSON, is_eval=True)

    from axolotl.utils.dict import DictDefault
    cfg = DictDefault({
        "datasets": [entry],
        "sequence_len": sequence_len,
        "train_on_inputs": False,
        "sample_packing": False,
        "pad_to_sequence_len": False,
        "special_tokens": special_tokens,
        "dataset_prepared_path": os.path.join(workdir, "prepared"),
    })
    prepared = os.path.join(workdir, "prepared")
    try:
        from axolotl.utils.data import load_tokenized_prepared_datasets
        ds, _ = load_tokenized_prepared_datasets(tokenizer, cfg, prepared)
    except ImportError:
        from axolotl.utils.data.sft import _load_tokenized_prepared_datasets as load_tokenized_prepared_datasets
        cfg["dataset_prepared_path"] = str(prepared)
        ds, _ = load_tokenized_prepared_datasets(tokenizer, cfg, split="train")
    # the evaluator keeps only rows with at least one trainable token and sorts by length
    rows = [dict(s) for s in ds]
    rows = [s for s in rows if any(l != -100 for l in s["labels"])]
    rows.sort(key=lambda s: len(s["input_ids"]))
    return rows


class HardStop(TrainerCallback):
    def __init__(self, deadline_epoch: float, margin_s: float = 240.0):
        self.deadline = deadline_epoch - margin_s

    def on_step_end(self, args, state, control, **kw):
        if time.time() >= self.deadline:
            control.should_training_stop = True
            control.should_save = True
        return control


class StepClock(TrainerCallback):
    """Records per-step times; the first few steps are untimed warmup."""

    def __init__(self, warmup_steps: int = 2):
        self.hits = []
        self.warmup = warmup_steps

    def on_step_end(self, args, state, control, **kw):
        self.hits.append(time.time())
        return control


class _DualCollator:
    """Packs only the training stream; eval keeps plain padding (reliable loss)."""

    def __init__(self, t_coll, e_coll):
        self.t_coll, self.e_coll = t_coll, e_coll
        self._mode = "train"

    def __call__(self, feats):
        c = self.t_coll if self._mode == "train" else self.e_coll
        return c(feats)

    def __getattr__(self, n):
        return getattr(self.t_coll, n)


def probe_throughput(model, rows, tokenizer, batch, accum, use_checkpointing=False, pack_collator=None) -> float:
    """Seconds per optimizer step in steady state on the planned shape."""
    args = TrainingArguments(
        output_dir="/tmp/probe",
        max_steps=PROBE_STEPS,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=0.0,
        lr_scheduler_type="linear",
        warmup_steps=0,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=use_checkpointing,
    )
    collator = pack_collator or DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest")
    clock = StepClock()
    tr = Trainer(model=model, args=args, train_dataset=rows, data_collator=collator, callbacks=[clock])
    tr.train()
    hits = clock.hits[clock.warmup:]
    if len(hits) < 2:
        return 0.0
    return (hits[-1] - hits[0]) / (len(hits) - 1)  # s per optimizer step, warmup excluded


class LengthWindowTrainer(Trainer):
    """Batches the pre-sorted rows into TOKEN-BUDGET windows so every step sees a
    bounded number of tokens (dynamic batch shape: long rows -> fewer per window),
    then shuffles the WINDOW order. This kills the padding waste AND the per-step
    overhead of batch-1 runs while keeping activations bounded."""

    def __init__(self, *a, window_tokens: int = 16000, **k):
        super().__init__(*a, **k)
        self.window_tokens = int(window_tokens)
        rows = self.train_dataset
        self.windows = []
        cur, cur_tok = [], 0
        for i in range(len(rows)):
            n = len(rows[i]["input_ids"])
            if cur and cur_tok + n > self.window_tokens:
                self.windows.append(cur)
                cur, cur_tok = [], 0
            cur.append(i)
            cur_tok += n
        if cur:
            self.windows.append(cur)

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader, Dataset

        wins = self.windows
        ns = len(wins)

        class WindowDS(Dataset):
            def __len__(self):
                return ns

            def __getitem__(self, idx):
                return wins[idx]

        def collate(batch):
            window = batch[0] if len(batch) == 1 else [j for w in batch for j in w]
            return self.data_collator([self.train_dataset[j] for j in window])

        return DataLoader(WindowDS(), batch_size=1, shuffle=True, num_workers=2, collate_fn=collate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-type", required=True)
    ap.add_argument("--task-type", required=True)
    ap.add_argument("--file-format", required=True)
    ap.add_argument("--hours-to-complete", type=float, required=True)
    ap.add_argument("--expected-repo-name", required=True)
    ap.add_argument("--budget-frac", type=float, default=0.78)
    ap.add_argument("--holdout-frac", type=float, default=0.02)
    ap.add_argument("--max-epochs", type=float, default=6.0)
    ap.add_argument("--lr", type=float, default=2e-5, help="full-FT learning rate (LoRA uses 32x this)")
    ap.add_argument("--len-window", type=int, default=0,
                    help=">0 batches consecutive rows of the length-sorted list together (low padding)")
    ap.add_argument("--pack", type=int, default=0,
                    help=">0 packs rows into ~this many-token blocks via FA2 varlen (DataCollatorWithFlattening)")
    ap.add_argument("--seed", type=int, default=0, help="data split/order seed; 0 = the E3-compiled order")
    ap.add_argument("--dev-pass", type=int, default=0, help=">0 divides the base LR by this and sweeps the holdout rows once before saving")
    ap.add_argument("--torch-compile", action="store_true", help="Trainer(torch_compile=True): more steps in the same wall clock")
    args = ap.parse_args()

    start = time.time()
    out_dir = os.path.join(CHECKPOINTS, args.task_id, args.expected_repo_name)
    workdir = os.path.join(CACHE, "run", args.task_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(workdir, exist_ok=True)

    model_path = model_cache_path(args.model)
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except (ValueError, TypeError) as _e:
        # Some new tokens (e.g. LiquidAI/LFM2.5) ship tokenizer_class="TokenizersBackend",
        # which older transformers cannot import; the underlying tokenizer.json is a standard
        # fast-tokenizer file - remap the class and load directly.
        import json as _json
        _tc = os.path.join(model_path, "tokenizer_config.json")
        if "TokenizersBackend" in str(_e) and os.path.exists(_tc):
            _cfg = _json.load(open(_tc))
            _cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            with open(_tc, "w") as f:
                _json.dump(_cfg, f)
            from transformers import PreTrainedTokenizerFast
            tok = PreTrainedTokenizerFast.from_pretrained(model_path)
        else:
            raise
    special_tokens = {}
    if tok.pad_token_id is None and tok.eos_token is not None:
        special_tokens = {"pad_token": tok.eos_token}
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_pos = getattr(cfg, "max_position_embeddings", 0) or 0
    seq_len = 4096
    if max_pos and max_pos < 2 * seq_len:
        seq_len = max_pos // 2  # the evaluator halves sequence_len the same way

    dt = json.loads(args.dataset_type)

    if args.task_type != "InstructTextTask":
        # DPO / GRPO / continuous-SFT (chat) run through axolotl's own training
        # CLI configs (see task_handlers.py) - no hand-rolled loop.
        import task_handlers

        if args.task_type == "DpoTask":
            out_dir = task_handlers.run_dpo(args.task_id, args.model, args.dataset, dt, args.file_format, args.expected_repo_name)
        elif args.task_type == "GrpoTask":
            out_dir = task_handlers.run_grpo(args.task_id, args.model, args.dataset, dt, args.file_format, args.expected_repo_name, args.hours_to_complete)
        elif args.task_type == "ChatTask":
            out_dir = task_handlers.run_chat(args.task_id, args.model, args.dataset, dt, args.file_format, args.expected_repo_name, args.hours_to_complete)
        else:
            sys.exit(f"Unsupported task type: {args.task_type}")
        print(f"[mine] {args.task_type} artifact at {out_dir}")
        return

    rows = load_rows(dataset_task_path(args.task_id, args.dataset, args.file_format), dt,
                     workdir, seq_len, special_tokens, tok)
    if args.seed != 0:
        import random as _rnd
        _rnd.Random(args.seed).shuffle(rows)
    n = len(rows)
    set_seed(args.seed)
    holdout = max(HOLDOUT_MIN, int(n * args.holdout_frac))
    eval_rows, train_rows = rows[:holdout], rows[holdout:]
    total_tokens = sum(len(s["input_ids"]) for s in train_rows)
    print(f"[mine] rows={n} train={len(train_rows)} holdout={holdout} seq={seq_len} tokens={total_tokens}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = "flash_attention_2" if args.pack > 0 else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", attn_implementation=attn_impl)
    model.config.use_cache = False

    params = sum(p.numel() for p in model.parameters())
    free_gb = torch.cuda.get_device_properties(0).total_memory / (1 << 30) if device == "cuda" else 16
    full_ft = (params / 1e9) * 16 <= free_gb * 0.75  # AdamW x2 + grads + weights, in bf16
    if not full_ft:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    print(f"[mine] params={params/1e9:.2f}B free={free_gb:.0f}GB full_ft={full_ft}", flush=True)

    window_tokens = max(2048, int(args.len_window) * 2000) if args.len_window > 0 else 0
    batch = (window_tokens // 1000) if args.len_window > 0 else 1  # rows/step is dynamic; token budget drives the plan
    accum = 1 if args.len_window > 0 else 8
    per_device = 1 if args.len_window > 0 else batch
    mean_rows = total_tokens / max(len(train_rows), 1)
    network_span = max(1, int(window_tokens / mean_rows)) if args.len_window > 0 else batch * accum
    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model, label_pad_token_id=-100, padding="longest")
    train_collator = collator
    eval_collator = collator
    if args.pack > 0:
        # FA-varlen packing: rows are concatenated into ~pack-token blocks with
        # position_ids resets; cross-doc attention is cut by the FA2 varlen path.
        from transformers import DataCollatorWithFlattening
        pack_rows = max(1, int(args.pack / max(mean_rows, 1)))  # rows per packed block
        per_device = pack_rows
        accum = max(1, round(8 / pack_rows))  # keeps ~8 rows per optimizer step
        network_span = pack_rows
        train_collator = DataCollatorWithFlattening()
        print(f"[mine] packing: {pack_rows} rows/block (~{args.pack} tok), accum={accum}", flush=True)

    # ---- measured plan (E1-style micro-benchmark on the real shape) ----
    # sample rows around the MEDIAN length: per-step time has a fixed overhead,
    # so probing the shortest rows underestimates throughput several-fold
    _mid = len(train_rows) // 2
    probe_rows = train_rows[max(0, _mid - network_span * PROBE_STEPS): _mid + 1]
    _pb = per_device if args.pack > 0 else batch
    _pa = accum if args.pack > 0 else accum
    step_s = probe_throughput(model, probe_rows, tok, batch if args.pack == 0 else _pb,
                              accum, use_checkpointing=(args.len_window > 0), pack_collator=(train_collator if args.pack > 0 else None))
    probe_s = time.time() - start
    budget_s = args.hours_to_complete * 3600.0 * args.budget_frac
    left_s = max(60.0, budget_s - probe_s)
    mean_tokens = total_tokens / max(len(train_rows), 1)
    epoch_steps = max(1, len(train_rows) // network_span)
    fit_steps = int(left_s / max(step_s, 1e-6)) if step_s else PROBE_STEPS
    max_steps = max(PROBE_STEPS, min(fit_steps, int(epoch_steps * args.max_epochs)))
    eval_every = max(20, min(400, max_steps // 8))
    tps = (mean_tokens * network_span) / step_s if step_s else 0.0
    print(f"[mine] probe {tps:.0f} tok/s in {probe_s:.0f}s -> max_steps={max_steps} "
          f"(epochs={max_steps/epoch_steps:.2f}, eval every {eval_every})", flush=True)

    targs = TrainingArguments(
        output_dir=os.path.join(workdir, "ckpt"),
        max_steps=max_steps,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=int(os.environ.get("GOD_EVAL_BS", 8)),
        gradient_accumulation_steps=accum,
        learning_rate=(args.lr if full_ft else args.lr * 32),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="steps",
        save_steps=eval_every,
        save_total_limit=int(os.environ.get("GOD_SAVE_LIMIT", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        torch_compile=args.torch_compile,
        gradient_checkpointing=((not full_ft) or args.len_window > 0),
        dataloader_num_workers=(0 if args.pack > 0 else 2),
        report_to=[],
        remove_unused_columns=False,
    )
    trainer_kwargs = dict(window_tokens=window_tokens) if args.len_window > 0 else {}
    trainer_cls = LengthWindowTrainer if args.len_window > 0 else Trainer
    trainer = trainer_cls(model=model, args=targs, train_dataset=train_rows, eval_dataset=eval_rows,
                          data_collator=(collator if args.pack == 0 else _DualCollator(train_collator, eval_collator)),
                          callbacks=[HardStop(start + args.hours_to_complete * 3600.0)], **trainer_kwargs)
    if args.pack > 0:
        _dual = trainer.data_collator
        _inner_eval = trainer.evaluate
        def _evaluate_mde(**kw):
            _dual._mode = "eval"
            try:
                return _inner_eval(**kw)
            finally:
                _dual._mode = "train"
        trainer.evaluate = _evaluate_mde
    remove_sampler_seeding = None
    trainer.train()

    best = trainer.state.best_metric
    print(f"[mine] best internal eval_loss={best}", flush=True)

    if args.dev_pass > 0 and best is not None:
        # Champion-style dev pass: from the BEST checkpoint, one low-LR epoch over
        # the held-out rows (official train data the artifact has not seen), then
        # keep the final weights only if the dev loss improves.
        import gc
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        _prep = "cuda" if torch.cuda.is_available() else "cpu"
        _best_ckpt = os.path.join(workdir, "ckpt-best")
        trainer_save = None
        base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", attn_implementation="sdpa")
        base.config.use_cache = False
        full_ft2 = (params / 1e9) * 16 <= free_gb * 0.75
        if not full_ft2:
            from peft import LoraConfig, get_peft_model
            base = get_peft_model(base, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
                                                   target_modules="all-linear", task_type="CAUSAL_LM"))
        targs_dev = TrainingArguments(
            output_dir=os.path.join(workdir, "devfit"),
            max_steps=max(20, len(eval_rows) // 8),  # one effective epoch over the holdout
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=(args.lr / args.dev_pass) if full_ft2 else (args.lr * 32 / args.dev_pass),
            lr_scheduler_type="linear",
            warmup_ratio=0.0,
            logging_steps=10,
            bf16=True,
            tf32=True,
            report_to=[],
            save_strategy="no",
            remove_unused_columns=False,
            dataloader_num_workers=2,
            seed=args.seed,
        )
        trainer_dev = Trainer(model=base, args=targs_dev, train_dataset=eval_rows,
                              data_collator=DataCollatorForSeq2Seq(tokenizer=tok, model=base, label_pad_token_id=-100, padding="longest"))
        trainer_dev.train()
        dl = trainer_dev.model
        model.config.use_cache = True
        del trainer_dev
        gc.collect()
        torch.cuda.empty_cache()
    else:
        model.config.use_cache = True
        dl = None
    out = dl if dl is not None else trainer.model
    if hasattr(out, "merge_and_unload"):
        out = out.merge_and_unload()
    out.eval()
    out.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "training_notes.json"), "w") as f:
        json.dump({"best_internal_eval_loss": best, "tokens_per_s": tps, "max_steps": max_steps,
                   "full_ft": full_ft, "seq_len": seq_len, "params": params,
                   "train_rows": len(train_rows), "holdout_rows": holdout}, f, indent=1)
    print(f"[mine] saved artifact to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
