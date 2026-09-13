# SN56 text-tournament trainer

Training repository for the Gradients on Demand (G.O.D, Bittensor netuid 56) text
tournaments: `InstructTextTask`, `DpoTask`, `GrpoTask` and `ChatTask` handlers with a
single unified pipeline.

Design (readable source, no obfuscation, no bundled datasets or pretrained models):

1. **Eval-parity tokenization.** Training rows are materialized through the same
   axolotl dataset pipeline the validator's evaluator uses
   (`god_core.training_config.create_dataset_entry` + `load_tokenized_prepared_datasets`,
   `train_on_inputs=false`), so the prompt format, BOS/EOS handling and the
   completion mask are byte-identical to the scored quantity. The source of the two
   vendored helpers is the G.O.D runtime (Apache-2.0) - see LICENSE.md and NOTICE.
2. **A measured time plan.** A short steady-state probe (warmup steps excluded)
   measures seconds-per-optimizer-step on the real shapes, then the run plans
   `max_steps` to fill a bounded fraction of `--hours-to-complete`, with a cosine
   schedule sized to that plan and a hard-stop callback that guarantees the deadline.
3. **Best checkpoint wins.** Evaluation runs on an internal held-out slice of the
   train file with the aggregator shape of the official metric; `load_best_model_at_end`
   restores the best checkpoint, which is what gets written to
   `/app/checkpoints/{task_id}/{expected_repo_name}`.
4. **Full fine-tune when it fits.** Models up to the free video-memory budget are trained
   full-parameter bf16 with fused AdamW; larger bases (the boss round uses 35-71B)
   fall back to LoRA on all linear layers. When `USE_KL=1`/`KL_COEF` are set the
   trainer switches loss and checkpoint selection to the KL-regularised objective.

Requirements honoured: no internet access from the container, `/cache` read-only,
model from `/cache/models/{model}`, task rows from `/cache/datasets/{task_id}_train_data.json`,
output `/app/checkpoints/{task_id}/{expected_repo_name}`, `max_position_embeddings`
never decreased.
