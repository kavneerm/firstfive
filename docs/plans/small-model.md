# Overnight search for a smaller ASL sign model (Fable 5.2 on EC2)

## Context
The deployed model `~/Downloads/arduino/aslModelENGS21-fold0-best.h5` (Kaggle asl-signs, 250 signs, `get_model(dim=192)`,
~1.9M params, 7.9 MB float32 TFLite, max_len 128, trained on fold 0 for 60 epochs at ~3 min/epoch) works, but we want a model that
uses **substantially less runtime memory at nearly the same accuracy**. a teammate's training code isn't available, so we rebuild the
pipeline ourselves. The new model must take the same input (same `Preprocess`, 118 landmarks, max_len 128) so the board code
(`sign_speaker.py`, `sign_infer_lite.py`) just loads a different `.tflite`.

Constraints from the user: EC2 overnight, **$75 cap for EC2**; Fable 5.2 runs via **Claude Code headless on the instance using
the user's Claude Pro login — no API credits**; data = `kaggle competitions download -c asl-signs` (user supplies the token);
memory metric = **sign model alone**; accuracy = very close to the baseline, and ≥75% correct on our own signing.

## Key design decisions
1. **Fable steers, a script does the work.** A deterministic `search.py` driver trains/evaluates trials from a queue. Fable is
   woken only between rounds (~every 1–2 h) to read `results.jsonl` and append the next configs. This keeps Pro usage small, and if
   Pro limits hit (or Fable isn't usable on Pro — *guess, unverified*), the driver keeps running a pre-written fallback queue.
2. **A frozen evaluation harness, not graded by the agent.** `eval_candidate.py` (sha256 recorded before launch, checked by
   the driver every round) exports TFLite and reports: val top-1/top-5, `.tflite` bytes, param count, and peak RSS of a fresh
   subprocess that loads the model with `tflite-runtime` and runs 50 inferences (memory measured as RSS minus that of an
   empty interpreter process). Fable may edit only `queue.jsonl` and `notes.md`.
3. **Fair baseline on a split we build.** Our folds won't match a teammate's. The baseline was trained on ~80% of signers, so we
   run it on all 21 participants: its held-out signers should score clearly lower. Those become our validation set, so both models
   are judged on signers neither trained on. Fallback if there's no clean gap: retrain dim-192 in our pipeline (one ~3 h slot) as the baseline.
4. **Search space** (same input/preprocess): `dim` {64, 96, 128, 160}, conv blocks per stage {2, 3}, stages {1, 2}, kernel {9, 17},
   transformer expand {1, 2}, ± **knowledge distillation** from the baseline `.h5` as teacher, then **int8 quantization**
   (dynamic-range and full-int8 with a calibration set) of finalists, evaluated after quantization.
5. **Proxy then promote.** Round 1: ~10–15-epoch proxy runs, several in parallel. Top 3 by (accuracy, memory) → full 60-epoch runs.
6. **Pass criterion (the overnight goal):** smallest candidate with val top-1 ≥ baseline − 1.0 pt **and** sign-model memory ≤ 50% of
   baseline (stretch ≤ 25%). The 75%-on-our-signing check happens tomorrow on the board.

## Instance & budget
- Recommended `g6.12xlarge` (4× L4, 48 vCPU): 4 trials in parallel, and enough CPU for the CPU-bound augmentation. Fallback `g6.4xlarge`
  (1× L4, 16 vCPU). **Prices are my estimate (~$4.6/h and ~$1.3/h on-demand); check the console before launch.**
- Hard stop: `shutdown -h +<minutes>` scheduled at boot, sized to $75 at the real hourly price (with `InstanceInitiatedShutdownBehavior=terminate`
  after results sync), plus an AWS Budget alarm. Results sync to S3 (or `scp` pull) every round so a shutdown loses nothing.
- Likely blocker: new accounts often have a **0 vCPU quota for G instances**. Request the increase today.

## Files (new project `~/Downloads/asl-small-model/`)
- `prepare_kaggle.py`: parquet → per-sequence float16 `.npy` shards + `meta.csv` with participant folds. Reads straight from the zip.
- `islr_model.py`: copied from `~/Downloads/arduino/islr_model.py`. Reuse `Preprocess`, `Conv1DBlock`, `TransformerBlock`, `ECA`,
  `LateDropout`, and generalize `get_model()` to take blocks/stages/kernel/expand (dim=192 defaults must rebuild the baseline exactly).
- `train.py`: our own training loop (AdamW + one-cycle LR, label smoothing 0.1, augmentation: flip, affine, time resample/crop,
  ±distillation loss). Augmentation runs in tf.data on the CPU.
- `eval_candidate.py`: the frozen harness (reuses the conversion logic from `~/Downloads/arduino/convert_sign_model.py`).
- `search.py`: queue driver, GPU slot scheduler, `results.jsonl`, sha check, wakes `claude -p --model claude-fable-5-2` with a fixed prompt.
- `setup_ec2.sh`: venv (TF 2.14.1, `numpy<2`, `pyarrow<19`, tflite-runtime, kaggle), Kaggle download, conversion, Claude Code install.
- `FABLE.md`: the agent's brief: goal, pass criterion, allowed files, budget clock, "don't touch the harness or data split".
- `docs/plans/small-model.md`: this plan, copied there after approval, with checkboxes.

## Steps
- [x] 0. Right after approval: request On-Demand G/VT quota 8 → 16 vCPU (us-east-1, `L-DB2E81BA`); save the Kaggle token to
      `~/.kaggle/access_token` (chmod 600, never in the project). Steps 1–3 run on the Mac during the ~30 min wait. Poll the quota during the wait:
      approved any time before +30 min → launch `g6.4xlarge` as soon as steps 1–3 pass; still not approved at +30 min → launch
      `g6.2xlarge` (8 vCPU) with a smaller queue (fewer parallel proxies, 2 finalists) and don't wait longer.
- [x] 1. Mac: write the code (Kaggle download blocked by 403 until the rules are accepted; used tests/make_fake_kaggle.py instead); pull ~300 sequences with `kaggle competitions download -c asl-signs -f ...` for a local smoke set.
- [x] 2. Mac check (tests/test_parity.py ALL PASS: preprocess diff 0, baseline logit diff 0): `get_model(dim=192)` loads `aslModelENGS21-fold0-best.h5` with no shape errors, and the harness reproduces the board
      model's predictions (max diff < 1e-4 vs `convert_sign_model.py`'s output).
- [x] 3. Mac check (search.py on fake data: trains, scores, rejects invalid config, writes SUMMARY.md): 2-epoch smoke train on the smoke set with a tiny config runs end to end through the harness and appends to `results.jsonl`.
- [ ] 4. You: Kaggle token, AWS quota/instance, `claude` login on the instance (OAuth, Pro).
- [x] 5. EC2: `setup_ec2.sh` → download + convert (~1 h, guess) → split check (step-3 decision) → baseline measured by the harness.
- [x] 6. EC2: launch `search.py` in tmux; I check in on round 1 before you go to sleep.
- [x] 7. Morning (done 04:10 EST, stopped early at the user's request): pull results, a fresh-context subagent reviews them against this plan, and the winner gets copied to the board.
- [ ] 8. Board: record ~20 signs × 2 takes with `record_clips.py`; `check_clips.py` on baseline vs winner (≥30/40 and no worse than the baseline by > 2 clips);
      measure sign-model RSS on the UNO Q.

## Verification
- Step 2/3 commands print pass/fail on the Mac before we spend anything on EC2.
- Each trial's line in `results.jsonl` comes from the frozen harness; the morning summary table prints baseline vs. every candidate
  (top-1, top-5, bytes, RSS) and flags the pass criterion.
- Board: `check_clips.py` PASS/FAIL table for both models; RSS of `sign_infer_lite.py` with each model.

## Risks / guesses
- Fable availability and limits on Pro are unverified. The driver doesn't depend on it.
- Only one night: maybe ~12–20 proxy runs + 3 full runs on 4 GPUs (estimate from the 3 min/epoch baseline; smaller models aren't
  much faster if the CPU is the bottleneck).
- A shrunken sign model may barely change total `sign_speaker` memory on the board (MediaPipe trackers likely dominate; not measured).

## Log
- 23:03 quota request 16 vCPU  CASE_OPENED. Instance choice switched to g5 (A10G) for TF 2.14/CUDA 11.8 safety: g5.4xlarge $1.624/h, g5.2xlarge $1.212/h (AWS pricing API).
- Fake-data finding: int8w of the *baseline* alone = 35% of its float32 runtime memory (20.1 → 6.9 MB, tf.lite on Mac).
- 23:19 launched [instance] g5.2xlarge (8 vCPU quota; 16 still pending). Auto-shutdown 16:00 UTC = 11:00 EST, stop not terminate.
- Data: 94,477 sequences, 3.57M frames, 0 skipped. Split: baseline per-signer top-1 gap → val = participants 29302, 49445, 27610 (13,965 clips; 59–72% vs 83–98% for the rest).
- Harness, baseline float32: top-1 0.6708, top-5 0.8574, 7.91 MB, 21.7 MB runtime. int8w: 0.6717, 2.07 MB, 9.2 MB (42%) → ≤50% goal met by quantization alone. Target raised to ≤25%.
- Bugs fixed on the box: CUDA requirement parsing; eager per-batch inference leaked GPU memory (→ model.predict).
- 23:45 search started; 74 s/epoch at dim 96. Fable not logged in on the box (rc=1) → my queue.jsonl of 6 proxies + 5 full 60-epoch runs (~9 h est).
- 03:12 EST user moved buildings; SSH rule swapped to each new IP (campus IP rotates).
- XNNPACK finding: default delegate keeps its own copy of the weights. int8w-plain added to the harness (scored on full val).
- Long runs (int8w-plain, KD): d96-s1 100 ep 0.6029 @5.4 MB (25%); d128 80 ep 0.6481 @6.3 MB (29%). Both short of 0.6608.
  Params→top-1: 333k 0.603, 880k 0.648, 1.84M 0.672. ~1.1–1.3M likely needed. Untested: d192-lean (stopped at epoch ~1), d160.
- 04:08 EST stopped at the user's request. **Winner: baseline int8w-plain: 0.6716 top-1, 7.4 MB (34% of 21.7 MB).**
  Deliverable: ~/Downloads/asl-best-model (model, patched sign_infer_lite.py with SIGN_NO_XNNPACK, board_check.py, run_mac.py, USAGE.md).
  Results copied to results_overnight/. Instance [instance] stopped (disk kept).
- [ ] 8. Board test (user): USAGE.md steps 1–4. Open risk: int8w ~5x slower than float+XNNPACK on the box.
