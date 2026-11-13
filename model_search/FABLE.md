# Brief for the overnight search agent

You are steering an architecture search for a smaller ASL sign-recognition model. A driver script (`search.py`) does
all training and scoring; you are woken between trials to decide what runs next. You cannot run commands.

## Goal
Find the export with the **least runtime memory** (`mem_delta_kb` in results.jsonl) whose validation top-1 is within
**1.0 point** of the baseline row (`"id": "baseline", "export": "float32"`: top-1 0.6708, 22172 KB), i.e. top-1 ≥ 0.6608.
Every trained model gets three rows: `float32`, `int8w` (8-bit weights, default XNNPACK runtime) and `int8w-plain`
(same file with XNNPACK turned off; scored separately because the int8 arithmetic differs). The board can use either runtime.
Measured at 05:25 (operator, repeated readings agree): XNNPACK keeps its own copy of the weights, so `int8w-plain` uses less memory.
Baseline int8w-plain ≈ 7.2 MB (33%), p-d96-kd ≈ 5.6 MB, m-d96-s1 ≈ 5.2 MB, smallest legal model ≈ 4.6 MB (new floor).
**Target ≤ 25% (≤ 5543 KB) is reachable again** if a d96-class model can be trained to top-1 ≥ 0.6608. Full 8-bit activations
were tried by the operator and broke accuracy (44–78% agreement), so they are not an option.

## What you may change
- `queue.jsonl`: one JSON object per line. The driver runs the first line whose `id` hasn't run yet, top to bottom,
  then falls back to `fallback_queue.jsonl`. Reorder, add or delete lines. Never reuse an id.
- `notes.md`: your reasoning, so the next wake-up (a fresh session with no memory) and the humans can follow it. Keep it short;
  append a dated section each time.
Do not try to change any other file. Changes to other files are detected, reverted, and switch you off.

## Config format (validated; invalid lines are logged in failures.jsonl and skipped)
`{"id": "...", "note": "...", "arch": {...}, "epochs": N, "distill": {"alpha": 0.5, "T": 2.0} | null, "lr": 5e-4,
"wd": 0.05, "batch": 64, "warmup": 0.1, "label_smoothing": 0.1, "seed": 42, "max_train": null}`
- `arch` keys (baseline values): dim 192 (32–256, divisible by heads), stages 2 (1–3), convs 3 (1–4, conv blocks per
  stage), ksize 17 (3–31), expand 2 (1–4, transformer MLP), heads 4 (1–8), conv_expand 2 (1–4), top_mult 2 (1–2).
- `epochs` 1–120. `distill` uses the baseline as a frozen teacher. `max_train` subsamples training sequences (faster proxies).

## Facts that matter
- The baseline was trained for 60 epochs. Short proxy runs (8 epochs) score well below the baseline. Compare proxies with each other and with
  `p-d192-ctrl` (the baseline architecture on the same short recipe). The gap between `p-d192-ctrl` and the baseline estimates what
  more epochs buy.
- Training speed is mostly limited by the CPU data pipeline, so small models aren't much faster per epoch. Check
  `runs/*/summary.json` `sec_per_epoch`. The driver skips any trial that can't finish before the deadline, so budget the night:
  a winner needs a full-length run (≈40+ epochs) to finish before the deadline. Start the full runs no later than halfway through the night.
- Runtime memory grows with activations (dim × 128 frames × layers) and weights; int8w roughly quarters the weight part.
- Validation: 13,965 sequences from 3 signers the baseline never saw; ±0.4 points is noise.
- Scoring a candidate takes several minutes (TFLite batch-1 on 14k clips, three times). It counts against the clock.
