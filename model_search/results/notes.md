# Search notes

## 2026-11-12 04:58 (first real wake; the 04:44 wake failed with "Not logged in")

**Data so far**
- baseline: float32 0.6708 / 22172 KB; int8w 0.6717 / 9400 KB.
- p-d96-kd (8 ep, KD): 0.5115; float32 11680 KB, int8w 7676 KB. int8w costs no accuracy (0.5112).
- Timing: 56 s/epoch at d96 (80.5k train seqs); eval ~6 min. So 8-ep proxy ≈ 14 min, 60-ep full run ≈ 63 min
  (driver estimates 72). 631 min left at 04:58 ≈ 9–10 full-run equivalents.

**Main finding: dim alone cannot reach the target.** Halving dim (192→96) only moved int8w memory 9400→7676 KB.
Fitting mem = a + slope × file size across the two models gives a ≈ 6.9 MB constant for *both* exports
(float32 slope ≈ 2.0, int8w ≈ 1.3). Extrapolated int8w at d48 ≈ 6.8–7.2 MB, still above 5540 KB.
Two points only, so this is a fit, not a measurement — the constant could be per-layer/op overhead,
ksize, attention, or genuinely fixed (input 128×708, interpreter). Unknown which.

**Plan**
1. Memory map first (queue ids `m-*`): 1-epoch runs, ~7 min each. Memory doesn't depend on training, so
   their accuracy is meaningless — read only `mem_delta_kb` (int8w). `m-min` = smallest legal model = floor.
   If `m-min` int8w > 5540 KB the 25% target is **unreachable in this search space**; say so plainly and
   switch to "least memory with top-1 ≥ 0.6608". The `m-d96-*` probes change one knob vs p-d96-kd (7676 KB).
   (Used full data rather than max_train: a max_train run would corrupt the driver's sec/epoch estimate for that dim.)
2. `p-d192-ctrl` to calibrate 8-ep proxies (p-d96-kd = 0.5115 means nothing until this is in).
3. Then 8-ep KD proxies only on shapes the memory map says are worth it; rewrite the `p-*` lines accordingly.
4. Full runs (60+ epochs, KD) should start by ~08:00–09:00; ~6 fit after that. Consider 90–120 epochs for the
   smallest candidates if proxies show them short on accuracy. Current `f-*` lines are placeholders/safety net.

**Next wake: to do**
- Each wake costs clock time; be quick. During the `m-*` block only act if `m-min` settles the floor question.
- After the m-block: tabulate int8w mem per knob here, delete placeholder p-/f- lines that the map rules out.

## 2026-11-12 05:05 (after m-min)

**RESULT: the ≤ 25% target (≤ 5540 KB) is not reachable in this search space.**
`m-min` (every arch knob at its legal minimum: d32, 1 stage, 1 conv, k3, expand 1, heads 1, 40.8k params)
measures **5764 KB int8w** (6244 KB float32) = 26.0% of baseline, and it is an untrained-quality model (top-1 0.044).
Any model that can actually classify sits above that. So no row tonight can honestly meet "≤ 25%"; the
shortfall is the harness's ~5.7 MB constant, not the architecture. Caveats: one measurement, and I don't know the
run-to-run repeatability of `mem_delta_kb` — a later row dipping under 5540 KB on a near-minimum model would be
measurement noise, not a win. Getting under 25% needs something outside my knobs (fewer input frames/features,
full int8 activations, or a different memory accounting) — a decision for the humans.

**Revised goal for tonight:** least int8w memory with top-1 ≥ 0.6608. The usable range is narrow:
5764 KB (floor) … 7676 KB (d96 baseline shape) … 9400 KB (baseline int8w). Everything above the floor at d96 is
only ~1.9 MB, so the realistic prize is roughly 27–35% of baseline, and accuracy is now the binding question.

**Timing update:** m-min trained at 28 s/epoch and its whole trial took 3 min (driver estimated 11); p-d96-kd took
14 min (estimated 45). Driver estimates are very conservative — late-night trials may be skipped earlier than needed.

**Queue now:** remaining m-probes (≈5 min each, read int8w `mem_delta_kb` only) → p-d192-ctrl → p-d64-kd, p-d96
→ `f-d96-kd` (first full run, moved up: tells us whether d96 can hold accuracy at all) → smaller proxies/full runs.

**Next wake: to do**
- After the m-block, tabulate int8w mem per knob here and pick shapes by KB saved per knob; drop f-/p- lines that
  save < ~200 KB vs a simpler neighbour (not worth an accuracy risk).
- `summary.json` reports `best_val_top1_keras`/`best_epoch`: if the exported checkpoint is the best epoch on the
  val signers, final top-1 is slightly optimistic (selection on the scoring set). Mention this with any winner
  that clears 0.6608 by less than ~0.4.

## 2026-11-12 05:12 (after m-d48; 620 min left)

**Data:** m-d48 (baseline shape at dim 48, 169k params): int8w **7152 KB** (32.3%), float32 8540 KB.

| model (int8w) | params | mem KB | % of 22172 |
|---|---|---|---|
| m-min (all knobs at minimum) | 41k | 5764 | 26.0 |
| m-d48 | 169k | 7152 | 32.3 |
| p-d96-kd | 531k | 7676 | 34.6 |
| baseline | 1837k | 9400 | 42.4 |

**Reading it with the operator's noise figure (0.2–0.6 MB run to run, section below):**
- d96 → d48 saves 524 KB = inside the noise. Dim below 96 buys essentially nothing measurable, and costs accuracy.
- d48 → m-min saves 1.4 MB although it removes only 128k params, so the remaining lever is structure
  (stages/convs = op count), not width. That is the one thing still worth probing.
- m-min's 5764 KB is only 224 KB above the 5540 KB line, i.e. within noise of it. My 05:05 statement stands in
  substance (no model that can classify will sit below m-min), but "26.0% vs 25%" is not a resolved difference;
  a single reading near the line proves nothing in either direction.
- Consequence for picking a winner: memory differences among d48/d64/d96-type models are not resolvable from single
  rows. Among candidates within ~0.6 MB of each other, prefer the more accurate one and report the memory as a tie.

**Queue changes:** dropped m-d96-ce1, m-d96-k3, m-d96-h1e1t1 (weight-only knobs; the whole int8w weight file at d96
is 0.68 MB, so they cannot move memory by more than noise). Kept m-d96-s1 and m-d96-c1 (≈5 min each). Dropped
p-d128-c2e1-kd proxy. Then p-d192-ctrl → p-d64-kd → p-d96 → f-d96-kd.

**Next wake: to do**
- If m-d96-s1 or m-d96-c1 is > 0.6 MB below 7676 KB, add an 8-ep KD proxy of that shape (and a d128/d192 version:
  if structure is the lever, a wide-but-shallow model may be both smaller and more accurate than a narrow deep one).
  If neither is, stop probing memory; the answer is "≈ 7–7.7 MB (32–35%), pick by accuracy".
- Full runs must be going by ~08:00. A 60-ep run is ≈ 63 min real (driver estimates 72); ~8 fit after 06:15.
- f-d96-kd is the first full run because d96 is the most likely to hold ≥ 0.6608; only go smaller if it clears
  with margin, otherwise go d128 / more epochs.

## 2026-11-12 05:13 (driver restarted at 05:11:39; 618 min left)

- The restart cut off m-d96-s1 at the start of epoch 1 (train.log stops at "steps/epoch"; there's no summary.json or result).
  It's still first in the queue, so it runs again. If m-d96-s1 shows up in skipped/failures, check this first.
- No new results. The 05:05/05:12 conclusions still hold: ≤25% can't be reached (m-min floor 5764 KB, within noise
  of the line); in practice the goal is least memory with top-1 ≥ 0.6608.
- Queue change: moved p-d64-kd to after f-d96-kd. By the 05:12 table d64 would be ≈7.3 MB vs 7676 KB for d96, which is
  inside the 0.6 MB noise, so its proxy only matters if f-d96-kd clears 0.6608 with margin. That gets the first
  full run going at ≈06:20 instead of ≈06:35.
- Budget: m×2 (~10 min) + p-d192-ctrl (~16) + p-d96 (~14) → f-d96-kd starts ~06:05–06:20, done ~07:20.

## 2026-11-12 06:12 (driver restart; 558 min left; deadline 15:30)

**New: `int8w-plain` rows (XNNPACK off) are now the lowest-memory export.** Official rows (results.jsonl):

| model | int8w KB | int8w-plain KB | plain % of 22172 |
|---|---|---|---|
| baseline | 9400 | 7552 | 34.1 |
| p-d96-kd (2 stages) | 7676 | 5932 | 26.8 |
| m-d96-c1 | 6808 | 5592 | 25.2 |
| m-d96-s1 (1 stage) | 6596 | **5536** | 25.0 |
| m-d48 | 7152 | **5416** | 24.4 |
| m-min (floor) | 5764 | **4924** | 22.2 |

- Supersedes my 05:05 "unreachable" call: with plain runtime ≤25% (≤5543 KB) is possible in principle.
- Caveat for the humans: the brief's 05:25 operator figures (baseline 7.2 MB, p-d96-kd 5.6, m-d96-s1 5.2, floor 4.6)
  sit ~0.3 MB *below* the official rows. By the official rows, d96-s1 beats the line by only 7 KB and d48 by 127 KB,
  well inside the 0.2–0.6 MB run-to-run noise. A "≤25%" claim on these shapes would need repeated readings.
  Full 2-stage d96 (5932) is above the line on either set of figures.
- Accuracy is the real question. p-d192-ctrl (8 ep) = 0.5533 vs baseline 0.6708 → 8→60 ep bought +11.8 pts at d192.
  p-d96-kd (8 ep) = 0.5115, 4.2 below ctrl; if that gap holds, d96 @60 ep ≈ 0.63, short. So use longer runs (90–100 ep).
- p-d96 (no-KD proxy) was in progress at the restart, with no run dir and no result. **Dropped it**: committing to KD
  (alpha 0.5, T 2) for all full runs without testing, to save 15 min. Risk: if KD hurts, every full run is affected.
- Timing (sec/epoch): d192 63, d96 2-stage 56, d96 1-stage ~36 (1-ep run, overheads included), m-min 28.

**Queue (proxies skipped, straight to full runs):**
1. f-d96-s1-kd, 100 ep: target candidate. ~60–70 min + eval, done ≈07:30.
2. f-d96-kd, 90 ep: safe candidate (26.8%). ~85 min + eval, done ≈09:05.
3. f-d128-s1-kd, 100 ep, then f-d64-kd, 100 ep: backups. Done ≈11:45. Leaves ~3.5 h for 1–2 more runs chosen from results.
Deleted the old placeholder p-/f- lines (d48/d64/d80/d128-c2e1 proxies).

**Next wake: to do**
- After f-d96-s1-kd: if ≥0.6608, it's the winner. Note best_epoch/selection-on-val (see 05:05) and the memory-noise
  caveat. Then go to f-d64-s1 / d80-s1 only if the margin is ≥0.5 pts. If it's 0.64–0.66, put a 120-ep d96-s1
  (new id) or f-d128-s1-kd ahead of f-d96-kd. If it's < 0.62, 1-stage is out: drop f-d128-s1-kd and keep f-d96-kd.
- Last full-length start ≈13:30 for a 100-ep 1-stage run; the driver's estimates run high, so leave margin.

## 2026-11-12 07:30 (after f-d96-s1-kd; ~480 min left)

**f-d96-s1-kd (d96, 1 stage, 333k params, 100 ep KD): fails.** float32 0.6095, int8w / int8w-plain **0.6029**
(−5.8 vs the 0.6608 bar). Plain mem 5552 KB (m-d96-s1 read 5536 for the same arch, so the readings agree; 9 KB over the line).
best_epoch 97/100: it had plateaued, so more epochs won't close 6 pts. Int8 costs this small model 0.66 pts
(agreement 0.931), but the baseline lost nothing. **Judge every candidate on its int8w-plain row.**
→ 1-stage is out (the 06:12 rule for results < 0.62). **The ≤25% target is effectively unreachable tonight**: hitting it needs
≤ ~330k params, and the 333k model is 6 pts short.

**What memory tracks under the plain runtime:** weights, roughly 1:1, not activations. Rows fit plain mem ≈ 4.8 MB + ~1.2–1.7 × int8 file size
(baseline 2.07 MB file→7552, d96 0.68→5932, d96-s1 0.45→5536). So the lever is params, and width can stay if the weights shrink.
Rough log-param fit (333k→0.603, 1.84M→0.672) says ~1.2–1.4M params would be needed. KD and longer training may lower that. Expect a
winner around 29–31% (6.4–6.9 MB), unless something smaller surprises.

**Queue (full runs only, 80 ep KD, ~85–90 min each incl. eval):**
1. f-d128-kd (~0.86M est), done ≈09:00. 2. f-d192-lean-kd (ce1/e1/tm1, ~1.13M est), done ≈10:30.
3. f-d160-kd (~1.28M est), done ≈12:00. 4. f-d96-kd 90 ep, done ≈13:35, conditional.
Deleted f-d128-s1-kd and f-d64-kd. Fallback winner if all fail: **baseline int8w-plain 0.6716 / 7552 KB (34.1%)**.

**Next wake: to do**
- After f-d128-kd: if ≥ 0.665 plain, move f-d96-kd up, maybe add d112, and drop f-d160-kd. If 0.655–0.665, keep the order.
  If < 0.645, drop f-d96-kd and consider d176 / d192-lean variants.
- Param counts here are estimates; check the `params` field in results to refine the KB/param fit.
- Last full-run start ≈13:45 (80 ep at ~60 s/ep + eval ≈ 90 min, deadline 15:30).

## 2026-11-12 09:05 (after f-d128-kd; ~385 min left)

**f-d128-kd (d128, 2 stages, 880k params, 80 ep KD): fails narrowly.** int8w-plain **0.6481** (−1.3 vs 0.6608), plain mem 6424 KB (29.0%).
float32 0.6491, so int8 costs only 0.1 pts here. best_epoch 79/80, but that's expected with a decaying LR, so it doesn't prove more epochs help.
64 s/epoch; the trial took 93 min including eval (the driver estimated 103).

**About the brief's "≤25% reachable with a d96-class model":** the data say no for tonight. The ≤25% line needs a
d96 *1-stage* shape (plain 5536–5552 KB). The 2-stage d96 is 5932 KB official (5.6 MB in the operator's reading),
which is over the line either way. d96-s1 at 100 ep reached 0.603, and d128 2-stage at 80 ep reached 0.648, so a d96 2-stage
would land around 0.62–0.63 and still be over the line. **Dropped f-d96-kd.** No row tonight will honestly meet ≤25%.

**Accuracy vs params (plain top-1):** 333k→0.603, 880k→0.648, 1.84M→0.672 (60 ep, no KD). Interpolating, 0.6608 needs ≈1.2–1.3M params.
**Mem vs file (plain):** d128 1.06 MB→6424, baseline 2.07→7552 (≈1.1 KB of memory per KB of file). So a winner probably sits at
≈6.7–6.9 MB (30–31%), only 0.6–0.9 MB under the baseline's 7552. That margin is near the 0.2–0.6 MB noise, so report it with that caveat.

**Queue (80 ep KD, ~93 min each):** f-d192-lean-kd (done ≈10:40) → f-d160-kd (≈12:15) → f-d144-kd placeholder (≈13:50).
A 4th full run won't fit: the driver's estimate (~103 min) from 13:50 goes past 15:30, so it would be skipped.

**Next wake: to do**
- After d192-lean: if ≥0.6608 plain, d160 is pointless (bigger). Replace it with something smaller in the same "lean" direction
  (e.g. d160 with conv_expand 1 / expand 1 / top_mult 1, or d192-lean with convs 2). If it's 0.650–0.66, keep d160.
  If it's < 0.650, the lean cuts hurt more than width: keep d160 and make slot 3 d176.
- Slot 3 must start by ~13:45. A d128 120-ep run (same 6424 KB, ~140 min) only fits if it starts by ~12:30. It's an option only if
  someone thinks epochs, not params, are the limit; d96-s1's plateau suggests params are the limit.
- Any winner that clears by < 0.4 pts: flag selection-on-val (best_epoch is picked on the scoring signers) and memory noise.
- Fallback winner stays **baseline int8w-plain 0.6716 / 7552 KB (34.1%)**.

## 05:10 note from the operator (Claude in the user's session), not Fable
- Memory probe noise, measured: baseline float32 read 22172 KB in the official row and 22796 KB on a re-run; other models vary
  about 200-600 KB between runs. Treat mem differences under ~0.6 MB as noise; near the 5540 KB line one reading can't settle it.
- The probe also counts ~0.2 MB for its own test input (same for every model). Harness unchanged so all rows stay comparable.
- The XNNPACK delegate (default in tflite-runtime, also on the board) is part of the fixed per-process cost; that is real on the UNO Q too.
