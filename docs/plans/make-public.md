# Make the repo ready to share publicly + re-run Cole's evaluation on the new model

Repo: github.com/kavneerm/firstfive (already public). Branches done 2026-11-19:
`main` = newest work + Cole's upload (f31b000), `old-main` = first version (5c53eb9).
`smaller-model-and-board-app` still exists (same commit as main) — delete only on the user's OK.

Scan result: no passwords, tokens, AWS IDs or home-folder paths in any commit. Board name `ENGS21.local` and SSH
alias `unoq` in `sign_app/USAGE.md` / `LEARNINGS.md` are harmless; kept.

User said "go" (2026-11-19): delete Cole's old charts, run his evaluation on the new model. No license named → MIT
(the option offered), stated in the reply.

## Facts that shape the evaluation
- Cole uploaded only outputs (report.txt, CSVs, 13 PNGs), no script. Not in his GitHub account either. So the
  script is rewritten here from his outputs' structure: same metrics, same 13 charts, same CSV columns.
- His test set "fold 0" = signers 29302, 49445, 27610, 62590 (18,528 clips; his per-signer n's sum to that).
  The new model `sign_app/sign_model_int8w.tflite` is the same trained network with 8-bit weights (LEARNINGS.md §1,
  no retraining), so fold 0 is unseen by it too → score it on the same 4 signers; directly comparable to his 70.97%.
  Also report the 3-signer subset (13,965) to tie back to the 67.16% in LEARNINGS.md.
- Inference goes through `sign_app/sign_infer_lite.py` with `SIGN_NO_XNNPACK=1`, i.e. the board's exact path.
- Full data is only on the stopped EC2 instance; the Mac has a 27-clip real sample (`asl-small-model/data/real_mini`).

## Steps
- [x] Delete Cole's 18 files from the repo root (git rm; still in history at f31b000).
- [x] `evaluation/evaluate.py`: model + data dir + split → predictions.npz, report.txt, summary.json,
      figure_stats.json, per_sign_accuracy.csv, top_confusions.csv, 13 charts. `--from-predictions` redraws without inference.
- [x] `evaluation/test_evaluate.py`: metrics checked against brute-force on synthetic logits; end-to-end on real_mini
      with predictions cross-checked against `LitePredictor` clip by clip.
- [x] README.md, LICENSE (MIT), .gitignore for data/zip/venv/eval outputs.
- [x] Full run on fold 0 (18,528 clips; needs EC2 or a Kaggle download — user's call; starting EC2 was blocked by the
      permission check 2026-11-19), commit results to
      `evaluation/results/int8w/`.

## Checks
- `python evaluation/test_evaluate.py` → ALL PASS.
- Existing tests unchanged: test_demo_parts ALL PASS, test_position ALL PASS; test_buffered_loop already FAILS
  before any change (missing `models/palm_detection_full.tflite`, not tracked).
- Bin counts don't depend on the model, so they must equal Cole's exactly (if not, his length / hand-share
  definitions differ, e.g. counted before empty frames were dropped): clip length 2235/4260/5357/3920/1701/1055,
  hand detection 2010/4994/4413/3492/3619.
- 3-signer subset: `--from-predictions <npz> --participants 29302,49445,27610`.
- Known display-only difference: tie order among equal-count mix-ups differs from Cole's (his rule unknown).
- Cross-checks on the full run: 3-signer subset top-1 = 67.16% (LEARNINGS.md, same model, same path); 4-signer
  top-1 within ~0.2 points of Cole's 70.97% for the float model (8-bit weights moved it +0.09 on the 3 signers).

## Result (2026-11-19, c6i.2xlarge — no g5 capacity; type restored to g5.2xlarge, instance stopped, SSH rule removed)
- 18,528 clips, 0 unusable, 1070 s. Top-1 70.93% (Cole, float: 70.97%), top-5 88.22% (88.10%), calibration 6.29 (6.34).
- Per signer within 0.4 points of Cole's, same clip counts. 3-signer subset 67.15% vs harness 0.6716 (≈1 clip; the
  harness preprocesses in TF, this path in numpy).
- Clip-length and hand-detection bin counts differ slightly from Cole's (e.g. ≤10 frames 2248 vs 2235; totals equal):
  here length/hand share count frames after prepare_kaggle.py dropped all-empty frames; his pipeline evidently
  counted differently. His script is unavailable, so left as is and noted.
