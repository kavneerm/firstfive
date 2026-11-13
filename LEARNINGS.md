# What we learned: smaller sign model, and the UNO Q sign program

Work done 2026-11-11 to 2026-11-12. Numbers are measurements unless marked as estimates.

## 1. Result in one table

Accuracy = top-1 on 13,965 Kaggle asl-signs clips from 3 signers the model never trained on.
Memory = what the sign model alone adds to a process (tflite-runtime 2.14, x86 EC2 box; not re-measured on the UNO Q).

| Model | Top-1 | Top-5 | File | Runtime memory |
|---|---|---|---|---|
| Deployed float32 model, default runtime | 67.08% | 85.74% | 7.91 MB | 21.7 MB |
| Same model, 8-bit weights | 67.17% | 85.93% | 2.07 MB | 9.2 MB (42%) |
| **Same model, 8-bit weights, XNNPACK off (chosen)** | **67.16%** | 85.92% | 2.07 MB | **7.4 MB (34%)** |
| Best newly trained model (width 128, 80 epochs, distilled) | 64.81% | | 1.06 MB | 6.3 MB (29%) |
| Width 96, 1 stage, 100 epochs, distilled | 60.29% | | 0.45 MB | 5.4 MB (25%) |

**No retraining was needed for the win.** Storing the existing model's weights as 8-bit numbers and turning off TFLite's
XNNPACK delegate cut memory to a third at the same accuracy. No smaller trained model got within 1 point of the original.

## 2. Model and memory findings

- **8-bit weights (dynamic-range quantization) cost the full-size model nothing** (67.08 → 67.17%). Small models lose more
  (0.1–0.7 points).
- **XNNPACK keeps its own repacked copy of the weights.** Turning it off
  (`experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES`) saved ~1.8 MB on the 8-bit model and ~9 MB
  on the float model. Accuracy must be re-scored without it, because the 8-bit arithmetic differs slightly (it held).
- **Full 8-bit quantization (activations too) broke the model**: only 44–78% of predictions matched float32. Not an option
  without quantization-aware training.
- **There is a floor of about 4.8 MB** (XNNPACK off; 5.6 MB with it): even a 90 KB model costs that much to load and run. So
  "25% of the original" (5.5 MB) sits right at the floor.
- **With XNNPACK off, memory tracks parameter count**, roughly `4.8 MB + 1.2–1.7 × 8-bit file size`. Width alone barely moved
  memory while XNNPACK was on; layer count mattered more there.
- **Accuracy vs parameters** (8-bit, distilled, long runs): 333k → 60.3%, 880k → 64.8%, 1.84M → 67.2%. Estimate: about
  1.1–1.3M parameters are needed to stay within 1 point, which would save only another 0.5–1 MB. Untested: width 160, and
  width 192 with slimmer blocks.
- **Speed is the cost of 8-bit weights**: ~67 ms per sign vs ~13 ms for float32 + XNNPACK on the EC2 box; 66–82 ms on the UNO Q.
  Fine for one inference per sign.
- **67% is the honest accuracy on new signers**, far below the ~89% Kaggle leaderboard figure, which is not signer-held-out.
- Memory readings vary ±0.2–0.6 MB between runs. Treat smaller differences as noise.

## 3. Training and search method

- **Finding the baseline's held-out signers without its training split:** score the baseline on all 21 signers. The 3 it never
  saw scored 59–72%; the other 18 scored 83–98%. Those 3 became the validation set, so old and new models are compared fairly.
- **A fixed harness scores everything** (`model_search/eval_candidate.py`): TFLite export, batch-1 accuracy on the full
  validation set, file size, memory in a fresh process. The agent steering the search could only edit the queue and its notes.
- **An agent steering a deterministic driver worked well** (`search.py`): it ran one-epoch "memory probes" first (memory doesn't
  depend on training), found the floor, and dropped an unreachable target on its own. Waking it once an hour instead of after
  every trial cut its usage with no loss.
- Short 8-epoch proxies understate long runs: the baseline architecture went 55.3% (8 epochs, our recipe) → 67.1% (60 epochs).
- Speed on one A10G with 8 vCPUs: 36–64 s per epoch on 80k clips. The whole night cost about $6 of EC2.
- Preprocessing must match the board exactly. Storing landmarks as float16 changed features by 8e-3; float32 gives 5e-7.
- TF 2.14 gotchas: an eager per-batch inference loop leaked GPU memory until it ran out (use `model.predict`); keep the
  landmark array on the CPU; `tensorflow[and-cuda]` can't install because of its TensorRT pin, so install the listed CUDA
  wheels without it.

## 4. Running on the UNO Q

- **The board runs one program, `~/sign_app`** (source: `sign_app/` here, plus `sign_speaker.py`, `demo_webcam.py`,
  `fast_landmarks.py` from the repo root): the 8-bit model, MediaPipe Holistic, the same display as the Mac demo, spoken words
  at ≥ 40% confidence. It starts at power-on as the system service `sign-app`. See `sign_app/USAGE.md`.
- **Use the board's name, not its IP**: `ENGS21.local` (SSH and `http://ENGS21.local:5000/`). The IP changes with every network.
- **One Holistic instance uses ~1.3 of 4 cores**: ~225 ms per frame, 4.5 fps, and its pose-quality setting makes no difference.
- **Three Holistic processes fed frames in turn** (`parallel_holistic.py`): 14 fps with nobody in view (the camera's rate),
  7–11 fps with a person, ~85% CPU. **Four collapse to ~5 fps**: MediaPipe degrades badly when cores are oversubscribed.
- **Open question:** with the LCD face check and the browser preview both active, a person in view dropped the rate to 3.4–4 fps.
  Likely the same oversubscription; the A/B test (`--no-lcd`, `--lcd-face-every 15`, `--no-preview`, `--workers 2`) was not run.
  It was not heat: 58 °C, all cores at full clock speed.
- **MediaPipe imports TensorFlow when it is installed**, though Holistic never uses it: ~300 MB and ~19 s per process. Setting
  `sys.modules['tensorflow'] = None` before importing mediapipe avoids it (workers: 275 MB each instead of ~700 MB).
- Whole program memory: ~0.7 GB with one tracker, ~1.3 GB with three. The sign model's 7 MB is a small part of that.

## 5. Grove LCD (RGB Backlight v5) position messages

- States: in position / no one in view / not in position (off-centre, face cut off) / too close / too far / too many people.
  Logic in `sign_app/position.py`, tested by `sign_app/tests/test_position.py`.
- Too close / too far use shoulder width as a fraction of the picture (defaults 0.55 and 0.18, **not tuned**). "Too many people"
  counts faces with MediaPipe's face detector, because Holistic only ever tracks one person.
- The LCD is on the microcontroller's I2C bus, so Linux can't drive it directly. The sketch `sign_app/lcd_status` takes text
  lines from Linux over the router's monitor port (`127.0.0.1:7500`), which needs no root. The router's RPC socket needs a
  group the `arduino` user is not in.
- **v5 quirk: send one character per I2C message.** A whole line in one message loses every second character.
  Text chip at 0x3E, backlight chip at 0x30 (registers 6/7/8 = red/green/blue).
- Uploading a sketch from the board itself needs `--discovery-timeout 60s`, or the network port isn't found.
- Not yet confirmed by eye: the fixed text and the backlight colours.

## 6. Operations lessons

- A process suspended with Ctrl+Z ignores SIGTERM and keeps the camera and port; send SIGCONT first or use SIGKILL.
- `pkill -f name` over SSH also matches the SSH command itself. Anchor the pattern to the program's full path.
- Starting a new copy before the old one has released port 5000 crashes the new one.
- A virtualenv's `python` can lose its executable bit when files are copied; `chmod +x` fixes "Permission denied".
- Campus Wi-Fi changed the laptop's public address several times an hour, which broke a single-address firewall rule.

## 7. Still open

- Confirm the LCD text and colours; tune the near/far limits with a person at real distances.
- Run the frame-rate A/B test with a person in view.
- Measure the sign model's memory and the 8-bit speed on the UNO Q itself (`sign_app/board_check.py`).
- Check accuracy on our own signing: about 20 signs × 2 takes, old model vs 8-bit model.
- Optional: install `tflite-runtime` in the board's environment so the main process doesn't load TensorFlow.
