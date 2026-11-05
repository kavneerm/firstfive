# Sign-to-speech on the Arduino Uno Q

Full plan: ~/.claude/plans/in-this-folder-i-synchronous-hennessy.md

Camera → MediaPipe Holistic → AutoSegmenter → model → speak the top-1 word (if p ≥ threshold) on the USB
speaker, with an MJPEG preview viewed from the laptop browser.

## Tasks
- [x] Copy model + demo files to `~/sign_demo/` on the board
- [x] Run `setup_uno_q.sh` (Python 3.11 venv at `~/demo_env`, TF 2.14.1, MediaPipe 0.10.14) — needed uv binary (no python3-venv) and jax/jaxlib/ml-dtypes pins
- [x] Write `sign_speaker.py` (camera by-id, Speaker based on `~/speak.py`, threshold/cooldown, stdlib MJPEG preview, `--video`, `--test-speaker`, `--benchmark`)
- [x] Copy `sign_speaker.py` to the board

## Verification
- [x] 1. Install: setup step 4 prints versions with no `!!`; dry run prints `model loaded: 250 classes` — passed after pinning
- [x] 2. Speaker: aplay exit 0 and user confirmed hearing "hello" (volume raised 40% → 90%)
- [ ] 3. Speed: FAILS target — Holistic ~5.6 fps (176–221 ms/frame); model+preprocess 737 → 116 ms after tf.function (outputs identical). Judge impact via live + clip tests
- [ ] 4. Fixture clips: recorded `<word>.mp4` clips → PASS/FAIL table, top-1/top-5 count; decide `--flip`
- [ ] 5. Live end to end: 3 signs spoken, preview shows REC + top-5
- [ ] 6. Fresh-subagent review against this plan

## Open assumptions
- `--max-len 128` equals training `CFG_MAX_LEN` — confirmed by the user

## Phase 2: parallel custom pipeline + lean deployment (approved 2026-11-03)
Why: ~3–4 fps with a person in view (Holistic ~270 ms/frame; split Hands+FaceMesh ~264, no gain); cores only
16–40% busy at max 2016 MHz; peak RSS ~704 MB (TF). Split landmarker gives near-flat guesses (2–8%).
New dependency approved: `tflite-runtime==2.14.0` in a slim venv `~/sign_env`.

- [x] 0. Stop live sign_speaker (holds the camera)
- [ ] 1. Record 3–5 fixture clips with record_clips.py (user signs) — NOT DONE YET
- [x] 2. convert_sign_model.py → sign_model.tflite (7.9 MB), builtins only — TF can be dropped
- [~] V1 Keras vs TFLite: PASS on 6 random clips (max 1.4e-05, same top-5); fixtures still to run
- [~] V2 numpy vs TF preprocess: PASS on random clips (max 6.9e-06); fixtures still to run
- [x] 3. fast_tracker.py written. Subgraph params are compiled into MediaPipe (not in files) → from MediaPipe source, verify via V3.
      Probed on real frames: palm in [0,1], hand in [0,1], face_det in [-1,1]; hand Identity_1 = presence, Identity_2 = handedness.
      Design change (measured): each model uses all 4 threads sequentially (2 hands seq 96 ms < parallel 106 ms).
- [ ] V3 custom vs Holistic landmarks: mean xy < 0.01, same-slot ≪ swapped, ≥ 90% detection agreement
- [ ] V4 model top-1 on custom == on Holistic per clip
- [x] 4. slim venv ~/sign_env (178 MB; numpy, opencv, tflite-runtime). sign_speaker defaults: --landmarker custom, sign_model.tflite
- [ ] V5 ≥ 8 fps active with signer in view; idle CPU% reported
- [ ] V6 peak RSS vs 704 MB (target < 300 MB)
- [ ] V7 check_clips top-1/top-5: custom vs holistic, flip on/off, resample on/off
- [ ] V8 live: 3 signs spoken
- [ ] Fresh-subagent review against the plan

## Status at pause (2026-11-03)
- Measured so far: TF-free sign model 54–90 ms/sign, peak RSS 79 MB (model alone); custom tracker benchmark 211 MB peak,
  but that run had NO person in view (camera pointed at ceiling lights; camera auto-exposure fell to 11.7 fps) → V5/V6 must be re-run.
- Model times (tflite-runtime, 4 threads): palm_full 75 ms, hand_full 48, hand_lite 23, face_det 7, face_mesh 10.
- Next session: aim the camera at face height → live check via preview → V5/V6 benchmark (full vs lite) → record clips →
  V3/V4 (compare_landmarkers.py --landmarker custom, in ~/demo_env) → V7 (check_clips.py) → V8 live → subagent review.
- Run live: `cd ~/sign_demo && ~/sign_env/bin/python sign_speaker.py` → http://<board-ip>:5000

## Phase 3: raise landmark fps (approved 2026-11-05) — user priority: preview smoothness / fps
Plan: ~/.claude/plans/in-this-folder-i-synchronous-hennessy.md
- [ ] F1 baseline benchmark with signer in view (full vs lite)
- [~] 2. CPU fixes: DONE newest-frame camera thread (stop-flag race found + fixed, verified), face every 2nd frame,
      per-stage timings in --benchmark. TODO: camera exposure (needs uncovered camera), full/lite choice, overlap
- [ ] F2 each change measured; keep only gains
- [ ] F3 target ≥ 12 fps with two hands + face
- [x] 3. GPU driver check: OpenCL works via rusticl with RUSTICL_ENABLE=msm (FD702, OpenCL 3.0, fp16, 1 CU @ 844 MHz);
      Vulkan 1.0 via turnip (Mesa 26.1.6). ai-edge-litert 2.3.0 aarch64 ships libLiteRtWebGpuAccelerator.so
      (CompiledModel + HardwareAccelerator.GPU|CPU). Risk: WebGPU may need Vulkan 1.1. Gate: drivers PASS; trial needs dep approval.
- [x] 4. GPU trial with prebuilt ai-edge-litert 2.3.0 (in ~/litert_test only): FAILS — SIGILL on load,
      "compiled with lse enabled, but this feature is not available on this processor". Board CPU flags have no
      `atomics` (ARMv8.0). Remaining GPU route = build the TFLite GPU delegate (OpenCL) from source for armv8.0.
- [ ] F4 custom vs Holistic landmarks still agree (needs clips)
- [ ] F6 live preview looks smooth (user)

### Live-test findings (2026-11-05)
- F1 baseline with user in view: custom+full 7.3 fps (329% CPU, 194 MB), custom+lite 10.1 fps (276%, 179 MB);
  palm detector ran on 74/147 frames (~95 ms each) = biggest cost. Camera alone gives 30 fps on auto exposure; it
  dropped to ~15 fps only because the models starve its thread (exposure test: restored to defaults, verified).
- "30 s behind" had TWO causes:
  1. guesses: AutoSegmenter only ends a clip when hands leave view (8 frames) or at max_frames=300 (= 30 s at 10 fps).
  2. video: board ↔ laptop go through a phone hotspot; raw throughput measured 267 KB/s earlier,
     19 KB/s later; 38 KB preview frames → 0.4–0.6 fps and up to 15 s stale. Plus a bug: on (re)connect the server
     sent the last frame from the previous viewer (21.75 s stale) — fixed (frames > 1 s old are never served).
- Preview now self-paced (/frame.jpg pull, one frame at a time) so frames cannot queue between board and browser.
- [ ] Shrink preview frames for slow links; [ ] time-based clip end / sliding-window prediction; [ ] palm throttle

## Phase 4: smooth recordable demo (approved 2026-11-05: "do what you think is best… display just needs to be smooth… record a video")
- [ ] A1 preview frames 320×240 (--preview-width), page stays self-paced
- [ ] A2 TimedSegmenter: end 0.4 s after hands drop, start after 0.15 s of hands, max clip 4 s (was 300 frames)
- [ ] A3 --record out.mp4: annotated frames written on the board at capture timing (no network → smooth)
- [ ] A4 palm detector throttle while ≥1 hand tracked (every 6th frame); measure before/after on the SAME recorded clip
- [ ] A5 record_clips.py silent by default (--cue to speak), per user preference
- [ ] A6 board Wi-Fi power save off (nmcli, reversible) — last, verify connectivity after
- [ ] Checks: unit test of TimedSegmenter on synthetic hand sequences; clip benchmark before/after; preview staleness
      via measure_poll.py; recorded demo plays smoothly (frame count / duration); fresh-subagent review

## Phase 5: accuracy — back to the known-good pipeline (2026-11-05)
Live accuracy is bad with the custom tracker (top-1 locked on "arm"/"weus"). User: the model worked well live on a
laptop with demo_webcam.py; no such env exists on this Mac any more. Clip-recording diagnostics rejected by the user.
- [ ] 1. Rebuild .venv-demo on the Mac from requirements-demo.txt; run the ORIGINAL demo_webcam.py (Holistic, .h5)
      with the laptop webcam; user confirms apple / pizza / thankyou / yes / no are recognised
- [ ] 2. Same pipeline + settings on the board (demo_env, Holistic, .h5) with the Arducam → if accuracy drops, the
      cause is camera (angle/blur/~15 fps) or board fps, not the model
- [ ] 3. Then speed up one step at a time, re-checking against the laptop result after each step
