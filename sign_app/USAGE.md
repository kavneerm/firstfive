# Smaller sign model for the UNO Q

## On the board now (set up 2026-11-12)

The board runs one program: `~/sign_app` (board_live.py = sign_speaker.py set up like the Mac demo: this 8-bit model,
MediaPipe Holistic complexity 1, XNNPACK off, words ≥ 40% spoken at volume 50, the same display as `run_mac.py`).

```bash
ssh unoq '~/sign_app/start.sh'      # start (extra options pass through, e.g. ~/sign_app/start.sh --volume 80)
ssh unoq '~/sign_app/stop.sh'       # stop (frees the camera and port 5000)
ssh unoq 'tail -f ~/sign_app/live.log'
```
Preview: open http://ENGS21.local:5000/ on the Mac (same network or hotspot as the board). Uses `~/demo_env` and `~/audio`.
Landmark tracking runs as 3 parallel Holistic processes (`parallel_holistic.py`): about 7-11 frames/second with a person in view
(was 2.5-5), ~80-87% of the four cores, ~1.3 GB memory in total. `start.sh --workers 1` goes back to a single instance; 4 overloads the board.
It starts by itself at power-on (system service `sign-app`, volume 100). Preview: http://ENGS21.local:5000/ on any network the Mac and board share.
`stop.sh` stops it until the next boot; `sudo systemctl disable sign-app` turns autostart off. Old code, sign_env, litert_test, the recordings,
audio_loud and the LCD sketch were removed from the board.


`sign_model_int8w.tflite` is **your current model with its weights stored as 8-bit numbers** (no retraining). It is a
drop-in replacement for `sign_model.tflite`: same input, same 250 signs, same `sign_to_prediction_index_map.json`.

## What it changes (measured 2026-11-12, EC2 box with tflite-runtime 2.14, not yet on the UNO Q)

Accuracy is top-1 on 13,965 Kaggle clips from 3 signers the model never trained on.

| Model / runtime | Top-1 | Top-5 | File | Sign-model memory |
|---|---|---|---|---|
| `sign_model.tflite` (deployed, float32), XNNPACK on | 67.08% | 85.74% | 7.91 MB | ~21.7 MB |
| `sign_model_int8w.tflite`, XNNPACK on | 67.17% | 85.93% | 2.07 MB | ~9.2 MB (42%) |
| `sign_model_int8w.tflite`, **XNNPACK off** | **67.16%** | 85.92% | 2.07 MB | **~7.4 MB (34%)** |

- **XNNPACK** is TFLite's default CPU accelerator. It keeps its own copy of the weights, so turning it off saves memory.
  `SIGN_NO_XNNPACK=1` turns it off (see below).
- **Speed is the open question.** On the busy EC2 box the 8-bit model took ~67 ms per sign against ~13 ms for the float model
  with XNNPACK. The UNO Q's CPU is slower, so measure it there first (step 2). If it's too slow for live use, keep the float model.
- This only shrinks the sign model. MediaPipe's landmark models, OpenCV and the camera use far more memory in `sign_speaker`.

## Files

| File | What it is |
|---|---|
| `sign_model_int8w.tflite` | the 8-bit model (sha256 `81401774…7090b`) |
| `sign_infer_lite.py` | your board copy plus one addition: `SIGN_NO_XNNPACK=1` turns XNNPACK off. Without the variable it behaves exactly as before |
| `sign_to_prediction_index_map.json` | unchanged sign map |
| `board_check.py` | compares models on the board: memory, speed, same answers |

## 0. Try it on the Mac first (webcam)

`run_mac.py` runs the existing webcam demo (`~/Downloads/arduino/demo_webcam.py`) with this model instead of the `.h5`.

```bash
cd sign_app/ in this repo
~/Downloads/arduino/.venv-demo/bin/python run_mac.py                      # SPACE record/stop · A auto · F flip · Q quit
SIGN_NO_XNNPACK=1 ~/Downloads/arduino/.venv-demo/bin/python run_mac.py    # same, XNNPACK off
~/Downloads/arduino/.venv-demo/bin/python run_mac.py --dry-run            # load check only, no camera
```

macOS asks for camera access the first time; allow it for Terminal.

## 1. Copy to the board

From the Mac (use the board's address and the usual login):

```bash
cd sign_app/ in this repo
scp sign_model_int8w.tflite board_check.py arduino@<board-address>:~/sign_demo/
ssh arduino@<board-address> 'cp ~/sign_demo/sign_infer_lite.py ~/sign_demo/sign_infer_lite.py.bak'
scp sign_infer_lite.py arduino@<board-address>:~/sign_demo/
```

The `.bak` copy lets you undo: `cp ~/sign_demo/sign_infer_lite.py.bak ~/sign_demo/sign_infer_lite.py`.

## 2. Measure on the board (no camera needed)

Stop `sign_speaker` first if it's running. Then:

```bash
cd ~/sign_demo && ~/sign_env/bin/python board_check.py sign_model.tflite sign_model_int8w.tflite
```

It prints four lines: each model with XNNPACK on and off, showing memory, ms per inference and "same top-1 as first: N/20".
Random inputs aren't real signs, so 18–20 of 20 agreeing is normal. On real clips the two models agree on 95% of top-1 answers and score the same.

## 3. Run it live

```bash
cd ~/sign_demo
SIGN_NO_XNNPACK=1 ~/sign_env/bin/python sign_speaker.py --model sign_model_int8w.tflite     # smallest memory
~/sign_env/bin/python sign_speaker.py --model sign_model_int8w.tflite                       # XNNPACK on, if that's faster
~/sign_env/bin/python sign_speaker.py                                                       # back to the original model
```

All the other `sign_speaker.py` options work as before.

## 4. What to compare

- **Latency:** does the spoken word come noticeably later than with `sign_model.tflite`?
- **Accuracy:** sign the same words with both models. A fair check: about 20 signs × 2 takes recorded with `record_clips.py`,
  then `check_clips.py` on each model. Pass = at least 30/40 correct, and no more than 2 worse than the original.
- **Memory:** `board_check.py` gives the sign model's share. For the whole program, watch `sign_speaker`'s RSS in `top`.

## Where it came from

Overnight search in `model_search/ in this repo` (plan and log: `docs/plans/small-model.md`). Smaller trained models were
still running when this folder was made. Width 128 is the best candidate so far, but it hadn't reached the accuracy bar yet. If one
wins, it will be a different `.tflite` that drops into the same place.
