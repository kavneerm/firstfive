# Sign-to-speech on the Arduino UNO Q

A camera watches someone sign one of 250 everyday American Sign Language words. The board recognizes the sign
and says the word out loud. It runs entirely on an Arduino UNO Q: MediaPipe Holistic finds 543 face, hand and body
points per frame, and a small transformer model (trained on Kaggle's
[asl-signs](https://www.kaggle.com/competitions/asl-signs) data) turns a clip of those points into a word.

ENGS 21 project.

## The model

`sign_app/sign_model_int8w.tflite` (2.07 MB) is the deployed model. It is the original trained network
(`aslModelENGS21-fold0-best.h5`) with its weights stored as 8-bit numbers and TFLite's XNNPACK delegate turned off.
That cut its memory to about a third with no loss of accuracy. See [LEARNINGS.md](LEARNINGS.md) for the whole story.

Accuracy on signers the model never saw during training (top-1 = its first guess is right):

| Test set | Clips | Top-1 | Top-5 | Source |
|---|---|---|---|---|
| Fold 0: 4 unseen signers | 18,528 | **70.93%** | 88.22% | [`evaluation/results/int8w/`](evaluation/results/int8w/report.txt) |
| 3 of those signers (the hardest) | 13,965 | 67.15% | 85.91% | [`evaluation/results/int8w/3_signers/`](evaluation/results/int8w/3_signers/report.txt) |

The original float model scored 70.97% on fold 0, so the 8-bit version costs 0.04 points (about 7 clips).
Chance is 0.4%. The ~89% figures on the Kaggle leaderboard are not on unseen signers, so they are not comparable.

## What's here

| Path | What it is |
|---|---|
| `sign_app/` | The program the board runs at power-on, and `run_mac.py` to run the same thing on a laptop. Start with `sign_app/USAGE.md`. |
| `sign_speaker.py`, `demo_webcam.py`, `sign_infer_lite.py` | Camera → landmarks → model → spoken word; webcam demo; TensorFlow-free inference. |
| `islr_model.py`, `convert_sign_model.py` | Model definition and preprocessing; `.h5` → `.tflite` conversion. |
| `evaluation/` | Full accuracy report (per word, per signer, confidence, mix-ups, 13 charts) for any `.tflite` model. |
| `model_search/` | The overnight search for a smaller model, its harness, and its results. |
| `setup_uno_q.sh`, `setup_lite.sh` | One-time board setup (Python 3.11 env; slim tflite-runtime env). |
| `tests/`, `sign_app/tests/` | Checks that need no camera. Plain scripts: run with `python <file>`. |
| `docs/plans/` | Plans and logs from the work. |

## Running it

On a laptop (Python 3.11):

```bash
python3.11 -m venv .venv-demo
.venv-demo/bin/pip install -r requirements-demo.txt
.venv-demo/bin/python sign_app/run_mac.py            # webcam window: A auto-record, F flip, Q quit
```

On the UNO Q: copy the folder to the board, run `bash setup_uno_q.sh`, then follow `sign_app/USAGE.md`.

## Getting the data

The dataset is not in this repo (it is ~40 GB, and Kaggle's competition rules do not allow redistributing it).
Accept the rules on the [competition page](https://www.kaggle.com/competitions/asl-signs/rules), then:

```bash
kaggle competitions download -c asl-signs
python model_search/prepare_kaggle.py --zip asl-signs.zip --out data/islr      # → landmarks.npy, meta.csv
python evaluation/evaluate.py --model sign_app/sign_model_int8w.tflite --data data/islr --out evaluation/results/int8w
```

`evaluate.py` needs numpy, matplotlib, and tflite-runtime or TensorFlow.

## License

Code: [MIT](LICENSE). The Kaggle asl-signs data and anything derived from it fall under the competition's own rules.
