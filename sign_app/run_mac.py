#!/usr/bin/env python3
"""Run the webcam demo from ~/Downloads/arduino with the 8-bit TFLite model instead of the Keras .h5.

    .venv-demo/bin/python sign_app/run_mac.py              # from the repo root; webcam window (A auto, F flip, Q quit)
    .venv-demo/bin/python sign_app/run_mac.py --dry-run    # load check only, no camera
    SIGN_NO_XNNPACK=1 .venv-demo/bin/python sign_app/run_mac.py   # XNNPACK off, as in the memory numbers

Every demo_webcam.py option works (--flip, --auto, --video clip.mp4, ...). --weights defaults to sign_model_int8w.tflite.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                            # this folder's sign_infer_lite (with SIGN_NO_XNNPACK)
sys.path.insert(1, os.path.dirname(HERE))                           # repo root: demo_webcam.py and its helpers

import demo_webcam as D
import sign_infer_lite as L


class TflitePredictor(L.LitePredictor):
    """Same constructor signature as demo_webcam.Predictor, so the demo code doesn't change."""

    def __init__(self, weights, signmap, max_len, dim):
        super().__init__(weights, signmap, max_len, num_threads=4)


D.Predictor = TflitePredictor

SPEAK_THRESHOLD = 0.40          # sign_speaker.py's default: words at or above this would be spoken on the board
_history, _last = [], {'res': None}
_base_overlay = D.draw_overlay


def _text(cv2, img, s, org, scale, color, thick):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)   # dark outline
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def overlay_with_words(cv2, img, state):
    """demo_webcam's overlay plus a large current guess and a strip of the last 8 guessed words."""
    res = state['result']
    if res is not None and res is not _last['res']:
        _last['res'] = res
        _history.append(res[0])
        del _history[:-8]
    img = _base_overlay(cv2, img, state)
    h, w = img.shape[:2]
    if res:
        name, p = res[0]
        col = (80, 220, 80) if p >= SPEAK_THRESHOLD else (200, 200, 200)
        _text(cv2, img, f'{name}  {100 * p:.0f}%', (12, 82), 1.5, col, 3)
    if _history:
        y = (h - 162) if res else (h - 16)
        x = 10
        _text(cv2, img, 'guesses:', (x, y), 0.6, (255, 255, 255), 1)
        x += cv2.getTextSize('guesses: ', cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        for name, p in _history:
            col = (80, 220, 80) if p >= SPEAK_THRESHOLD else (160, 160, 160)
            _text(cv2, img, name, (x, y), 0.7, col, 2)
            x += cv2.getTextSize(name + '  ', cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
            if x > w - 40:
                break
    return img


D.draw_overlay = overlay_with_words
if '--auto' not in sys.argv and '--manual' not in sys.argv:
    sys.argv.append('--auto')         # guess continuously; press A in the window to switch to SPACE-to-record
if '--manual' in sys.argv:
    sys.argv.remove('--manual')
if '--weights' not in sys.argv:
    sys.argv += ['--weights', os.path.join(HERE, 'sign_model_int8w.tflite')]
if '--signmap' not in sys.argv:
    sys.argv += ['--signmap', os.path.join(os.path.dirname(HERE), 'sign_to_prediction_index_map.json')]
D.main()
