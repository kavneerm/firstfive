#!/usr/bin/env python3
"""sign_speaker.py set up like the Mac demo (run_mac.py): same 8-bit model, MediaPipe Holistic at complexity 1,
and the same display (big current guess, strip of recent guesses, top-5 bars), viewed in a browser on the Mac.
Words at or above 40% are spoken through the speaker, as in sign_speaker.py.

    cd ~/sign_demo && SIGN_NO_XNNPACK=1 ~/demo_env/bin/python -u board_live.py --volume 50
    then open http://<board-address>:5000/ on the Mac

Any sign_speaker.py option can be added or overridden (e.g. --threshold 2 for silence).
--workers N (default 3) runs N Holistic instances in parallel processes; --workers 1 is the plain single instance.
The Grove LCD shows whether the signer is in position (position.py); --no-lcd turns that off, --near / --far set the limits.
"""
import os, sys

import sign_speaker as S

def _arg(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv[:-1] else default


SPEAK_THRESHOLD = _arg('--threshold', 0.40)      # green = would be spoken; follows --threshold when it is overridden
_history, _last = [], {'res': None}
_base_overlay = S.draw_overlay


def _text(cv2, img, s, org, scale, color, thick):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def overlay_with_words(cv2, img, state):
    """Identical to run_mac.py's overlay: demo_webcam's overlay plus a big current guess and the last 8 guesses."""
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


def _pop_int(name, default):
    """Take `name N` out of sys.argv (sign_speaker.py doesn't know it) and return N."""
    if name in sys.argv[:-1]:
        i = sys.argv.index(name)
        v = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
        return v
    return default


def _pop_flag(name):
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    return False


def _pop_float(name, default):
    if name in sys.argv[:-1]:
        i = sys.argv.index(name)
        v = float(sys.argv[i + 1]); del sys.argv[i:i + 2]
        return v
    return default


def _sigterm(*_):
    raise KeyboardInterrupt                # stop.sh sends SIGTERM: leave through sign_speaker's cleanup (workers, LCD)


if __name__ == '__main__':                 # worker processes re-import this file; only the real program runs main()
    import signal
    signal.signal(signal.SIGTERM, _sigterm)
    S.draw_overlay = overlay_with_words
    workers = _pop_int('--workers', 3)     # Holistic instances, one process each (1 = the plain single instance)
    no_lcd = _pop_flag('--no-lcd')         # skip the position messages on the Grove LCD
    near, far = _pop_float('--near', 0.55), _pop_float('--far', 0.18)   # shoulder width limits for TOO CLOSE / TOO FAR
    lcd_face_every = _pop_int('--lcd-face-every', 4)                    # count faces (for TOO MANY PEOPLE) every N frames
    defaults = {'--model': 'sign_model_int8w.tflite', '--landmarker': 'holistic', '--complexity': '1',
                '--preview-width': '640'}
    for k, v in defaults.items():
        if k not in sys.argv:
            sys.argv += [k, v]
    _plain = S.landmarker
    is_holistic = sys.argv[sys.argv.index('--landmarker') + 1] == 'holistic'

    def _landmarker(a):
        if workers > 1 and is_holistic:
            from parallel_holistic import ParallelHolistic
            lm = ParallelHolistic(workers, a.complexity)
            mp = lm.mp
            print(f'Holistic running in {workers} processes', flush=True)
        else:
            mp, lm = _plain(a)
        if not no_lcd and is_holistic:
            from position import PositionWatcher
            lm = PositionWatcher(lm, mp, near=near, far=far, face_every=lcd_face_every)
            print(f'position messages on the LCD (too close above {near}, too far below {far} shoulder width)', flush=True)
        return mp, lm

    S.landmarker = _landmarker
    S.main()
