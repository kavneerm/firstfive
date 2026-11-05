#!/usr/bin/env python3
"""
Speak demo_webcam.py's predictions on the Uno Q's USB speaker without touching the demo: follows the demo's log
and, for each prediction whose top guess is at least --threshold confident, plays ~/audio/<word>.wav on the board
over SSH (same aplay device as ~/speak.py).

    python demo_webcam.py ... > demo.log 2>&1 &
    python speak_predictions.py demo.log
"""
import argparse, re, subprocess, time

LINE = re.compile(r'^(?:manual|auto) \(\d+ frames\): (\S+) (\d+)%')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log')
    ap.add_argument('--threshold', type=float, default=0.40)
    ap.add_argument('--host', default='unoq')
    ap.add_argument('--device', default='plughw:CARD=UACDemoV10,DEV=0')
    a = ap.parse_args()

    with open(a.log) as fh:
        fh.seek(0, 2)                                   # only predictions made from now on
        print(f'speaking predictions >= {100 * a.threshold:.0f}% on {a.host}', flush=True)
        while True:
            line = fh.readline()
            if not line:
                time.sleep(0.1)
                continue
            m = LINE.match(line)
            if not m:
                continue
            word, pct = m.group(1), int(m.group(2))
            if pct < 100 * a.threshold:
                print(f'not spoken: {word} {pct}%', flush=True)
                continue
            # words are model labels (letters only), so they are safe inside the remote command
            subprocess.Popen(['ssh', '-o', 'BatchMode=yes', a.host,
                              f'aplay -q -D {a.device} ~/audio/{word}.wav'])
            print(f'speaking: {word} {pct}%', flush=True)


if __name__ == '__main__':
    main()
