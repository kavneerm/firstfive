#!/usr/bin/env python3
"""
Record one fixture clip per sign for check_clips.py / compare_landmarkers.py: for each word it prints a cue (or
says the word through the speaker with --cue), waits --lead seconds, then records --seconds of raw camera video
(no landmark work, so the clip keeps the camera's full frame rate) to <out>/<word>.mp4.

    python record_clips.py hello mom thankyou --out clips
"""
import argparse, os, sys, time

from sign_speaker import ARDUCAM, AUDIO_DEVICE, AUDIO_DIR, Speaker, open_capture


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('words', nargs='+')
    ap.add_argument('--out', default='clips'); ap.add_argument('--camera', default=ARDUCAM)
    ap.add_argument('--seconds', type=float, default=4.0); ap.add_argument('--lead', type=float, default=1.5)
    ap.add_argument('--width', type=int, default=640); ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--cue', action='store_true', help='say each word through the speaker (default: print only)')
    a = ap.parse_args()

    import cv2
    speaker = Speaker(AUDIO_DIR, AUDIO_DEVICE)
    if a.cue and speaker.missing(a.words):
        sys.exit(f'no audio cue for: {", ".join(speaker.missing(a.words))}')
    os.makedirs(a.out, exist_ok=True)
    cap, _ = open_capture(cv2, a.camera, a.width, a.height)
    for word in a.words:
        print(f'NEXT: {word} -- sign it in {a.lead:.1f} s', flush=True)
        if a.cue:
            speaker.say(word).wait()
        t_end = time.time() + a.lead
        while time.time() < t_end:                 # keep draining the camera so the clip starts with fresh frames
            cap.read()
        frames, t0 = [], time.time()
        while time.time() - t0 < a.seconds:
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        fps = len(frames) / max(time.time() - t0, 1e-6)
        path, k = os.path.join(a.out, f'{word}.mp4'), 2
        while os.path.exists(path):                 # another take of the same sign: word_2.mp4, word_3.mp4, ...
            path, k = os.path.join(a.out, f'{word}_{k}.mp4'), k + 1
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        print(f'{path}: {len(frames)} frames at {fps:.1f} fps', flush=True)
    cap.release()


if __name__ == '__main__':
    main()
