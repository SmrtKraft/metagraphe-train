"""Pack GAPS (classical/fingerstyle guitar, 404 pieces) into training HDF5.

GAPS ships full recordings (.wav) with tightly-aligned .mid transcriptions and
an official train/val/test split in gaps_metadata_with_splits.csv. This is the
fingerstyle/waltz material the model most needs. Audio → 16 kHz mono; MIDI →
note events (onset, offset, pitch, velocity). Long pieces are stored whole;
the training loader samples 10 s segments.
"""
import os, sys
import numpy as np
import pandas as pd
import librosa
import mido
import h5py

SR = 16000
ROOT = 'data/gaps'
OUT = 'data/packed'
os.makedirs(OUT, exist_ok=True)


def midi_notes(path):
    """Flatten a MIDI file to (onset_s, offset_s, pitch, vel) events."""
    mid = mido.MidiFile(path)
    events = []
    open_notes = {}  # (pitch) -> (onset_s, vel)
    t = 0.0
    for msg in mid:  # mido iteration yields delta-time in SECONDS already
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            open_notes.setdefault(msg.note, []).append((t, msg.velocity))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            stack = open_notes.get(msg.note)
            if stack:
                onset, vel = stack.pop(0)
                if 21 <= msg.note <= 108 and t - onset > 0.01:
                    events.append((onset, t, msg.note, vel))
    events.sort()
    return np.array(events, dtype=np.float32) if events else np.zeros((0, 4), np.float32)


def main():
    df = pd.read_csv(f'{ROOT}/gaps_metadata_with_splits.csv')
    split_col = 'split' if 'split' in df.columns else df.columns[-1]
    print('splits:', df[split_col].value_counts().to_dict())
    counts = {}
    handles = {}
    for _, row in df.iterrows():
        split = str(row[split_col]).strip().lower()
        if split not in ('train', 'val', 'valid', 'validation', 'test'):
            continue
        split = 'val' if split.startswith('val') else split
        wav = os.path.join(ROOT, str(row['audio_path']))
        mid = os.path.join(ROOT, str(row['midi_path']))
        if not (os.path.exists(wav) and os.path.exists(mid)):
            continue
        try:
            ev = midi_notes(mid)
            if len(ev) == 0:
                continue
            audio, _ = librosa.load(wav, sr=SR, mono=True)
        except Exception as e:
            print('skip', row['id'], e); continue
        if split not in handles:
            handles[split] = h5py.File(f'{OUT}/gaps_{split}.h5', 'w')
            counts[split] = 0
        g = handles[split].create_group(f'clip{counts[split]:04d}')
        g.create_dataset('audio', data=audio.astype(np.float32), compression='gzip')
        g.create_dataset('events', data=ev)
        g.attrs['name'] = str(row['id'])
        g.attrs['instrument'] = 'guitar'
        counts[split] += 1
        if counts[split] % 25 == 0:
            print(f'{split}: {counts[split]} packed…', flush=True)
    for split, h in handles.items():
        sz = os.path.getsize(h.filename) / 1e6
        print(f'{split}: {counts[split]} clips -> {h.filename} ({sz:.0f} MB)')
        h.close()
    print('done')


if __name__ == '__main__':
    main()
