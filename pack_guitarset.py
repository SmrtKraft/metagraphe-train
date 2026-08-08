"""Pack GuitarSet into training HDF5: 16 kHz mono audio + note events.

JAMS is plain JSON — the six `note_midi` annotations (one per string) merge
into one event list per file. Split by player (GuitarSet convention:
filename prefix 00–05): 00–03 train, 04 val, 05 test — held-out players,
never held-out excerpts, so eval measures generalization to unseen hands.
"""
import json, sys, glob, os
import numpy as np
import librosa
import h5py

SR = 16000
AUDIO_DIR = 'data/guitarset/audio'
ANN_DIR = 'data/guitarset/annotation'
OUT = 'data/packed'
os.makedirs(OUT, exist_ok=True)

def jams_notes(path):
    with open(path) as f:
        j = json.load(f)
    events = []
    for ann in j['annotations']:
        if ann['namespace'] != 'note_midi':
            continue
        for d in ann['data']:
            onset = float(d['time']); dur = float(d['duration'])
            pitch = int(round(float(d['value'])))
            if 21 <= pitch <= 108 and dur > 0.01:
                events.append((onset, onset + dur, pitch, 100))
    events.sort()
    return events

def split_of(name):
    player = int(name[:2])
    return 'train' if player <= 3 else ('val' if player == 4 else 'test')

files = {'train': [], 'val': [], 'test': []}
for wav in sorted(glob.glob(f'{AUDIO_DIR}/*.wav')):
    base = os.path.basename(wav).replace('_mic.wav', '')
    jam = f'{ANN_DIR}/{base}.jams'
    if not os.path.exists(jam):
        print('no jams for', base); continue
    files[split_of(os.path.basename(wav))].append((wav, jam))

for split, pairs in files.items():
    out = f'{OUT}/guitarset_{split}.h5'
    with h5py.File(out, 'w') as h:
        for i, (wav, jam) in enumerate(pairs):
            audio, _ = librosa.load(wav, sr=SR, mono=True)
            ev = np.array(jams_notes(jam), dtype=np.float32)
            g = h.create_group(f'clip{i:04d}')
            g.create_dataset('audio', data=audio.astype(np.float32), compression='gzip')
            g.create_dataset('events', data=ev)  # onset, offset, pitch, vel
            g.attrs['name'] = os.path.basename(wav)
            g.attrs['instrument'] = 'guitar'
    n = len(pairs)
    print(f'{split}: {n} clips -> {out} ({os.path.getsize(out)/1e6:.0f} MB)')
print('done')
