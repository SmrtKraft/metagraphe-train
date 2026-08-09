"""Pack MusicNet (real chamber recordings — piano/strings/winds) into training HDF5.

MusicNet (Thickstun et al.) is ~34 h of real classical recordings with per-note
labels. License: Creative Commons Attribution 4.0 (CC-BY-4.0) — commercial use
is permitted with attribution (see DATASETS.md). This is the real
multi-instrument complement to our synth + guitar mixes.

The tar extracts to  <root>/{train,test}_{data,labels}/  where *_data holds
44.1 kHz mono .wav and *_labels holds one .csv per recording with columns:

    start_time,end_time,instrument,note,start_beat,end_beat,note_value

start_time / end_time are SAMPLE indices at 44100 Hz (NOT seconds) — we divide
by 44100 to get seconds. `note` is already a MIDI pitch; `instrument` is a GM
program number. Velocity is not labelled, so we use a constant default.

Output schema matches pack_guitarset.py exactly:
    group clipNNNN/
        audio   float32, 16 kHz mono, gzip
        events  float32 [N,4] = (onset_sec, offset_sec, midi_pitch, velocity)
        attrs   name, instrument

Splits: MusicNet ships an official train/test directory split. We pack the
official train recordings, carve a small deterministic val slice out of them,
and (unless --no-test) also pack the official test recordings to musicnet_test.h5
as a held-out eval set. Recordings are minutes long — stored whole; the training
loader samples 10 s windows. Packing streams recording-by-recording and prints
progress. Use --limit N to cap recordings processed if disk is tight.

Usage:
    .venv/bin/python pack_musicnet.py [--root data/musicnet] [--limit N] [--no-test]
"""
import argparse
import glob
import os

import h5py
import librosa
import numpy as np
import pandas as pd

SR = 16000
SRC_SR = 44100          # MusicNet audio + label sample clock
DEFAULT_VELOCITY = 100
PITCH_LO, PITCH_HI = 21, 108

# GM program number -> readable instrument name (the set MusicNet actually uses).
GM_NAME = {
    1: 'piano', 7: 'harpsichord', 41: 'violin', 42: 'viola', 43: 'cello',
    44: 'contrabass', 61: 'horn', 69: 'oboe', 71: 'bassoon', 72: 'clarinet',
    74: 'flute',
}


def find_split_dirs(root):
    """Locate the train/test data+label dirs anywhere under root (the tar nests
    them one level deep in a `musicnet/` folder)."""
    dirs = {}
    for want in ('train_data', 'train_labels', 'test_data', 'test_labels'):
        hits = glob.glob(os.path.join(root, '**', want), recursive=True)
        hits = [h for h in hits if os.path.isdir(h)]
        if hits:
            dirs[want] = sorted(hits, key=len)[0]
    return dirs


def load_events(csv_path):
    """Read a MusicNet label CSV -> float32 [N,4] (onset_s, offset_s, pitch, vel)."""
    df = pd.read_csv(csv_path)
    onset = df['start_time'].to_numpy(dtype=np.float64) / SRC_SR
    offset = df['end_time'].to_numpy(dtype=np.float64) / SRC_SR
    pitch = df['note'].to_numpy(dtype=np.int64)
    keep = (pitch >= PITCH_LO) & (pitch <= PITCH_HI) & (offset - onset > 0.01)
    ev = np.stack([
        onset[keep], offset[keep], pitch[keep].astype(np.float64),
        np.full(keep.sum(), DEFAULT_VELOCITY, dtype=np.float64),
    ], axis=1).astype(np.float32)
    ev = ev[np.argsort(ev[:, 0], kind='stable')] if len(ev) else ev
    return ev


def instrument_label(csv_path, ensembles):
    """A per-clip instrument attr: metadata ensemble name if known, else the set
    of GM instruments present in the labels, else 'ensemble'."""
    rid = os.path.splitext(os.path.basename(csv_path))[0]
    ens = ensembles.get(rid)
    if ens:
        return ens.lower()
    try:
        progs = sorted(set(int(x) for x in pd.read_csv(csv_path)['instrument'].unique()))
        names = [GM_NAME.get(p, f'gm{p}') for p in progs]
        return '+'.join(names) if names else 'ensemble'
    except Exception:
        return 'ensemble'


def recordings(data_dir, label_dir):
    """Yield (id, wav_path, csv_path) pairs present in both dirs, sorted by id."""
    for wav in sorted(glob.glob(os.path.join(data_dir, '*.wav'))):
        rid = os.path.splitext(os.path.basename(wav))[0]
        csv = os.path.join(label_dir, f'{rid}.csv')
        if os.path.exists(csv):
            yield rid, wav, csv


def carve_val(train_ids, frac=0.08, min_n=4, max_n=30):
    """Deterministically pick evenly-spaced ids from sorted train ids for val,
    so the slice spans composers rather than a contiguous block."""
    n = len(train_ids)
    if n == 0:
        return set()
    k = max(min_n, min(max_n, round(n * frac)))
    k = min(k, n - 1)  # keep at least one training recording
    if k <= 0:
        return set()
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return {train_ids[i] for i in sorted(set(idx.tolist()))}


def pack(pairs, out_path, ensembles, tag):
    if not pairs:
        print(f'{tag}: nothing to pack'); return 0
    n = 0
    with h5py.File(out_path, 'w') as h:
        for rid, wav, csv in pairs:
            try:
                ev = load_events(csv)
                if len(ev) == 0:
                    print(f'  skip {rid}: no in-range notes'); continue
                audio, _ = librosa.load(wav, sr=SR, mono=True)
            except Exception as e:
                print(f'  skip {rid}: {e}'); continue
            g = h.create_group(f'clip{n:04d}')
            g.create_dataset('audio', data=audio.astype(np.float32), compression='gzip')
            g.create_dataset('events', data=ev)  # onset, offset, pitch, vel
            g.attrs['name'] = f'musicnet_{rid}'
            g.attrs['instrument'] = instrument_label(csv, ensembles)
            n += 1
            if n % 10 == 0:
                print(f'{tag}: {n} packed…', flush=True)
    sz = os.path.getsize(out_path) / 1e6
    print(f'{tag}: {n} clips -> {out_path} ({sz:.0f} MB)')
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/musicnet', help='dir the tar was extracted into')
    ap.add_argument('--out', default='data/packed', help='output dir for *.h5')
    ap.add_argument('--limit', type=int, default=0, help='cap TRAIN recordings read (0 = all); disk-safety')
    ap.add_argument('--no-test', action='store_true', help='skip packing the official test split')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dirs = find_split_dirs(args.root)
    if 'train_data' not in dirs or 'train_labels' not in dirs:
        raise SystemExit(
            f'MusicNet train dirs not found under {args.root!r}. Extract musicnet.tar.gz '
            f'there first (tar xzf musicnet.tar.gz -C {args.root}). Found: {sorted(dirs)}')

    # metadata ensemble names (optional, improves the per-clip instrument attr)
    ensembles = {}
    for meta in glob.glob(os.path.join(args.root, '**', 'musicnet_metadata.csv'), recursive=True):
        try:
            m = pd.read_csv(meta)
            ensembles = {str(r['id']): str(r['ensemble']) for _, r in m.iterrows()}
            break
        except Exception:
            pass

    train_all = list(recordings(dirs['train_data'], dirs['train_labels']))
    if args.limit and args.limit > 0:
        train_all = train_all[:args.limit]
        print(f'--limit {args.limit}: using {len(train_all)} train recordings')

    val_ids = carve_val([rid for rid, _, _ in train_all])
    train_pairs = [p for p in train_all if p[0] not in val_ids]
    val_pairs = [p for p in train_all if p[0] in val_ids]
    print(f'official train: {len(train_all)} recordings -> {len(train_pairs)} train / {len(val_pairs)} val')

    pack(train_pairs, os.path.join(args.out, 'musicnet_train.h5'), ensembles, 'train')
    pack(val_pairs, os.path.join(args.out, 'musicnet_val.h5'), ensembles, 'val')

    if not args.no_test and 'test_data' in dirs and 'test_labels' in dirs:
        test_pairs = list(recordings(dirs['test_data'], dirs['test_labels']))
        pack(test_pairs, os.path.join(args.out, 'musicnet_test.h5'), ensembles, 'test')

    print('done')


if __name__ == '__main__':
    main()
