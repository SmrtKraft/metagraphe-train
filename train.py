"""Fine-tune the Kong onset/offset-regression note model on packed GuitarSet.

WHY this exists: the guitar-domain paper (arXiv 2402.15258) showed that
fine-tuning this exact architecture from the MAESTRO-pretrained checkpoint
beats basic-pitch by ~18 F1 on guitar. This script replicates that recipe on
our packed HDF5 data (data/packed/guitarset_*.h5) so we can (a) prove the
loop moves F1 on real data and (b) measure whether Apple-Silicon MPS is fast
enough for the full 30-50 epoch run.

Design decisions:
- Targets are built with logic transplanted line-for-line from the vendored
  utils/utilities.py TargetProcessor (same rounding, same J-shape regression
  via TargetProcessor.get_regression, same cross-segment mask semantics).
  A --check-targets mode PROVES parity by feeding the same notes through the
  vendored MIDI-string path and diffing every roll. Silent target drift is
  the classic way these models lose accuracy, so parity is tested, not assumed.
- Only the note branch (Regress_onset_offset_frame_velocity_CRNN) is trained.
  Guitar has no sustain pedal, so the pedal net is carried through untouched
  in saved checkpoints purely to keep them drop-in compatible with the
  vendored Note_pedal inference path (smoke_test.py works on ft_epoch*.pth).
- Eval = full-clip inference (enframe/deframe exactly like vendor inference.py)
  + RegressionPostProcessor + mir_eval note-onset F1 (+-50ms, offset ignored),
  on 8 fixed val clips. The untuned checkpoint is scored first as baseline.

Run:
  .venv/bin/python train.py --epochs 2            # the real thing
  .venv/bin/python train.py --check-targets       # target parity proof
Outputs: checkpoints/ft_epoch{N}.pth, runs/log.jsonl (one JSON line/epoch).
"""
import os
# MPS misses a few ops (e.g. some GRU internals historically); fall back to
# CPU per-op instead of crashing. Must be set before torch import.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import sys
import json
import time
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'vendor/piano_transcription/pytorch'))
sys.path.insert(0, os.path.join(ROOT, 'vendor/piano_transcription/utils'))

import numpy as np
import h5py
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import config
from models import Regress_onset_offset_frame_velocity_CRNN
from losses import regress_onset_offset_frame_velocity_bce
from utilities import TargetProcessor, RegressionPostProcessor
from pytorch_utils import forward as vendor_forward
import mir_eval

SEGMENT_SECONDS = 10.0
SAMPLE_RATE = config.sample_rate            # 16000
FPS = config.frames_per_second              # 100
CLASSES_NUM = config.classes_num            # 88
BEGIN_NOTE = config.begin_note              # 21
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)

TRAIN_H5 = os.path.join(ROOT, 'data/packed/guitarset_train.h5')
VAL_H5 = os.path.join(ROOT, 'data/packed/guitarset_val.h5')
BASE_CKPT = os.path.join(ROOT, 'checkpoints/note_model.pth')


# ---------------------------------------------------------------------------
# Target construction — transplanted from vendored TargetProcessor.process()
# (utils/utilities.py). The vendored path parses MIDI-event strings; ours takes
# (onset_sec, offset_sec, pitch, velocity) rows, but every frame computation
# below mirrors the vendored lines exactly (int(round(...)), fin_frame>=0
# gate, unpaired-onset clipping to segment end + masking, J-shape last).
# ---------------------------------------------------------------------------
def pair_note_events(events, start_time, end_time):
    """Replay the vendored note_on/note_off pairing (TargetProcessor.process
    step 1) over our (onset, offset, pitch, velocity) rows — including its
    scan-window quirks, because we promised bit-exact targets, not idealized
    ones (--check-targets enforces this):

    - buffer_dict keeps ONE onset per pitch: GuitarSet has genuinely
      overlapping same-pitch notes (two strings, one pitch); a second note_on
      overwrites the first and an unmatched note_off is dropped, exactly as a
      piano-roll (which cannot express same-pitch overlap) requires.
    - the scan runs range(ex_bgn_idx, fin_idx) where fin_idx comes from
      enumerate+break: if no event lies past the segment end (segment at the
      clip tail) fin_idx lands on len-1 and the final event is NOT scanned;
      ex_bgn_idx only looks back (fin_idx - bgn_idx) events before the segment.
    - onsets left in the buffer get their offset clipped to segment end and
      are flagged unpaired -> the caller masks them out of the loss.

    Returns [(onset, offset, pitch, velocity, unpaired), ...] in vendored
    append order (paired notes in offset order, then leftover onsets).
    """
    msgs = []
    for (onset_time, offset_time, midi_pitch, velocity) in events:
        msgs.append((float(onset_time), 0, int(midi_pitch), float(velocity)))
        msgs.append((float(offset_time), 1, int(midi_pitch), 0.0))
    msgs.sort(key=lambda m: m[0])   # stable, like the packed MIDI event stream
    if not msgs:
        return []

    times = [m[0] for m in msgs]
    last = len(times) - 1           # enumerate leaves the index here w/o break
    bgn_idx = next((i for i, t in enumerate(times) if t > start_time), last)
    fin_idx = next((i for i, t in enumerate(times) if t > end_time), last)
    ex_bgn_idx = max(bgn_idx - (fin_idx - bgn_idx), 0)

    buffer_dict = {}
    note_events = []
    for i in range(ex_bgn_idx, fin_idx):
        (t, kind, pitch, vel) = msgs[i]
        if kind == 0:
            buffer_dict[pitch] = (t, vel)
        elif pitch in buffer_dict:
            (onset_t, onset_vel) = buffer_dict.pop(pitch)
            note_events.append((onset_t, t, pitch, onset_vel, False))
    for pitch, (onset_t, onset_vel) in buffer_dict.items():
        note_events.append((onset_t, end_time, pitch, onset_vel, True))
    return note_events


def build_note_targets(events, start_time, target_processor):
    """events: (N, 4) float array [onset_sec, offset_sec, midi_pitch, velocity]
    (absolute clip time). Returns the note-branch target dict."""
    # float64 like the vendored MIDI-time path: NumPy 2 keeps float32-scalar
    # minus python-float in float32, which drifts the regression distances
    # ~1e-7 vs vendored (caught by --check-targets before this cast existed).
    events = np.asarray(events, dtype=np.float64)
    tp = target_processor
    seg = tp.segment_seconds
    end_time = start_time + seg
    frames_num = int(round(seg * tp.frames_per_second)) + 1

    onset_roll = np.zeros((frames_num, tp.classes_num))
    offset_roll = np.zeros((frames_num, tp.classes_num))
    reg_onset_roll = np.ones((frames_num, tp.classes_num))
    reg_offset_roll = np.ones((frames_num, tp.classes_num))
    frame_roll = np.zeros((frames_num, tp.classes_num))
    velocity_roll = np.zeros((frames_num, tp.classes_num))
    mask_roll = np.ones((frames_num, tp.classes_num))

    for (onset_time, offset_eff, midi_pitch, velocity, unpaired) in \
            pair_note_events(events, start_time, end_time):
        # 'unpaired': the note_off falls beyond the segment, so the offset was
        # clipped to segment end AND the note is masked out of the loss from
        # its onset frame on (we can't supervise an unseen offset).
        piano_note = int(np.clip(midi_pitch - tp.begin_note, 0, tp.max_piano_note))
        bgn_frame = int(round((onset_time - start_time) * tp.frames_per_second))
        fin_frame = int(round((offset_eff - start_time) * tp.frames_per_second))
        if fin_frame < 0:
            continue    # note ended before this segment

        frame_roll[max(bgn_frame, 0): fin_frame + 1, piano_note] = 1
        offset_roll[fin_frame, piano_note] = 1
        velocity_roll[max(bgn_frame, 0): fin_frame + 1, piano_note] = velocity
        # Vector from frame center to true offset (pre-J-shape)
        reg_offset_roll[fin_frame, piano_note] = \
            (offset_eff - start_time) - (fin_frame / tp.frames_per_second)

        if bgn_frame >= 0:
            onset_roll[bgn_frame, piano_note] = 1
            reg_onset_roll[bgn_frame, piano_note] = \
                (onset_time - start_time) - (bgn_frame / tp.frames_per_second)
        else:
            # Note began before the segment: no onset supervision, mask head
            mask_roll[: fin_frame + 1, piano_note] = 0

        if unpaired:
            # Vendored: mask_roll[bgn_frame:, note] = 0 for buffer_dict leftovers
            if bgn_frame >= 0:
                mask_roll[bgn_frame:, piano_note] = 0
            else:
                mask_roll[:, piano_note] = 0    # spans whole segment

    for k in range(tp.classes_num):
        reg_onset_roll[:, k] = tp.get_regression(reg_onset_roll[:, k])
        reg_offset_roll[:, k] = tp.get_regression(reg_offset_roll[:, k])

    return {
        'onset_roll': onset_roll, 'offset_roll': offset_roll,
        'reg_onset_roll': reg_onset_roll, 'reg_offset_roll': reg_offset_roll,
        'frame_roll': frame_roll, 'velocity_roll': velocity_roll,
        'mask_roll': mask_roll}


def check_target_parity(n_cases=24, seed=1234):
    """PROOF that build_note_targets == vendored TargetProcessor.process.

    Renders packed note events into the MIDI-string format the vendored code
    parses, runs both paths on random segments of real train clips, and diffs
    every roll elementwise. Exits nonzero on any mismatch.
    """
    tp = TargetProcessor(SEGMENT_SECONDS, FPS, BEGIN_NOTE, CLASSES_NUM)
    rng = np.random.default_rng(seed)
    keys_to_check = ['onset_roll', 'offset_roll', 'reg_onset_roll',
                     'reg_offset_roll', 'frame_roll', 'velocity_roll', 'mask_roll']
    n_fail = 0
    with h5py.File(TRAIN_H5, 'r') as hf:
        clip_names = sorted(hf.keys())
        for case in range(n_cases):
            name = clip_names[rng.integers(len(clip_names))]
            events = hf[name]['events'][:]
            dur = hf[name]['audio'].shape[0] / SAMPLE_RATE
            start_time = float(rng.uniform(0, dur - SEGMENT_SECONDS))

            # Render events as vendored midi_events strings (time-sorted)
            msgs = []
            for (on, off, pitch, vel) in events:
                msgs.append((float(on), 'note_on channel=0 note=%d velocity=%d time=0'
                             % (int(pitch), int(vel))))
                msgs.append((float(off), 'note_off channel=0 note=%d velocity=0 time=0'
                             % int(pitch)))
            msgs.sort(key=lambda m: m[0])
            midi_events_time = np.array([m[0] for m in msgs])
            midi_events = [m[1] for m in msgs]

            ref_dict, _, _ = tp.process(start_time, midi_events_time, midi_events,
                                        extend_pedal=False, note_shift=0)
            ours = build_note_targets(events, start_time, tp)

            for key in keys_to_check:
                if not np.allclose(ours[key], ref_dict[key], atol=1e-9):
                    n_diff = int(np.sum(~np.isclose(ours[key], ref_dict[key], atol=1e-9)))
                    print('MISMATCH case %d (%s @ %.2fs) key=%s cells=%d'
                          % (case, name, start_time, key, n_diff))
                    n_fail += 1
    if n_fail:
        print('PARITY FAILED: %d roll mismatches' % n_fail)
        sys.exit(1)
    print('PARITY OK: %d random segments, all 7 rolls identical to vendored '
          'TargetProcessor' % n_cases)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GuitarSegmentDataset(Dataset):
    """Random 10s segments from packed train clips across ONE OR MORE HDF5
    files (guitar + classical + whatever we pack next — the model is GENERAL,
    not guitar-only). len = clips * segments_per_clip so one 'epoch' sees each
    clip ~segments_per_clip times."""

    def __init__(self, h5_paths, segments_per_clip=2):
        if isinstance(h5_paths, str):
            h5_paths = [h5_paths]
        self.h5_paths = list(h5_paths)
        self.segments_per_clip = segments_per_clip
        self.hfs = None         # opened lazily per worker (h5py + fork safety)
        self.rng = None
        self.index = []         # flat list of (path, clip_name)
        self.durations = {}     # (path, clip_name) -> seconds
        for p in self.h5_paths:
            with h5py.File(p, 'r') as hf:
                for k in sorted(hf.keys()):
                    self.index.append((p, k))
                    self.durations[(p, k)] = hf[k]['audio'].shape[0] / SAMPLE_RATE
        self.target_processor = TargetProcessor(
            SEGMENT_SECONDS, FPS, BEGIN_NOTE, CLASSES_NUM)

    def __len__(self):
        return len(self.index) * self.segments_per_clip

    def __getitem__(self, index):
        if self.hfs is None:
            self.hfs = {p: h5py.File(p, 'r') for p in self.h5_paths}
        if self.rng is None:
            info = torch.utils.data.get_worker_info()
            seed = (torch.initial_seed() + (info.id if info else 0)) % (2 ** 31)
            self.rng = np.random.default_rng(seed)

        path, name = self.index[index // self.segments_per_clip]
        dur = self.durations[(path, name)]
        start_time = float(self.rng.uniform(0, max(0.0, dur - SEGMENT_SECONDS)))
        start_sample = int(start_time * SAMPLE_RATE)

        hf = self.hfs[path]
        waveform = hf[name]['audio'][start_sample: start_sample + SEGMENT_SAMPLES]
        if len(waveform) < SEGMENT_SAMPLES:     # clip tail / short-clip guard
            waveform = np.pad(waveform, (0, SEGMENT_SAMPLES - len(waveform)))
        events = hf[name]['events'][:]

        data = build_note_targets(events, start_time, self.target_processor)
        data['waveform'] = waveform.astype(np.float32)
        return data


def collate(list_data):
    return {key: torch.from_numpy(
        np.stack([d[key] for d in list_data]).astype(np.float32))
        for key in list_data[0]}


# ---------------------------------------------------------------------------
# Evaluation — full-clip inference exactly like vendor inference.py
# ---------------------------------------------------------------------------
def enframe(x, segment_samples):
    """(1, samples) -> (N, segment_samples), hop = half segment. Adapted from
    vendor pytorch/inference.py PianoTranscription.enframe."""
    assert x.shape[1] % segment_samples == 0
    batch = []
    pointer = 0
    while pointer + segment_samples <= x.shape[1]:
        batch.append(x[:, pointer: pointer + segment_samples])
        pointer += segment_samples // 2
    return np.concatenate(batch, axis=0)


def deframe(x):
    """(N, segment_frames, classes) -> (frames, classes). Adapted from vendor
    pytorch/inference.py PianoTranscription.deframe (50% overlap, keep centers)."""
    if x.shape[0] == 1:
        return x[0]
    x = x[:, 0:-1, :]
    (N, segment_frames, _) = x.shape
    assert segment_frames % 4 == 0
    y = [x[0, 0: int(segment_frames * 0.75)]]
    for i in range(1, N - 1):
        y.append(x[i, int(segment_frames * 0.25): int(segment_frames * 0.75)])
    y.append(x[-1, int(segment_frames * 0.25):])
    return np.concatenate(y, axis=0)


def transcribe_notes(model, audio, forward_batch_size=4):
    """audio (samples,) float32 -> est note events via RegressionPostProcessor.
    No pedal outputs -> post-processor takes the notes-only path."""
    audio = audio[None, :]
    audio_len = audio.shape[1]
    pad_len = int(np.ceil(audio_len / SEGMENT_SAMPLES)) * SEGMENT_SAMPLES - audio_len
    audio = np.concatenate((audio, np.zeros((1, pad_len), dtype=np.float32)), axis=1)
    segments = enframe(audio, SEGMENT_SAMPLES).astype(np.float32)

    output_dict = vendor_forward(model, segments, batch_size=forward_batch_size)
    for key in output_dict:
        output_dict[key] = deframe(output_dict[key])

    post = RegressionPostProcessor(
        FPS, classes_num=CLASSES_NUM, onset_threshold=0.3,
        offset_threshold=0.3, frame_threshold=0.1, pedal_offset_threshold=0.2)
    est_note_events, _ = post.output_dict_to_midi_events(output_dict)
    return est_note_events


def load_val_clips(n_clips, val_paths=None):
    """Load up to n_clips from EACH val file so the reported F1 covers every
    instrument in the mix, not just whichever sorts first."""
    if val_paths is None:
        val_paths = [VAL_H5]
    clips = []
    for p in val_paths:
        tag = os.path.basename(p).replace('_val.h5', '')
        with h5py.File(p, 'r') as hf:
            for name in sorted(hf.keys())[:n_clips]:
                clips.append({
                    'name': f'{tag}/{name}',
                    'audio': hf[name]['audio'][:].astype(np.float32),
                    'events': hf[name]['events'][:]})
    return clips


def evaluate(model, val_clips):
    """Note-onset F1, +-50ms onset tolerance, offsets ignored (the standard
    'note w/ onset' metric). Micro = pooled over clips; macro = mean of per-clip."""
    was_training = model.training
    total_match, total_ref, total_est = 0, 0, 0
    per_clip_f1 = []
    for clip in val_clips:
        est = transcribe_notes(model, clip['audio'])
        ref = clip['events']
        ref_intervals = ref[:, 0:2].astype(np.float64)
        ref_pitches = mir_eval.util.midi_to_hz(ref[:, 2])
        if est:
            est_intervals = np.array(
                [[e['onset_time'], max(e['offset_time'], e['onset_time'] + 1e-4)]
                 for e in est], dtype=np.float64)
            est_pitches = mir_eval.util.midi_to_hz(
                np.array([e['midi_note'] for e in est], dtype=np.float64))
            matching = mir_eval.transcription.match_notes(
                ref_intervals, ref_pitches, est_intervals, est_pitches,
                onset_tolerance=0.05, offset_ratio=None)
        else:
            matching = []
        n_match, n_est = len(matching), len(est)
        n_ref = len(ref)
        total_match += n_match
        total_ref += n_ref
        total_est += n_est
        p = n_match / n_est if n_est else 0.0
        r = n_match / n_ref if n_ref else 0.0
        per_clip_f1.append(2 * p * r / (p + r) if (p + r) else 0.0)

    precision = total_match / total_est if total_est else 0.0
    recall = total_match / total_ref if total_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    if was_training:
        model.train()
    return {'f1': f1, 'precision': precision, 'recall': recall,
            'macro_f1': float(np.mean(per_clip_f1)),
            'n_ref': total_ref, 'n_est': total_est}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TARGET_KEYS = ['reg_onset_roll', 'reg_offset_roll', 'frame_roll',
               'velocity_roll', 'onset_roll', 'mask_roll']


def pick_device(requested):
    if requested != 'auto':
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(args):
    device = pick_device(args.device)
    print('device:', device)

    # Bare note model, seeded from the pretrained Note_pedal checkpoint.
    # The pedal branch is irrelevant for guitar and is never instantiated for
    # training; its weights are only carried into saved checkpoints so they
    # stay loadable by the vendored Note_pedal inference class.
    ckpt = torch.load(BASE_CKPT, map_location='cpu')
    model = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=FPS, classes_num=CLASSES_NUM)
    model.load_state_dict(ckpt['model']['note_model'])
    pedal_sd = ckpt['model']['pedal_model']
    model.to(device)

    import glob
    train_paths = sorted(glob.glob(os.path.join(args.data, '*_train.h5'))) or [TRAIN_H5]
    val_paths = sorted(glob.glob(os.path.join(args.data, '*_val.h5'))) or [VAL_H5]
    print('train files:', [os.path.basename(p) for p in train_paths])
    print('val files:', [os.path.basename(p) for p in val_paths])
    dataset = GuitarSegmentDataset(train_paths, segments_per_clip=args.segments_per_clip)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate,
                        persistent_workers=args.num_workers > 0, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0.0, amsgrad=True)

    val_clips = load_val_clips(args.val_clips, val_paths)
    os.makedirs(os.path.join(ROOT, 'runs'), exist_ok=True)
    log_path = os.path.join(ROOT, 'runs/log.jsonl')

    def log(record):
        record['time'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record))

    # Baseline: untuned piano weights scored on guitar val — the number the
    # fine-tune has to beat.
    if not args.skip_baseline:
        t0 = time.time()
        metrics = evaluate(model, val_clips)
        log({'event': 'baseline', 'epoch': 0, **metrics,
             'eval_minutes': round((time.time() - t0) / 60, 2)})

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        losses = []
        for step, batch in enumerate(loader):
            waveform = batch['waveform'].to(device)
            target_dict = {k: batch[k].to(device) for k in TARGET_KEYS}

            output_dict = model(waveform)
            loss = regress_onset_offset_frame_velocity_bce(
                model, output_dict, target_dict)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())
            if step % 10 == 0:
                print('epoch %d step %d/%d loss %.4f (%.1fs)'
                      % (epoch, step, len(loader), loss.item(),
                         time.time() - t_epoch), flush=True)
        train_minutes = (time.time() - t_epoch) / 60

        t_eval = time.time()
        metrics = evaluate(model, val_clips)
        eval_minutes = (time.time() - t_eval) / 60

        ckpt_path = os.path.join(ROOT, 'checkpoints/ft_epoch%d.pth' % epoch)
        torch.save({'model': {
            'note_model': {k: v.cpu() for k, v in model.state_dict().items()},
            'pedal_model': pedal_sd}}, ckpt_path)

        log({'event': 'epoch', 'epoch': epoch,
             'train_loss': float(np.mean(losses)), **metrics,
             'train_minutes': round(train_minutes, 2),
             'eval_minutes': round(eval_minutes, 2),
             'checkpoint': os.path.relpath(ckpt_path, ROOT)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=3.0)
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu', 'cuda'])
    parser.add_argument('--data', default=os.path.join(ROOT, 'data/packed'),
                        help='dir of packed *_train.h5 / *_val.h5 (all instruments)')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--segments-per-clip', type=int, default=2)
    parser.add_argument('--val-clips', type=int, default=8)
    parser.add_argument('--skip-baseline', action='store_true')
    parser.add_argument('--check-targets', action='store_true',
                        help='run target-parity proof against vendored code and exit')
    args = parser.parse_args()

    if args.check_targets:
        check_target_parity()
        return
    train(args)


if __name__ == '__main__':
    main()
