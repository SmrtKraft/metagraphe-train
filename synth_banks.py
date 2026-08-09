"""Render unlimited, perfectly-labeled polyphonic training audio from KANON's
single-note orchestral sample banks — for the general audio->MIDI model.

Why this exists
---------------
The real datasets (GAPS, GuitarSet, MusicNet) are guitar/chamber-heavy and
FIXED in size. The banks give us exact ground truth for FREE: we know every
note we place to the sample, so we can mint as much labeled multi-instrument
polyphony as we want. The one thing free-labeled synth data must get right for
THIS model is RHYTHM — the (onset, offset) labels ARE the rhythm supervision.
So the clip generator places notes on a real beat grid at a real tempo (swing
and straight), with bass<->chord accompaniment textures (waltz, boom-chick,
arpeggiated splits) front and center, varied rhythmic values, rests, syncopation
and small human-timing jitter. The emitted labels are the ACTUAL jittered
placement, sample-exact.

Sampler (per instrument)
------------------------
Given (target_midi, target_velocity, duration): pick the nearest sample by
pitch (preferring the closest velocity layer), pitch-shift it to the exact
target by varispeed resampling (+ cents tuning correction), cap the shift to
+-5 semitones (farther => out of range => skip), then shape the amplitude to
the requested duration — decaying instruments (piano) ride their own decay and
fade the tail; sustained instruments (strings/winds/brass) hold a crossfaded
sustain loop then release. Never clicks.

Output: HDF5 in the SAME schema as pack_guitarset.py — groups clip0000..,
each with `audio` (float32 16k mono, gzip) and `events` (N x 4:
onset_s, offset_s, midi_pitch, velocity 1..127), attrs `name` + `instrument`.
"""
import os, sys, glob, argparse, math
from math import gcd
import numpy as np
import soundfile as sf
import h5py
from scipy.signal import resample_poly, fftconvolve, butter, lfilter

SR = 16000
BANKS = 'data/banks'
OUT = 'data/packed'
INSTRUMENTS = ['piano', 'violin', 'viola', 'cello', 'bass',
               'flute', 'oboe', 'clarinet', 'horn', 'trumpet']
DECAYING = {'piano'}          # rides its own decay; everything else sustains
MAX_SHIFT = 5                 # semitones; nearest sample farther than this = out of range

# ── low-level sample handling ───────────────────────────────────────────────

def _load16(path):
    """Load a bank WAV as 16 kHz mono float32 (rational resample, fast + clean)."""
    x, sr = sf.read(path)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def _varispeed(x, ratio):
    """Pitch-shift by ratio via linear-interpolation varispeed (fast, ~sample-exact
    pitch — verified within a cent of equal temperament). Aliasing on up-shift is
    negligible after the augmentation lowpass and is fine for training material."""
    if abs(ratio - 1.0) < 1e-6:
        return x.copy()
    n = int(np.floor(len(x) / ratio))
    if n < 4:
        return x[:4].copy()
    idx = np.arange(n, dtype=np.float64) * ratio
    return np.interp(idx, np.arange(len(x), dtype=np.float64), x).astype(np.float32)


def _fade(seg, a_ms, r_ms):
    """Apply attack fade-in and release fade-out (raised-cosine) in place-ish."""
    seg = seg.copy()
    n = len(seg)
    a = min(int(a_ms * SR / 1000), n // 2)
    r = min(int(r_ms * SR / 1000), n // 2)
    if a > 1:
        seg[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a, dtype=np.float32))
    if r > 1:
        seg[-r:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r, dtype=np.float32))
    return seg


class Sampler:
    """One instrument's playable range, built from its bank + manifest."""

    def __init__(self, name):
        self.name = name
        self.decaying = name in DECAYING
        self.samples = []          # list of dicts: midi, vel, cents, path, x16(lazy)
        d = os.path.join(BANKS, name)
        for line in open(os.path.join(d, 'manifest.txt')):
            parts = line.split()
            if len(parts) < 4:
                continue
            fn, midi, vel, cents = parts[0], int(parts[1]), float(parts[2]), float(parts[3])
            self.samples.append({'midi': midi, 'vel': vel, 'cents': cents,
                                 'path': os.path.join(d, fn), 'x16': None})
        self.samp_midis = np.array(sorted(set(s['midi'] for s in self.samples)))
        lo, hi = int(self.samp_midis.min()), int(self.samp_midis.max())
        # A pitch is reachable if some sample sits within +-MAX_SHIFT of it. This
        # correctly excludes holes (e.g. horn's 13-semitone gap) as out of range.
        self.reachable = np.array([m for m in range(lo - MAX_SHIFT, hi + MAX_SHIFT + 1)
                                   if np.min(np.abs(self.samp_midis - m)) <= MAX_SHIFT])
        self._shift_cache = {}     # (sample_index, target_midi) -> pitch-shifted x16

    def _x16(self, i):
        s = self.samples[i]
        if s['x16'] is None:
            s['x16'] = _load16(s['path'])
        return s['x16']

    def _pick(self, target_midi, target_vel):
        """Nearest sample by pitch, then closest velocity layer. Returns index or None."""
        best_i, best_key = None, None
        for i, s in enumerate(self.samples):
            pd = abs(s['midi'] - target_midi)
            if pd > MAX_SHIFT:
                continue
            key = (pd, abs(s['vel'] - target_vel))
            if best_key is None or key < best_key:
                best_key, best_i = key, i
        return best_i

    def _shifted(self, i, target_midi):
        key = (i, target_midi)
        c = self._shift_cache.get(key)
        if c is None:
            s = self.samples[i]
            semis = (target_midi - s['midi']) + s['cents'] / 100.0   # + cents correction
            c = _varispeed(self._x16(i), 2.0 ** (semis / 12.0))
            self._shift_cache[key] = c
        return c

    def _shape_decay(self, y, N):
        """Piano-style: let the sample's decay ride, fade the tail at note-off."""
        out = np.zeros(N, dtype=np.float32)
        L = min(N, len(y))
        seg = _fade(y[:L], 4.0, min(60.0, L / SR * 1000 / 3))
        out[:L] = seg
        return out

    def _shape_sustain(self, y, N):
        """Sustained: attack from the sample, crossfaded sustain loop to fill,
        synthetic release. Loops the body so arbitrary durations never click."""
        L = len(y)
        a = max(1, int(0.25 * L))          # sustain-loop start (past the attack)
        b = max(a + 1, int(0.78 * L))      # sustain-loop end (before natural release)
        xf = min(int(0.03 * SR), (b - a) // 2)
        if N <= L:
            core = y[:N].copy()
        else:
            parts = [y[:b].copy()]
            cur = b
            loop = y[a:b]
            while cur < N:
                if xf > 1 and len(parts[-1]) >= xf and len(loop) >= xf:
                    tail = parts[-1][-xf:]
                    head = loop[:xf]
                    w = np.linspace(0, 1, xf, dtype=np.float32)
                    parts[-1][-xf:] = tail * np.cos(w * np.pi / 2) + head * np.sin(w * np.pi / 2)
                    parts.append(loop[xf:].copy())
                    cur += len(loop) - xf
                else:
                    parts.append(loop.copy())
                    cur += len(loop)
            core = np.concatenate(parts)[:N]
        rel = min(120.0, N / SR * 1000 / 3)
        return _fade(core, 8.0, rel)

    def render(self, target_midi, target_vel, dur_s):
        """16 kHz mono audio of this note, or None if out of range."""
        i = self._pick(target_midi, target_vel)
        if i is None:
            return None
        y = self._shifted(i, target_midi)
        N = max(4, int(round(dur_s * SR)))
        out = self._shape_decay(y, N) if self.decaying else self._shape_sustain(y, N)
        amp = 0.15 + 0.85 * float(np.clip(target_vel, 0.0, 1.0))   # dynamic loudness
        return out * amp


# ── music theory (just enough for plausible streams) ────────────────────────

SCALES = {
    'major':      [0, 2, 4, 5, 7, 9, 11],
    'minor':      [0, 2, 3, 5, 7, 8, 10],
    'dorian':     [0, 2, 3, 5, 7, 9, 10],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
}
# progressions as scale-degree indices (0=tonic)
PROGRESSIONS = [[0, 3, 4, 0], [0, 4, 5, 3], [0, 5, 3, 4],
                [1, 4, 0, 0], [0, 3, 0, 4], [5, 3, 0, 4], [0, 0, 3, 4]]


def scale_pitches(root_pc, scale, lo, hi):
    """All MIDI pitches of the scale within [lo, hi]."""
    ps = []
    for m in range(lo, hi + 1):
        if (m - root_pc) % 12 in scale:
            ps.append(m)
    return ps


def chord_tones(root_pc, scale, degree):
    """Triad (root, third, fifth) pitch-classes for a scale degree."""
    s = SCALES.get(scale, scale) if isinstance(scale, str) else scale
    return [(root_pc + s[(degree + k) % 7] + 12 * ((degree + k) // 7)) % 12 for k in (0, 2, 4)]


def nearest_in(pool, target):
    """Nearest reachable pitch in `pool` to target."""
    if len(pool) == 0:
        return None
    pool = np.asarray(pool)
    return int(pool[np.argmin(np.abs(pool - target))])


def voice_chord(pcs, center, pool, n=3):
    """Realize chord pitch-classes as concrete pitches near `center`, drawn from
    the instrument's reachable pool. Returns up to n distinct pitches."""
    out = []
    for k, pc in enumerate(pcs[:n]):
        target = center + (k - 1) * 4          # spread voices
        best = None
        for octv in range(-2, 4):
            cand = pc + 12 * ((center // 12) + octv)
            m = nearest_in(pool, cand)
            if m is None:
                continue
            if best is None or abs(m - target) < abs(best - target):
                best = m
        if best is not None and best not in out:
            out.append(best)
    return out


# ── rhythm: a real beat grid, swing, human jitter ───────────────────────────

def make_clock(rng, bars, beats_per_bar, bpm, swing):
    """Return beat_to_sec(beat) mapping a musical beat position (float) to seconds,
    applying swing to off-beat eighths. Straight when swing==0.5."""
    spb = 60.0 / bpm

    def beat_to_sec(beat):
        base = math.floor(beat)
        frac = beat - base
        if 0.45 < frac < 0.55:                 # off-beat eighth -> swing it
            frac = swing
        return (base + frac) * spb
    return beat_to_sec, spb


def jitter(rng, ms=12.0, cap=0.025):
    return float(np.clip(rng.normal(0, ms / 1000.0), -cap, cap))


# duration menu in BEATS (value, weight) — whole..sixteenth, dotted, triplet
DUR_MENU = [(4.0, 2), (3.0, 2), (2.0, 5), (1.5, 5), (1.0, 9),
            (0.75, 5), (0.5, 9), (1 / 3, 3), (0.25, 4)]
_DV = np.array([d for d, _ in DUR_MENU])
_DW = np.array([w for _, w in DUR_MENU], dtype=float); _DW /= _DW.sum()


def rhythm_fill(rng, total_beats, rest_p=0.12, allow_short=True):
    """Fill total_beats with (start_beat, dur_beats, is_rest) tokens on the grid."""
    toks = []
    t = 0.0
    while total_beats - t > 1e-3:
        choices = _DV[_DV <= (total_beats - t) + 1e-6]
        if len(choices) == 0:
            break
        w = _DW[:len(choices)].copy()
        if not allow_short:
            w[_DV[:len(choices)] < 0.5] *= 0.2
        w = w / w.sum()
        d = float(rng.choice(choices, p=w))
        is_rest = rng.random() < rest_p and t > 0
        toks.append((t, d, is_rest))
        t += d
    return toks


# ── pattern generators: each yields (onset_beat, dur_beats, midi, vel01, role) ─
# role 'bass' vs 'upper' only affects nothing downstream except realism; kept for clarity.

def accent(rng, beat_in_bar, beats_per_bar, base=0.6):
    """Velocity accent marking the pulse: strong downbeat, medium mid-bar."""
    v = base
    if abs(beat_in_bar - round(beat_in_bar)) < 0.05:      # on a beat
        v += 0.12
    if round(beat_in_bar) == 0:                            # downbeat
        v += 0.16
    elif beats_per_bar == 4 and round(beat_in_bar) == 2:   # backbeat-ish
        v += 0.08
    return float(np.clip(v + rng.normal(0, 0.05), 0.15, 1.0))


def gen_waltz(rng, prog, root_pc, scale, bass_pool, chord_pool, bars):
    """3/4: bass root on beat 1, chord on beats 2 & 3. The #1 target texture."""
    notes = []
    for bar in range(bars):
        deg = prog[bar % len(prog)]
        pcs = chord_tones(root_pc, scale, deg)
        bass_center = (bass_pool[0] + bass_pool[-1]) // 2 - 3
        root_m = nearest_in(bass_pool, pcs[0] + 12 * (bass_center // 12))
        if root_m is not None:
            notes.append((bar * 3 + 0.0, 1.0, root_m, accent(rng, 0, 3, 0.7), 'bass'))
        ccenter = (chord_pool[0] + chord_pool[-1]) // 2 + 2
        chord = voice_chord(pcs, ccenter, chord_pool, n=3)
        for beat in (1.0, 2.0):
            for m in chord:
                notes.append((bar * 3 + beat, 0.9, m, accent(rng, beat, 3, 0.5), 'upper'))
    return notes, 3


def gen_boom_chick(rng, prog, root_pc, scale, bass_pool, chord_pool, bars):
    """4/4 boom-chick: alternating bass (root/fifth) on 1 & 3, chords on 2 & 4."""
    notes = []
    for bar in range(bars):
        deg = prog[bar % len(prog)]
        pcs = chord_tones(root_pc, scale, deg)
        bass_center = (bass_pool[0] + bass_pool[-1]) // 2 - 2
        root_m = nearest_in(bass_pool, pcs[0] + 12 * (bass_center // 12))
        fifth_m = nearest_in(bass_pool, pcs[2] + 12 * (bass_center // 12))
        for beat, bm in ((0.0, root_m), (2.0, fifth_m)):
            if bm is not None:
                notes.append((bar * 4 + beat, 0.9, bm, accent(rng, beat, 4, 0.7), 'bass'))
        ccenter = (chord_pool[0] + chord_pool[-1]) // 2 + 2
        chord = voice_chord(pcs, ccenter, chord_pool, n=3)
        for beat in (1.0, 3.0):
            for m in chord:
                notes.append((bar * 4 + beat, 0.8, m, accent(rng, beat, 4, 0.5), 'upper'))
    return notes, 4


def gen_arp_split(rng, prog, root_pc, scale, bass_pool, chord_pool, bars):
    """LH/RH split: low root struck on the downbeat, upper voices arpeggiated
    across the bar with time offsets (fingerstyle-ish)."""
    bpb = rng.choice([3, 4])
    notes = []
    for bar in range(bars):
        deg = prog[bar % len(prog)]
        pcs = chord_tones(root_pc, scale, deg)
        bass_center = (bass_pool[0] + bass_pool[-1]) // 2 - 3
        root_m = nearest_in(bass_pool, pcs[0] + 12 * (bass_center // 12))
        if root_m is not None:
            notes.append((bar * bpb + 0.0, float(bpb) * 0.9, root_m,
                          accent(rng, 0, bpb, 0.72), 'bass'))
        ccenter = (chord_pool[0] + chord_pool[-1]) // 2 + 3
        chord = voice_chord(pcs, ccenter, chord_pool, n=3)
        if not chord:
            continue
        step = 0.5
        pos = 1.0 if bpb == 3 else 0.5
        k = 0
        while pos < bpb:
            m = chord[k % len(chord)]
            notes.append((bar * bpb + pos, step * 0.95, m, accent(rng, pos, bpb, 0.5), 'upper'))
            pos += step
            k += 1
    return notes, bpb


def gen_block_chords(rng, prog, root_pc, scale, bass_pool, chord_pool, bars):
    """Homophonic block triads changing on a rhythm, optional bass doubling."""
    bpb = rng.choice([3, 4])
    notes = []
    for bar in range(bars):
        deg = prog[bar % len(prog)]
        pcs = chord_tones(root_pc, scale, deg)
        ccenter = (chord_pool[0] + chord_pool[-1]) // 2 + 2
        chord = voice_chord(pcs, ccenter, chord_pool, n=rng.choice([3, 3, 4]))
        for (t, d, rest) in rhythm_fill(rng, bpb, rest_p=0.10, allow_short=False):
            if rest:
                continue
            for m in chord:
                notes.append((bar * bpb + t, d * 0.95, m, accent(rng, t, bpb, 0.55), 'upper'))
        if bass_pool is not None and rng.random() < 0.6:
            bass_center = (bass_pool[0] + bass_pool[-1]) // 2 - 2
            rm = nearest_in(bass_pool, pcs[0] + 12 * (bass_center // 12))
            if rm is not None:
                notes.append((bar * bpb + 0.0, float(bpb) * 0.9, rm,
                              accent(rng, 0, bpb, 0.68), 'bass'))
    return notes, bpb


def gen_arpeggio(rng, prog, root_pc, scale, bass_pool, chord_pool, bars):
    """A chord broken into a running eighth/sixteenth pattern up/down."""
    bpb = rng.choice([3, 4])
    step = rng.choice([0.5, 0.25, 1 / 3])
    notes = []
    for bar in range(bars):
        deg = prog[bar % len(prog)]
        pcs = chord_tones(root_pc, scale, deg)
        ccenter = (chord_pool[0] + chord_pool[-1]) // 2
        tones = voice_chord(pcs, ccenter, chord_pool, n=3)
        if not tones:
            continue
        seq = tones + tones[-2:0:-1]           # up then down
        pos, k = 0.0, 0
        while pos < bpb - 1e-6:
            m = seq[k % len(seq)]
            notes.append((bar * bpb + pos, step * 0.9, m, accent(rng, pos, bpb, 0.5), 'upper'))
            pos += step
            k += 1
    return notes, bpb


def gen_melody(rng, prog, root_pc, scale, bass_pool, chord_pool, bars):
    """Monophonic melody: scale steps/leaps, varied rhythmic values, rests,
    syncopation, and an optional pickup (anacrusis)."""
    bpb = rng.choice([3, 4])
    lo, hi = chord_pool[0], chord_pool[-1]
    spool = scale_pitches(root_pc, SCALES.get(scale, [0, 2, 4, 5, 7, 9, 11]), lo, hi)
    if not spool:
        spool = list(range(lo, hi + 1))
    notes = []
    cur = spool[len(spool) // 2]
    # optional pickup: 1-2 short notes leading into bar 1 (placed in the last beat)
    pickup = []
    if rng.random() < 0.4:
        for j, pb in enumerate([-0.5] if rng.random() < 0.5 else [-1.0, -0.5]):
            cur = _step_melody(rng, spool, cur)
            pickup.append((pb, 0.5, cur, accent(rng, 0.5, bpb, 0.5), 'upper'))
    for bar in range(bars):
        for (t, d, rest) in rhythm_fill(rng, bpb, rest_p=0.16):
            if rest:
                continue
            cur = _step_melody(rng, spool, cur)
            notes.append((bar * bpb + t, d * rng.uniform(0.6, 0.95), cur,
                          accent(rng, t, bpb, 0.55), 'upper'))
    return pickup + notes, bpb


def _step_melody(rng, spool, cur):
    i = int(np.argmin(np.abs(np.array(spool) - cur)))
    move = rng.choice([-3, -2, -1, 1, 2, 3, 4], p=[0.08, 0.16, 0.28, 0.28, 0.12, 0.05, 0.03])
    j = int(np.clip(i + move, 0, len(spool) - 1))
    return spool[j]


def gen_counterpoint(rng, prog, root_pc, scale, bass_pool, chord_pool, bars):
    """Two independent voices (upper melody + lower line) with different rhythms."""
    bpb = rng.choice([3, 4])
    up, _ = gen_melody(rng, prog, root_pc, scale, bass_pool, chord_pool, bars)
    lo_pool = bass_pool if bass_pool is not None else chord_pool
    lo_lo, lo_hi = lo_pool[0], lo_pool[min(len(lo_pool) - 1, len(lo_pool) // 2)]
    spool = scale_pitches(root_pc, SCALES.get(scale, [0, 2, 4, 5, 7, 9, 11]), lo_lo, lo_hi)
    if not spool:
        spool = list(range(lo_lo, lo_hi + 1))
    low = []
    cur = spool[len(spool) // 2]
    for bar in range(bars):
        for (t, d, rest) in rhythm_fill(rng, bpb, rest_p=0.14, allow_short=False):
            if rest:
                continue
            cur = _step_melody(rng, spool, cur)
            low.append((bar * bpb + t, d * rng.uniform(0.7, 0.95), cur,
                        accent(rng, t, bpb, 0.5), 'bass'))
    return up + low, bpb


def gen_pad(rng, prog, root_pc, scale, bass_pool, chord_pool, bars):
    """Sustained chord tones, whole/half notes, tied across bars."""
    bpb = rng.choice([3, 4])
    notes = []
    bar = 0
    while bar < bars:
        deg = prog[bar % len(prog)]
        pcs = chord_tones(root_pc, scale, deg)
        ccenter = (chord_pool[0] + chord_pool[-1]) // 2 + 2
        chord = voice_chord(pcs, ccenter, chord_pool, n=rng.choice([2, 3, 3]))
        hold = rng.choice([1, 2]) if bars - bar > 1 else 1     # tie over 1-2 bars
        dur = bpb * hold - 0.1
        for m in chord:
            notes.append((bar * bpb, dur, m, accent(rng, 0, bpb, 0.5), 'upper'))
        if bass_pool is not None and rng.random() < 0.7:
            bc = (bass_pool[0] + bass_pool[-1]) // 2 - 2
            rm = nearest_in(bass_pool, pcs[0] + 12 * (bc // 12))
            if rm is not None:
                notes.append((bar * bpb, dur, rm, accent(rng, 0, bpb, 0.55), 'bass'))
        bar += hold
    return notes, bpb


# accompaniment / bass-chord textures get the lion's share of the weight
PATTERNS = [
    ('waltz',        gen_waltz,        6, True),
    ('boom_chick',   gen_boom_chick,   6, True),
    ('arp_split',    gen_arp_split,    6, True),
    ('block_chords', gen_block_chords, 3, False),
    ('arpeggio',     gen_arpeggio,     3, False),
    ('melody',       gen_melody,       4, False),
    ('counterpoint', gen_counterpoint, 3, False),
    ('pad',          gen_pad,          2, False),
]
_PW = np.array([w for _, _, w, _ in PATTERNS], dtype=float); _PW /= _PW.sum()


# ── augmentation ────────────────────────────────────────────────────────────

def synth_ir(rng):
    """Short synthetic room IR: a few early reflections + exponential noise tail."""
    tau = rng.uniform(0.12, 0.45)
    L = int(tau * 3 * SR)
    t = np.arange(L) / SR
    ir = rng.standard_normal(L).astype(np.float32) * np.exp(-t / tau).astype(np.float32)
    for _ in range(rng.integers(2, 5)):        # early reflections
        d = int(rng.uniform(0.005, 0.05) * SR)
        if d < L:
            ir[d] += rng.uniform(0.3, 0.7)
    ir[0] = 1.0
    ir /= np.max(np.abs(ir)) + 1e-9
    return ir


def pink_noise(n, rng):
    """Cheap pink-ish noise via one-pole filtered white."""
    w = rng.standard_normal(n).astype(np.float32)
    b, a = butter(1, 0.03)
    p = lfilter(b, a, w).astype(np.float32)
    return p / (np.std(p) + 1e-9)


def augment(x, rng):
    y = x
    if rng.random() < 0.6:                     # light room reverb
        ir = synth_ir(rng)
        wet = fftconvolve(y, ir)[:len(y)].astype(np.float32)
        mix = rng.uniform(0.08, 0.25)
        y = (1 - mix) * y + mix * wet
    # broadband + pink noise at low level (always a touch, so it isn't sterile)
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    nlev = rms * rng.uniform(0.001, 0.02)
    y = y + nlev * pink_noise(len(y), rng) + nlev * 0.5 * rng.standard_normal(len(y)).astype(np.float32)
    if rng.random() < 0.35:                     # occasional gentle lowpass
        fc = rng.uniform(3500, 7000) / (SR / 2)
        b, a = butter(2, fc)
        y = lfilter(b, a, y).astype(np.float32)
    y = y * rng.uniform(0.5, 1.0)               # random overall gain
    peak = np.max(np.abs(y)) + 1e-9             # normalize to avoid clipping
    if peak > 0.99:
        y = y * (0.99 / peak)
    elif peak < 0.3:
        y = y * (0.7 / peak)
    return y.astype(np.float32)


# ── one clip ────────────────────────────────────────────────────────────────

def register_window(reachable, span, rng, bias='mid'):
    """Pick a contiguous playable window of ~span semitones for a role."""
    lo, hi = int(reachable[0]), int(reachable[-1])
    if hi - lo <= span:
        return reachable
    if bias == 'low':
        start = lo + int(rng.uniform(0, (hi - lo - span) * 0.35))
    elif bias == 'high':
        start = lo + int(rng.uniform((hi - lo - span) * 0.55, hi - lo - span))
    else:
        start = lo + int(rng.uniform((hi - lo - span) * 0.25, (hi - lo - span) * 0.75))
    return reachable[(reachable >= start) & (reachable <= start + span)]


def make_clip(rng, samplers, primary, second=None):
    """Return (audio float32 16k, events Nx4, meta). meta has bpm + pattern."""
    clip_dur = float(np.clip(rng.normal(6.0, 2.5), 2.0, 12.0))
    bpm = float(rng.uniform(50, 180))
    swing = 0.5 if rng.random() < 0.6 else rng.uniform(0.58, 0.66)

    root_pc = int(rng.integers(0, 12))
    scale = rng.choice(list(SCALES.keys()))
    prog = PROGRESSIONS[int(rng.integers(len(PROGRESSIONS)))]

    pi = int(rng.choice(len(PATTERNS), p=_PW))
    pname, pfn, _, is_accomp = PATTERNS[pi]

    def build(inst_name, pat_fn, pat_name, is_ac, force_low=False):
        s = samplers[inst_name]
        reach = s.reachable
        # bass pool = low third; chord/melody pool = mid-high
        bass_pool = register_window(reach, 16, rng, 'low')
        chord_pool = register_window(reach, 22, rng, 'low' if force_low else 'mid')
        # meter: waltz forces 3, else pattern picks; derive bars from tempo+dur
        beats_per_bar = 3 if pat_name == 'waltz' else 4
        spb = 60.0 / bpm
        bars = max(1, int(clip_dur / (beats_per_bar * spb)) + 1)
        notes, bpb = pat_fn(rng, prog, root_pc, scale,
                            np.asarray(bass_pool), np.asarray(chord_pool), bars)
        clock, spb = make_clock(rng, bars, bpb, bpm, swing)
        placed = []
        for (onb, durb, midi, vel, role) in notes:
            on = clock(onb) + jitter(rng)
            off = clock(onb + durb) + jitter(rng)
            if on < 0:                          # pickup before t0 -> shift into clip
                on = max(0.0, on + spb)
                off = off + spb
            off = max(on + 0.04, off)           # never zero/negative length
            if on >= clip_dur:
                continue
            off = min(off, clip_dur)
            if off - on < 0.03:
                continue
            placed.append((on, off, int(midi), float(vel), inst_name))
        return placed

    events = build(primary, pfn, pname, is_accomp)
    if second is not None:
        # second instrument: complementary role (bass line or pad), same key
        alt_fn = gen_pad if rng.random() < 0.5 else gen_counterpoint
        events += build(second, alt_fn, 'pad', False, force_low=rng.random() < 0.5)

    # render
    tail = 1.2
    total = int((clip_dur + tail) * SR)
    buf = np.zeros(total, dtype=np.float32)
    gt = []
    for (on, off, midi, vel, inst) in events:
        wav = samplers[inst].render(midi, vel, off - on)
        if wav is None:
            continue
        i0 = int(round(on * SR))
        i1 = min(total, i0 + len(wav))
        if i1 <= i0:
            continue
        buf[i0:i1] += wav[:i1 - i0]
        gt.append((on, off, midi, int(np.clip(round(vel * 127), 1, 127))))

    audio = augment(buf, rng)
    ev = np.array(sorted(gt), dtype=np.float32) if gt else np.zeros((0, 4), np.float32)
    return audio, ev, {'bpm': bpm, 'pattern': pname, 'accomp': is_accomp,
                       'swing': swing != 0.5, 'primary': primary, 'second': second}


# ── driver ──────────────────────────────────────────────────────────────────

def build_split(split, n, seed_base, samplers, verify=False):
    out = f'{OUT}/synth_{split}.h5'
    os.makedirs(OUT, exist_ok=True)
    # balance instruments roughly evenly as the PRIMARY
    order = (INSTRUMENTS * (n // len(INSTRUMENTS) + 1))[:n]
    rng0 = np.random.default_rng(seed_base)
    rng0.shuffle(order)

    per_inst_notes = {i: 0 for i in INSTRUMENTS}
    per_inst_primary = {i: 0 for i in INSTRUMENTS}
    pitch_lo = {i: 999 for i in INSTRUMENTS}
    pitch_hi = {i: -1 for i in INSTRUMENTS}
    bpms, accomp_flags, note_total = [], [], 0
    verify_clips = []

    with h5py.File(out, 'w') as h:
        for i in range(n):
            rng = np.random.default_rng(seed_base * 1000003 + i)
            primary = order[i]
            second = None
            if rng.random() < 0.30:
                second = rng.choice([x for x in INSTRUMENTS if x != primary])
            audio, ev, meta = make_clip(rng, samplers, primary, second)
            g = h.create_group(f'clip{i:04d}')
            g.create_dataset('audio', data=audio.astype(np.float32), compression='gzip')
            g.create_dataset('events', data=ev)
            g.attrs['name'] = f'synth_{split}_{i:04d}_{meta["pattern"]}'
            g.attrs['instrument'] = primary if second is None else f'{primary}+{second}'
            # stats
            per_inst_primary[primary] += 1
            bpms.append(meta['bpm']); accomp_flags.append(meta['accomp'])
            note_total += len(ev)
            for (on, off, midi, vel) in ev:
                # attribute pitch range to whichever instrument owns that pitch pool
                for inst in ([primary] if second is None else [primary, second]):
                    if samplers[inst].reachable[0] <= midi <= samplers[inst].reachable[-1]:
                        per_inst_notes[inst] += 1
                        pitch_lo[inst] = min(pitch_lo[inst], int(midi))
                        pitch_hi[inst] = max(pitch_hi[inst], int(midi))
                        break
            if verify and len(verify_clips) < 3 and len(ev) >= 6:
                verify_clips.append((audio.copy(), ev.copy()))
            if (i + 1) % 200 == 0:
                print(f'  {split}: {i+1}/{n} clips…', flush=True)

    sz = os.path.getsize(out) / 1e6
    print(f'{split}: {n} clips, {note_total} notes -> {out} ({sz:.0f} MB)')
    stats = {'per_inst_primary': per_inst_primary, 'per_inst_notes': per_inst_notes,
             'pitch_lo': pitch_lo, 'pitch_hi': pitch_hi, 'bpms': bpms,
             'accomp_flags': accomp_flags, 'note_total': note_total, 'n': n}
    return stats, verify_clips


# ── verification (PROVE IT) ─────────────────────────────────────────────────

def goertzel_power(x, f, sr=SR):
    """Normalized single-bin (Goertzel) power of frequency f in signal x."""
    n = len(x)
    if n < 8:
        return 0.0
    k = 2.0 * math.cos(2.0 * math.pi * f / sr)
    s0 = s1 = s2 = 0.0
    for v in x:
        s0 = v + k * s1 - s2
        s2 = s1; s1 = s0
    power = s1 * s1 + s2 * s2 - k * s1 * s2
    return power / (n * (np.mean(x ** 2) + 1e-12))     # normalize by length + energy


def verify_alignment(clips):
    """For sampled events, Goertzel power at the event's f0 in its window should be
    elevated vs the same f0 measured in the clip's quietest (silent) window."""
    rng = np.random.default_rng(0)
    passed = tested = 0
    for audio, ev in clips:
        # find a quiet reference window (~120 ms) of lowest RMS
        W = int(0.12 * SR)
        best_r, best_i = 1e9, 0
        for st in range(0, max(1, len(audio) - W), W // 2):
            r = np.mean(audio[st:st + W] ** 2)
            if r < best_r:
                best_r, best_i = r, st
        silent = audio[best_i:best_i + W]
        pick = ev[rng.choice(len(ev), size=min(4, len(ev)), replace=False)]
        for (on, off, midi, vel) in pick:
            if off - on < 0.12:
                continue
            f0 = 440.0 * 2 ** ((midi - 69) / 12.0)
            w = audio[int(on * SR):int(off * SR)]
            # subsample to keep the pure-python Goertzel fast
            if len(w) > 4000:
                w = w[:4000]
            sref = silent if len(silent) >= 8 else w
            p_sig = goertzel_power(w, f0)
            p_ref = goertzel_power(sref, f0)
            tested += 1
            if p_sig > 4.0 * (p_ref + 1e-9):
                passed += 1
    frac = passed / max(1, tested)
    ok = frac >= 0.7
    print(f'(b) event->audio alignment: {passed}/{tested} events elevated at their f0 '
          f'({frac*100:.0f}%)  {"PASS" if ok else "FAIL"}')
    return ok


def report(stats, samplers, label):
    print(f'\n── {label} stats ──')
    print(f'clips: {stats["n"]}   notes: {stats["note_total"]}   '
          f'notes/clip: {stats["note_total"]/max(1,stats["n"]):.1f}')
    bpms = np.array(stats['bpms'])
    print(f'(d) tempo range: {bpms.min():.0f}-{bpms.max():.0f} bpm')
    frac_ac = np.mean(stats['accomp_flags'])
    print(f'    accompaniment/bass-chord clips: {frac_ac*100:.0f}%')
    print('    per-instrument (primary clips / notes / pitch range):')
    cover_ok = True
    for inst in INSTRUMENTS:
        lo, hi = stats['pitch_lo'][inst], stats['pitch_hi'][inst]
        reach = samplers[inst].reachable
        rr = f'{lo}-{hi}' if hi >= 0 else 'none'
        # coverage: emitted span should cover a good chunk of the reachable range
        span = (hi - lo) if hi >= 0 else 0
        full = reach[-1] - reach[0]
        frac = span / max(1, full)
        # only judge coverage where we have enough notes for it to be meaningful
        if stats['per_inst_notes'][inst] >= 30 and frac < 0.4:
            cover_ok = False
        print(f'      {inst:9s} {stats["per_inst_primary"][inst]:3d} / '
              f'{stats["per_inst_notes"][inst]:5d} / midi {rr:>7s} '
              f'(reachable {reach[0]}-{reach[-1]}, {frac*100:.0f}% covered)')
    print(f'(c) pitch coverage spans instrument ranges: '
          f'{"PASS" if cover_ok else "FAIL"}')
    return cover_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=1500)
    ap.add_argument('--val', type=int, default=150)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()

    print('loading samplers…')
    samplers = {name: Sampler(name) for name in INSTRUMENTS}
    for name, s in samplers.items():
        print(f'  {name:9s} {len(s.samples):3d} samples, reachable midi '
              f'{s.reachable[0]}-{s.reachable[-1]} ({len(s.reachable)} pitches)')

    tr_stats, tr_vclips = build_split('train', args.train, args.seed, samplers,
                                      verify=args.verify)
    va_stats, va_vclips = build_split('val', args.val, args.seed + 777, samplers,
                                      verify=args.verify)

    a = report(tr_stats, samplers, 'TRAIN')
    report(va_stats, samplers, 'VAL')

    if args.verify:
        print('\n── verification ──')
        vclips = (tr_vclips + va_vclips)[:4]
        # (a) audio isn't silent / sane RMS
        rms_ok = True
        for k, (audio, ev) in enumerate(vclips):
            rms = float(np.sqrt(np.mean(audio ** 2)))
            peak = float(np.max(np.abs(audio)))
            silent = rms < 1e-4
            print(f'(a) clip{k}: rms={rms:.4f} peak={peak:.3f} notes={len(ev)} '
                  f'{"SILENT!" if silent else "ok"}')
            if silent or rms > 1.0:
                rms_ok = False
        print(f'(a) audio non-silent + sane RMS: {"PASS" if rms_ok else "FAIL"}')
        b = verify_alignment(vclips)
        print(f'\nSUMMARY: rms={"PASS" if rms_ok else "FAIL"}  '
              f'align={"PASS" if b else "FAIL"}  coverage={"PASS" if a else "FAIL"}')


if __name__ == '__main__':
    main()
