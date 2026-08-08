# METAGRAPHE-TRAIN — the general transcription model for KANON

Goal (user): audio→MIDI that makes people say WOW — **general-purpose** (any
instrument), accurate, running 100% offline inside KANON. Acceptance test #1:
the fingerstyle waltz (bass note beat 1, chords beats 2–3) must FEEL right.

## Architecture
Kong et al. high-resolution onset/offset regression (arXiv 2010.01815),
~20M params: logmel → ConvBlocks → biGRU → per-frame regression heads
(onset, offset, frame, velocity). Vendored: vendor/piano_transcription
(bytedance, **Apache-2.0**). The guitar-domain paper (2402.15258) proved this
arch +18 F1 over basic-pitch on guitar by fine-tuning — we replicate that
recipe but GENERAL, not guitar-only.

## Strategy: fine-tune, not scratch
Start from bytedance's released pretrained checkpoint (Apache-2.0 weight
release; trained by them on MAESTRO — their license grant covers the weights,
we never touch MAESTRO audio ourselves). Fine-tune on commercially-clean
multi-instrument data only.

## Datasets (license-gated — commercial product)
| set | content | license | status |
|---|---|---|---|
| GAPS (HF xavriley/GAPS) | 14h classical/fingerstyle guitar | MIT | to download |
| GuitarSet | 3h guitar w/ hexaphonic truth | CC-BY 4.0 | to download |
| MusicNet | 34h chamber (strings/winds/piano) | CC-BY 4.0 | verify + download |
| URMP | ensemble strings/winds | CHECK license (likely CC-BY-NC → skip if NC) |
| Slakh2100 | synth multi-instrument renders | CC-BY (check variant) |
| Iowa/Symphonia samples (in kanon repo) | single notes, all orchestral | public domain | synth-render mixes w/ exact truth |
| KS/synth renders (own code) | unlimited procedurally-labeled audio | ours | augmentation |
**MAESTRO: NC license — never in OUR training runs.**

## Pipeline
1. venv + torch/MPS (Apple Silicon) — in progress
2. bytedance pretrained checkpoint (Zenodo) → verify inference runs on a wav
3. Dataset download + unified HDF5 packing (audio 16k + note events)
4. Fine-tune: full 21–108 range, mixed-instrument batches, MPS (fallback:
   scope a cloud GPU day if MPS too slow — be honest about it)
5. Eval harness: note-F1 (onset±50ms, w/ and w/o offset) per instrument vs
   basic-pitch on held-out splits + the user's waltz (kanon scratchpad howl.wav)
6. Export: torch.onnx → int8 quantize (~15–25MB) → onnxruntime-web WASM
7. KANON integration: new engine beside basic-pitch (fallback stays);
   port the high-res regression note-decoder to TS; reuse the whole KANON
   post-stack (onset snap, harmonic verify, merge, bends)

## Guardrails
- Every training claim measured on held-out data; never eval on train.
- License check BEFORE a dataset enters the mix (commercial product).
- KANON app keeps working the whole time — model swaps in only on a
  measured, decisive win (per-instrument + waltz).
