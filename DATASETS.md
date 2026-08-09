# Datasets — license verdicts & sources

METAGRAPHE-TRAIN ships inside a **commercial** product (KANON), so every dataset
that touches a training run must be commercially licensable. This file records
the verdict and source for each real-recording set considered as a
multi-instrument complement to our synth renders + guitar sets. Verified
2026-08-09.

Quick map of what feeds the model today:

| set | content | license | commercial? | in the mix |
|---|---|---|---|---|
| GuitarSet | 3 h guitar, hexaphonic truth | CC-BY-4.0 | ✅ yes | packed |
| GAPS | 14 h classical/fingerstyle guitar | MIT | ✅ yes | packed |
| **MusicNet** | **~34 h real chamber: piano/strings/winds** | **CC-BY-4.0** | **✅ yes** | **packed (new)** |
| synth banks (Iowa/Symphonia) | our own procedural renders | ours | ✅ yes | packed |
| URMP | 3.75 h real chamber (strings/winds) | unlabeled (Dryad) | ⚠️ unconfirmed | skipped — see below |
| Slakh2100 | synthetic multitrack renders | ambiguous (see below) | ⚠️/❌ | skipped — see below |
| MAESTRO | piano | **NC** | ❌ no | never — NC |

---

## MusicNet — PACKED ✅ (primary target)

- **Content:** 330 freely-licensed classical recordings (~34 h), >1M per-note
  labels; real piano, violin, viola, cello, contrabass, horn, oboe, bassoon,
  clarinet, flute. The real chamber material our synth + guitar mixes lack.
- **License verdict: CC-BY-4.0 (Creative Commons Attribution 4.0).**
  Commercial use is **permitted** with attribution. Confirmed directly on the
  Zenodo record's license field ("re-distribution and re-use ... on the
  condition that the creator is appropriately credited").
  - Source of truth: https://zenodo.org/records/5120004 (license: CC-BY-4.0)
  - Project homepage: https://johnthickstun.com/musicnet.html (redirect from the
    old `homes.cs.washington.edu/~thickstn/musicnet.html`)
  - **Attribution to carry into the product:** Thickstun, Harchaoui, Kakade,
    *"Learning Features of Music from Scratch"* (ICLR 2017); MusicNet, CC-BY-4.0.
  - ⚠️ Do **not** confuse with **MusicNet-EM** / **MusicNet-16k-EM** (refined EM
    labels, Zenodo 8021437 / 10009959) — those are **CC-BY-NC-SA-4.0 (NC)**. We
    use the ORIGINAL `musicnet.tar.gz` labels only.
- **Download (verified reachable 2026-08-09, HTTP 200):**
  - `musicnet.tar.gz` — 11.1 GB — https://zenodo.org/records/5120004/files/musicnet.tar.gz
  - `musicnet_metadata.csv` — 44 kB — https://zenodo.org/records/5120004/files/musicnet_metadata.csv
  - (`musicnet_midis.tar.gz` — 2.6 MB — unaligned MIDI; not used)
  - Extracts to ~50 GB: `{train,test}_{data,labels}/`. *_data = 44.1 kHz mono
    WAV; *_labels = one CSV per recording.
- **Label format (handled by `pack_musicnet.py`):** columns
  `start_time,end_time,instrument,note,start_beat,end_beat,note_value`.
  `start_time`/`end_time` are **sample indices at 44100 Hz** — we divide by
  44100 to get seconds. `note` is already a MIDI pitch; `instrument` is a GM
  program number. Velocity is unlabeled → constant default (100).
- **Packer:** `pack_musicnet.py` → resample to 16 kHz mono, events
  `(onset_s, offset_s, midi_pitch, vel)`, same HDF5 schema as
  `pack_guitarset.py` (`clipNNNN/{audio,events}`, attrs `name`,`instrument`).
  Uses MusicNet's **official train/test split** (the tar's directory split),
  carves a small evenly-spaced val out of train, and also packs the official
  test set → `musicnet_{train,val,test}.h5`. Streams recording-by-recording with
  progress; `--limit N` caps recordings if disk is tight.
- **On the pod:** `cloud/bootstrap.sh` downloads + extracts + packs it (guarded;
  `MUSICNET=0` to skip; deletes raw WAVs after packing to reclaim ~50 GB).

## URMP — skipped ⚠️ (license not conclusively commercial-clean)

- **Content:** University of Rochester Multi-modal Music Performance — 44 real
  chamber pieces (~3.75 h), strings + winds, per-note truth. Same instrument
  families MusicNet already covers, so limited marginal value once MusicNet is in.
- **License verdict: UNCONFIRMED.** The Dryad record shows only a generic
  "Content on this site is licensed for reuse" and no explicit CC tag on the
  dataset page. Dryad's submission policy defaults to **CC0** (which *would* be
  commercial-OK), but the record is not explicitly labeled, and the audio is
  performer recordings. For a commercial product we don't ship on an inferred
  license.
  - Source: https://datadryad.org/dataset/doi:10.5061/dryad.ng3r749 (DOI
    10.5061/dryad.ng3r749); homepage https://labsites.rochester.edu/air/projects/URMP.html
  - Download: `Dataset.tar.gz` ~12.1 GB.
- **Decision:** skipped to stay conservative. Revisit only if we confirm CC0/CC-BY
  in writing from the authors — and even then it's largely redundant with MusicNet.

## Slakh2100 — skipped ❌ (redundant + license ambiguity + size)

- **Content:** 2100 **synthetic** multitrack renders (Lakh MIDI → sample-based
  synthesis), ~145 h. "Redux" dedupes to 1710 tracks (1289/270/151).
- **License verdict: AMBIGUOUS across mirrors.** The Zenodo record (4599666)
  labels it **CC-BY-4.0**, but redistributed variants (e.g. HF
  `Slakh2100-FLAC-Redux-Reduced`) are tagged **CC-BY-NC-SA-4.0**, and the
  underlying compositions come from web-scraped Lakh MIDI (murky compositional
  provenance). Not clean enough to bet a commercial release on.
  - Sources: https://zenodo.org/records/4599666 · http://www.slakh.com/ ·
    https://huggingface.co/datasets/DreamyWanderer/Slakh2100-FLAC-Redux-Reduced
- **Also disqualifying regardless of license:** it's **synthetic** — exactly what
  our own synth-bank pipeline already generates with exact truth — and it's
  ~105 GB compressed / ~500 GB uncompressed, far outside this pod's disk budget.
- **Decision:** skipped. No marginal value over our own renders; not worth the
  license risk or the terabyte.

## MAESTRO — never

Piano, non-commercial license. We only ever consume ByteDance's Apache-2.0
*weights* (pretrained on MAESTRO by them); MAESTRO audio never enters OUR runs.

---

**Bottom line:** MusicNet (CC-BY-4.0) is the real multi-instrument win and is
packed. URMP and Slakh are skipped — URMP on unconfirmed licensing, Slakh on
license ambiguity + redundancy with our synth data + prohibitive size. The
commercially-clean real-recording mix is therefore **GuitarSet + GAPS + MusicNet**
plus our own synth banks.
