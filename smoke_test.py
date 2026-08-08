"""Smoke test: the pretrained Kong checkpoint transcribes real guitar audio
end-to-end (proves checkpoint + arch + inference pipeline before any training)."""
import sys, time
sys.path.insert(0, 'vendor/piano_transcription/pytorch')
sys.path.insert(0, 'vendor/piano_transcription/utils')
import librosa
from inference import PianoTranscription

wav = 'data/guitarset/audio/00_BN1-129-Eb_comp_mic.wav'
audio, _ = librosa.load(wav, sr=16000, mono=True)
print(f'audio {len(audio)/16000:.1f}s')
t0 = time.time()
tr = PianoTranscription('Note_pedal', device='cpu', checkpoint_path='checkpoints/note_model.pth')
out = tr.transcribe(audio, 'smoke_out.mid')
notes = out['est_note_events']
print(f'transcribed in {time.time()-t0:.1f}s -> {len(notes)} notes')
print('first 5:', [(round(n["onset_time"],2), int(n["midi_note"])) for n in notes[:5]])
