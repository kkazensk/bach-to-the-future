# generate_1.py

import pickle
import numpy as np
from music21 import stream, note, key as m21key, chord
from music21 import roman
from tensorflow.keras.models import load_model

# Helper to map durations to nearest valid value
def closest_duration(dur):
    valid_durations = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.125]
    return min(valid_durations, key=lambda x: abs(x - dur))

def get_chord_for_note(n, key_obj):
    try:
        scale_degree = key_obj.getScaleDegreeFromPitch(n.pitch)
        chord_map_major = {
            1: ['I', 'vi'],
            2: ['ii'],
            3: ['iii', 'I'],
            4: ['IV', 'ii'],
            5: ['V', 'I'],
            6: ['vi', 'IV'],
            7: ['vii°', 'V'],
        }
        choices = chord_map_major.get(scale_degree, ['I'])
        selected_chord = np.random.choice(choices)
        return roman.RomanNumeral(selected_chord, key_obj).pitches
    except:
        return None

# === Parameters ===
SEQUENCE_LENGTH = 35

# === Load model and mappings ===
print("Loading model and mappings...")
model = load_model('train2.keras')

with open('note_1_mappings.pkl', 'rb') as f:
    note_to_int, int_to_note = pickle.load(f)

with open('duration_1_mappings.pkl', 'rb') as f:
    duration_to_int, int_to_duration = pickle.load(f)

num_notes = len(note_to_int)
num_durations = len(duration_to_int)

# === Ask user for key ===
user_key_input = input("Enter a key (e.g., 'C major', 'D minor'): ")

# === Seed sequence ===
# We'll just pick a random starting point
start_idx = np.random.randint(0, len(note_to_int))
pattern = [start_idx] * SEQUENCE_LENGTH

# === Generate melody ===
generated_notes = []
num_generate = 300  # Number of notes to generate

print(f"Generating melody in {user_key_input}...")

for _ in range(num_generate):
    input_seq = np.array(pattern)
    input_seq = np.reshape(input_seq, (1, SEQUENCE_LENGTH, 1))
    input_seq = input_seq / float(num_notes)

    # Predict pitch and duration using only the first part
    pitch_preds = model.predict(input_seq, verbose=0)[0]  # pitch_0 output
    dur_preds = model.predict(input_seq, verbose=0)[1]    # duration_0 output

    def sample_with_temperature(preds, temperature=1.0):
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds + 1e-8) / temperature  # add small epsilon to prevent log(0)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        preds = preds.flatten()  # ensures 1D
        return np.random.choice(len(preds), p=preds)


    temperature = 1.0  # Try values between 0.7 and 1.2

    pitch_index = sample_with_temperature(pitch_preds, temperature)
    dur_index = sample_with_temperature(dur_preds, temperature)


    pitch = int_to_note[pitch_index]
    duration = float(int_to_duration[dur_index])

    generated_notes.append((pitch, duration))

    # Update pattern
    pattern.append(pitch_index)
    pattern = pattern[1:]

# === Build music21 stream ===
# === Build music21 stream ===
melody_part = stream.Part()
chord_part = stream.Part()

try:
    tonic, mode = user_key_input.strip().split()
    user_key = m21key.Key(tonic, mode)
    melody_part.append(user_key)
    chord_part.append(user_key)
except ValueError:
    print(f"Invalid key input: '{user_key_input}'. Please enter like 'C major' or 'A minor'.")
    exit(1)

chord_probability = 0.3  # 30% of the time, insert a chord

for pitch, dur in generated_notes:
    dur = closest_duration(dur)

    # Add melody
    if '.' in pitch:
        chord_notes = [note.Note(p) for p in pitch.split('.')]
        new_chord = chord.Chord(chord_notes)
        new_chord.quarterLength = dur
        melody_part.append(new_chord)
        lead_note = chord_notes[0]
    else:
        new_note = note.Note(pitch)
        new_note.quarterLength = dur
        melody_part.append(new_note)
        lead_note = new_note

    # Add chord part (optional backing)
    if np.random.rand() < chord_probability:
        chord_pitches = get_chord_for_note(lead_note, user_key)
        if chord_pitches:
            backing_chord = chord.Chord(chord_pitches)
            backing_chord.quarterLength = dur
            chord_part.append(backing_chord)
        else:
            chord_part.append(note.Rest(quarterLength=dur))
    else:
        chord_part.append(note.Rest(quarterLength=dur))




# === Save to MusicXML ===
# === Save to MusicXML with chords ===
score = stream.Score()
score.insert(0, melody_part)
score.insert(0, chord_part) # comment this out to avoid producing chords

output_file = 'generated_melody.xml'
score.write('musicxml', fp=output_file)
print(f"Melody with chords saved to {output_file}!")

