# generate.py
from music21 import converter, note, chord, stream, key, interval, roman
import numpy as np
import random
import pickle
from tensorflow.keras.models import load_model

# --- Duration Helper ---
def get_valid_duration(d):
    valid_durations = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
    return min(valid_durations, key=lambda x: abs(x - d))

# --- Chord Generation ---
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

# --- MusicXML Construction ---
def generate_musicxml_with_chords(melody_notes, key_obj, chord_probability=0.3):
    score = stream.Score()
    melody_part = stream.Part()
    chord_part = stream.Part()

    for n, dur in melody_notes:
        dur = get_valid_duration(dur)
        if '.' in n:
            try:
                chord_obj = chord.Chord(n.split('.'))
                note_obj = chord_obj.notes[0]
            except:
                continue
        else:
            try:
                note_obj = note.Note(n)
            except:
                continue

        note_obj.quarterLength = dur
        melody_part.append(note_obj)

        if random.random() < chord_probability:
            chord_pitches = get_chord_for_note(note_obj, key_obj)
            if chord_pitches:
                backing_chord = chord.Chord(chord_pitches)
                backing_chord.quarterLength = dur
                chord_part.append(backing_chord)
            else:
                chord_part.append(note.Rest(quarterLength=dur))
        else:
            chord_part.append(note.Rest(quarterLength=dur))

    score.insert(0, melody_part)
    score.insert(0, chord_part)
    return score

# --- Melody Generation ---
def generate_music_multi(model, seed_sequence, all_unique_notes, length=200, temperature=1.0):
    output_notes = []
    sequence = seed_sequence

    for _ in range(length):
        pred_pitch_probs, pred_dur = model.predict(sequence, verbose=0)
        pred_pitch_probs = pred_pitch_probs[0].astype(np.float64)

        pred_pitch_probs = np.log(pred_pitch_probs + 1e-8) / temperature
        exp_preds = np.exp(pred_pitch_probs)
        pred_pitch_probs = exp_preds / np.sum(exp_preds)

        pred_index = np.random.choice(len(pred_pitch_probs), p=pred_pitch_probs)
        predicted_note = all_unique_notes[pred_index]
        predicted_duration = max(0.25, float(pred_dur[0][0]))
        predicted_duration = get_valid_duration(predicted_duration)

        output_notes.append((predicted_note, predicted_duration))

        next_input = np.zeros_like(sequence)
        next_input[0][:-1] = sequence[0][1:]
        next_input[0][-1] = pred_index / float(len(all_unique_notes))
        sequence = next_input

    return output_notes

# --- Transposition ---
def transpose_notes(notes, interval_obj):
    transposed_notes = []
    for n in notes:
        try:
            if isinstance(n, note.Note):
                transposed_notes.append(n.transpose(interval_obj))
            elif isinstance(n, chord.Chord):
                transposed_chord = chord.Chord([p.transpose(interval_obj) for p in n.pitches])
                transposed_notes.append(transposed_chord)
            else:
                transposed_notes.append(n)
        except Exception as e:
            print(f"Error transposing note/chord {n}: {e}")
            continue
    return transposed_notes

# === Main ===
target_key = input("Enter desired output key (e.g., 'C major', 'D minor'): ").strip()
try:
    user_key_obj = key.Key(target_key.split()[0], target_key.split()[1])
except:
    print(f"Invalid key format '{target_key}'.")
    exit()

print("Loading model and mappings...")
model = load_model("lstm_music_model.keras")
with open("note_mappings.pkl", "rb") as f:
    note_to_int, int_to_note, all_unique_notes = pickle.load(f)

print("Generating music...")
sequence_length = 35
seed = np.random.randint(0, len(all_unique_notes), sequence_length)
seed_input = np.reshape(seed, (1, sequence_length, 1)) / float(len(all_unique_notes))
generated_sequence = generate_music_multi(model, seed_input, all_unique_notes, length=200)

print("Transposing to target key...")
original_key = key.Key("C")  # Assume original training key was C
interval_obj = interval.Interval(original_key.tonic, user_key_obj.tonic)
transposed_sequence = []
for n, d in generated_sequence:
    try:
        temp_note = note.Note(n)
        transposed = transpose_notes([temp_note], interval_obj)[0]
        transposed_sequence.append((transposed.nameWithOctave, d))
    except:
        continue

print("Saving output...")
final_score = generate_musicxml_with_chords(transposed_sequence, user_key_obj)
output_path = "new_music.xml"
final_score.write('musicxml', fp=output_path)
print(f"Saved: {output_path}")
