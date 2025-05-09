# generate.py

from music21 import converter, note, chord, stream, key, interval, roman
import numpy as np
import random
import pickle
from tensorflow.keras.models import load_model
from file_converter import convert_musicxml_to_pdf

# --- Duration Helper ---
# This function ensures that the note durations are mapped to a valid value
def get_valid_duration(d):
    valid_durations = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
    # Returns the closest valid duration
    return min(valid_durations, key=lambda x: abs(x - d))

# --- Chord Generation ---
# This function generates a chord based on a given note and key
def get_chord_for_note(n, key_obj):
    try:
        # Get scale degree of the note in the given key
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
        # Get possible chords based on scale degree
        choices = chord_map_major.get(scale_degree, ['I'])
        selected_chord = np.random.choice(choices)
        return roman.RomanNumeral(selected_chord, key_obj).pitches
    except:
        return None

# --- MusicXML Construction ---
# This function creates a MusicXML score using the generated melody and key
def generate_musicxml_with_chords(melody_notes, key_obj, chord_probability=0.3):
    score = stream.Score()  # Initialize a new score object
    melody_part = stream.Part()  # Create part for the melody
    chord_part = stream.Part()  # Create part for the chords

    # Loop through each note and its corresponding duration in the melody
    for n, dur in melody_notes:
        dur = get_valid_duration(dur)  # Ensure valid duration
        # Handle chord if note contains multiple pitches
        if '.' in n:
            try:
                chord_obj = chord.Chord(n.split('.'))  # Create chord from the note string
                note_obj = chord_obj.notes[0]  # Take the first note of the chord
            except:
                continue
        else:
            try:
                note_obj = note.Note(n)  # Create a single note
            except:
                continue

        note_obj.quarterLength = dur  # Set note duration
        melody_part.append(note_obj)  # Append note to the melody part

        # With a probability, add a chord as a backing to the melody
        if random.random() < chord_probability:
            chord_pitches = get_chord_for_note(note_obj, key_obj)  # Get chord for the note
            if chord_pitches:
                backing_chord = chord.Chord(chord_pitches)  # Create the backing chord
                backing_chord.quarterLength = dur  # Set duration
                chord_part.append(backing_chord)  # Add to chord part
            else:
                chord_part.append(note.Rest(quarterLength=dur))  # If no chord, add a rest
        else:
            chord_part.append(note.Rest(quarterLength=dur))  # If no chord, add a rest

    score.insert(0, melody_part)  # Insert melody part into score
    score.insert(0, chord_part)   # Insert chord part into score
    return score  # Return the constructed score

# --- Melody Generation ---
# This function generates the melody using the trained model
def generate_music_multi(model, seed_sequence, all_unique_notes, length=200, temperature=1.0):
    output_notes = []  # List to hold generated notes
    sequence = seed_sequence  # Initial sequence for prediction

    # Generate a sequence of notes
    for _ in range(length):
        pred_pitch_probs, pred_dur = model.predict(sequence, verbose=0)  # Predict pitch and duration
        pred_pitch_probs = pred_pitch_probs[0].astype(np.float64)  # Convert pitch predictions to float

        # Apply temperature scaling for diversity in predictions
        pred_pitch_probs = np.log(pred_pitch_probs + 1e-8) / temperature
        exp_preds = np.exp(pred_pitch_probs)
        pred_pitch_probs = exp_preds / np.sum(exp_preds)  # Normalize the probabilities

        # Select the note based on predicted probabilities
        pred_index = np.random.choice(len(pred_pitch_probs), p=pred_pitch_probs)
        predicted_note = all_unique_notes[pred_index]  # Get the predicted note
        predicted_duration = max(0.25, float(pred_dur[0][0]))  # Ensure minimum duration
        predicted_duration = get_valid_duration(predicted_duration)  # Map to valid duration

        output_notes.append((predicted_note, predicted_duration))  # Append the note and duration to output

        # Update sequence for next prediction
        next_input = np.zeros_like(sequence)
        next_input[0][:-1] = sequence[0][1:]
        next_input[0][-1] = pred_index / float(len(all_unique_notes))
        sequence = next_input  # Update sequence for next iteration

    return output_notes  # Return the generated notes

# --- Transposition ---
# This function transposes the notes to a different key
def transpose_notes(notes, interval_obj):
    transposed_notes = []
    # Transpose each note or chord
    for n in notes:
        try:
            if isinstance(n, note.Note):
                transposed_notes.append(n.transpose(interval_obj))  # Transpose single note
            elif isinstance(n, chord.Chord):
                # Transpose each pitch in the chord
                transposed_chord = chord.Chord([p.transpose(interval_obj) for p in n.pitches])
                transposed_notes.append(transposed_chord)  # Append transposed chord
            else:
                transposed_notes.append(n)
        except Exception as e:
            print(f"Error transposing note/chord {n}: {e}")
            continue
    return transposed_notes  # Return transposed notes

# === Main ===
# Get user input for target key (e.g., 'C major', 'D minor')
target_key = input("Enter desired output key (e.g., 'C major', 'D minor'): ").strip()
try:
    # Parse the input into tonic and mode (major or minor)
    user_key_obj = key.Key(target_key.split()[0], target_key.split()[1])
except:
    print(f"Invalid key format '{target_key}'.")  # Handle invalid input format
    exit()

print("Loading model and mappings...")
model = load_model("lstm_music_model.keras")  # Load the trained model
with open("note_mappings.pkl", "rb") as f:
    note_to_int, int_to_note, all_unique_notes = pickle.load(f)  # Load mappings for notes

print("Generating music...")
# Generate a seed sequence for the melody generation
sequence_length = 35
seed = np.random.randint(0, len(all_unique_notes), sequence_length)  # Random initial seed
seed_input = np.reshape(seed, (1, sequence_length, 1)) / float(len(all_unique_notes))  # Reshape and normalize
generated_sequence = generate_music_multi(model, seed_input, all_unique_notes, length=200)  # Generate melody

print("Transposing to target key...")
# Assume original key is C, transpose to target key
original_key = key.Key("C")  # Original training key (C major)
interval_obj = interval.Interval(original_key.tonic, user_key_obj.tonic)  # Compute transposition interval
transposed_sequence = []
for n, d in generated_sequence:
    try:
        # Transpose each note in the generated sequence
        temp_note = note.Note(n)
        transposed = transpose_notes([temp_note], interval_obj)[0]
        transposed_sequence.append((transposed.nameWithOctave, d))  # Append transposed note with duration
    except:
        continue

print("Saving output...")
# Generate the final MusicXML with chords
final_score = generate_musicxml_with_chords(transposed_sequence, user_key_obj)
output_path = "new_music"  # Set output path
final_score.write('musicxml', fp=output_path+'.xml')  # Save the generated music to MusicXML file
convert_musicxml_to_pdf(output_path+'.xml', output_path, 'generated'+output_path)
print(f"Saved: {output_path}")  # Print confirmation message
