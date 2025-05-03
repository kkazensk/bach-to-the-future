# generate_1.py

import pickle
import numpy as np
from music21 import stream, note, key as m21key, chord
from music21 import roman
from tensorflow.keras.models import load_model

# Helper function to map a duration to the closest valid duration from a predefined list
def closest_duration(dur):
    valid_durations = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.125]
    return min(valid_durations, key=lambda x: abs(x - dur))  # Return the duration that is closest to the input

# Helper function to get a chord for a note based on its scale degree in the given key
def get_chord_for_note(n, key_obj):
    try:
        scale_degree = key_obj.getScaleDegreeFromPitch(n.pitch)  # Get the scale degree of the pitch
        # Define possible chords for each scale degree in a major key
        chord_map_major = {
            1: ['I', 'vi'],
            2: ['ii'],
            3: ['iii', 'I'],
            4: ['IV', 'ii'],
            5: ['V', 'I'],
            6: ['vi', 'IV'],
            7: ['vii°', 'V'],
        }
        # Choose a chord for the scale degree from the predefined list
        choices = chord_map_major.get(scale_degree, ['I'])
        selected_chord = np.random.choice(choices)
        # Return the pitches of the selected chord in Roman numeral form
        return roman.RomanNumeral(selected_chord, key_obj).pitches
    except:
        return None  # If something goes wrong, return None

# === Parameters ===
SEQUENCE_LENGTH = 35  # Length of the input sequence for the model

# === Load model and mappings ===
print("Loading model and mappings...")
model = load_model('train2.keras')  # Load the pre-trained model

# Load mappings for note-to-int and int-to-note for pitch and duration
with open('note_1_mappings.pkl', 'rb') as f:
    note_to_int, int_to_note = pickle.load(f)

with open('duration_1_mappings.pkl', 'rb') as f:
    duration_to_int, int_to_duration = pickle.load(f)

num_notes = len(note_to_int)  # Number of possible notes
num_durations = len(duration_to_int)  # Number of possible durations

# === Ask user for key ===
user_key_input = input("Enter a key (e.g., 'C major', 'D minor'): ")  # Prompt user to enter a key

# === Seed sequence ===
# Randomly pick a starting point for the melody sequence
start_idx = np.random.randint(0, len(note_to_int))
pattern = [start_idx] * SEQUENCE_LENGTH  # Initialize the sequence with the starting note

# === Generate melody ===
generated_notes = []  # List to store the generated notes
num_generate = 300  # Number of notes to generate

print(f"Generating melody in {user_key_input}...")

for _ in range(num_generate):
    input_seq = np.array(pattern)  # Convert pattern to a numpy array
    input_seq = np.reshape(input_seq, (1, SEQUENCE_LENGTH, 1))  # Reshape the sequence for model input
    input_seq = input_seq / float(num_notes)  # Normalize the sequence

    # Predict pitch and duration using the trained model
    pitch_preds = model.predict(input_seq, verbose=0)[0]  # Predict pitch for part 0
    dur_preds = model.predict(input_seq, verbose=0)[1]    # Predict duration for part 0

    # Function to sample a prediction using temperature-based scaling
    def sample_with_temperature(preds, temperature=1.0):
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds + 1e-8) / temperature  # Apply temperature scaling (log to prevent very small numbers)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)  # Normalize to get a probability distribution
        preds = preds.flatten()  # Flatten the predictions to 1D
        return np.random.choice(len(preds), p=preds)  # Sample a note/duration based on probabilities

    temperature = 1.0  # Set temperature for sampling (1.0 means no scaling)

    # Sample pitch and duration
    pitch_index = sample_with_temperature(pitch_preds, temperature)
    dur_index = sample_with_temperature(dur_preds, temperature)

    pitch = int_to_note[pitch_index]  # Get the note name from index
    duration = float(int_to_duration[dur_index])  # Get the duration in float

    generated_notes.append((pitch, duration))  # Add the note and duration to the list

    # Update the sequence pattern
    pattern.append(pitch_index)
    pattern = pattern[1:]  # Remove the oldest note to maintain sequence length

# === Build music21 stream ===
melody_part = stream.Part()  # Create a part for the melody
chord_part = stream.Part()  # Create a part for the chords

# Try to parse the user input key into tonic and mode
try:
    tonic, mode = user_key_input.strip().split()
    user_key = m21key.Key(tonic, mode)  # Create a music21 Key object
    melody_part.append(user_key)  # Add the key signature to the melody part
    chord_part.append(user_key)   # Add the key signature to the chord part
except ValueError:
    print(f"Invalid key input: '{user_key_input}'. Please enter like 'C major' or 'A minor'.")
    exit(1)  # Exit the program if key input is invalid

chord_probability = 0.3  # Probability of inserting a chord (30%)

# Generate melody and add chords (if applicable)
for pitch, dur in generated_notes:
    dur = closest_duration(dur)  # Map the duration to the closest valid value

    # Add melody note or chord
    if '.' in pitch:
        chord_notes = [note.Note(p) for p in pitch.split('.')]  # Split the pitch into multiple notes (chord)
        new_chord = chord.Chord(chord_notes)
        new_chord.quarterLength = dur  # Set the duration for the chord
        melody_part.append(new_chord)  # Append the chord to the melody part
        lead_note = chord_notes[0]  # Set the first note of the chord as the lead note
    else:
        new_note = note.Note(pitch)  # Create a new note
        new_note.quarterLength = dur  # Set the duration
        melody_part.append(new_note)  # Append the note to the melody part
        lead_note = new_note  # Set the note as the lead note

    # 30% chance of adding a chord for backing
    if np.random.rand() < chord_probability:
        chord_pitches = get_chord_for_note(lead_note, user_key)  # Get chord pitches based on the lead note
        if chord_pitches:
            backing_chord = chord.Chord(chord_pitches)  # Create a chord from the pitches
            backing_chord.quarterLength = dur  # Set the duration for the chord
            chord_part.append(backing_chord)  # Append the chord to the chord part
        else:
            chord_part.append(note.Rest(quarterLength=dur))  # If no chord, add a rest
    else:
        chord_part.append(note.Rest(quarterLength=dur))  # Add a rest if no chord is generated

# === Save to MusicXML ===
# Save the generated melody and chords to a MusicXML file
score = stream.Score()  # Create a score
score.insert(0, melody_part)  # Add the melody part to the score
score.insert(0, chord_part)   # Add the chord part to the score (comment out to avoid chords)

output_file = 'generated_melody.xml'  # Output file name
score.write('musicxml', fp=output_file)  # Write the score to a MusicXML file
print(f"Melody with chords saved to {output_file}!")  # Notify the user
