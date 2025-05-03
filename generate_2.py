import pickle
import numpy as np
from music21 import stream, note, chord, metadata, key
from tensorflow.keras.models import load_model

# === Constants ===
SEQUENCE_LENGTH = 35  # Length of the input sequence for the model
NUM_PARTS = 4  # Number of parts to generate (e.g., 4 parts for a string quartet)
VALID_DURATIONS = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.125]  # Valid note durations (in quarter lengths)

# === Load model and mappings ===
model = load_model('train2.keras')  # Load the trained model from a file

# Load the mappings for notes and durations
with open('note_1_mappings.pkl', 'rb') as f:
    note_to_int, int_to_note = pickle.load(f)  # Mapping of note names to integers and vice versa

with open('duration_1_mappings.pkl', 'rb') as f:
    duration_to_int, int_to_duration = pickle.load(f)  # Mapping of durations to integers and vice versa

num_notes = len(note_to_int)  # Total number of distinct notes
num_durations = len(duration_to_int)  # Total number of distinct durations

# === User Input for Key ===
key_input = input("Enter the desired key (e.g., C major, A minor): ")  # Prompt the user for a key input
try:
    # Split the input into tonic (note) and mode (major/minor)
    tonic, mode = key_input.split()
    if mode not in ['major', 'minor']:
        raise ValueError("Invalid mode. Use 'major' or 'minor'.")
    music_key = key.Key(tonic, mode)  # Set the key based on user input
    print(f"Setting key to: {music_key}")
except ValueError as e:
    print(f"Error: {e}. Defaulting to C major.")  # Default to C major if input is invalid
    music_key = key.Key('C', 'major')  # Default key

# === Function to add randomness ===
def sample_prediction(predictions, temperature=1.0):
    """
    Sample a prediction from the model's output using a probability distribution.
    The temperature parameter controls the randomness: higher values increase randomness.
    """
    predictions = np.log(predictions + 1e-7) / temperature  # Apply temperature scaling
    predictions = np.exp(predictions) / np.sum(np.exp(predictions))  # Convert to probability distribution
    
    # Ensure predictions is 1D
    predictions = predictions.flatten()  # Flatten in case it's multidimensional
    
    # Sample a note or duration based on the probabilities
    return np.random.choice(len(predictions), p=predictions)


# === Generate notes ===
generated_parts = [[] for _ in range(NUM_PARTS)]  # Initialize a list to hold generated notes for each part

# Generate each part independently
for part in range(NUM_PARTS):
    # Create a unique seed pattern for each part (starting sequence of notes)
    start = np.random.randint(0, len(note_to_int) - SEQUENCE_LENGTH)
    seed_notes = list(note_to_int.values())[start:start + SEQUENCE_LENGTH]
    pattern = seed_notes.copy()

    for _ in range(100):  # Number of steps to generate per part (can adjust for longer/shorter pieces)
        input_seq = np.reshape(pattern, (1, SEQUENCE_LENGTH, 1))  # Reshape the pattern for the model
        input_seq = input_seq / float(num_notes)  # Normalize input values

        predictions = model.predict(input_seq, verbose=0)  # Get predictions from the model

        pitch_pred = predictions[part * 2]  # Pitch prediction for this part
        dur_pred = predictions[part * 2 + 1]  # Duration prediction for this part

        # Sample pitch and duration using the predictions
        pitch_index = sample_prediction(pitch_pred, temperature=1.0)  # Adjust temperature for randomness
        dur_index = np.argmax(dur_pred)  # Pick the most likely duration

        pitch = int_to_note[pitch_index]  # Convert index to pitch (note name)
        dur_str = int_to_duration[dur_index]  # Convert index to duration

        try:
            dur = float(dur_str)  # Convert the duration to a float
        except ValueError:
            dur = 1.0  # Fallback in case of error

        # Ensure the duration is valid
        if dur not in VALID_DURATIONS:
            dur = min(VALID_DURATIONS, key=lambda x: abs(x - dur))  # Choose the closest valid duration

        generated_parts[part].append((pitch, dur))  # Append the generated note and duration to the part

        # Update the seed pattern for the next note
        pattern.append(pitch_index)  # Add the new note to the pattern
        pattern = pattern[1:]  # Remove the first note to maintain sequence length

# === Create Music21 Stream ===
score = stream.Score()  # Create a new Music21 score object
score.insert(0, metadata.Metadata())  # Add metadata (composer, title, etc.)
score.metadata.title = "Generated Composition"
score.metadata.composer = "AI Composer"
score.insert(0, music_key)  # Set the key signature for the score

# Add each part (generated notes) to the score
for i in range(NUM_PARTS):
    part = stream.Part()  # Create a new part for each generated part

    for p, d in generated_parts[i]:
        # If the pitch is a chord (contains a dot), create a chord object
        if '.' in p or p.isdigit():
            notes_in_chord = p.split('.')
            chord_notes = [note.Note(n) for n in notes_in_chord]
            for n in chord_notes:
                n.duration.quarterLength = d  # Set the duration of each note in the chord
            new_chord = chord.Chord(chord_notes)
            new_chord.duration.quarterLength = d  # Set the chord's duration
            part.append(new_chord)  # Add the chord to the part
        else:
            # Otherwise, create a single note object
            n = note.Note(p)
            n.duration.quarterLength = d  # Set the note's duration
            part.append(n)  # Add the note to the part

    score.append(part)  # Add the part to the score

# === Save the output ===
score.write('musicxml', fp='generated_music.xml')  # Write the score to a MusicXML file
print("Generated music saved to generated_music.xml")  # Confirmation message
