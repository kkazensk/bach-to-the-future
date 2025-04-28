# train.py

# === Import Libraries ===
from music21 import converter, note, chord
import os
import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# === Data Extraction Functions ===

def get_notes_and_chords(xml_file):
    # Parse a MusicXML file and extract (pitch, duration) tuples
    score = converter.parse(xml_file)
    notes = []
    for element in score.flat.notes:
        try:
            pitch_str = ""
            duration = float(element.quarterLength)  # Get duration in quarter lengths
            if isinstance(element, note.Note):
                pitch_str = str(element.pitch)  # Single note
            elif isinstance(element, chord.Chord):
                pitch_str = '.'.join(str(p) for p in element.pitches)  # Multiple pitches joined by '.'
            if pitch_str:
                notes.append((pitch_str, duration))
        except:
            continue  # Skip any element that raises an error
    return notes

def process_files(input_dir):
    # Process all MusicXML files in the input directory
    all_notes = []
    for file in os.listdir(input_dir):
        if file.endswith(('.xml', '.mxl', '.musicxml')):
            try:
                notes = get_notes_and_chords(os.path.join(input_dir, file))
                all_notes.append(notes)
            except:
                print(f"Skipping {file} due to parsing error.")  # Handle broken files gracefully
    return all_notes

def prepare_sequences(note_tuples, sequence_length=25):
    # Prepare input-output sequences for model training
    input_seq, output_seq_pitch, output_seq_dur = [], [], []
    for i in range(len(note_tuples) - sequence_length):
        seq = note_tuples[i:i+sequence_length]
        input_seq.append(seq)
        output_seq_pitch.append(note_tuples[i+sequence_length][0])  # Next pitch to predict
        output_seq_dur.append(note_tuples[i+sequence_length][1])    # Next duration to predict
    return input_seq, output_seq_pitch, output_seq_dur

def build_model(input_shape, num_pitch_classes):
    # Build the LSTM model with two outputs: pitch and duration
    inp = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)
    pitch_output = Dense(num_pitch_classes, activation='softmax', name='pitch')(x)  # Categorical output for pitch
    dur_output = Dense(1, activation='linear', name='duration')(x)  # Regression output for duration
    model = Model(inputs=inp, outputs=[pitch_output, dur_output])
    model.compile(loss={'pitch': 'categorical_crossentropy', 'duration': 'mse'}, optimizer='adam')
    return model

# === Main Training Flow ===

# Get dataset path from user
input_dir = input("Enter the path to your MusicXML dataset: ").strip()

print("Loading dataset...")
dataset = process_files(input_dir)

# Flatten dataset into one list of notes and durations
all_notes = [item[0] for sublist in dataset for item in sublist]
all_durations = [item[1] for sublist in dataset for item in sublist]

# Map notes to integers for model training
all_unique_notes = sorted(set(all_notes))
note_to_int = {note: number for number, note in enumerate(all_unique_notes)}
int_to_note = {number: note for note, number in note_to_int.items()}

# Save mappings to a pickle file for later use (during generation)
with open("note_mappings.pkl", "wb") as f:
    pickle.dump((note_to_int, int_to_note, all_unique_notes), f)

print("Preparing sequences...")
# Create input-output sequences
note_tuples = list(zip(all_notes, all_durations))
sequence_length = 35
input_seq_tuples, output_seq_pitch, output_seq_dur = prepare_sequences(note_tuples, sequence_length)

# Prepare input array (normalized) and output arrays
X = np.array([[note_to_int[n[0]] for n in seq] for seq in input_seq_tuples])
X = np.reshape(X, (len(X), sequence_length, 1)) / float(len(all_unique_notes))  # Normalize input

y_pitch = to_categorical([note_to_int[n] for n in output_seq_pitch], num_classes=len(all_unique_notes))  # One-hot encoding for pitch
y_dur = np.array(output_seq_dur).reshape(-1, 1)  # Duration as a regression target

print("Building and training model...")
# Build and train the LSTM model
model = build_model((sequence_length, 1), len(all_unique_notes))
model.fit(X, {'pitch': y_pitch, 'duration': y_dur}, epochs=10, batch_size=64)

print("Saving model...")
# Save the trained model in TensorFlow's native format
model.save("lstm_music_model.keras")
print("Training complete! Model and mappings saved.")
