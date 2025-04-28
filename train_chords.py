# train_chords.py
from music21 import converter, note, chord
import os
import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# === Data Extraction ===
def get_notes_and_chords(xml_file):
    """Extracts notes and chords from a MusicXML file, returning (pitch, duration) pairs."""
    score = converter.parse(xml_file)
    notes = []
    for element in score.flat.notes:
        try:
            pitch_str = ""
            duration = float(element.quarterLength)
            if isinstance(element, note.Note):
                pitch_str = str(element.pitch)  # Single note
            elif isinstance(element, chord.Chord):
                pitch_str = '.'.join(str(p) for p in element.pitches)  # Chord as joined pitches
            if pitch_str:
                notes.append((pitch_str, duration))
        except:
            continue  # Skip elements that cause errors
    return notes

def process_files(input_dir):
    """Processes all MusicXML files in a directory, extracting note sequences."""
    all_notes = []
    for file in os.listdir(input_dir):
        if file.endswith(('.xml', '.mxl', '.musicxml')):
            try:
                notes = get_notes_and_chords(os.path.join(input_dir, file))
                all_notes.append(notes)
            except:
                print(f"Skipping {file} due to parsing error.")
    return all_notes

def prepare_sequences(note_tuples, sequence_length=25):
    """
    Splits the note data into input-output sequences for training.
    - input_seq: sequences of notes and durations
    - output_seq_pitch: next note (pitch) after sequence
    - output_seq_dur: next note (duration) after sequence
    """
    input_seq, output_seq_pitch, output_seq_dur = [], [], []
    for i in range(len(note_tuples) - sequence_length):
        seq = note_tuples[i:i+sequence_length]
        input_seq.append(seq)
        output_seq_pitch.append(note_tuples[i+sequence_length][0])  # Predict pitch
        output_seq_dur.append(note_tuples[i+sequence_length][1])    # Predict duration
    return input_seq, output_seq_pitch, output_seq_dur

def build_model(input_shape, num_pitch_classes, num_parts=4):
    """
    Builds a Keras model for multi-part music generation.
    Each part predicts both pitch (categorical) and duration (continuous).
    """
    inp = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)
    
    outputs = []
    for part in range(num_parts):
        pitch_output = Dense(num_pitch_classes, activation='softmax', name=f'pitch_{part}')(x)
        dur_output = Dense(1, activation='linear', name=f'duration_{part}')(x)
        outputs.extend([pitch_output, dur_output])
    
    model = Model(inputs=inp, outputs=outputs)
    
    # Compile model with appropriate losses and weights for pitch and duration
    loss_dict = {}
    for i in range(num_parts):
        loss_dict[f'pitch_{i}'] = 'categorical_crossentropy'
        loss_dict[f'duration_{i}'] = 'mean_squared_error'  # MSE for duration
    
    loss_weights = {f'pitch_{i}': 1.0 for i in range(num_parts)}
    loss_weights.update({f'duration_{i}': 1.0 for i in range(num_parts)})
    
    model.compile(loss=loss_dict,
                  loss_weights=loss_weights,
                  optimizer='adam')
    
    return model

# === Main Training Flow ===

# Prompt user for dataset location
input_dir = input("Enter the path to your MusicXML dataset: ").strip()

print("Loading dataset...")
dataset = process_files(input_dir)

# Flatten the dataset to get all notes and durations
all_notes = [item[0] for sublist in dataset for item in sublist]
all_durations = [item[1] for sublist in dataset for item in sublist]

# Create mappings from notes to integers (and back) for categorical encoding
all_unique_notes = sorted(set(all_notes))
note_to_int = {note: number for number, note in enumerate(all_unique_notes)}
int_to_note = {number: note for note, number in note_to_int.items()}

# Save mappings for use during generation
with open("note_mappings.pkl", "wb") as f:
    pickle.dump((note_to_int, int_to_note, all_unique_notes), f)

print("Preparing sequences...")
note_tuples = list(zip(all_notes, all_durations))
sequence_length = 25

# Create sequences of notes and durations for model input/output
input_seq_tuples, output_seq_pitch, output_seq_dur = prepare_sequences(note_tuples, sequence_length)

# Convert input sequences to numerical format
X = np.array([[note_to_int[n[0]] for n in seq] for seq in input_seq_tuples])
X = np.reshape(X, (len(X), sequence_length, 1)) / float(len(all_unique_notes))  # Normalize

# Prepare multi-part outputs
y_pitch = [to_categorical([note_to_int[n] for n in output_seq_pitch], num_classes=len(all_unique_notes))] * 4
y_dur = [np.array(output_seq_dur).reshape(-1, 1)] * 4

print("Building and training model...")
model = build_model((sequence_length, 1), len(all_unique_notes), num_parts=4)

# Train the model with multi-part outputs
model.fit(X, {'pitch_0': y_pitch[0], 'duration_0': y_dur[0],
              'pitch_1': y_pitch[1], 'duration_1': y_dur[1],
              'pitch_2': y_pitch[2], 'duration_2': y_dur[2],
              'pitch_3': y_pitch[3], 'duration_3': y_dur[3]},
          epochs=5, batch_size=64)

print("Saving model...")
model.save("train2.keras")  # Save model in Keras native format
print("Training complete! Model and mappings saved.")
