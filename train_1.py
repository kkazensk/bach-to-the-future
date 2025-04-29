# train_1.py

import os
import pickle
import numpy as np
from collections import Counter
from music21 import converter, note, chord
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === Parameters ===
SEQUENCE_LENGTH = 35
NUM_PARTS = 4
DATASET_FOLDER = './musicxml_dataset/'

# === Functions ===

def get_notes_and_durations(xml_file):
    """Extract notes/chords and durations from a MusicXML file."""
    score = converter.parse(xml_file)
    notes_durations = []
    for element in score.flat.notes:
        if isinstance(element, note.Note):
            pitch = str(element.pitch)
        elif isinstance(element, chord.Chord):
            pitch = '.'.join(str(p) for p in element.pitches)
        else:
            continue
        duration = round(float(element.quarterLength), 2)
        notes_durations.append((pitch, duration))
    return notes_durations

def process_dataset(dataset_folder):
    """Process all files in the dataset folder and collect note-duration pairs."""
    all_notes_durations = []
    for filename in os.listdir(dataset_folder):
        if filename.endswith(('.xml', '.mxl', '.musicxml')):
            filepath = os.path.join(dataset_folder, filename)
            try:
                notes_durations = get_notes_and_durations(filepath)
                all_notes_durations.extend(notes_durations)
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
    return all_notes_durations

def create_mappings(all_notes_durations):
    """Create mappings for notes and durations."""
    notes = sorted(set(note for note, dur in all_notes_durations))
    durations = sorted(set(str(dur) for note, dur in all_notes_durations))

    note_to_int = {n: i for i, n in enumerate(notes)}
    int_to_note = {i: n for n, i in note_to_int.items()}
    duration_to_int = {d: i for i, d in enumerate(durations)}
    int_to_duration = {i: d for d, i in duration_to_int.items()}

    return note_to_int, int_to_note, duration_to_int, int_to_duration

def prepare_sequences(all_notes_durations, sequence_length, note_to_int, duration_to_int):
    """Prepare sequences for input and output."""
    inputs = []
    outputs_pitch = []
    outputs_dur = []

    for i in range(len(all_notes_durations) - sequence_length):
        seq_in = all_notes_durations[i:i+sequence_length]
        seq_out = all_notes_durations[i+sequence_length]

        inputs.append([note_to_int[note] for note, dur in seq_in])
        outputs_pitch.append(note_to_int[seq_out[0]])
        outputs_dur.append(duration_to_int[str(seq_out[1])])

    return inputs, outputs_pitch, outputs_dur

def build_model(sequence_length, num_notes, num_durations, num_parts=4):
    """Build the multi-output LSTM model."""
    inp = Input(shape=(sequence_length, 1))

    x = LSTM(256, return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x = LSTM(256)(x)
    x = Dropout(0.3)(x)

    outputs = []
    for part in range(num_parts):
        pitch_out = Dense(num_notes, activation='softmax', name=f'pitch_{part}')(x)
        dur_out = Dense(num_durations, activation='softmax', name=f'duration_{part}')(x)
        outputs.extend([pitch_out, dur_out])

    model = Model(inputs=inp, outputs=outputs)

    losses = {f'pitch_{i}': 'categorical_crossentropy' for i in range(num_parts)}
    losses.update({f'duration_{i}': 'categorical_crossentropy' for i in range(num_parts)})

    model.compile(optimizer='adam', loss=losses)

    return model

# === Main Program ===
def main():
    print("Loading dataset...")
    all_notes_durations = process_dataset(DATASET_FOLDER)

    print("Creating mappings...")
    note_to_int, int_to_note, duration_to_int, int_to_duration = create_mappings(all_notes_durations)

    print(f"Unique notes: {len(note_to_int)}, Unique durations: {len(duration_to_int)}")

    print("Preparing sequences...")
    X, y_pitch, y_dur = prepare_sequences(all_notes_durations, SEQUENCE_LENGTH, note_to_int, duration_to_int)

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], SEQUENCE_LENGTH, 1))
    X = X / float(len(note_to_int))  # Normalize input

    # Convert outputs to categorical
    y_pitch_cat = to_categorical(y_pitch, num_classes=len(note_to_int))
    y_dur_cat = to_categorical(y_dur, num_classes=len(duration_to_int))

    # Duplicate labels for each part
    y_outputs = {}
    for part in range(NUM_PARTS):
        y_outputs[f'pitch_{part}'] = y_pitch_cat
        y_outputs[f'duration_{part}'] = y_dur_cat

    print("Building model...")
    model = build_model(SEQUENCE_LENGTH, len(note_to_int), len(duration_to_int), num_parts=NUM_PARTS)

    print("Training model...")
    model.fit(X, y_outputs, epochs=10, batch_size=64, verbose=1)

    print("Saving model and mappings...")
    model.save('train2.keras')
    with open('note_1_mappings.pkl', 'wb') as f:
        pickle.dump((note_to_int, int_to_note), f)
    with open('duration_1_mappings.pkl', 'wb') as f:
        pickle.dump((duration_to_int, int_to_duration), f)

    print("Training complete!")

if __name__ == '__main__':
    main()
