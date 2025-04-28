from music21 import converter, note, chord
import os
import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# === Data Extraction ===
def get_notes_and_chords(xml_file):
    score = converter.parse(xml_file)
    notes = []
    for element in score.flat.notesAndRests:
        pitch_str = ""
        duration = float(element.quarterLength)
        if isinstance(element, note.Note):
            pitch_str = str(element.pitch)
        elif isinstance(element, chord.Chord):
            pitch_str = '.'.join(str(p) for p in element.pitches)
        elif isinstance(element, note.Rest):
            pitch_str = 'rest'
        if pitch_str:
            notes.append((pitch_str, duration))
    return notes

def process_files(input_dir):
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
    input_seq, output_seq_pitch, output_seq_dur = [], [], []
    for i in range(len(note_tuples) - sequence_length):
        seq = note_tuples[i:i+sequence_length]
        input_seq.append(seq)
        output_seq_pitch.append(note_tuples[i+sequence_length][0])
        output_seq_dur.append(note_tuples[i+sequence_length][1])
    return input_seq, output_seq_pitch, output_seq_dur

# === Duration Classification ===
duration_classes = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
duration_to_int = {dur: i for i, dur in enumerate(duration_classes)}
int_to_duration = {i: dur for dur, i in duration_to_int.items()}

def quantize_duration(d):
    return min(duration_classes, key=lambda x: abs(x - d))

def build_model(input_shape, num_pitch_classes, num_dur_classes, num_parts=4):
    inp = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)

    outputs = []
    for part in range(num_parts):
        pitch_output = Dense(num_pitch_classes, activation='softmax', name=f'pitch_{part}')(x)
        dur_output = Dense(num_dur_classes, activation='softmax', name=f'duration_{part}')(x)
        outputs.extend([pitch_output, dur_output])

    model = Model(inputs=inp, outputs=outputs)
    
    loss_dict = {f'pitch_{i}': 'categorical_crossentropy' for i in range(num_parts)}
    loss_dict.update({f'duration_{i}': 'categorical_crossentropy' for i in range(num_parts)})

    loss_weights = {k: 1.0 for k in loss_dict.keys()}

    model.compile(loss=loss_dict, loss_weights=loss_weights, optimizer='adam')
    return model

# === Main Training Flow ===
input_dir = input("Enter the path to your MusicXML dataset: ").strip()

print("Loading dataset...")
dataset = process_files(input_dir)
all_notes = [item[0] for sublist in dataset for item in sublist]
all_durations = [item[1] for sublist in dataset for item in sublist]

all_unique_notes = sorted(set(all_notes))
note_to_int = {note: number for number, note in enumerate(all_unique_notes)}
int_to_note = {number: note for note, number in note_to_int.items()}

print(f"Number of unique notes: {len(all_unique_notes)}")

with open("note_mappings.pkl", "wb") as f:
    pickle.dump((note_to_int, int_to_note, all_unique_notes, duration_to_int, int_to_duration, duration_classes), f)

print("Preparing sequences...")
note_tuples = list(zip(all_notes, all_durations))
sequence_length = 35
input_seq_tuples, output_seq_pitch, output_seq_dur = prepare_sequences(note_tuples, sequence_length)

X = np.array([[note_to_int[n[0]] for n in seq] for seq in input_seq_tuples])
X = np.reshape(X, (len(X), sequence_length, 1)) / float(len(all_unique_notes))

y_pitch = [to_categorical([note_to_int[n] for n in output_seq_pitch], num_classes=len(all_unique_notes))] * 4
y_dur = [to_categorical([duration_to_int[quantize_duration(d)] for d in output_seq_dur], num_classes=len(duration_classes))] * 4

print("Building and training model...")
model = build_model((sequence_length, 1), len(all_unique_notes), len(duration_classes), num_parts=4)
model.fit(X, {
    'pitch_0': y_pitch[0], 'duration_0': y_dur[0],
    'pitch_1': y_pitch[1], 'duration_1': y_dur[1],
    'pitch_2': y_pitch[2], 'duration_2': y_dur[2],
    'pitch_3': y_pitch[3], 'duration_3': y_dur[3]},
    epochs=6, batch_size=32, verbose=1)

print("Saving model...")
model.save("lstm_four_part_music_model.keras")
print("Training complete! Model and mappings saved.")
