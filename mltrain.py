from music21 import converter, note, chord, stream, key, interval, roman
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import random

# --- Helper to Clamp and Round Durations ---
def get_valid_duration(d):
    valid_durations = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
    return min(valid_durations, key=lambda x: abs(x - d))

# --- Transpose Notes Helper ---
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

# --- Data Extraction ---
def get_notes_and_chords(xml_file):
    score = converter.parse(xml_file)
    notes = []
    for element in score.flat.notes:
        try:
            pitch_str = ""
            duration = float(element.quarterLength)
            if isinstance(element, note.Note):
                pitch_str = str(element.pitch)
            elif isinstance(element, chord.Chord):
                pitch_str = '.'.join(str(p) for p in element.pitches)
            if pitch_str:
                notes.append((pitch_str, duration))
        except:
            continue
    return notes

# --- Dataset Processing ---
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

# --- Sequence Preparation ---
def prepare_sequences(note_tuples, sequence_length=25):
    input_seq, output_seq_pitch, output_seq_dur = [], [], []
    for i in range(len(note_tuples) - sequence_length):
        seq = note_tuples[i:i+sequence_length]
        input_seq.append(seq)
        output_seq_pitch.append(note_tuples[i+sequence_length][0])
        output_seq_dur.append(note_tuples[i+sequence_length][1])
    return input_seq, output_seq_pitch, output_seq_dur

# --- LSTM Model ---
def build_multi_output_model(input_shape, num_pitch_classes):
    inp = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)
    pitch_output = Dense(num_pitch_classes, activation='softmax', name='pitch')(x)
    dur_output = Dense(1, activation='linear', name='duration')(x)
    model = Model(inputs=inp, outputs=[pitch_output, dur_output])
    model.compile(loss={'pitch': 'categorical_crossentropy', 'duration': 'mse'}, optimizer='adam')
    return model

# --- Music Generation ---
def generate_music_multi(model, seed_sequence, length=200, temperature=1.0):
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

# --- Score Construction ---
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

# === MAIN EXECUTION ===
input_dir = input("Enter the path to your MusicXML dataset: ").strip()
target_key = input("Enter desired output key (e.g., 'C major', 'D minor'): ").strip()

try:
    user_key_obj = key.Key(target_key.split()[0], target_key.split()[1])
except:
    print(f"Invalid key format '{target_key}'.")
    exit()

print("Loading dataset...")
dataset = process_files(input_dir)
all_notes = [item[0] for sublist in dataset for item in sublist]
all_durations = [item[1] for sublist in dataset for item in sublist]
all_unique_notes = sorted(set(all_notes))
note_to_int = {note: number for number, note in enumerate(all_unique_notes)}
int_to_note = {number: note for note, number in note_to_int.items()}

print("Preparing sequences...")
note_tuples = list(zip(all_notes, all_durations))
sequence_length = 25
input_seq_tuples, output_seq_pitch, output_seq_dur = prepare_sequences(note_tuples, sequence_length)

X = np.array([[note_to_int[n[0]] for n in seq] for seq in input_seq_tuples])
X = np.reshape(X, (len(X), sequence_length, 1)) / float(len(all_unique_notes))

y_pitch = to_categorical([note_to_int[n] for n in output_seq_pitch], num_classes=len(all_unique_notes))
y_dur = np.array(output_seq_dur).reshape(-1, 1)

print("Building and training model...")
model = build_multi_output_model((sequence_length, 1), len(all_unique_notes))
model.fit(X, {'pitch': y_pitch, 'duration': y_dur}, epochs=5, batch_size=64)

print("Generating music...")
seed = X[0:1]
generated_sequence = generate_music_multi(model, seed, length=200)

print("Transposing...")
training_score = converter.parse(os.path.join(input_dir, os.listdir(input_dir)[0]))
original_key = training_score.analyze('key')
transposition_interval = interval.Interval(original_key.tonic, user_key_obj.tonic)
transposed_sequence = []
for n, d in generated_sequence:
    try:
        temp_note = note.Note(n)
        transposed = transpose_notes([temp_note], transposition_interval)[0]
        transposed_sequence.append((transposed.nameWithOctave, d))
    except Exception as e:
        print(f"Skipping invalid note '{n}' due to: {e}")
        continue


print("Saving to file...")
final_score = generate_musicxml_with_chords(transposed_sequence, user_key_obj)
output_path = "generated_music.xml"
final_score.write('musicxml', fp=output_path)
print(f"Music generated and saved as: {output_path} in {target_key}")
