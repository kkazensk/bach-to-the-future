from music21 import converter, note, chord, stream, key, interval, roman
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
import random

# Extract notes and chords from a MusicXML file
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

# Process all files in the dataset folder
def process_files(input_dir):
    all_notes = []
    for file in os.listdir(input_dir):
        if file.endswith('.xml') or file.endswith('.mxl') or file.endswith('.musicxml'):
            file_path = os.path.join(input_dir, file)
            try:
                notes = get_notes_and_chords(file_path)
                all_notes.append(notes)
            except:
                print(f"Skipping {file} due to parsing error.")
    return all_notes

# Prepare note sequences for training
def prepare_sequences(note_tuples, sequence_length=25):
    input_seq = []
    output_seq_pitch = []
    output_seq_dur = []
    for i in range(len(note_tuples) - sequence_length):
        seq = note_tuples[i:i+sequence_length]
        input_seq.append(seq)
        output_seq_pitch.append(note_tuples[i+sequence_length][0])
        output_seq_dur.append(note_tuples[i+sequence_length][1])
    return input_seq, output_seq_pitch, output_seq_dur

# Define the LSTM model
def build_model(input_shape, output_size):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dropout(0.3))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Transpose notes by an interval
def transpose_notes(notes, interval):
    transposed = []
    for n in notes:
        try:
            if '.' in n:
                chord_obj = chord.Chord(n.split('.'))
                chord_obj.transpose(interval, inPlace=True)
                transposed.append('.'.join(str(p) for p in chord_obj.normalOrder))
            else:
                note_obj = note.Note(n)
                note_obj.transpose(interval, inPlace=True)
                transposed.append(str(note_obj.pitch))
        except:
            transposed.append(n)
    return transposed

# Predict notes
def generate_music(model, seed_sequence, length=200, temperature=1.0):
    prediction_output = []
    sequence = seed_sequence

    for _ in range(length):
        predicted_probs = model.predict(sequence, verbose=0)[0]
        predicted_probs = predicted_probs.astype(np.float64)

        predicted_probs = np.log(predicted_probs + 1e-8) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)

        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        predicted_note = all_unique_notes[predicted_index]
        prediction_output.append(predicted_note)

        if not predicted_note or any(not part[-1].isdigit() for part in predicted_note.split('.') if part):
            continue


        next_input = np.zeros_like(sequence)
        next_input[0][:-1] = sequence[0][1:]
        next_input[0][-1] = predicted_index
        sequence = next_input

    return prediction_output

# Generate a MusicXML score with backing chords
def generate_musicxml_with_chords(melody_notes, key_obj, chord_probability=0.3):
    score = stream.Score()
    melody_part = stream.Part()
    chord_part = stream.Part()

    for n in melody_notes:
        if n == '':
            continue
        if '.' in n:
            try:
                pitches = n.split('.')
                if len(pitches) < 2:
                    print(f"Skipping invalid chord: {n}")
                    continue
                chord_obj = chord.Chord(pitches)
                note_obj = chord_obj.notes[0]
            except Exception as e:
                print(f"Error processing chord {n}: {e}")
                continue
        else:
            try:
                if n[-1].isdigit():
                    note_obj = note.Note(n[:-1])
                    note_obj.octave = int(n[-1])
                else:
                    note_obj = note.Note(n)
            except Exception as e:
                print(f"Error processing note {n}: {e}")
                continue

        melody_part.append(note_obj)

        if random.random() < chord_probability:
            chord_pitches = get_chord_for_note(note_obj, key_obj)
            if chord_pitches:
                backing_chord = chord.Chord(chord_pitches)
                backing_chord.quarterLength = note_obj.quarterLength
                chord_part.append(backing_chord)
            else:
                chord_part.append(note.Rest(quarterLength=note_obj.quarterLength))
        else:
            chord_part.append(note.Rest(quarterLength=note_obj.quarterLength))

    score.insert(0, melody_part)
    score.insert(0, chord_part)
    return score

# Get a chord for the given note and key
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

# === Main Execution ===
input_dir = input("Enter the path to your MusicXML dataset: ").strip()
target_key = input("Enter desired output key (e.g., 'C major', 'D minor'): ").strip()

try:
    user_key_obj = key.Key(target_key.split()[0], target_key.split()[1])
    print(user_key_obj)
except Exception as e:
    print(f"Invalid key format '{target_key}'. Please use formats like 'C major' or 'D minor'.")
    exit()

print("Loading dataset")
dataset = process_files(input_dir)
all_notes = [item[0] for sublist in dataset for item in sublist]
all_durations = [item[1] for sublist in dataset for item in sublist]
all_unique_notes = sorted(set(all_notes))
note_to_int = {note: number for number, note in enumerate(all_unique_notes)}
int_to_note = {number: note for note, number in note_to_int.items()}

print("Encoding")
note_tuples = list(zip(all_notes, all_durations))
sequence_length = 25
input_seq_tuples, output_seq_pitch, output_seq_dur = prepare_sequences(note_tuples, sequence_length)

X = []
for seq in input_seq_tuples:
    X.append([note_to_int[n[0]] for n in seq])
X = np.reshape(X, (len(X), sequence_length, 1))
X = X / float(len(all_unique_notes))

y = [note_to_int[n] for n in output_seq_pitch]
y = to_categorical(y, num_classes=len(all_unique_notes))

print("Training model")
model = build_model((X.shape[1], X.shape[2]), len(all_unique_notes))
model.fit(X, y, epochs=5, batch_size=64)

print("Generate and Transpose")
seed = X[0:1]
generated_notes = generate_music(model, seed, length=200)

print("Get key")
training_score = converter.parse(os.path.join(input_dir, os.listdir(input_dir)[0]))
original_key = training_score.analyze('key')
transposition_interval = interval.Interval(original_key.tonic, user_key_obj.tonic)
transposed_notes = transpose_notes(generated_notes, transposition_interval)

print("SAVING")
final_score = generate_musicxml_with_chords(transposed_notes, user_key_obj)
output_path = "generated_music.xml"
final_score.write('musicxml', fp=output_path)
print(f"Music generated and saved as: {output_path} in {target_key}")