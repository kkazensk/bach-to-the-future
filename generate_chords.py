# generate.py
from music21 import converter, note, chord, stream, key, interval, roman, metadata, meter, key
import numpy as np
import random
import pickle
from tensorflow.keras.models import load_model

# --- Duration Helper ---
def get_valid_duration(d, duration_classes):
    valid_duration = min(duration_classes, key=lambda x: abs(x - d))
    return valid_duration

def transpose_notes(notes, interval_obj):
    return [n.transpose(interval_obj) for n in notes]

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
def generate_musicxml_with_chords(four_part_notes, key_obj, duration_classes):
    score = stream.Score()
    part_streams = [stream.Part() for _ in range(4)]  # Violin 1, Violin 2, Viola, Cello

    score.insert(0, metadata.Metadata())
    for part in part_streams:
        part.append(meter.TimeSignature('4/4'))
        part.append(key.KeySignature(user_key_obj.sharps))


    for timestep in four_part_notes:
        for i, (n, dur) in enumerate(timestep):
            dur = get_valid_duration(dur, duration_classes)
            
            # For melody (part 0), create a note
            if i == 0:  # Melody part
                try:
                    note_obj = note.Note(n)
                    note_obj.quarterLength = dur
                    part_streams[i].append(note_obj)
                except:
                    # If the note is invalid, add a rest
                    part_streams[i].append(note.Rest(quarterLength=dur))
            
            # For accompaniment (other parts), generate a chord
            else:  # Chord parts (Violin 2, Viola, Cello)
                chord_pitches = get_chord_for_note(n, key_obj)
                if chord_pitches:
                    try:
                        chord_obj = chord.Chord(chord_pitches)
                        chord_obj.quarterLength = dur
                        part_streams[i].append(chord_obj)
                    except:
                        # If there's an issue with the chord, add a rest
                        part_streams[i].append(note.Rest(quarterLength=dur))
                else:
                    # If no chord was generated, add a rest
                    part_streams[i].append(note.Rest(quarterLength=dur))

    for part in part_streams:
        score.insert(0, part)

    return score

# --- Melody Generation ---
# --- Melody Generation ---
def generate_music_multi(model, seed_sequence, all_unique_notes, duration_classes, length=200, temperature=1.0):
    output_notes = []
    sequence = seed_sequence

    for _ in range(length):
        predictions = model.predict(sequence, verbose=0)
        
        # Separate pitch and duration outputs (assuming alternating in order)
        pred_pitch_probs = predictions[::2]   # pitch_0, pitch_1, pitch_2, pitch_3
        pred_dur = predictions[1::2]          # duration_0, duration_1, duration_2, duration_3

        # Process each part
        step_notes = []
        for i in range(2):  # Only two parts (melody and back chords)
            pitch_prob = pred_pitch_probs[i][0]  # Get softmax output
            
            # Ensure pitch_prob is a 1D array matching all_unique_notes
            pitch_prob = np.array(pitch_prob).flatten()  # Flatten if necessary
            
            # Ensure that pitch_prob has the same size as the number of unique notes
            if len(pitch_prob) != len(all_unique_notes):
                # Fix: Align probabilities to match the number of unique notes
                
                # Assuming model output is a smaller set of pitches and we need to map them to all_unique_notes:
                # Here, we will interpolate the pitch_prob to match the number of unique notes
                # This is only a workaround if the model output size is smaller (e.g., a subset of notes).
                pitch_prob = np.resize(pitch_prob, len(all_unique_notes))

            # Normalize probabilities with temperature scaling
            pitch_prob = np.log(pitch_prob + 1e-8) / temperature
            exp_preds = np.exp(pitch_prob)
            pitch_prob = exp_preds / np.sum(exp_preds)  # Ensure the sum of probabilities equals 1

            # Randomly select a note based on probabilities
            predicted_index = np.random.choice(len(all_unique_notes), p=pitch_prob)

            duration = pred_dur[i][0][0]
            duration = np.clip(duration, 0.25, 4.0)  # Clamp
            predicted_duration = get_valid_duration(duration, duration_classes)


            predicted_note = all_unique_notes[predicted_index]  # <- GET the note name
            step_notes.append((predicted_note, predicted_duration))


        output_notes.append(step_notes)

        # Update seed (slide window with last note of first part)
        next_input = np.zeros_like(sequence)
        next_input[0][:-1] = sequence[0][1:]
        next_input[0][-1] = predicted_index / float(len(all_unique_notes))
        sequence = next_input

    return output_notes



# === Main ===
target_key = input("Enter desired output key (e.g., 'C major', 'D minor'): ").strip()
try:
    user_key_obj = key.Key(target_key.split()[0], target_key.split()[1])
except:
    print(f"Invalid key format '{target_key}'.")
    exit()

print("Loading model and mappings...")
model = load_model("train2.keras")
with open("note_mappings_with_durations.pkl", "rb") as f:
    note_to_int, int_to_note, all_unique_notes, duration_to_int, int_to_duration, duration_classes, beat_to_int, int_to_beat, beat_positions, unique_durations = pickle.load(f)

print("Generating music...")
sequence_length = 25
seed = np.random.randint(0, len(all_unique_notes), sequence_length)
seed_input = np.reshape(seed, (1, sequence_length, 1)) / float(len(all_unique_notes))
generated_sequence = generate_music_multi(model, seed_input, all_unique_notes, duration_classes, length=200)

print("Transposing to target key...")
original_key = key.Key("C")  # Assume original training key was C
interval_obj = interval.Interval(original_key.tonic, user_key_obj.tonic)
transposed_sequence = []
for part in generated_sequence:
    transposed_part = []
    for n, d in part:
        try:
            temp_note = note.Note(n)
            transposed = transpose_notes([temp_note], interval_obj)[0]
            transposed_part.append((transposed.nameWithOctave, d))
        except:
            continue
    transposed_sequence.append(transposed_part)

print("Saving output...")
final_score = generate_musicxml_with_chords(transposed_sequence, user_key_obj, duration_classes)
output_path = "generated_music.xml"

small_score = stream.Score()
small_score.append(final_score.parts[0].measures(0, 5))  # first few measures
small_score.write('musicxml', fp="small_test.xml")

try:
    # Test with a single part and note
    test_score = stream.Score()
    test_part = stream.Part()
    test_part.append(meter.TimeSignature('4/4'))
    test_part.append(key.KeySignature(user_key_obj.sharps))
    test_part.append(note.Note("C4", quarterLength=1))
    test_score.append(test_part)
    test_score.write('musicxml', fp="test_output.xml")
    print("Test MusicXML saved successfully!")
except Exception as e:
    print(f"Error: {e}")


print("here...")
print(len(final_score.parts))
for part in final_score.parts:
    print(len(part.flat.notesAndRests))

#final_score.coreElementsChanged()
print("writing...")
final_score.write('musicxml', fp=output_path)
print(f"Saved: {output_path}")
