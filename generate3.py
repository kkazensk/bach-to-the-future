from music21 import converter, note, chord, stream, key, interval
import numpy as np
import pickle
from tensorflow.keras.models import load_model

def get_valid_duration(d, duration_classes):
    """
    Returns the closest valid duration for the given duration.
    """
    return min(duration_classes, key=lambda x: abs(x - d))

def transpose_notes(notes, interval_obj):
    """
    Transposes the notes by the given interval.
    """
    return [n.transpose(interval_obj) for n in notes]

def generate_musicxml(four_part_notes, key_obj, duration_classes):
    """
    Generates a MusicXML score with the given four-part notes and key.
    """
    score = stream.Score()
    part_streams = [stream.Part() for _ in range(4)]

    for timestep in four_part_notes:
        for i, (n, dur) in enumerate(timestep):
            dur = get_valid_duration(dur, duration_classes)
            if n == 'rest':
                part_streams[i].append(note.Rest(quarterLength=dur))
            else:
                try:
                    note_obj = note.Note(n)
                    note_obj.quarterLength = dur
                    part_streams[i].append(note_obj)
                except:
                    part_streams[i].append(note.Rest(quarterLength=dur))

    for part in part_streams:
        score.insert(0, part)

    return score

def generate_music_multi(model, seed_sequence, all_unique_notes, duration_classes, length=200, temperature=1.0):
    """
    Generates a sequence of music with the model.
    """
    output_notes = []
    sequence = seed_sequence
    predicted_index_for_seed = None

    for _ in range(length):
        predictions = model.predict(sequence, verbose=0)

        # Split the predictions for pitch and duration
        pred_pitch_probs = predictions[::2]
        pred_dur = predictions[1::2]

        step_notes = []
        for i in range(4):
            pitch_prob = pred_pitch_probs[i][0]

            # Ensure pitch probabilities match the number of unique notes
            if len(pitch_prob) != len(all_unique_notes):
                if len(pitch_prob) < len(all_unique_notes):
                    pitch_prob = np.pad(pitch_prob, (0, len(all_unique_notes) - len(pitch_prob)), 'constant')
                elif len(pitch_prob) > len(all_unique_notes):
                    pitch_prob = pitch_prob[:len(all_unique_notes)]

            duration = pred_dur[i][0][0]
            # Use valid duration
            predicted_duration = get_valid_duration(duration, duration_classes)

            pitch_prob = np.log(pitch_prob + 1e-8) / temperature
            exp_preds = np.exp(pitch_prob)
            pitch_prob = exp_preds / np.sum(exp_preds)

            predicted_index = np.random.choice(len(all_unique_notes), p=pitch_prob)

            if predicted_index_for_seed is None:
                predicted_index_for_seed = predicted_index

            predicted_note = all_unique_notes[predicted_index]
            step_notes.append((predicted_note, predicted_duration))

        output_notes.append(step_notes)

        # Update sequence with predicted notes
        next_input = np.zeros_like(sequence)
        next_input[0][:-1] = sequence[0][1:]
        if predicted_index_for_seed is not None:
            next_input[0][-1] = predicted_index_for_seed / float(len(all_unique_notes))
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
model = load_model("lstm_four_part_music_model.keras")
with open("note_mappings_with_durations.pkl", "rb") as f:
    note_to_int, int_to_note, all_unique_notes, duration_to_int, int_to_duration, duration_classes, beat_to_int, int_to_beat, beat_positions, all_unique_durations = pickle.load(f)


print("Generating music...")
sequence_length = 35
seed_sequence = np.random.randint(0, len(all_unique_notes), (1, sequence_length, 1)) / float(len(all_unique_notes))
output_notes = generate_music_multi(model, seed_sequence, all_unique_notes, duration_classes, length=200)

print("Generating MusicXML...")
generated_score = generate_musicxml(output_notes, user_key_obj, duration_classes)
print("done generating. saving...")
print(f"Writing MusicXML with {len(output_notes)} time steps to 'new_score.musicxml'...")

generated_score.write('musicxml', fp='new_score.musicxml')
print("Saved generated music to 'new_score.musicxml'. You can open it in MuseScore.")

print("Music generation complete!")
