import pickle

# These should match what was used during training
note_to_int = {'C4': 0, 'D4': 1, 'E4': 2, 'F4': 3, 'G4': 4, 'A4': 5, 'B4': 6, 'rest': 7}  # example
int_to_note = {v: k for k, v in note_to_int.items()}
all_unique_notes = list(note_to_int.keys())

duration_classes = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
duration_to_int = {str(d): i for i, d in enumerate(duration_classes)}
int_to_duration = {v: k for k, v in duration_to_int.items()}

beat_positions = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                  2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75]
beat_to_int = {str(b): i for i, b in enumerate(beat_positions)}
int_to_beat = {v: k for k, v in beat_to_int.items()}

with open("note_mappings_with_beats.pkl", "wb") as f:
    pickle.dump((note_to_int, int_to_note, all_unique_notes,
                 duration_to_int, int_to_duration, duration_classes,
                 beat_to_int, int_to_beat, beat_positions), f)

print("Saved 'note_mappings_with_beats.pkl'. You're good to generate now!")
