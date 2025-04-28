import os
import pickle
import xml.etree.ElementTree as ET

dataset_folder = './musicxml_dataset/'

unique_notes = set()
unique_durations = set()

# --- Debug: check folder and files ---
if not os.path.exists(dataset_folder):
    print(f"Folder {dataset_folder} does not exist!")
    exit()

files = [f for f in os.listdir(dataset_folder) if f.endswith('.xml') or f.endswith('.musicxml')]
if not files:
    print("No MusicXML files found!")
    exit()

print(f"Found {len(files)} MusicXML files.")

# --- Function to Extract Namespace ---
def get_namespace(tag):
    # Assumes the format is '{namespace}tagname'
    if '}' in tag:
        return tag.split('}')[0].strip('{')
    return ''

# --- Function to Map Duration to Nearest Valid Value ---
def get_valid_duration(duration_value, duration_classes):
    """
    Map the duration to the nearest valid class.
    If the duration is out of range, log it and return the closest valid duration.
    """
    closest_duration = min(duration_classes, key=lambda x: abs(x - duration_value))
    
    # Log any durations that are too far outside the valid range
    if duration_value not in duration_classes and not (duration_value in duration_classes):
        print(f"Warning: Unexpected duration {duration_value:.2f} found. Mapping to {closest_duration:.2f}.")
        
    return closest_duration

# --- Main loop ---
for filename in files:
    filepath = os.path.join(dataset_folder, filename)
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Automatically extract namespace from root tag
        namespace = get_namespace(root.tag)
        
        # Debug: how many notes found
        notes = root.findall('.//note')
        notes_simple = list(root.iter('note'))

        if len(notes) == 0 and len(notes_simple) > 0:
            print(f"[{filename}] Namespace didn't work, using no namespace.")
            notes = notes_simple
            namespace = None
        else:
            print(f"[{filename}] Found {len(notes)} notes using namespace.")

        for note in notes:
            if namespace:
                pitch = note.find(f'{{{namespace}}}pitch')
                rest = note.find(f'{{{namespace}}}rest')
                duration = note.find(f'{{{namespace}}}duration')
            else:
                pitch = note.find('pitch')
                rest = note.find('rest')
                duration = note.find('duration')

            if pitch is not None:
                step = pitch.find(f'{{{namespace}}}step') if namespace else pitch.find('step')
                octave = pitch.find(f'{{{namespace}}}octave') if namespace else pitch.find('octave')
                alter = pitch.find(f'{{{namespace}}}alter') if namespace else pitch.find('alter')

                if step is not None and octave is not None:
                    note_name = step.text + octave.text
                    if alter is not None:
                        alter_val = int(alter.text)
                        if alter_val == 1:
                            note_name = step.text + '#' + octave.text  # sharp
                        elif alter_val == -1:
                            note_name = step.text + 'b' + octave.text  # flat
                    unique_notes.add(note_name)
            elif rest is not None:
                unique_notes.add('rest')

            if duration is not None:
                try:
                    duration_value = float(duration.text)  # Get the duration value
                    unique_durations.add(duration_value)  # Add raw duration to the set
                except ValueError:
                    print(f"Error: Invalid duration value '{duration.text}' in {filename}. Skipping note.")

    except Exception as e:
        print(f"Error reading {filename}: {e}")

# --- After processing ---
if not unique_notes:
    print("No notes found across all files!")
else:
    print(f"Found {len(unique_notes)} unique notes total.")

if not unique_durations:
    print("No valid durations found across all files!")
else:
    print(f"Found {len(unique_durations)} unique durations total.")

# --- Create duration_classes based on extracted data ---
# Optionally round the durations to a certain precision, e.g., 2 decimal places
duration_classes = sorted(set(round(d, 2) for d in unique_durations))

# --- Mappings ---
note_to_int = {note: i for i, note in enumerate(sorted(unique_notes))}
int_to_note = {i: note for note, i in note_to_int.items()}

# Duration mappings
duration_to_int = {str(d): i for i, d in enumerate(duration_classes)}
int_to_duration = {v: k for k, v in duration_to_int.items()}

# Beat position mappings
beat_positions = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                  2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75]
beat_to_int = {str(b): i for i, b in enumerate(beat_positions)}
int_to_beat = {v: k for k, v in beat_to_int.items()}

# --- Save ---
with open("note_mappings_with_durations.pkl", "wb") as f:
    pickle.dump((note_to_int, int_to_note, list(unique_notes),
                 duration_to_int, int_to_duration, duration_classes,
                 beat_to_int, int_to_beat, beat_positions,
                 list(unique_durations)), f)

print(f"Saved mappings with {len(unique_notes)} unique notes, "
      f"{len(duration_classes)} unique durations, and other data.")
