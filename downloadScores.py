from music21 import corpus
import os
import re

# Folder where MusicXML files will be saved
output_dir = 'musicxml_dataset'
os.makedirs(output_dir, exist_ok=True)

# Choose composers
composers = ['bach', 'beethoven']

# Loop and download
for composer in composers:
    works = corpus.getComposer(composer)
    print(f"\nFound {len(works)} works by {composer}")

    for i, work in enumerate(works):
        try:
            score = corpus.parse(work)
            # Get work title (metadata or identifier)
            title = score.metadata.title if score.metadata.title else str(work)
            
            # Clean up title by replacing problematic characters
            title = re.sub(r'[^\w\-]', '_', title)
            filename = f"{composer}_{i}_{title}.musicxml"
            filepath = os.path.join(output_dir, filename)

            # Write to MusicXML
            score.write('musicxml', fp=filepath)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Skipped {work} - Error: {e}")

