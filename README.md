1. Set up venv: python3 -m venv bachvenv (Done once, do not repeat)

2. activate: source bachvenv/bin/activate

3. requirements.txt for dependencies... to install all: python3 -m pip install -r requirements.txt


The training files save training to .keras so that you can run the generate file multiple times without needing to retrain


VERSION 0:
Similar to VERSION 1, but was the first iteration with less training influence.

VERSION 1:
Generates a melody part with backing chords

    python3 train_1.py
        - Will prompt for training directory of musicxml files (e.g. ./music_xml_dataset)
        - When testing, we copied 10 training files into a new directory called ./small_music_dataset  to make training times quicker for iterative design
    python3 generate_1.py
        - Will prompt for key to generate music (e.g. C major)


VERSION 2:
Generates a 4-part score, but needs more improvement in inter-part relatability and note durations.

    python3 train_1.py
        - Same training as VERSION 1, so you do not need to retrain if already
           trained.
    python3 generate_2.py
        - Will prompt for key to generate music (e.g. C major)



Tune Training Tuning:
    - You can change the number of epochs. For quicker training but less iterations, use 1 to 5. For mid-range pcik from 5-10. For better, pick 15 or higher.
    - You can change the sequence_length. This affects training time by dictating how many notes in one sequence should influence the generation of one note. For quicker training, pick 10 to 25. For better training, consider 30 to 40.
    - You can also change the LSTM. For our purposes we used 128. You can also try 256, but my laptop heated up too much while training, hence the bump down. If you wanted to lower this more to 64, you could try that as well.


downloadScores.py creates the musicxml_dataset for training from music21 library

To run with GUI 

python GUI.py
