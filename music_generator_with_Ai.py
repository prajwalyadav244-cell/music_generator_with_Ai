# pip install music21 tensorflow
from music21 import converter, instrument, note, chord
import glob, numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Collect MIDI data
def get_notes():
    notes = []
    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# 2. Prepare sequences (music21 logic)
notes = get_notes()
pitchnames = sorted(set(notes))
n_vocab = len(pitchnames)
# ... create input/output sequences for LSTM ...

# 3. Build Model (RNN/LSTM)
model = Sequential([
    LSTM(256, input_shape=(100, 1), return_sequences=True),
    Dropout(0.3),
    LSTM(512, return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(256, activation='relu'),
    Dense(n_vocab, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')
# model.fit() on your dataset

# 4. Generate new music and save to MIDI
print("Model trained. Generating new sequence and saving to output.mid")
# Use model.predict() to generate notes and convert to MIDI via music21 stream
