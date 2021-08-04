import glob
import pretty_midi
import convert
import numpy as np


def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in files:
        try:
            data = pretty_midi.PrettyMIDI(f)
            song = data.get_piano_roll(fs=16)
            song = convert.forward(song)
            #print(np.shape(song))
            #song = np.transpose(song) - #if your code matrices aren't working, try uncommenting this. the convert.py file might not be updated
            songs.append(song)
        except Exception as e:
            raise e
    return songs


def get_song(f):

    data = pretty_midi.PrettyMIDI(f)
    song = data.get_piano_roll(fs=16)
    song = convert.forward(song)

    return song



# custom function to extract chroma features from song
def get_chromas(songs):
    chromas = []
    for song in songs:
        chroma = np.zeros(shape=(np.shape(song)[0], 12))
        for i in np.arange(np.shape(song)[0]):
            for j in np.arange(78):
                if song[i][j] > 0:
                    chroma[i][np.mod(j, 12)] += 1
        # print(np.shape(chroma))
        chromas.append(chroma)

    return chromas