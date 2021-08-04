import pretty_midi
import reverse_pianoroll
import convert
import librosa
import numpy as np
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from os import listdir
import glob
import read
import argparse

lowest_note = 0 #the index of the lowest note on the piano roll
highest_note = 78 #the index of the highest note on the piano roll
note_range = highest_note-lowest_note #the note range

num_timesteps  = 4 #This is the number of timesteps that we will create at a time
X_dim = 2*note_range*num_timesteps #This is the size of the visible layer.
Z_dim = 12*num_timesteps
n_hidden = 50 #This is the size of the hidden layer


parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir", default="haydn", type=str)
parser.add_argument("--output_dir", default="converted", type=str)
parser.add_argument("--checkpoint_dir", default="model", type=str)

args = parser.parse_args()


imported_graph = tf.train.import_meta_graph("./"+args.checkpoint_dir+"/model.ckpt.meta")
#for testing, i'll be using a different dataset of MIDI files to input into the generator here.
test_songs = read.get_songs(args.dataset_dir)
test_chromas = read.get_chromas(test_songs)

sess = tf.Session()
imported_graph.restore(sess, "./"+args.checkpoint_dir+"/model.ckpt")
graph = tf.get_default_graph()
G_W1g = graph.get_operation_by_name('gen/G_W1').outputs[0]
G_b1g = graph.get_operation_by_name('gen/G_b1').outputs[0]
G_W2g = graph.get_operation_by_name('gen/G_W2').outputs[0]
G_b2g = graph.get_operation_by_name('gen/G_b2').outputs[0]


G_W1 = sess.run(G_W1g)
G_W2 = sess.run(G_W2g)
G_b1 = sess.run(G_b1g)
G_b2 = sess.run(G_b2g)


print("Successfully loaded checkpoint variables.")

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def generator(z):
  z = z.astype(np.float32)
  G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
  G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
  G_prob = tf.nn.sigmoid(G_log_prob)

  return G_prob


i = 0

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

for c in test_chromas:
    test_chroma = np.array(c)

    test_chroma = test_chroma[:np.floor(test_chroma.shape[0] / num_timesteps).astype(int) * num_timesteps]
    test_chroma = np.reshape(test_chroma,
                             [int(test_chroma.shape[0] / num_timesteps), test_chroma.shape[1] * num_timesteps])

    out_samples = generator(test_chroma)
    outt = sess.run(out_samples)

    #print(np.shape(outt))

    S = np.reshape(outt, (np.floor(outt.shape[0] * outt.shape[1] / 2 / note_range).astype(int), 2 * note_range))
    C = np.reshape(test_chroma, (test_chroma.shape[0] * num_timesteps, 12))

    thresh_S = S >= 0.5


    test = reverse_pianoroll.piano_roll_to_pretty_midi(convert.back(thresh_S), fs=16)
    test.write(args.output_dir+'/{}.mid'.format(i))
    i += 1


print("Successfully transferred pieces.")
print("Done.")