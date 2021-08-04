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
import midi_manipulation
import numpy as np
import pandas as pd
import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

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
parser.add_argument("--gan_checkpoint_dir", default="gansaved", type=str)
parser.add_argument("--vae_checkpoint_dir", default="vaesaved", type=str)
args = parser.parse_args()


imported_gan = tf.train.import_meta_graph("./"+args.gan_checkpoint_dir+"/model.ckpt.meta")
#for testing, i'll be using a different dataset of MIDI files to input into the generator here.
test_songs = read.get_songs(args.dataset_dir)
test_chromas = read.get_chromas(test_songs)

sess = tf.Session()
imported_gan.restore(sess, "./"+args.gan_checkpoint_dir+"/model.ckpt")
graph = tf.get_default_graph()
G_W1g = graph.get_operation_by_name('gen/G_W1').outputs[0]
G_b1g = graph.get_operation_by_name('gen/G_b1').outputs[0]
G_W2g = graph.get_operation_by_name('gen/G_W2').outputs[0]
G_b2g = graph.get_operation_by_name('gen/G_b2').outputs[0]


G_W1 = sess.run(G_W1g)
G_W2 = sess.run(G_W2g)
G_b1 = sess.run(G_b1g)
G_b2 = sess.run(G_b2g)


for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
  print(i.name)   # i.name if you want just a name


print("Successfully loaded checkpoint variables.")


#quit()

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
    test.write(args.output_dir+'/{}_piano1.mid'.format(i))
    i += 1


print("Successfully transferred pieces.")
sess.close()
tf.reset_default_graph()


imported_vae = tf.train.import_meta_graph("./"+args.vae_checkpoint_dir+"/modelv.ckpt.meta")

for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
  print(i.name)   # i.name if you want just a name
sess = tf.Session()
imported_vae.restore(sess, "./"+args.vae_checkpoint_dir+"/modelv.ckpt")
graph = tf.get_default_graph()
Q_b1g = graph.get_operation_by_name('S/Q_b1').outputs[0]
Q_W1g = graph.get_operation_by_name('S/Q_W1').outputs[0]
Q_b2_mug = graph.get_operation_by_name('S/Q_b2_mu').outputs[0]
Q_W2_mug = graph.get_operation_by_name('S/Q_W2_mu').outputs[0]
Q_b2_sigmag = graph.get_operation_by_name('S/Q_b2_sigma').outputs[0]
Q_W2_sigmag = graph.get_operation_by_name('S/Q_W2_sigma').outputs[0]
P_W1g = graph.get_operation_by_name('S/P_W1').outputs[0]
P_b1g = graph.get_operation_by_name('S/P_b1').outputs[0]
P_W2g = graph.get_operation_by_name('S/P_W2').outputs[0]
P_b2g = graph.get_operation_by_name('S/P_b2').outputs[0]



Q_b1 = sess.run(Q_b1g)
Q_W1 = sess.run(Q_W1g)
Q_b2_mu = sess.run(Q_b2_mug)
Q_W2_mu = sess.run(Q_W2_mug)
Q_b2_sigma = sess.run(Q_b2_sigmag)
Q_W2_sigma = sess.run(Q_W2_sigmag)
P_W1 = sess.run(P_W1g)
P_b1 = sess.run(P_b1g)
P_W2 = sess.run(P_W2g)
P_b2 = sess.run(P_b2g)


def Q(X):
    X = X.astype(np.float32)
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

def P(z):
    z = z.astype(np.float32)
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits




lowest_note = midi_manipulation.lowerBound #the index of the lowest note on the piano roll
highest_note = midi_manipulation.upperBound #the index of the highest note on the piano roll
note_range = highest_note-lowest_note #the note range

num_timesteps  = 4 #64 #32 #16 #This is the number of timesteps that we will create at a time  (16 = one bar)
n_visible      = 2*note_range*num_timesteps #This is the size of the visible layer.
n_hidden       = 500 #50 #This is the size of the hidden layer

z_dim = n_hidden #100
X_dim = n_visible #mnist.train.images.shape[1]
h_dim = n_hidden #128


for f in os.listdir(args.output_dir):
    print(f)
    q = args.output_dir + "/" + f
    querysong = read.get_song(q)
    #querysong = np.array(midi_manipulation.midiToNoteStateMatrix(q))
    song = np.array(querysong)
    zeropadsong = np.zeros(((np.floor(song.shape[0] / num_timesteps).astype(int) + 1) * num_timesteps, song.shape[1]))
    zeropadsong[:song.shape[0], :song.shape[1]] = song
    # song = song[:(np.floor(song.shape[0]/num_timesteps).astype(int)+1)*num_timesteps]
    song = np.reshape(zeropadsong, [int(song.shape[0] / num_timesteps) + 1, song.shape[1] * num_timesteps])
    print(np.shape(song))

    decode_bars = np.shape(song)[0]
    S_reconstruct = np.reshape(song, (decode_bars * num_timesteps, 2 * note_range))

    midi_manipulation.noteStateMatrixToMidi(S_reconstruct, "out/song_reconstruct" + f)

    decode_bars = np.shape(song)[0]
    S_reconstruct = np.reshape(song, (decode_bars * num_timesteps, 2 * note_range))
    Xq = song
    zs = True

    z_mutensor, z_logvartensor = Q(Xq)
    z_mu = sess.run(z_mutensor)
    z_logvar = sess.run(z_logvartensor)

    zq_sampletensor = sample_z(z_mu, z_logvar)
    zq_sample = sess.run(zq_sampletensor)
    print(np.shape(zq_sample))
    print(type(zq_sample))
    samplestensor, _ = P(zq_sample)
    samples = sess.run(samplestensor)
    S = np.reshape(samples, (decode_bars * num_timesteps, 2 * note_range))
    thresh_S = S >= 0.66  # 0.857 #0.5)
    fout = f[0:7]
    midi_manipulation.noteStateMatrixToMidi(thresh_S, args.output_dir + "/" + fout+"2")

