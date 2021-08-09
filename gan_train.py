#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import tensorflow as tf
import convert
import read
import reverse_pianoroll
import time

lowest_note = 0  # the index of the lowest note on the piano roll
highest_note = 78  # the index of the highest note on the piano roll
note_range = highest_note - lowest_note  # the note range

num_timesteps = 4  # This is the number of timesteps that we will create at a time

# This is the size of the visible layer.

X_dim = 2 * note_range * num_timesteps
Z_dim = 12 * num_timesteps
n_hidden = 50  # This is the size of the hidden layer

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_dir', default='pop', type=str)
parser.add_argument('--output_dir', default='out', type=str)
parser.add_argument('--checkpoint_dir', default='gansaved', type=str)

parser.add_argument('--epochs', default=150000, type=int)
parser.add_argument('--l', default=100, type=int,
                    help='lambda value for generator loss')
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--threshold', default=0.5, type=float,
                    help='confidence threshold for thresh_S, note output'
                    )

args = parser.parse_args()
songs = read.get_songs(args.dataset_dir)
chromas = read.get_chromas(songs)
print('Successfully processed {} songs and chroma.'.format(len(songs)))


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim + Z_dim, 512]))
D_b1 = tf.Variable(tf.zeros(shape=[512]))

D_W2 = tf.Variable(xavier_init([512, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

with tf.variable_scope('gen', reuse=True):
    G_W1 = tf.Variable(xavier_init([Z_dim, 128]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

    G_W2 = tf.Variable(xavier_init([128, X_dim]), name='G_W2')
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x, c):
    D_h1 = tf.nn.relu(tf.matmul(tf.concat([x, c], 1), D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


G_sample = generator(Z)
(D_real, D_logit_real) = discriminator(X, Z)
(D_fake, D_logit_fake) = discriminator(G_sample, Z)

# Alternative losses:
# -------------------

D_loss_real = \
    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                   labels=tf.ones_like(D_logit_real)))
D_loss_fake = \
    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                   labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss_fake = \
    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                   labels=tf.ones_like(D_logit_fake)))
G_loss_L1 = tf.reduce_mean(tf.losses.mean_squared_error(X, G_sample))
G_loss = G_loss_fake + args.l * G_loss_L1

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                       'gen'))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

i = 0
num_epochs = args.epochs
batch_size = args.batch_size
start_time = time.time()
while i <= num_epochs:
    for (song, chroma) in zip(songs, chromas):

        song = np.array(song)

        song_steps = np.floor(song.shape[0] / num_timesteps).astype(int)
        song = song[:song_steps * num_timesteps]
        song = np.reshape(song, [song_steps, song.shape[1]
                          * num_timesteps])
        chroma = np.array(chroma)
        chroma = chroma[:song_steps * num_timesteps]
        chroma = np.reshape(chroma, [song_steps, chroma.shape[1]
                            * num_timesteps])
        batch_size = min(batch_size, len(song))

        # Train the RBM on batch_size examples at a time

        for ind in range(0, len(song), batch_size):
            X_mb = song[ind:ind + batch_size]
            ch = chroma[ind:ind + batch_size]

            #            _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb})

            (_, D_loss_curr) = sess.run([D_solver, D_loss],
                    feed_dict={X: X_mb, Z: ch})
            (_, G_loss_curr) = sess.run([G_solver, G_loss],
                    feed_dict={X: X_mb, Z: ch})
            if i % 1000 == 0:

                dloss = 'D_Loss: {:.4}'.format(D_loss_curr)
                gloss = 'G_Loss: {:.4}'.format(G_loss_curr)
                # D(x) --> D's estimate of the probability that real data instance x is real
                # G(z) --> G's output when given noise z
                print('(DLoss, GLoss): (%s, %s)' % (D_loss_curr, G_loss_curr), "[Iter. %s]" % (i))
                samples = sess.run(G_sample, feed_dict={Z: ch})

                S = np.reshape(samples, (ch.shape[0] * num_timesteps, 2
                               * note_range))
                thresh_S = S >= 0.5
                C = np.reshape(ch, (ch.shape[0] * num_timesteps, 12))

                test = \
                    reverse_pianoroll.piano_roll_to_pretty_midi(convert.back(thresh_S),
                        fs=16)
                test.write(args.output_dir + '/{}.mid'.format(i))

            i += 1

end_time = time.time()
total_time = start_time - end_time
print('Model trained successfully in %s seconds' % (total_time))

# Save just in case

mname = 'GANmodel'
save_path = saver.save(sess, './' + args.checkpoint_dir + '/' + mname
                       + '.ckpt')
print('Model saved in path: %s' % save_path)
sess.close()
