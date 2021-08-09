import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import midi_manipulation
import read

# the index of the lowest note on the piano roll
lowest_note = midi_manipulation.lowerBound
# the index of the highest note on the piano roll
highest_note = midi_manipulation.upperBound
note_range = highest_note - lowest_note  # the note range
# 64 #32 #16 #This is the number of timesteps that we will create at a
# time  (16 = one bar)
num_timesteps = 4
# This is the size of the visible layer.
n_visible = 2 * note_range * num_timesteps
n_hidden = 500  # 50 #This is the size of the hidden layer

z_dim = n_hidden  # 100
X_dim = n_visible  # mnist.train.images.shape[1]
h_dim = n_hidden  # 128


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# 64 #32 #16 #This is the number of timesteps that we will create at a
# time  (16 = one bar)
num_timesteps = 4
# This is the size of the visible layer.
n_visible = 2 * note_range * num_timesteps
n_hidden = 500  # 50 #This is the size of the hidden layer

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir", default="mozart", type=str)
parser.add_argument("--output_dir", default="out", type=str)
parser.add_argument("--checkpoint_dir", default="vaesaved", type=str)

parser.add_argument(
    "--epochs",
    default=20000,
    type=int,
    help="The number of training epochs that we are going to run. For each epoch we go through the entire data set.")
parser.add_argument(
    "--batch_size",
    default=100,
    type=int,
    help="The number of training examples that we are going to send through the model at a time. ")
parser.add_argument(
    "--threshold",
    default=0.67,
    type=float,
    help="confidence threshold for thresh_S, note output")

args = parser.parse_args()

songs = read.get_songs(args.dataset_dir)
print("Successfully processed {} songs.".format(len(songs)))


X = tf.placeholder(tf.float32, shape=[None, X_dim], name="X")
z = tf.placeholder(tf.float32, shape=[None, z_dim], name="z")

with tf.variable_scope("S", reuse=True):
    Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]), name="Q_b1")
    Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]), name="Q_W1")

    Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]), name="Q_W2_mu")
    Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]), name="Q_b2_mu")

    Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]), name="Q_W2_sigma")
    Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]), name="Q_b2_sigma")

    P_W1 = tf.Variable(xavier_init([z_dim, h_dim]), name="P_W1")
    P_b1 = tf.Variable(tf.zeros(shape=[h_dim]), name="P_b1")

    P_b2 = tf.Variable(tf.zeros(shape=[X_dim]), name="P_b2")
    P_W2 = tf.Variable(xavier_init([h_dim, X_dim]), name="P_W2")


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


z_mu, z_logvar = Q(X)
z_sample = sample_z(z_mu, z_logvar)
_, logits = P(z_sample)

# Sampling from random z
X_samples, _ = P(z)

# E[log P(X|z)]
recon_loss = tf.reduce_sum(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=X), 1)
# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)  # check learning rate

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "S"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# The number of training epochs that we are going to run. For each epoch
# we go through the entire data set.
num_epochs = 10000
# The number of training examples that we are going to send through the
# model at a time.
batch_size = 100
# lr         = tf.constant(0.005, tf.float32) #The learning rate of our model

i = 0
loss_value = np.array([])
iter_value = np.array([])
songs = [songs[0]]
while i <= num_epochs:
    for song in songs:
        # The songs are stored in a time x notes format. The size of each song is timesteps_in_song x 2*note_range
        # Here we reshape the songs so that each training example is a vector
        # with num_timesteps x 2*note_range elements
        song = np.array(song)
        song = song[:np.floor(
            song.shape[0] / num_timesteps).astype(int) * num_timesteps]
        song = np.reshape(song,
                          [int(song.shape[0] / num_timesteps),
                           song.shape[1] * num_timesteps])

        # Train the VAE on batch_size examples at a time
        for ind in range(0, len(song), batch_size):
            X_mb = song[ind:ind + batch_size]
            _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb})

    if i % 100 == 0:
        log_str = '[Iter: {}] '.format(i)
        log_str += '[Loss: {}]'.format(loss)
        print(log_str)
        iter_value = np.append(iter_value, i)

        loss_value = np.append(loss_value, loss)

        # print(iter_value)
        # print(loss_value)
        #plt.figure()

    if i % 1000 == 0:
        # here yass bessties. X_samples only thing you need
        samples = sess.run(X_samples, feed_dict={z: np.random.randn(1, z_dim)})
        S = np.reshape(samples, (num_timesteps, 2 * note_range))
        thresh_S = S >= 0.5
        '''
        plt.figure(figsize=(12, 2))
        plt.subplot(1, 2, 1)
        plt.imshow(S)
        plt.subplot(1, 2, 2)
        plt.imshow(thresh_S)
        plt.tight_layout()
        plt.pause(0.1)
        plt.plot(iter_value, loss_value, 'o-')

        plt.show()
        '''

        # midi_manipulation.noteStateMatrixToMidi(thresh_S, "out/generated_chord_{}".format(i))
        #                 print(i)


    i += 1

print("Model trained successfully.")


x_samp = iter_value
y_samp = loss_value

mname = 'VAEmodel'
save_path = saver.save(sess, './' + args.checkpoint_dir + '/' + mname + ".ckpt")
print("Model saved in path: %s" % save_path)
sess.close()
