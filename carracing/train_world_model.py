#### INITIAL SETTINGS

cuda_device="-1" # CPU "-1" or "0" for GPU
z_size = 32
batch_size = 2 # usually batch size is 100

################################# Extract frames ################################# Extract frames

'''
saves ~ X episodes generated from a random policy
'''

import numpy as np
import random
import os
import gym

from model import make_model

# MAX_FRAMES = 1000 # max length of carracing
# MAX_TRIALS = 10000 # just use this to extract one trial.
print("change extract settings back in extract.py")
MAX_FRAMES = 100 # max length of carracing
MAX_TRIALS = 4 # just use this to extract one trial.

render_mode = False # for debugging.

DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

model = make_model(load_model=False)

total_frames = 0
model.make_env(render_mode=render_mode, full_episode=False)
for trial in range(int(MAX_TRIALS*1.25)): # increase by 25% due to possible errors during training.
  try:
    random_generated_int = random.randint(0, 2**31-1)
    filename = DIR_NAME+"/"+str(random_generated_int)+".npz"
    recording_obs = []
    recording_action = []

    np.random.seed(random_generated_int)
    model.env.seed(random_generated_int)

    # random policy
    model.init_random_model_params(stdev=np.random.rand()*0.01)

    model.reset()
    obs = model.env.reset() # pixels

    for frame in range(MAX_FRAMES):
      model.env.render("rgb_array")

      recording_obs.append(obs)
      z, mu, logvar = model.encode_obs(obs)
      action = model.get_action(z)

      recording_action.append(action)
      obs, reward, done, info = model.env.step(action)

      if done:
        break

    total_frames += (frame+1)
    print("dead at", frame+1, "total recorded frames for this worker", total_frames)
    recording_obs = np.array(recording_obs, dtype=np.uint8)
    recording_action = np.array(recording_action, dtype=np.float16)
    np.savez_compressed(filename, obs=recording_obs, action=recording_action)
  except gym.error.Error:
    print("stupid gym error, life goes on")
    model.env.close()
    model.make_env(render_mode=False)
    continue
model.env.close()

################################# Train VAE ################################# Train VAE

'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device  # can override for multi-gpu systems
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from vae.vae import ConvVAE, reset_graph

# Hyperparameters for ConvVAE
learning_rate = 0.0001
kl_tolerance = 0.5

# Parameters for training
NUM_EPOCH = 10
DATA_DIR = "record"

model_save_path = "tf_vae"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)


def count_length_of_filelist(filelist):
  # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
  N = len(filelist)
  total_length = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    total_length += l
    if (i % 1000 == 0):
      print("loading file", i)
  return total_length

  # def create_dataset(filelist, N=10000, M=1000): # N is 10000 episodes, M is number of timesteps


def create_dataset(filelist, N=MAX_TRIALS, M=MAX_FRAMES):  # N is 10 episodes, M is number of timesteps
  print("change create_dataset settings back in vae train")

  data = np.zeros((M * N, 96, 96, 3), dtype=np.uint8)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    if (idx + l) > (M * N):
      data = data[0:idx]
      print('premature break')
      break
    data[idx:idx + l] = raw_data
    idx += l
    if ((i + 1) % 100 == 0):
      print("loading file", i + 1)
  return data


# load dataset from record/*. only use first 10K, sorted by filename.
filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:MAX_TRIALS]
# print("check total number of images:", count_length_of_filelist(filelist))
dataset = create_dataset(filelist)

# split into batches:
total_length = len(dataset)
num_batches = int(np.floor(total_length / batch_size))
print("num_batches", num_batches)

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True)

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
for epoch in range(NUM_EPOCH):
  print(epoch, NUM_EPOCH)
  np.random.shuffle(dataset)
  for idx in range(num_batches):
    print(idx, num_batches)
    batch = dataset[idx * batch_size:(idx + 1) * batch_size]

    obs = batch.astype(np.float) / 255.0

    feed = {vae.x: obs}

    (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
      vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
    ], feed)

    if ((train_step + 1) % 500 == 0):
      print("step", (train_step + 1), train_loss, r_loss, kl_loss)
    if ((train_step + 1) % 5000 == 0):
      vae.save_json("tf_vae/vae.json")

# finished, final model:
vae.save_json("tf_vae/vae.json")


################################# Train MDN-RNN ################################# Train MDN-RNN

'''
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
'''

import json
import time

from rnn.rnn import HyperParams, MDNRNN

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

DATA_DIR = "series"
model_save_path = "tf_rnn"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

initial_z_save_path = "tf_initial_z"
if not os.path.exists(initial_z_save_path):
  os.makedirs(initial_z_save_path)


def random_batch():
  indices = np.random.permutation(N_data)[0:batch_size]
  mu = data_mu[indices]
  logvar = data_logvar[indices]
  action = data_action[indices]
  s = logvar.shape
  z = mu + np.exp(logvar / 2.0) * np.random.randn(*s)
  return z, action


def default_hps():
  return HyperParams(num_steps=4000,
                     max_seq_len=(MAX_FRAMES-1), # train on sequences of 1000 (so 999 + teacher forcing shift)
                     input_seq_width=(z_size+3),    # width of our data (32 + 3 actions) 32 for encoder size
                     output_seq_width=z_size,    # width of our data is 32
                     rnn_size=256,    # number of rnn cells
                     batch_size=batch_size,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=5,   # number of mixtures in MDN
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)

raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))

# load preprocessed data
data_mu = raw_data["mu"]
data_logvar = raw_data["logvar"]
data_action = raw_data["action"]
max_seq_len = hps_model.max_seq_len

N_data = len(data_mu)  # should be 10k
batch_size = hps_model.batch_size

# save 1000 initial mu and logvars:
initial_mu = np.copy(data_mu[:1000, 0, :] * 10000).astype(np.int).tolist()
initial_logvar = np.copy(data_logvar[:1000, 0, :] * 10000).astype(np.int).tolist()
with open(os.path.join("tf_initial_z", "initial_z.json"), 'wt') as outfile:
  json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))

reset_graph()
rnn = MDNRNN(hps_model)

# train loop:
hps = hps_model
start = time.time()
for local_step in range(hps.num_steps):

  step = rnn.sess.run(rnn.global_step)
  curr_learning_rate = (hps.learning_rate - hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate

  raw_z, raw_a = random_batch()
  inputs = np.concatenate((raw_z[:, :-1, :], raw_a[:, :-1, :]), axis=2)
  outputs = raw_z[:, 1:, :]  # teacher forcing (shift by one predictions)

  feed = {rnn.input_x: inputs, rnn.output_x: outputs, rnn.lr: curr_learning_rate}
  (train_cost, state, train_step, _) = rnn.sess.run([rnn.cost, rnn.final_state, rnn.global_step, rnn.train_op], feed)
  if (step % 20 == 0 and step > 0):
    end = time.time()
    time_taken = end - start
    start = time.time()
    output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (
    step, curr_learning_rate, train_cost, time_taken)
    print(output_log)

# save the model (don't bother with tf checkpoints json all the way ...)
rnn.save_json(os.path.join(model_save_path, "rnn.json"))

