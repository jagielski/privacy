# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Training a deep NN on IMDB reviews with differentially private Adam optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer


from absl import app
from absl import flags

import audit

#### FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('d', 250, 'Data dimension')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 250, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('attack_type', "clip_aware", 'clip_aware or backdoor')
flags.DEFINE_integer('num_jobs', 10, 'Number of jobs for parallelism')
flags.DEFINE_integer('num_trials', 100, 'Number of trials for auditing')
flags.DEFINE_float('attack_l2_norm', 10, 'Size of poisoning data')
flags.DEFINE_float('alpha', 0.05, '1-confidence')
FLAGS = flags.FLAGS


def compute_epsilon(train_size):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / train_size
  steps = FLAGS.epochs * train_size / FLAGS.batch_size
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to approximate 1 / (number of training points).
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

def build_model(x, y):
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(FLAGS.d,),
      use_bias=False, kernel_initializer=tf.keras.initializers.Zeros())])
  return model


def train_model(model, train_x, train_y):
  """Train the model on given data."""
  optimizer = dp_optimizer_vectorized.VectorizedDPSGD(
  #optimizer = DPKerasSGDOptimizer(
      l2_norm_clip=FLAGS.l2_norm_clip,
      noise_multiplier=FLAGS.noise_multiplier,
      num_microbatches=FLAGS.microbatches,
      learning_rate=FLAGS.learning_rate)

  #(.5-x.w)^2 -> 2(.5-x.w)x
  loss = tf.keras.losses.MeanSquaredError(reduction=tf.losses.Reduction.NONE)
  #loss = tf.keras.losses.MeanSquaredError()

  # Compile model with Keras
  model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

  # Train model with Keras
  model.fit(train_x, train_y,
            epochs=FLAGS.epochs,
            validation_data=(train_x, train_y),
            batch_size=FLAGS.batch_size,
            verbose=0)
  return model


def membership_test(model, pois_x, pois_y):
  """Membership inference - detect poisoning."""
  return model.predict(pois_x)

  #return model.trainable_weights[0].numpy().ravel()


def gen_data(n, d):
  """Make binomial dataset."""
  x = np.random.normal(size=(n, d))  # (np.random.uniform(size=(n, d)) > 0.5)*2 - 1
  y = np.ones(shape=(n,))/2.
  return x, y

def train_and_score(dataset):
  """Complete training run with membership inference score."""
  x, y, pois_x, pois_y, i = dataset
  np.random.seed(i)
  tf.set_random_seed(i)
  model = build_model(x, y)
  model = train_model(model, x, y)
  return membership_test(model, pois_x, pois_y)


def main(unused_argv):
  del unused_argv
  # Load training and test data.
  np.random.seed(0)

  x, y = gen_data(1 + FLAGS.batch_size, FLAGS.d) 
  
  auditor = audit.AuditAttack(x, y, train_and_score)  #"mean_audit", f"mean_audit_trainer.py {train_args}")
  pois_x1, pois_x2 = x[:-1].copy(), x[:-1].copy()
  pois_x1[-1] = x[-1]
  pois_y = y[:-1]
  target_x =  x[-1][None, :]
  assert np.unique(np.nonzero(pois_x1 - pois_x2)[0]).size==1

  pois_data = (pois_x1, pois_y), (pois_x2, pois_y), (target_x, y[-1]) 
  poisoning = {}
  poisoning["data"] = (pois_data[0], pois_data[1])
  poisoning["pois"] = pois_data[2]
  auditor.poisoning = poisoning
  
  pois_scores, unpois_scores = auditor.run_experiments(FLAGS.num_trials,
                                                       FLAGS.num_jobs)

  thresh, _, _ = audit.compute_results(pois_scores, unpois_scores, 1,
                              alpha=FLAGS.alpha, threshold=None)
  
  pois_scores, unpois_scores = auditor.run_experiments(FLAGS.num_trials,
                                                       FLAGS.num_jobs)
  _, eps, acc = audit.compute_results(pois_scores, unpois_scores, 1,
                              alpha=FLAGS.alpha, threshold=thresh)
 
  epsilon_ub = compute_epsilon(FLAGS.batch_size)

  print(f"Analysis epsilon is {epsilon_ub}.")
  print(f"At threshold={thresh}, epsilon={eps}.")
  print(f"The best accuracy at distinguishing poisoning is {acc}.")

if __name__ == '__main__':
  app.run(main)
