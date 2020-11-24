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
import tensorflow as tf

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer


from absl import app
from absl import flags

import audit

#### FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_integer('i', None, 'training index')
flags.DEFINE_integer('is_pois', None, 'whether to use poisoned data')
flags.DEFINE_string('name', None, 'name of dataset')
flags.DEFINE_integer(
    'microbatches', 250, 'Number of microbatches '
    '(must evenly divide batch_size)')
FLAGS = flags.FLAGS


def train_model(train_x, train_y):
  """Train the model on given data."""
  
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(1, input_shape=(train_x.shape[1],), use_bias=False,
                            kernel_initializer=tf.keras.initializers.Zeros())])
  
  #optimizer = dp_optimizer_vectorized.VectorizedDPSGD(
  optimizer = DPKerasSGDOptimizer(
      l2_norm_clip=FLAGS.l2_norm_clip,
      noise_multiplier=FLAGS.noise_multiplier,
      num_microbatches=FLAGS.microbatches,
      learning_rate=FLAGS.learning_rate)

  #(.5-x.w)^2 -> 2(.5-x.w)x
  #x.w = 0: 2(.5-x.w)x = x
  loss = tf.keras.losses.MeanSquaredError()

  # Compile model with Keras
  model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

  # Train model with Keras
  model.fit(train_x, train_y,
            epochs=FLAGS.epochs,
            validation_data=(train_x, train_y),
            batch_size=FLAGS.batch_size,
            verbose=1)
  return model


def membership_test(model, pois_x, pois_y):
  """Membership inference - detect poisoning."""
  return model.predict(pois_x)
  #return model.trainable_weights[0].numpy().ravel()


def train_and_score(dataset):
  """Complete training run with membership inference score."""
  x, y, (pois_x, pois_y) = dataset
  model = train_model(x, y)
  return membership_test(model, pois_x, pois_y)


def main(unused_argv):
  del unused_argv
  
  np.random.seed(FLAGS.i)
  tf.random.set_seed(FLAGS.i)
  
  dataset = np.load(f"{FLAGS.name}.npy", allow_pickle=True)
  (pois_x1, pois_y1), (pois_x2, pois_y2), (sample_x, sample_y) = dataset
  if FLAGS.is_pois:
    dataset = pois_x1, pois_y1, (sample_x, sample_y)
  else:
    dataset = pois_x2, pois_y2, (sample_x, sample_y)
  print(train_and_score(dataset).item())
    

if __name__ == '__main__':
  app.run(main)
