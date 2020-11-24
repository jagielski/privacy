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

import multiprocessing as mp
import numpy as np
from scipy import stats
import subprocess

import attacks

def clopper_pearson(count, trials, alpha):
  """
  Computes clopper pearson confidence interval.
  Code from https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
  """
  count, trials, alpha = np.array(count), np.array(trials), np.array(alpha)
  q = count / trials
  ci_low = stats.beta.ppf(alpha / 2., count, trials - count + 1)
  ci_upp = stats.beta.isf(alpha / 2., count + 1, trials - count)

  if np.ndim(ci_low) > 0:
    ci_low[q == 0] = 0
    ci_upp[q == 1] = 1
  else:
    ci_low = ci_low if (q != 0) else 0
    ci_upp = ci_upp if (q != 1) else 1
  return ci_low, ci_upp


def compute_results(poison_scores, unpois_scores, pois_ct,
                    alpha=0.05, threshold=None):
  """
  Searches over thresholds for the best epsilon lower bound and accuracy.
  poison_scores: list of scores from poisoned models
  unpois_scores: list of scores from unpoisoned models
  pois_ct: number of poison points
  alpha: confidence parameter
  threshold: if None, search over all thresholds, else use given threshold
  """
  if threshold is None:  # search for best threshold
    all_thresholds = np.unique(poison_scores + unpois_scores)
  else:
    all_thresholds = [threshold]

  poison_arr = np.array(poison_scores)
  unpois_arr = np.array(unpois_scores)

  best_threshold, best_epsilon, best_acc = None, 0, 0
  for thresh in all_thresholds:
    epsilon, acc = compute_epsilon_and_acc(poison_arr, unpois_arr, thresh,
                                           alpha, pois_ct)
    if epsilon > best_epsilon:
      best_epsilon, best_threshold = epsilon, thresh
    best_acc = max(best_acc, acc)
  return best_threshold, best_epsilon, best_acc


def compute_epsilon_and_acc(poison_arr, unpois_arr, threshold, alpha, pois_ct):
  """For a given threshold, compute epsilon and accuracy."""
  poison_ct = (poison_arr > threshold).sum()
  unpois_ct = (unpois_arr > threshold).sum()

  p1, _ = clopper_pearson(poison_ct, poison_arr.size, alpha)
  _, p0 = clopper_pearson(unpois_ct, unpois_arr.size, alpha)

  print(poison_ct, unpois_ct, p1, p0)

  if (p1 <= 1e-5) or (p0 >= 1 - 1e-5):
    return 0, 0

  if (p0 + p1) > 1:  # see Appendix A
    p0, p1 = (1-p1), (1-p0)

  epsilon = np.log(p1/p0)/pois_ct
  acc = (p1 + (1-p0))/2

  return epsilon, acc


def run_experiment(args):
  script, name, i, is_pois = args
  cmd = f"python {script} --name={name} --is_pois={is_pois} --i={i}"
  print(cmd)
  process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out, err = process.communicate()
  print(out)
  return float(out.split()[-1])

class AuditAttack:
  """Audit attack class. Generates poisoning, then runs auditing algorithm."""
  def __init__(self, trn_x, trn_y, name, train_script):
    """
    trn_x: training features
    trn_y: training labels
    name: identifier for the attack
    train_script: command for running the attack
    """
    self.trn_x, self.trn_y = trn_x, trn_y
    self.train_script = train_script
    self.name = name

  def make_poisoning(self, pois_ct, attack_type, l2_norm=10):
    return attacks.make_many_pois(self.trn_x, self.trn_y, [pois_ct],
                                  attack=attack_type, l2_norm=l2_norm)

  def run_experiments(self, num_trials, num_jobs):
    """Uses multiprocessing to run all training experiments."""
    pool = mp.Pool(num_jobs)
    
    poison_args = [(self.train_script, self.name, i, 1) for i in range(num_trials)]
    unpois_args = [(self.train_script, self.name, i, 0) for i in range(num_trials)]

    poison_scores = pool.map(run_experiment, poison_args)
    unpois_scores = pool.map(run_experiment, unpois_args)

    return poison_scores, unpois_scores

  def run(self, pois_ct, attack_type, num_trials, num_jobs, alpha=0.05,
          threshold=None, l2_norm=10):
    """Complete auditing algorithm."""
    pois_datasets = self.make_poisoning(pois_ct, attack_type, l2_norm=l2_norm)
    (pois_x1, pois_y1), (pois_x2, pois_y2) = pois_datasets[pois_ct]
    assert np.allclose(pois_x1, pois_x2)
    pois_diff = (pois_y1 - pois_y2)
    assert np.unique(np.nonzero(pois_diff)[0]).size == pois_ct
    sample_x, sample_y = pois_datasets["pois"]
    np.save(self.name, (pois_x1, pois_y1, pois_x2, pois_y2, sample_x, sample_y))

    poison_scores, unpois_scores = self.run_experiments(num_trials, num_jobs)

    results = compute_results(poison_scores, unpois_scores, pois_ct,
                              alpha=alpha, threshold=threshold)
    return results
