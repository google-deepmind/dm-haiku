# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Single-process IMPALA wiring."""

import threading
from typing import List

from absl import app
from bsuite.experiments.catch import catch
from haiku.examples.impala import actor as actor_lib
from haiku.examples.impala import agent as agent_lib
from haiku.examples.impala import haiku_nets
from haiku.examples.impala import learner as learner_lib
from haiku.examples.impala import util
import jax
from jax.experimental import optix

ACTION_REPEAT = 1
BATCH_SIZE = 2
DISCOUNT_FACTOR = 0.99
MAX_ENV_FRAMES = 20000
NUM_ACTORS = 2
UNROLL_LENGTH = 20

FRAMES_PER_ITER = ACTION_REPEAT * BATCH_SIZE * UNROLL_LENGTH


def run_actor(actor: actor_lib.Actor, stop_signal: List[bool]):
  """Runs an actor to produce num_trajectories trajectories."""
  while not stop_signal[0]:
    frame_count, params = actor.pull_params()
    actor.unroll_and_push(frame_count, params)


def main(_):
  # A thunk that builds a new environment.
  # Substitute your environment here!
  build_env = catch.Catch

  # Construct the agent. We need a sample environment for its spec.
  env_for_spec = build_env()
  num_actions = env_for_spec.action_spec().num_values
  agent = agent_lib.Agent(num_actions, env_for_spec.observation_spec(),
                          haiku_nets.CatchNet)

  # Construct the optimizer.
  max_updates = MAX_ENV_FRAMES / FRAMES_PER_ITER
  opt = optix.rmsprop(1e-1, decay=0.99, eps=0.1)

  # Construct the learner.
  learner = learner_lib.Learner(
      agent,
      jax.random.PRNGKey(428),
      opt,
      BATCH_SIZE,
      DISCOUNT_FACTOR,
      FRAMES_PER_ITER,
      max_abs_reward=1.,
      logger=util.AbslLogger(),  # Provide your own logger here.
  )

  # Construct the actors on different threads.
  # stop_signal in a list so the reference is shared.
  actor_threads = []
  stop_signal = [False]
  for i in range(NUM_ACTORS):
    actor = actor_lib.Actor(
        agent,
        build_env(),
        UNROLL_LENGTH,
        learner,
        rng_seed=i,
        logger=util.AbslLogger(),  # Provide your own logger here.
    )
    args = (actor, stop_signal)
    actor_threads.append(threading.Thread(target=run_actor, args=args))

  # Start the actors and learner.
  for t in actor_threads:
    t.start()
  learner.run(int(max_updates))

  # Stop.
  stop_signal[0] = True
  for t in actor_threads:
    t.join()


if __name__ == '__main__':
  app.run(main)
