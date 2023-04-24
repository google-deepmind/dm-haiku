# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for haiku._src.config."""

from concurrent import futures
import inspect
import threading

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import config


class ConfigTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._before = config.main_thread_config
    config.tls.config = config.main_thread_config = config.Config.default()

  def tearDown(self):
    super().tearDown()
    config.tls.config = config.main_thread_config = self._before
    del self._before

  def test_check_jax_usage(self):
    cfg = config.get_config()
    config.check_jax_usage()
    self.assertTrue(cfg.check_jax_usage)
    config.check_jax_usage(False)
    self.assertFalse(cfg.check_jax_usage)
    config.check_jax_usage(True)
    self.assertTrue(cfg.check_jax_usage)

  @parameterized.parameters(True, False)
  def test_inherits_default_from_main_thread(self, default):
    e1 = threading.Event()
    e2 = threading.Event()

    config.get_config().check_jax_usage = default

    def f():
      self.assertEqual(config.get_config().check_jax_usage, default)
      config.get_config().check_jax_usage = True
      e1.set()
      e2.wait()
      self.assertTrue(config.get_config().check_jax_usage)

    def g():
      e1.wait()
      self.assertEqual(config.get_config().check_jax_usage, default)
      config.get_config().check_jax_usage = False
      e2.set()
      self.assertFalse(config.get_config().check_jax_usage)

    with futures.ThreadPoolExecutor() as tpe:
      f1 = tpe.submit(g)
      f2 = tpe.submit(f)
      f2.result()
      f1.result()

    self.assertEqual(config.get_config().check_jax_usage, default)

  def test_with_config(self):
    ran_f = [False]

    @config.with_config(check_jax_usage=False)
    def f():
      ran_f[0] = True
      return config.get_config().check_jax_usage

    cfg = config.get_config()
    cfg.check_jax_usage = True
    self.assertFalse(f())
    self.assertTrue(ran_f[0])
    self.assertTrue(cfg.check_jax_usage)

  def test_assign(self):
    cfg = config.get_config()
    cfg.check_jax_usage = False
    with config.assign(check_jax_usage=True):
      self.assertTrue(cfg.check_jax_usage)
    self.assertFalse(cfg.check_jax_usage)

  def test_assign_with_error(self):
    cfg = config.get_config()
    cfg.check_jax_usage = False
    try:
      with config.assign(check_jax_usage=True):
        self.assertTrue(cfg.check_jax_usage)
        # Raise an exception to test that config is reset on error.
        raise ValueError("expected")
    except ValueError:
      pass
    self.assertFalse(cfg.check_jax_usage)

  def test_context_matches_set(self):
    context_sig = inspect.signature(config.context)
    set_sig = inspect.signature(config.set)
    self.assertEqual(context_sig.parameters, set_sig.parameters)

  def test_context(self):
    cfg = config.get_config()
    cfg.check_jax_usage = False
    with config.context(check_jax_usage=True):
      self.assertTrue(cfg.check_jax_usage)
    self.assertFalse(cfg.check_jax_usage)

  def test_set(self):
    cfg = config.get_config()
    cfg.check_jax_usage = False
    config.set(check_jax_usage=True)
    self.assertTrue(cfg.check_jax_usage)
    config.set(check_jax_usage=False)
    self.assertFalse(cfg.check_jax_usage)

  def test_rng_reserve_size(self):
    cfg = config.get_config()
    prev = config.rng_reserve_size(3)
    self.assertEqual(cfg.rng_reserve_size, 3)
    self.assertEqual(prev, 1)
    prev = config.rng_reserve_size(10)
    self.assertEqual(cfg.rng_reserve_size, 10)
    self.assertEqual(prev, 3)

  def test_rng_reserve_size_error(self):
    with self.assertRaisesRegex(ValueError, "RNG reserve size"):
      config.rng_reserve_size(0)

    with self.assertRaisesRegex(ValueError, "RNG reserve size"):
      config.rng_reserve_size(-1)

if __name__ == "__main__":
  absltest.main()
