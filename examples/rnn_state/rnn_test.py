"""Tests for RNN state handling example."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rnn import RNN


class RNNTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.rng = jax.random.PRNGKey(42)

    @parameterized.parameters(1, 2, 4)
    def test_rnn_shapes(self, batch_size):
        """Tests that RNN outputs have the expected shapes."""
        def net_fn(batch_size):
            model = RNN(hidden_size=4)
            return model.initial_state(batch_size)

        def forward(x, state):
            model = RNN(hidden_size=4)
            return model(x, state)

        init_fn = hk.transform(net_fn)
        forward_fn = hk.transform_with_state(forward)

        seq_length = 5
        inputs = jnp.ones((batch_size, seq_length))
        
        # Test initialization
        state = init_fn.apply({}, self.rng, batch_size=batch_size)
        params = forward_fn.init(self.rng, inputs, state)
        
        # Test forward pass
        output, new_state = forward_fn.apply(params, state, self.rng, inputs, state)
        
        # Check shapes
        self.assertEqual(output.shape, (batch_size, 4))
        
        # Check state structure
        self.assertIsInstance(new_state, hk.LSTMState)
        self.assertEqual(new_state.hidden.shape, (batch_size, 4))
        self.assertEqual(new_state.cell.shape, (batch_size, 4))

    def test_state_evolution(self):
        """Tests that RNN state properly evolves."""
        def net_fn(batch_size):
            model = RNN(hidden_size=4)
            return model.initial_state(batch_size)

        def forward(x, state):
            model = RNN(hidden_size=4)
            return model(x, state)

        init_fn = hk.transform(net_fn)
        forward_fn = hk.transform_with_state(forward)

        batch_size = 1
        seq_length = 5
        inputs = jnp.ones((batch_size, seq_length))
        
        # Initialize
        state = init_fn.apply({}, self.rng, batch_size=batch_size)
        params = forward_fn.init(self.rng, inputs, state)
        
        # Run two forward passes
        output1, state1 = forward_fn.apply(params, state, self.rng, inputs, state)
        output2, state2 = forward_fn.apply(params, state1, self.rng, inputs, state1)
        
        # Check that states are different
        state_diff = jax.tree_util.tree_map(
            lambda x, y: jnp.any(x != y), state1, state2)
        self.assertTrue(
            any(jax.tree_util.tree_leaves(state_diff)),
            "State should change between steps")


if __name__ == '__main__':
    absltest.main() 