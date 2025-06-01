"""Example demonstrating RNN state handling in Haiku.

This example shows how to properly initialize and manage RNN state in Haiku,
specifically focusing on LSTM state handling and reinitialization.
"""

import haiku as hk
import jax
import jax.numpy as jnp


class RNN(hk.RNNCore):
    """Simple RNN wrapper around LSTM demonstrating proper state handling."""
    
    def __init__(self, hidden_size=4, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.rnn = hk.LSTM(hidden_size)
    
    def __call__(self, inputs, state):
        out, h = self.rnn(inputs, state)
        return out, h
    
    def initial_state(self, batch_size):
        return self.rnn.initial_state(batch_size) 