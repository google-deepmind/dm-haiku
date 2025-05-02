"""Training example demonstrating RNN state handling in Haiku."""

import haiku as hk
import jax
import jax.numpy as jnp
from rnn import RNN


def net_fn(batch_size: int = 1):
    """Creates the model and returns initial state."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model = RNN(hidden_size=4)
    return model.initial_state(batch_size)


def forward(x, state):
    """Forward pass of the model."""
    if x.ndim != 2:
        raise ValueError(f"Input must be 2D (batch_size, features), got shape {x.shape}")
    model = RNN(hidden_size=4)
    return model(x, state)


def main():
    """Main training loop demonstrating RNN state handling."""
    # Transform functions
    init_fn = hk.transform(net_fn)
    forward_fn = hk.transform_with_state(forward)
    
    # Initialize model
    rng = jax.random.PRNGKey(42)
    batch_size = 2
    seq_length = 5
    
    # Create test data
    inputs = jnp.ones((batch_size, seq_length))
    
    # Get initial state and parameters
    state = init_fn.apply({}, rng, batch_size=batch_size)
    params = forward_fn.init(rng, inputs, state)
    
    # Run forward pass
    output, new_state = forward_fn.apply(params, state, rng, inputs, state)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")


if __name__ == "__main__":
    main() 