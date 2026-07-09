# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for DFA core implementation."""

import haiku as hk
from haiku._src.dfa import core
import jax
import jax.numpy as jnp


def test_dfa_linear_forward():
    """Test that DFA linear layer produces correct forward pass output."""
    
    def network(x):
        return core.DFALinear(output_size=10, output_dim=5, activation=jax.nn.tanh)(x)
    
    network = hk.transform(network)
    rng = jax.random.PRNGKey(42)
    x = jnp.ones((2, 8))
    
    params = network.init(rng, x)
    output = network.apply(params, rng, x)
    
    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
    print("✓ Forward pass shape test passed")


def test_dfa_network_forward():
    """Test multi-layer DFA network forward pass."""
    
    def network(x):
        # DFA layers with built-in activation
        x = core.DFALinear(output_size=64, output_dim=10, activation=jax.nn.tanh)(x)
        x = core.DFALinear(output_size=32, output_dim=10, activation=jax.nn.tanh)(x)
        # Output layer (standard, no activation)
        x = hk.Linear(10)(x)
        return x
    
    network = hk.transform(network)
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (4, 20))
    
    params = network.init(rng, x)
    output = network.apply(params, rng, x)
    
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
    print("✓ Multi-layer DFA network forward pass test passed")


def test_dfa_gradient_computation():
    """Test that DFA computes gradients correctly."""
    
    def network(x):
        x = core.DFALinear(output_size=64, output_dim=10, activation=jax.nn.tanh)(x)
        x = core.DFALinear(output_size=32, output_dim=10, activation=jax.nn.tanh)(x)
        x = hk.Linear(10)(x)
        return x
    
    def loss_fn(params, x, y):
        output = network.apply(params, None, x)
        return jnp.mean((output - y) ** 2)
    
    network = hk.transform(network)
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (4, 20))
    y = jax.random.normal(rng, (4, 10))
    
    params = network.init(rng, x)
    
    # Compute gradients
    try:
        grads = jax.grad(loss_fn)(params, x, y)
        
        # Check that gradients exist for all parameters
        assert "dfa_linear/w" in grads
        assert "dfa_linear/b" in grads
        assert "dfa_linear/feedback" in grads
        assert "dfa_linear_1/w" in grads
        assert "dfa_linear_1/feedback" in grads
        assert "linear/w" in grads
        
        # Check that feedback gradients are zero
        fb1_grad = grads["dfa_linear/feedback"]
        fb2_grad = grads["dfa_linear_1/feedback"]
        assert jnp.allclose(fb1_grad, 0.0), "Feedback matrix 1 should have zero gradient"
        assert jnp.allclose(fb2_grad, 0.0), "Feedback matrix 2 should have zero gradient"
        
        # Check that weight gradients are non-zero
        w1_grad = grads["dfa_linear/w"]
        w2_grad = grads["dfa_linear_1/w"]
        assert not jnp.allclose(w1_grad, 0.0), "Weight gradients should be non-zero"
        assert not jnp.allclose(w2_grad, 0.0), "Weight gradients should be non-zero"
        
        print("✓ Gradient computation test passed")
        print(f"  Layer 1 weight gradient norm: {jnp.linalg.norm(w1_grad):.4f}")
        print(f"  Layer 2 weight gradient norm: {jnp.linalg.norm(w2_grad):.4f}")
        print(f"  Layer 1 feedback gradient norm: {jnp.linalg.norm(fb1_grad):.4f}")
        print(f"  Layer 2 feedback gradient norm: {jnp.linalg.norm(fb2_grad):.4f}")
        
    except Exception as e:
        print(f"✗ Gradient computation failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_dfa_training_step():
    """Test that DFA network can perform a training step."""
    
    def network(x):
        x = core.DFALinear(output_size=64, output_dim=10, activation=jax.nn.tanh)(x)
        x = core.DFALinear(output_size=32, output_dim=10, activation=jax.nn.tanh)(x)
        x = hk.Linear(10)(x)
        return x
    
    def loss_fn(params, x, y):
        output = network.apply(params, None, x)
        return jnp.mean((output - y) ** 2)
    
    network = hk.transform(network)
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (8, 20))
    y = jax.random.normal(rng, (8, 10))
    
    params = network.init(rng, x)
    
    # Initial loss
    initial_loss = loss_fn(params, x, y)
    
    # Compute gradients and update
    grads = jax.grad(loss_fn)(params, x, y)
    
    # Simple SGD update
    learning_rate = 0.01
    updated_params = jax.tree.map(
        lambda p, g: p - learning_rate * g, params, grads
    )
    
    # New loss
    new_loss = loss_fn(updated_params, x, y)
    
    print("✓ Training step test passed")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Loss after update: {new_loss:.4f}")
    print(f"  Loss decreased: {new_loss < initial_loss}")


if __name__ == "__main__":
    print("Running DFA core tests...\n")
    test_dfa_linear_forward()
    print()
    test_dfa_network_forward()
    print()
    test_dfa_gradient_computation()
    print()
    test_dfa_training_step()
    print("\nAll tests passed!")
