"""Tests for Feedback Alignment implementation."""

import jax
import jax.numpy as jnp
import haiku as hk
from haiku._src.dfa.core import FALinear, fa_linear


def test_fa_linear_forward():
    """Test that FA linear layer produces correct forward pass output."""
    print("Testing FA linear forward pass...")
    
    def network(x):
        return FALinear(32, name="fa_layer")(x)
    
    network = hk.transform(network)
    
    rng = jax.random.PRNGKey(42)
    x = jnp.ones((4, 10))
    
    params = network.init(rng, x)
    output = network.apply(params, rng, x)
    
    assert output.shape == (4, 32), f"Expected shape (4, 32), got {output.shape}"
    print(f"✓ Forward pass output shape: {output.shape}")
    print(f"✓ Output range: [{output.min():.3f}, {output.max():.3f}]")


def test_fa_linear_gradient():
    """Test that FA linear layer computes gradients correctly."""
    print("\nTesting FA linear gradients...")
    
    def network(x):
        x = FALinear(32, name="fa1")(x)
        x = jax.nn.relu(x)
        x = FALinear(16, name="fa2")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(10, name="output")(x)
        return x
    
    def loss_fn(params, x, y):
        network_fn = hk.transform(network)
        pred = network_fn.apply(params, None, x)
        return jnp.mean((pred - y) ** 2)
    
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (4, 20))
    y = jax.random.normal(rng, (4, 10))
    
    network_fn = hk.transform(network)
    params = network_fn.init(rng, x)
    
    # Compute gradients
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    
    print(f"✓ Loss: {loss:.4f}")
    print(f"✓ Gradients computed successfully")
    
    # Check that feedback matrices have zero gradient
    assert jnp.allclose(grads["fa1"]["feedback"], 0.0), "Feedback should have zero gradient"
    assert jnp.allclose(grads["fa2"]["feedback"], 0.0), "Feedback should have zero gradient"
    print("✓ Feedback matrices have zero gradient (as expected)")
    
    # Check that weight gradients are non-zero
    assert not jnp.allclose(grads["fa1"]["w"], 0.0), "Weight gradients should be non-zero"
    assert not jnp.allclose(grads["fa2"]["w"], 0.0), "Weight gradients should be non-zero"
    print("✓ Weight gradients are non-zero")


def test_fa_vs_standard_backprop():
    """Compare FA with standard backpropagation."""
    print("\nComparing FA with standard backprop...")
    
    def fa_network(x):
        x = FALinear(64, name="hidden1")(x)
        x = jax.nn.tanh(x)
        x = FALinear(32, name="hidden2")(x)
        x = jax.nn.tanh(x)
        x = hk.Linear(10, name="output")(x)
        return x
    
    def standard_network(x):
        x = hk.Linear(64, name="hidden1")(x)
        x = jax.nn.tanh(x)
        x = hk.Linear(32, name="hidden2")(x)
        x = jax.nn.tanh(x)
        x = hk.Linear(10, name="output")(x)
        return x
    
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (8, 20))
    y = jax.random.normal(rng, (8, 10))
    
    # Initialize both networks
    fa_net = hk.transform(fa_network)
    std_net = hk.transform(standard_network)
    
    fa_params = fa_net.init(rng, x)
    std_params = std_net.init(rng, x)
    
    # Forward pass
    fa_out = fa_net.apply(fa_params, None, x)
    std_out = std_net.apply(std_params, None, x)
    
    print(f"✓ FA output shape: {fa_out.shape}")
    print(f"✓ Standard output shape: {std_out.shape}")
    
    # Both should produce valid outputs
    assert not jnp.any(jnp.isnan(fa_out)), "FA output contains NaN"
    assert not jnp.any(jnp.isnan(std_out)), "Standard output contains NaN"
    print("✓ Both networks produce valid outputs")


def test_fa_training_loop():
    """Test FA in a simple training loop."""
    print("\nTesting FA in training loop...")
    
    def network(x):
        x = FALinear(32, name="hidden")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(1, name="output")(x)
        return x
    
    def loss_fn(params, x, y):
        net = hk.transform(network)
        pred = net.apply(params, None, x)
        return jnp.mean((pred - y) ** 2)
    
    # Create simple dataset
    rng = jax.random.PRNGKey(42)
    x_train = jax.random.normal(rng, (100, 10))
    y_train = jnp.sum(x_train, axis=1, keepdims=True)
    
    # Initialize network
    net = hk.transform(network)
    params = net.init(rng, x_train[:1])
    
    # Training loop
    learning_rate = 0.01
    losses = []
    
    for step in range(10):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_train, y_train)
        losses.append(float(loss))
        
        # Simple SGD update
        params = jax.tree_util.tree_map(
            lambda p, g: p - learning_rate * g, params, grads
        )
    
    print(f"✓ Initial loss: {losses[0]:.4f}")
    print(f"✓ Final loss: {losses[-1]:.4f}")
    print(f"✓ Loss decreased: {losses[-1] < losses[0]}")
    
    assert losses[-1] < losses[0], "Loss should decrease during training"
    print("✓ FA training successful")


def test_functional_interface():
    """Test the functional fa_linear interface."""
    print("\nTesting functional interface...")
    
    def network(x):
        layer = fa_linear(32, name="fa_layer")
        return layer(x)
    
    net = hk.transform(network)
    
    rng = jax.random.PRNGKey(42)
    x = jnp.ones((4, 10))
    
    params = net.init(rng, x)
    output = net.apply(params, rng, x)
    
    assert output.shape == (4, 32), f"Expected shape (4, 32), got {output.shape}"
    print(f"✓ Functional interface works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Feedback Alignment Tests")
    print("=" * 60)
    
    test_fa_linear_forward()
    test_fa_linear_gradient()
    test_fa_vs_standard_backprop()
    test_fa_training_loop()
    test_functional_interface()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
