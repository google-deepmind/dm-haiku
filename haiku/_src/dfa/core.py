"""Feedback Alignment implementation for Haiku.

This module implements Feedback Alignment (FA), a biologically plausible
alternative to backpropagation. Instead of using the transpose of forward
weights for backpropagation, FA uses fixed random feedback matrices.

This solves the "weight transport problem" - the biological implausibility
of neurons having access to the transpose of their forward weights.

Reference:
    Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
    error backpropagation for deep learning. Nature Communications, 7, 13276.
"""

from typing import Callable, Optional
import jax
import jax.numpy as jnp
import haiku as hk


class FALinear(hk.Module):
    """Linear layer with Feedback Alignment.
    
    This layer uses fixed random feedback weights instead of transposed
    forward weights during backpropagation. This addresses the biological
    implausibility of the weight transport problem while maintaining
    learning performance comparable to standard backpropagation.
    
    The layer can be used as a drop-in replacement for hk.Linear in most
    cases. Apply activation functions separately after the layer.
    
    Example:
        ```python
        def network(x):
            x = hk.FALinear(128, name="hidden1")(x)
            x = jax.nn.relu(x)
            x = hk.FALinear(64, name="hidden2")(x)
            x = jax.nn.relu(x)
            x = hk.Linear(10, name="output")(x)
            return x
        ```
    
    Note:
        The feedback matrix is initialized once and remains fixed during
        training. It is not updated by gradient descent.
    """

    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        feedback_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        """Initializes the FA linear layer.
        
        Args:
            output_size: Number of output features.
            with_bias: Whether to include a bias term.
            w_init: Optional weight initializer. Defaults to truncated normal
                with stddev = 1/sqrt(input_size).
            b_init: Optional bias initializer. Defaults to zeros.
            feedback_init: Optional feedback matrix initializer. Defaults to
                random normal with stddev = 0.05.
            name: Optional name for this module.
        """
        super().__init__(name=name)
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        self.feedback_init = feedback_init

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Applies the FA linear transformation.
        
        Args:
            inputs: Input array of shape (batch_size, input_size).
        
        Returns:
            Output array of shape (batch_size, output_size).
        
        Raises:
            ValueError: If inputs is a scalar.
        """
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        # Initialize forward weights
        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / jnp.sqrt(input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        
        w = hk.get_parameter(
            "w", 
            [input_size, output_size], 
            dtype, 
            init=w_init
        )

        # Initialize fixed random feedback matrix
        # This has the same shape as w.T to replace it during backprop
        feedback_init = self.feedback_init
        if feedback_init is None:
            feedback_init = hk.initializers.RandomNormal(stddev=0.05)
        
        feedback = hk.get_parameter(
            "feedback",
            [output_size, input_size],  # Transposed shape
            dtype,
            init=feedback_init,
        )

        # Initialize bias if needed
        if self.with_bias:
            b = hk.get_parameter(
                "b", 
                [output_size], 
                dtype, 
                init=self.b_init
            )
        else:
            b = None

        # Apply FA transformation
        return _fa_linear_impl(inputs, w, b, feedback)


def _fa_linear_impl(
    inputs: jax.Array,
    weights: jax.Array,
    bias: Optional[jax.Array],
    feedback: jax.Array,
) -> jax.Array:
    """Implements linear layer with feedback alignment gradient.
    
    Forward pass: standard linear transformation
    Backward pass: use feedback matrix instead of weight transpose
    
    Args:
        inputs: Input array, shape (batch, input_size).
        weights: Forward weight matrix, shape (input_size, output_size).
        bias: Bias vector, shape (output_size), or None.
        feedback: Feedback matrix, shape (output_size, input_size).
    
    Returns:
        Output array, shape (batch, output_size).
    """

    @jax.custom_vjp
    def fa_linear(x, w, b, fb):
        """Forward pass: standard linear transformation."""
        out = jnp.dot(x, w)
        if b is not None:
            out = out + b
        return out

    def fwd(x, w, b, fb):
        """Forward pass with residual saving."""
        out = jnp.dot(x, w)
        if b is not None:
            out = out + b
        return out, (x, w, b, fb)

    def bwd(residuals, output_grad):
        """Backward pass using feedback matrix.
        
        Standard backprop would compute:
            grad_x = output_grad @ w.T
        
        Feedback alignment computes:
            grad_x = output_grad @ feedback
        
        This replaces w.T with a fixed random matrix, solving the
        weight transport problem.
        
        Args:
            residuals: Saved values (x, w, b, fb) from forward pass.
            output_grad: Gradient from layer above, shape (batch, output_size).
        
        Returns:
            Tuple of gradients for (x, w, b, fb).
        """
        x, w, b, fb = residuals
        
        # Compute parameter gradients using standard backprop
        # These use the actual activations and errors
        grad_w = jnp.dot(x.T, output_grad)
        
        if b is not None:
            grad_b = jnp.sum(output_grad, axis=0)
        else:
            grad_b = None
        
        # KEY DIFFERENCE: Use feedback matrix instead of w.T
        # Standard: grad_x = output_grad @ w.T
        # FA: grad_x = output_grad @ feedback
        grad_x = jnp.dot(output_grad, fb)
        
        # Feedback matrix has zero gradient (it's fixed)
        grad_fb = jnp.zeros_like(fb)
        
        return (grad_x, grad_w, grad_b, grad_fb)

    fa_linear.defvjp(fwd, bwd)
    return fa_linear(inputs, weights, bias, feedback)


def fa_linear(
    output_size: int,
    with_bias: bool = True,
    w_init: Optional[hk.initializers.Initializer] = None,
    b_init: Optional[hk.initializers.Initializer] = None,
    feedback_init: Optional[hk.initializers.Initializer] = None,
    name: Optional[str] = None,
) -> Callable[[jax.Array], jax.Array]:
    """Functional interface for FA linear layer.
    
    Creates and applies an FALinear module. This is a convenience function
    for functional-style code.
    
    Args:
        output_size: Number of output features.
        with_bias: Whether to include a bias term.
        w_init: Optional weight initializer.
        b_init: Optional bias initializer.
        feedback_init: Optional feedback matrix initializer.
        name: Optional name for the module.
    
    Returns:
        A function that applies the FA linear transformation.
    
    Example:
        ```python
        def network(x):
            x = fa_linear(128, name="hidden1")(x)
            x = jax.nn.relu(x)
            return fa_linear(10, name="output")(x)
        ```
    """
    layer = FALinear(
        output_size=output_size,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        feedback_init=feedback_init,
        name=name,
    )
    return layer
