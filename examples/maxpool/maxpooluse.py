import haiku as hk
from haiku._src import pool   


def forward_fn(x):
    net = hk.Sequential([
        hk.Conv2D(output_channels=16, kernel_shape=3, stride=1, padding="SAME"),
        pool.MinPool(window_shape=2, strides=2, padding="VALID"),
        hk.Flatten(),
        hk.Linear(10)
    ])
    return net(x)
forward = hk.transform(forward_fn)
import jax
import jax.numpy as jnp

x = jnp.ones([1, 28, 28, 3])

rng = jax.random.PRNGKey(42)
params = forward.init(rng, x)

output = forward.apply(params, rng, x)
print(output.shape)
import jax.numpy as jnp
from haiku._src import pooling

x = jnp.array([[[[1.0], [2.0]],
                [[3.0], [4.0]]]])  # shape: (1, 2, 2, 1)

min_pool_layer = pooling.MinPool(window_shape=2, strides=1, padding="VALID")
init, apply = hk.transform(lambda x: min_pool_layer(x))

rng = jax.random.PRNGKey(0)
params = init(rng, x)
out = apply(params, rng, x)

print("Output:", out)
