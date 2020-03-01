
.. _transforms-caveats:

Limitations of using JAX transforms inside of networks
======================================================

Once a Haiku network has been transformed to a pure function using
:func:`hk.transform() <haiku.transform>`, it's possible to freely combine it with
any JAX transformations like :func:`jax.jit`, :func:`jax.grad`, and so on.
It's also possible to use JAX transformations inside of a Haiku network, but
there can be issues with JAX's tracing mechanism since Haiku functions can have
side effects (through :func:`hk.next_rng_key() <haiku.next_rng_key>`,
:func:`hk.get_parameter() <haiku.get_parameter>` and other stateful Haiku
calls).
To work around this, Haiku provides wrapped versions of JAX transforms under
the :mod:`haiku` namespace. You can access these as :func:`hk.jit() <haiku.jit>`,
:func:`hk.grad() <haiku.grad>`, and so on.

These wrappers turn the underlying Haiku function into a pure function, apply
the corresponding JAX transformation and re-package the result as a stateful
Haiku function.
The wrapped transforms are currently considered experimental.
