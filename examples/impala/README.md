# JAX IMPALA

This is a re-implementation of the canonical IMPALA in JAX.
See: https://arxiv.org/abs/1802.01561

It is provided as an example of a nontrivial Haiku codebase for reinforcement
learning.

## Running the code.

We provide a minimal example of how to wire the learners and actors together in
a single-process single-GPU Catch example; this can be forked and modified for
other environments.

If the learner is wrapped in an RPC server, the same Learner and Actor classes
may be used to run a distributed IMPALA.
