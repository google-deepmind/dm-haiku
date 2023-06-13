load("//haiku/_src:build_defs.bzl", "hk_py_binary", "hk_py_library", "hk_py_test")

package(
    default_applicable_licenses = ["//haiku:license"],
    default_visibility = [":__subpackages__"],
)

licenses(["notice"])

hk_py_binary(
    name = "run_catch",
    srcs = ["run_catch.py"],
    deps = [
        ":actor",
        ":agent",
        ":haiku_nets",
        ":learner",
        ":util",
        # pip: absl:app
        # pip: bsuite/environments:catch
        # pip: jax
        # pip: optax
    ],
)

hk_py_library(
    name = "actor",
    srcs = ["actor.py"],
    deps = [
        ":agent",
        ":learner",
        ":util",
        # pip: dm_env
        "//haiku",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "actor_test",
    srcs = ["actor_test.py"],
    deps = [
        ":actor",
        ":agent",
        ":haiku_nets",
        ":learner",
        ":util",
        # pip: absl/testing:absltest
        # pip: bsuite/environments:catch
        # pip: dm_env
        # pip: jax
        # pip: mock
        # pip: numpy
        # pip: tree
    ],
)

hk_py_library(
    name = "agent",
    srcs = ["agent.py"],
    deps = [
        ":util",
        # pip: dm_env
        "//haiku",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "haiku_nets",
    srcs = ["haiku_nets.py"],
    deps = [
        # pip: dm_env
        "//haiku",
        # pip: jax
    ],
)

hk_py_library(
    name = "learner",
    srcs = ["learner.py"],
    deps = [
        ":agent",
        ":util",
        # pip: dm_env
        "//haiku",
        # pip: jax
        # pip: jax:optimizers
        # pip: numpy
        # pip: optax
        # pip: rlax
    ],
)

hk_py_test(
    name = "learner_test",
    srcs = ["learner_test.py"],
    deps = [
        ":actor",
        ":agent",
        ":haiku_nets",
        ":learner",
        # pip: absl/testing:absltest
        # pip: bsuite/environments:catch
        # pip: jax
        # pip: optax
    ],
)

hk_py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        # pip: absl/logging
        # pip: dm_env
        # pip: numpy
        # pip: tree
    ],
)
