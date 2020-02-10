package(default_visibility = [":__subpackages__"])

licenses(["notice"])

exports_files(["LICENSE"])

py_binary(
    name = "run_catch",
    srcs = ["run_catch.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":actor",
        ":agent",
        ":haiku_nets",
        ":learner",
        ":util",
        # pip: absl:app
        # pip: bsuite/experiments/catch
        # pip: jax
        # pip: jax:optix
    ],
)

py_library(
    name = "actor",
    srcs = ["actor.py"],
    srcs_version = "PY3",
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

py_test(
    name = "actor_test",
    srcs = ["actor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":actor",
        ":agent",
        ":haiku_nets",
        ":learner",
        ":util",
        # pip: absl/testing:absltest
        # pip: bsuite/experiments/catch
        # pip: dm_env
        # pip: jax
        # pip: mock
        # pip: numpy
        # pip: tree
    ],
)

py_library(
    name = "agent",
    srcs = ["agent.py"],
    srcs_version = "PY3",
    deps = [
        ":util",
        # pip: dm_env
        "//haiku",
        # pip: jax
    ],
)

py_library(
    name = "haiku_nets",
    srcs = ["haiku_nets.py"],
    srcs_version = "PY3",
    deps = [
        # pip: dm_env
        "//haiku",
        # pip: jax
        # pip: tree
    ],
)

py_library(
    name = "learner",
    srcs = ["learner.py"],
    srcs_version = "PY3",
    deps = [
        ":agent",
        ":util",
        # pip: dm_env
        "//haiku",
        # pip: jax
        # pip: jax:optimizers
        # pip: jax:optix
        # pip: numpy
        # pip: rlax
        # pip: six
    ],
)

py_test(
    name = "learner_test",
    srcs = ["learner_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":actor",
        ":agent",
        ":haiku_nets",
        ":learner",
        # pip: absl/testing:absltest
        # pip: bsuite/experiments/catch
        # pip: jax
        # pip: jax:optix
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
    deps = [
        # pip: absl/logging
        # pip: dm_env
        # pip: numpy
        # pip: tree
    ],
)
