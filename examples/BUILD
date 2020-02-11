package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

py_binary(
    name = "mnist",
    srcs = ["mnist.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//haiku",
        # pip: jax
        # pip: jax:optix
        # pip: numpy
        # pip: tensorflow_datasets
    ],
)

py_binary(
    name = "impala_lite",
    srcs = ["impala_lite.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        # pip: absl/logging
        # pip: bsuite/experiments/catch
        # pip: dm_env
        "//haiku",
        # pip: jax
        # pip: jax:optix
        # pip: numpy
        # pip: rlax
    ],
)
