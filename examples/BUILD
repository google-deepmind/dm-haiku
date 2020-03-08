load("//haiku/_src:build_defs.bzl", "hk_py_binary")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

hk_py_binary(
    name = "mnist",
    srcs = ["mnist.py"],
    deps = [
        # pip: absl:app
        "//haiku",
        # pip: jax
        # pip: jax:optix
        # pip: numpy
        # pip: tensorflow_datasets
    ],
)

hk_py_binary(
    name = "impala_lite",
    srcs = ["impala_lite.py"],
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
