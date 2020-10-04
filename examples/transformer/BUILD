load("//haiku/_src:build_defs.bzl", "hk_py_binary", "hk_py_library")

package(default_visibility = [":__subpackages__"])

licenses(["notice"])

hk_py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":dataset",
        ":model",
        # pip: absl:app
        # pip: absl/flags
        # pip: absl/logging
        "//haiku",
        # pip: jax
        # pip: numpy
        # pip: optax
        # pip: tensorflow
    ],
)

hk_py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        # pip: tensorflow
        # pip: tensorflow_datasets
    ],
)

hk_py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//haiku",
        # pip: jax
        # pip: numpy
    ],
)
