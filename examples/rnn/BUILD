load("//haiku/_src:build_defs.bzl", "hk_py_binary", "hk_py_library")

licenses(["notice"])

exports_files(["LICENSE"])

hk_py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        # pip: numpy
        # pip: tensorflow
        # pip: tensorflow_datasets
    ],
)

hk_py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":dataset",
        # pip: absl:app
        # pip: absl/flags
        # pip: absl/logging
        "//haiku",
        # pip: jax
        # pip: jax:optix
        # pip: numpy
    ],
)
