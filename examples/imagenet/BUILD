load("//haiku/_src:build_defs.bzl", "hk_py_binary", "hk_py_library")

package(
    default_applicable_licenses = ["//haiku:license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

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
        # pip: jmp
        # pip: numpy
        # pip: optax
        # pip: tree
    ],
)

hk_py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        # pip: jax
        # pip: numpy
        # pip: packaging
        # pip: tensorflow
        # pip: tensorflow_datasets
    ],
)
