load("//haiku/_src:build_defs.bzl", "hk_py_library", "hk_py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

hk_py_library(
    name = "rnn",
    srcs = ["rnn.py"],
    deps = [
        "//haiku",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":rnn",
        "//haiku",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "rnn_test",
    srcs = ["rnn_test.py"],
    deps = [
        ":rnn",
        "//haiku",
        # pip: absl/testing
        # pip: jax
        # pip: numpy
    ],
) 