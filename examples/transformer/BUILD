package(default_visibility = [":__subpackages__"])

licenses(["notice"])

exports_files(["LICENSE"])

py_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset",
        ":model",
        # link: gpu_support
        # pip: absl:app
        # pip: absl/flags
        # pip: absl/logging
        "//haiku",
        # pip: jax
        # pip: jax:optix
        # pip: tensorflow
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    srcs_version = "PY3",
    deps = [
        # pip: tensorflow
        # pip: tensorflow_datasets
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
    deps = [
        "//haiku",
        # pip: jax
        # pip: numpy
    ],
)
