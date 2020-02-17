package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

py_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset",
        # link: tpu_device
        # link: gpu_support
        # link: tpu_support
        # pip: absl:app
        # pip: absl/logging
        "//haiku",
        # pip: jax
        # pip: jax:optix
        # pip: numpy
        # pip: tqdm
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    srcs_version = "PY3",
    deps = [
        # pip: numpy
        # pip: tensorflow
        # pip: tensorflow_datasets
    ],
)
