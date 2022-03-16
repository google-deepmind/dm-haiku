# Description: Haiku is a neural network library for JAX.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

# This should be the only public build rule for Haiku, containing everything
# you need to build neural networks with JAX.
py_library(
    name = "haiku",
    srcs = [
        "__init__.py",
        "config.py",
        "data_structures.py",
        "experimental.py",
        "initializers.py",
        "mixed_precision.py",
        "nets.py",
        "pad.py",
        "testing.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//haiku/_src:attention",
        "//haiku/_src:base",
        "//haiku/_src:basic",
        "//haiku/_src:batch_norm",
        "//haiku/_src:bias",
        "//haiku/_src:config",
        "//haiku/_src:conv",
        "//haiku/_src:data_structures",
        "//haiku/_src:deferred",
        "//haiku/_src:depthwise_conv",
        "//haiku/_src:dot",
        "//haiku/_src:embed",
        "//haiku/_src:filtering",
        "//haiku/_src:group_norm",
        "//haiku/_src:initializers",
        "//haiku/_src:layer_norm",
        "//haiku/_src:layer_stack",
        "//haiku/_src:lift",
        "//haiku/_src:mixed_precision",
        "//haiku/_src:module",
        "//haiku/_src:moving_averages",
        "//haiku/_src:multi_transform",
        "//haiku/_src:pad",
        "//haiku/_src:pool",
        "//haiku/_src:random",
        "//haiku/_src:recurrent",
        "//haiku/_src:reshape",
        "//haiku/_src:rms_norm",
        "//haiku/_src:spectral_norm",
        "//haiku/_src:stateful",
        "//haiku/_src:summarise",
        "//haiku/_src:test_utils",
        "//haiku/_src:transform",
        "//haiku/_src:typing",
        "//haiku/_src:utils",
        "//haiku/_src/nets:mlp",
        "//haiku/_src/nets:mobilenetv1",
        "//haiku/_src/nets:resnet",
        "//haiku/_src/nets:vqvae",
    ],
)
