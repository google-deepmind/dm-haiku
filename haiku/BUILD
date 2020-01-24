# Description: Haiku is a neural network library for JAX.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

# Currently Haiku is under active development, please send us a CL adding your
# project(s) here if you would like to use it (so we're aware and try to avoid
# breaking your usecases!).
# TODO(tomhennigan) Change to public once haiku is more stable.
package_group(
    name = "friends",
    includes = ["//learning/deepmind:experimental"],
    packages = [
        "//learning/brain/research/dune/...",
        "//learning/brain/research/sequin/...",
        "//learning/deepmind/atlas/...",
        "//learning/deepmind/dm_python/...",
        "//learning/deepmind/research/jax/...",
        "//platforms/xla/tests/...",
        "//research/simulation/whirl/...",
        "//haiku/...",
        "//waymo/experimental/...",
    ],
)

# This should be the only public build rule for Haiku, containing everything
# you need to build neural networks with JAX.
py_library(
    name = "haiku",
    srcs = [
        "__init__.py",
        "data_structures.py",
        "graph.py",
        "initializers.py",
        "nets.py",
        "pad.py",
        "testing.py",
        "typing.py",
    ],
    srcs_version = "PY3",
    visibility = [":friends"],
    deps = [
        "//haiku/_src:base",
        "//haiku/_src:basic",
        "//haiku/_src:batch_norm",
        "//haiku/_src:bias",
        "//haiku/_src:conv",
        "//haiku/_src:data_structures",
        "//haiku/_src:embed",
        "//haiku/_src:graph",
        "//haiku/_src:initializers",
        "//haiku/_src:layer_norm",
        "//haiku/_src:module",
        "//haiku/_src:moving_averages",
        "//haiku/_src:pad",
        "//haiku/_src:pool",
        "//haiku/_src:recurrent",
        "//haiku/_src:reshape",
        "//haiku/_src:spectral_norm",
        "//haiku/_src:stateful",
        "//haiku/_src:test_utils",
        "//haiku/_src:typing",
        "//haiku/_src/nets:mlp",
        "//haiku/_src/nets:resnet",
    ],
)
