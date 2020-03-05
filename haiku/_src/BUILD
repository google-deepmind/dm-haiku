load("//haiku/_src:build_defs.bzl", "hk_py_library", "hk_py_test")

package(default_visibility = ["//haiku:__subpackages__"])

licenses(["notice"])

exports_files(["LICENSE"])

hk_py_library(
    name = "analytics",
    srcs = ["analytics.py"],
)

hk_py_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        ":analytics",
        ":data_structures",
        ":typing",
        # pip: jax
    ],
)

hk_py_test(
    name = "base_test",
    srcs = ["base_test.py"],
    deps = [
        ":base",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
    ],
)

hk_py_library(
    name = "bias",
    srcs = ["bias.py"],
    deps = [
        ":base",
        ":module",
        ":utils",
        # pip: jax
    ],
)

hk_py_test(
    name = "bias_test",
    srcs = ["bias_test.py"],
    deps = [
        ":base",
        ":bias",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "module",
    srcs = ["module.py"],
    deps = [
        ":base",
        ":utils",
        # pip: jax
    ],
)

hk_py_test(
    name = "module_test",
    srcs = ["module_test.py"],
    deps = [
        ":base",
        ":module",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
    ],
)

hk_py_library(
    name = "basic",
    srcs = ["basic.py"],
    deps = [
        ":base",
        ":initializers",
        ":module",
        # pip: jax
        # pip: tree
    ],
)

hk_py_test(
    name = "basic_test",
    srcs = ["basic_test.py"],
    deps = [
        ":base",
        ":basic",
        ":initializers",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "batch_norm",
    srcs = ["batch_norm.py"],
    deps = [
        ":base",
        ":module",
        ":moving_averages",
        ":utils",
        # pip: jax
    ],
)

hk_py_test(
    name = "batch_norm_test",
    srcs = ["batch_norm_test.py"],
    deps = [
        ":batch_norm",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "depthwise_conv",
    srcs = ["depthwise_conv.py"],
    deps = [
        ":base",
        ":initializers",
        ":module",
        ":utils",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "depthwise_conv_test",
    srcs = ["depthwise_conv_test.py"],
    deps = [
        ":base",
        ":depthwise_conv",
        ":initializers",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "conv",
    srcs = ["conv.py"],
    deps = [
        ":base",
        ":initializers",
        ":module",
        ":pad",
        ":utils",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "conv_test",
    srcs = ["conv_test.py"],
    deps = [
        ":base",
        ":conv",
        ":initializers",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "lift",
    srcs = ["lift.py"],
    deps = [
        ":base",
        ":data_structures",
        ":module",
    ],
)

hk_py_test(
    name = "lift_test",
    srcs = ["lift_test.py"],
    deps = [
        ":base",
        ":lift",
        ":module",
        # pip: absl/testing:absltest
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "pad",
    srcs = ["pad.py"],
    deps = [
        ":typing",
        ":utils",
    ],
)

hk_py_test(
    name = "pad_test",
    srcs = ["pad_test.py"],
    tpu = False,  # TODO(tamaranorman) fix build failure
    deps = [
        ":pad",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
    ],
)

hk_py_library(
    name = "pool",
    srcs = ["pool.py"],
    deps = [
        ":module",
        # pip: jax
    ],
)

hk_py_test(
    name = "pool_test",
    srcs = ["pool_test.py"],
    deps = [
        ":pool",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "data_structures",
    srcs = ["data_structures.py"],
    deps = [
        ":utils",
        # pip: jax
    ],
)

hk_py_test(
    name = "data_structures_test",
    srcs = ["data_structures_test.py"],
    deps = [
        ":data_structures",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: cloudpickle
        # pip: dill
        # pip: jax
        # pip: tree
    ],
)

hk_py_library(
    name = "utils",
    srcs = ["utils.py"],
)

hk_py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    tpu = False,  # TODO(tomhennigan) Debug TPU build error.
    deps = [
        ":utils",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
    ],
)

hk_py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    deps = [
        ":base",
        ":typing",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "initializers_test",
    srcs = ["initializers_test.py"],
    deps = [
        ":initializers",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "layer_norm_test",
    srcs = ["layer_norm_test.py"],
    deps = [
        ":initializers",
        ":layer_norm",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "layer_norm",
    srcs = ["layer_norm.py"],
    deps = [
        ":base",
        ":module",
        ":utils",
        # pip: jax
    ],
)

hk_py_library(
    name = "moving_averages",
    srcs = ["moving_averages.py"],
    deps = [
        ":base",
        ":data_structures",
        ":initializers",
        ":module",
        # pip: jax
    ],
)

hk_py_test(
    name = "moving_averages_test",
    srcs = ["moving_averages_test.py"],
    deps = [
        ":base",
        ":basic",
        ":moving_averages",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: jax
        # pip: numpy
        # pip: tree
    ],
)

hk_py_library(
    name = "recurrent",
    srcs = ["recurrent.py"],
    deps = [
        ":base",
        ":basic",
        ":conv",
        ":initializers",
        ":module",
        # pip: jax
    ],
)

hk_py_test(
    name = "recurrent_test",
    timeout = "long",
    srcs = ["recurrent_test.py"],
    deps = [
        ":base",
        ":basic",
        ":recurrent",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
        # pip: numpy
        # pip: tree
    ],
)

hk_py_test(
    name = "spectral_norm_test",
    srcs = ["spectral_norm_test.py"],
    deps = [
        ":base",
        ":basic",
        ":spectral_norm",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "embed_test",
    srcs = ["embed_test.py"],
    deps = [
        ":embed",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        ":base",
        # pip: absl/testing:parameterized
        # pip: jax
    ],
)

hk_py_library(
    name = "spectral_norm",
    srcs = ["spectral_norm.py"],
    deps = [
        ":base",
        ":data_structures",
        ":initializers",
        ":module",
        # pip: jax
    ],
)

hk_py_library(
    name = "embed",
    srcs = ["embed.py"],
    deps = [
        ":base",
        ":basic",
        ":initializers",
        ":module",
        # pip: jax
    ],
)

hk_py_library(
    name = "reshape",
    srcs = ["reshape.py"],
    deps = [
        ":module",
        ":typing",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "reshape_test",
    srcs = ["reshape_test.py"],
    deps = [
        ":base",
        ":reshape",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "stateful",
    srcs = ["stateful.py"],
    deps = [
        ":base",
        # pip: jax
    ],
)

hk_py_test(
    name = "stateful_test",
    srcs = ["stateful_test.py"],
    deps = [
        ":base",
        ":module",
        ":stateful",
        ":test_utils",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "typing",
    srcs = ["typing.py"],
    deps = [
        # pip: jax
    ],
)
