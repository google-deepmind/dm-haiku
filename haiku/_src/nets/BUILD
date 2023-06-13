load("//haiku/_src:build_defs.bzl", "hk_py_library", "hk_py_test")

package(
    default_applicable_licenses = ["//haiku:license"],
    default_visibility = ["//haiku:__subpackages__"],
)

licenses(["notice"])

hk_py_library(
    name = "resnet",
    srcs = ["resnet.py"],
    deps = [
        "//haiku/_src:basic",
        "//haiku/_src:batch_norm",
        "//haiku/_src:conv",
        "//haiku/_src:module",
        "//haiku/_src:pool",
        # pip: jax
    ],
)

hk_py_test(
    name = "resnet_test",
    timeout = "long",
    srcs = ["resnet_test.py"],
    deps = [
        ":resnet",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku",
        "//haiku/_src:test_utils",
        "//haiku/_src:transform",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "mlp",
    srcs = ["mlp.py"],
    deps = [
        "//haiku/_src:base",
        "//haiku/_src:basic",
        "//haiku/_src:initializers",
        "//haiku/_src:module",
        # pip: jax
    ],
)

hk_py_test(
    name = "mlp_test",
    timeout = "long",
    srcs = ["mlp_test.py"],
    deps = [
        ":mlp",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku/_src:test_utils",
        # pip: jax
    ],
)

hk_py_library(
    name = "mobilenetv1",
    srcs = ["mobilenetv1.py"],
    deps = [
        "//haiku/_src:basic",
        "//haiku/_src:batch_norm",
        "//haiku/_src:conv",
        "//haiku/_src:depthwise_conv",
        "//haiku/_src:module",
        "//haiku/_src:pool",
        "//haiku/_src:reshape",
        # pip: jax
    ],
)

hk_py_test(
    name = "mobilenetv1_test",
    timeout = "long",
    srcs = ["mobilenetv1_test.py"],
    deps = [
        ":mobilenetv1",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku/_src:test_utils",
        # pip: jax
    ],
)

hk_py_library(
    name = "vqvae",
    srcs = ["vqvae.py"],
    deps = [
        "//haiku/_src:base",
        "//haiku/_src:initializers",
        "//haiku/_src:module",
        "//haiku/_src:moving_averages",
        "//haiku/_src:typing",
        # pip: jax
    ],
)

hk_py_test(
    name = "vqvae_test",
    timeout = "long",
    srcs = ["vqvae_test.py"],
    deps = [
        ":vqvae",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku/_src:stateful",
        "//haiku/_src:test_utils",
        "//haiku/_src:transform",
        # pip: jax
        # pip: numpy
    ],
)
