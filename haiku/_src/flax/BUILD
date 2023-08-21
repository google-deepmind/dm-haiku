# Utilities for working with a mixture of Haiku and Flax.

load("//haiku/_src:build_defs.bzl", "hk_py_library", "hk_py_test")

package(
    default_applicable_licenses = ["//haiku:license"],
    default_visibility = ["//haiku:__subpackages__"],
)

licenses(["notice"])

hk_py_library(
    name = "flax_module",
    srcs = ["flax_module.py"],
    deps = [
        ":utils",
        # pip: flax:core
        "//haiku/_src:filtering",
        "//haiku/_src:transform",
        "//haiku/_src:typing",
    ],
)

hk_py_test(
    name = "flax_module_test",
    srcs = ["flax_module_test.py"],
    deps = [
        ":flax_module",
        ":utils",
        "//testing/pybase",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: flax:core
        "//haiku/_src:base",
        "//haiku/_src:filtering",
        "//haiku/_src:module",
        "//haiku/_src:transform",
        "//haiku/_src/nets:resnet",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = ["//haiku/_src:typing"],
)

hk_py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "//testing/pybase",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
    ],
)

hk_py_library(
    name = "transform_flax",
    srcs = ["transform_flax.py"],
    deps = [
        ":utils",
        # pip: flax:core
        "//haiku/_src:base",
        "//haiku/_src:filtering",
        "//haiku/_src:lift",
        "//haiku/_src:transform",
        "//haiku/_src:typing",
    ],
)

hk_py_test(
    name = "transform_flax_test",
    srcs = ["transform_flax_test.py"],
    deps = [
        ":transform_flax",
        "//testing/pybase",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: flax:core
        "//haiku/_src:base",
        "//haiku/_src:module",
        "//haiku/_src:transform",
        # pip: jax
    ],
)
