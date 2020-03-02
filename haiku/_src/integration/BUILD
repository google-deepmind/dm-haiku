load("//haiku/_src:build_defs.bzl", "hk_py_library", "hk_py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

hk_py_library(
    name = "descriptors",
    srcs = ["descriptors.py"],
    deps = [
        "//haiku",
        "//haiku/_src:typing",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "descriptors_test",
    srcs = ["descriptors_test.py"],
    deps = [
        ":descriptors",
        # pip: absl/testing:absltest
        "//haiku",
        "//haiku/_src:test_utils",
    ],
)

hk_py_test(
    name = "jax_transforms_test",
    timeout = "long",
    srcs = ["jax_transforms_test.py"],
    shard_count = 50,
    deps = [
        ":descriptors",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku",
        "//haiku/_src:test_utils",
        "//haiku/_src:typing",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "hk_transforms_test",
    timeout = "long",
    srcs = ["hk_transforms_test.py"],
    shard_count = 50,
    deps = [
        ":descriptors",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku",
        "//haiku/_src:test_utils",
        "//haiku/_src:typing",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "doctest_test",
    srcs = ["doctest_test.py"],
    deps = [
        # pip: absl/logging
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku",
        "//haiku/_src:test_utils",
        # pip: jax
    ],
)

hk_py_test(
    name = "bfloat16_test",
    srcs = ["bfloat16_test.py"],
    gpu = False,
    shard_count = 50,
    deps = [
        ":descriptors",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku",
        "//haiku/_src:test_utils",
        "//haiku/_src:typing",
        # pip: jax
    ],
)

hk_py_test(
    name = "float64_test",
    srcs = ["float64_test.py"],
    gpu = False,
    tpu = False,
    shard_count = 50,
    deps = [
        ":bfloat16_test",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
    ],
)
