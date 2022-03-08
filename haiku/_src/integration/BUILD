load("//haiku/_src:build_defs.bzl", "hk_py_binary", "hk_py_library", "hk_py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

hk_py_library(
    name = "descriptors",
    srcs = ["descriptors.py"],
    deps = [
        "//haiku",
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
    name = "check_tracer_leaks_test",
    timeout = "long",
    srcs = ["check_tracer_leaks_test.py"],
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
        "//haiku/_src:stateful",
        "//haiku/_src:test_utils",
        "//haiku/_src:typing",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "doctest_test",
    timeout = "long",
    srcs = ["doctest_test.py"],
    cpu = False,  # Takes 10x longer than GPU.
    tpu = False,  # Sometimes generates different numerical results
    deps = [
        # pip: absl/logging
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: chex
        "//haiku",
        "//haiku/_src:test_utils",
        # pip: jax
        # pip: jmp
    ],
)

hk_py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
        ":descriptors",
        # pip: absl/testing:parameterized
        "//haiku",
        # pip: jax
        # pip: numpy
        # pip: tree
    ],
)

hk_py_test(
    name = "half_test",
    timeout = "long",
    srcs = ["half_test.py"],
    gpu = False,
    shard_count = 50,
    tpu = False,
    deps = [
        ":common",
        ":descriptors",
        # pip: absl/testing:absltest
        "//haiku/_src:test_utils",
        "//haiku/_src:typing",
        # pip: jax
    ],
)

hk_py_binary(
    name = "checkpoint_generate",
    srcs = ["checkpoint_generate.py"],
    deps = [
        ":checkpoint_utils",
        ":descriptors",
        # pip: absl:app
        # pip: absl/flags
        "//haiku/_src:utils",
    ],
)

hk_py_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
    deps = [
        ":descriptors",
        "//haiku",
        "//haiku/_src:utils",
        # pip: jax
    ],
)

hk_py_test(
    name = "checkpoint_test",
    srcs = ["checkpoint_test.py"],
    data = glob(["checkpoints/**"]),
    gpu = False,
    tpu = False,
    deps = [
        ":checkpoint_utils",
        ":common",
        ":descriptors",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku/_src:test_utils",
        "//haiku/_src:typing",
        "//haiku/_src:utils",
        # pip: jax
    ],
)

hk_py_test(
    name = "float64_test",
    srcs = ["float64_test.py"],
    gpu = False,
    shard_count = 50,
    tpu = False,
    deps = [
        ":common",
        ":descriptors",
        # pip: absl/testing:absltest
        "//haiku/_src:test_utils",
        "//haiku/_src:typing",
        # pip: jax
    ],
)

hk_py_test(
    name = "numpy_inputs_test",
    srcs = ["numpy_inputs_test.py"],
    shard_count = 50,
    deps = [
        ":descriptors",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku",
        "//haiku/_src:test_utils",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "rank_promotion_test",
    srcs = ["rank_promotion_test.py"],
    gpu = False,
    shard_count = 50,
    tpu = False,
    deps = [
        ":descriptors",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku",
        "//haiku/_src:test_utils",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "to_dot_test",
    srcs = ["to_dot_test.py"],
    gpu = False,
    shard_count = 50,
    tpu = False,
    deps = [
        ":descriptors",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku",
        "//haiku/_src:test_utils",
        # pip: jax
        # pip: numpy
    ],
)

hk_py_test(
    name = "summarise_test",
    srcs = ["summarise_test.py"],
    gpu = False,
    shard_count = 50,
    tpu = False,
    deps = [
        ":descriptors",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku",
        "//haiku/_src:test_utils",
        # pip: jax
    ],
)

hk_py_test(
    name = "shim_hk_test",
    srcs = ["shim_hk_test.py"],
    gpu = False,
    tpu = False,
    deps = [
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku",
    ],
)

hk_py_test(
    name = "jax2tf_test",
    timeout = "long",
    srcs = ["jax2tf_test.py"],
    shard_count = 50,
    deps = [
        ":descriptors",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//haiku",
        "//haiku/_src:test_utils",
        # pip: jax
        # pip: jax/experimental/jax2tf
        # pip: numpy
        # pip: tensorflow
    ],
)
