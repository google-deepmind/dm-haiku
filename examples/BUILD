load("//haiku/_src:build_defs.bzl", "hk_py_binary", "hk_py_test")

package(
    default_applicable_licenses = ["//haiku:license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

hk_py_binary(
    name = "mnist",
    srcs = ["mnist.py"],
    deps = [
        # pip: absl:app
        "//haiku",
        # pip: jax
        # pip: numpy
        # pip: optax
        # pip: tensorflow_datasets
    ],
)

hk_py_binary(
    name = "vae",
    srcs = ["vae.py"],
    deps = [
        # pip: absl:app
        # pip: absl/flags
        # pip: absl/logging
        "//haiku",
        # pip: jax
        # pip: numpy
        # pip: optax
        # pip: tensorflow_datasets
    ],
)

hk_py_binary(
    name = "mnist_pruning",
    srcs = ["mnist_pruning.py"],
    deps = [
        # pip: absl:app
        "//haiku",
        # pip: jax
        # pip: numpy
        # pip: optax
        # pip: tensorflow_datasets
    ],
)

hk_py_binary(
    name = "impala_lite",
    srcs = ["impala_lite.py"],
    test_lib = True,
    deps = [
        # pip: absl:app
        # pip: absl/logging
        # pip: bsuite/environments:catch
        # pip: dm_env
        "//haiku",
        # pip: jax
        # pip: numpy
        # pip: optax
        # pip: rlax
    ],
)

hk_py_test(
    name = "impala_lite_test",
    srcs = ["impala_lite_test.py"],
    deps = [
        ":impala_lite.testonly_lib",
        # pip: absl/testing:absltest
    ],
)
