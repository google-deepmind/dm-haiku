package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

py_binary(
    name = "mnist",
    srcs = ["mnist.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//haiku",
        # pip: jax
        # pip: jax:optix
        # pip: numpy
        # pip: tensorflow_datasets
        # pip: tree
    ],
)

py_binary(
    name = "sat_using_graphnet",
    srcs = ["sat_using_graphnet.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//haiku",
        # pip: jax
        # pip: jax:optix
    ],
)

py_binary(
    name = "zacharys_karate_club_using_graphnet",
    srcs = ["zacharys_karate_club_using_graphnet.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//haiku",
        # pip: jax
        # pip: jax:optix
    ],
)

py_binary(
    name = "game_of_life_using_graphnets",
    srcs = ["game_of_life_using_graphnets.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//haiku",
        # pip: jax
        # pip: numpy
    ],
)

py_binary(
    name = "impala_lite",
    srcs = ["impala_lite.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        # pip: absl/logging
        # pip: bsuite/experiments/catch
        # pip: dm_env
        "//haiku",
        # pip: jax
        # pip: jax:optix
        # pip: numpy
        # pip: rlax
        # pip: tree
    ],
)
