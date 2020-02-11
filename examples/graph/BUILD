package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

py_binary(
    name = "sat",
    srcs = ["sat.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//haiku",
        # pip: jax
        # pip: jax:optix
        # pip: numpy
    ],
)

py_binary(
    name = "zacharys_karate_club",
    srcs = ["zacharys_karate_club.py"],
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
    name = "game_of_life",
    srcs = ["game_of_life.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//haiku",
        # pip: jax
        # pip: numpy
    ],
)
