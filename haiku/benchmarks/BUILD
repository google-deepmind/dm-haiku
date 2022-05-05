# Haiku Benchmarks

load("//haiku/_src:build_defs.bzl", "hk_py_binary")

licenses(["notice"])

hk_py_binary(
    name = "init",
    srcs = ["init.py"],
    deps = [
        # pip: google_benchmark
        "//haiku",
        # pip: jax
    ],
)

hk_py_binary(
    name = "eval_shape",
    srcs = ["eval_shape.py"],
    deps = [
        # pip: google_benchmark
        "//haiku",
        # pip: jax
    ],
)
