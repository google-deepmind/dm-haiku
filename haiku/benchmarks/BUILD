# Haiku Benchmarks

load("//haiku/_src:build_defs.bzl", "hk_py_binary")

licenses(["notice"])

hk_py_binary(
    name = "init",
    srcs = ["init.py"],
    deps = [
        "//third_party/benchmark/bindings/python/google_benchmark",
        "//haiku",
        # pip: jax
    ],
)
