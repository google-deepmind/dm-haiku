# Haiku Benchmarks

## Haiku initialization benchmarks
These benchmarks try to capture how long JAX tracing and JAX compilation take
on some sample models (currently an MLP and a ResNet).

Sample output for the CPU target:

```{.no-copy}
------------------------------------------------------------
Benchmark                  Time             CPU   Iterations
------------------------------------------------------------
trace_mlp           53383109 ns     53378968 ns           12
compile_mlp        206958966 ns    128557777 ns            5
run_mlp              1297425 ns      1296412 ns          484
trace_resnet_50   1977248429 ns   1976793581 ns            1
compile_resnet_50 7552480230 ns   5049206996 ns            1
run_resnet_50       51843911 ns     51837717 ns           13
```

Tracing (`trace_`) includes JAX preparing the function for XLA compilation,
which mainly consists of python-side bookkeeping and running the model once.
Compilation (`compile_`) includes XLA compiling the traced function.
Running (`run_`) includes a single compiled forward pass of the model.

## Running benchmarks

Install `google-benchmark` and run `init.py`:

```shell
pip install google-benchmark
python benchmarks/init.py
```

