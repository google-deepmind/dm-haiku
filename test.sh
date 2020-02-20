#!/bin/bash
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Pip installs the relevant dependencies and runs the Haiku tests on CPU

set -e
set -x

virtualenv -p python3 .
source bin/activate
python --version

# Install JAX.
python -m pip install -r requirements-jax.txt
python -c 'import jax; print(jax.__version__)'

# Run setup.py, this installs the python dependencies
python -m pip install .

# Python test dependencies.
python -m pip install -r requirements-test.txt

# CPU count on macos or linux
if [ "$(uname)" == "Darwin" ]; then
  N_JOBS=$(sysctl -n hw.logicalcpu)
else
  N_JOBS=$(grep -c ^processor /proc/cpuinfo)
fi

# Run tests using pytest.
python -m pytest -n "${N_JOBS}" haiku

# Test docs still build.
cd docs/
pip install -r requirements.txt
# TODO(tomhennigan) Add `make doctest`.
make html
