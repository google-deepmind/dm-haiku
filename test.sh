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

readonly VENV_DIR=/tmp/haiku-env

if [ -z "$(which pandoc)" ]; then
  echo 1>&2 "Requesting to install pandoc"
  sudo apt install -y pandoc
fi

# Install deps in a virtual env.
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
python -m pip install --upgrade pip setuptools
pip install \
  -r requirements.txt \
  -r requirements-jax.txt \
  -r requirements-flax.txt \
  -r requirements-test.txt
python -c 'import jax; print(jax.__version__)'

# Run setup.py to install Haiku itself.
python -m pip install .

# Run tests using pytest.
TEST_OPTS=()
if [[ "${INTEGRATION}" == "false" ]]; then
  TEST_OPTS+=("--ignore=haiku/_src/integration/")
else
  # Isolate the float64 test because it needs to set a flag at start-up, so no
  # other tests can be run before it.
  python -m pytest haiku/_src/integration/float64_test.py
  TEST_OPTS+=("--ignore=haiku/_src/integration/float64_test.py")
fi
python -m pytest -n auto haiku "${TEST_OPTS[@]}"

# Test docs still build.
cd docs/
pip install -r requirements.txt
make coverage_check
make doctest
make html
