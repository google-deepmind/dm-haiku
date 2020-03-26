# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Common networks built using Haiku modules."""

from haiku._src.nets.mlp import MLP
from haiku._src.nets.mobilenetv1 import MobileNetV1
from haiku._src.nets.resnet import ResNet
from haiku._src.nets.resnet import ResNet101
from haiku._src.nets.resnet import ResNet152
from haiku._src.nets.resnet import ResNet200
from haiku._src.nets.resnet import ResNet50

__all__ = (
    "ResNet",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNet200",
    "MLP",
    "MobileNetV1"
)
