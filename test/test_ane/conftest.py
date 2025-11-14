# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch


@pytest.fixture(scope="session")
def mps_available():
    """Check if MPS (Metal Performance Shaders) is available."""
    return torch.backends.mps.is_available()


@pytest.fixture(scope="session")
def mps_device():
    """Get MPS device if available."""
    if torch.backends.mps.is_available():
        return torch.device("mps:0")
    return None

