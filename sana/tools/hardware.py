# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

"""GPU hardware detection helpers (shared by the realtime inference entrypoints)."""

from __future__ import annotations


def is_blackwell() -> bool:
    """Return ``True`` on Blackwell-class GPUs.

    Blackwell compute capabilities: ``sm_100`` (B200 / GB200) and ``sm_120``
    (RTX 5090 / GB10). Hopper (H100) is ``sm_90`` and Ada (L40S / 4090) is
    ``sm_89`` — both return ``False``. NVFP4 and fp8-blockscale kernels require
    Blackwell, so this gates the realtime NVFP4 preset; on everything else the
    pipeline falls back to the bf16 path.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return False
    return major >= 10
