# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ADCTM server package.

Keep template-only `openenv.core` server components optional so importing the
submission app does not fail in environments where that SDK is not installed.
"""

__all__ = []

try:
    from .ADCTMEnv_environment import AdctmenvEnvironment
except ModuleNotFoundError:
    AdctmenvEnvironment = None
else:
    __all__ = ["AdctmenvEnvironment"]
