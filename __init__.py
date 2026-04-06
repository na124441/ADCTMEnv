# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ADCTM package exports.

The OpenEnv starter template shipped package-level imports that depend on the
`openenv.core` client runtime. That runtime is not needed for the hackathon
submission server, so we keep these imports lazy to avoid breaking local
validation, pytest collection, or `uvicorn server.app:app`.
"""

__all__ = []

try:
    from .client import AdctmenvEnv
    from utils.models import AdctmenvAction, AdctmenvObservation
except ModuleNotFoundError:
    AdctmenvEnv = None
    AdctmenvAction = None
    AdctmenvObservation = None
else:
    __all__ = [
        "AdctmenvAction",
        "AdctmenvObservation",
        "AdctmenvEnv",
    ]
