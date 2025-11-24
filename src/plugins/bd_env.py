import torch
import os
import socket
from typing_extensions import override
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.plugins.environments.lightning import LightningEnvironment


class BDEnvironment(LightningEnvironment):
    pass
    # def __init__(self) -> None:
    #     super().__init__()
    #     self._global_rank: int = 0
    #     self._world_size: int = 1
    #
    # @property
    # @override
    # def creates_processes_externally(self) -> bool:
    #     """Returns whether the cluster creates the processes or not.
    #
    #     If at least :code:`LOCAL_RANK` is available as environment variable, Lightning assumes the user acts as the
    #     process launcher/job scheduler and Lightning will not launch new processes.
    #
    #     """
    #     return "LOCAL_RANK" in os.environ
    #
    # @staticmethod
    # @override
    # def detect() -> bool:
    #     assert "ARNOLD_WORKER_0_HOST" in os.environ.keys()
    #     assert "ARNOLD_WORKER_0_PORT" in os.environ.keys()
    #     return True
    #
    # @override
    # def world_size(self) -> int:
    #     return self._world_size
    #
    # @override
    # def set_world_size(self, size: int) -> None:
    #     self._world_size = size
    #
    # @override
    # def global_rank(self) -> int:
    #     return self._global_rank
    #
    # @override
    # def set_global_rank(self, rank: int) -> None:
    #     self._global_rank = rank
    #     rank_zero_only.rank = rank
    #
    # @override
    # def local_rank(self) -> int:
    #     return int(os.environ.get("LOCAL_RANK", 0))
    #
    # @override
    # def node_rank(self) -> int:
    #     return int(os.environ.get("ARNOLD_ID"))
    #
    # @override
    # def teardown(self) -> None:
    #     if "WORLD_SIZE" in os.environ:
    #         del os.environ["WORLD_SIZE"]
    #
    # @property
    # def main_address(self) -> str:
    #     return os.environ.get("ARNOLD_WORKER_0_HOST")
    #
    # @property
    # def main_port(self) -> int:
    #     return int(os.environ.get("ARNOLD_WORKER_0_PORT"))
