import sys

from .megablock_dmoe import MegaBlockdMoE
from .megablock_moe import MegaBlockMoE

try:
    from megablocks import ops  # noqa # pylint: disable=W0611
except (ModuleNotFoundError, ImportError):
    print(
        "MegaBlocks not found, please see "
        "https://github.com/stanford-futuredata/megablocks/. "
        "Note that MegaBlocks depends on mosaicml-turbo, which only "
        "supports python 3.10.",
        flush=True,
    )
    sys.exit()

try:
    import stk  # noqa # pylint: disable=W0611
except (ModuleNotFoundError, ImportError):
    print("STK not found: please see https://github.com/stanford-futuredata/stk", flush=True)
    sys.exit()

__all__ = ["MegaBlockMoE", "MegaBlockdMoE"]
