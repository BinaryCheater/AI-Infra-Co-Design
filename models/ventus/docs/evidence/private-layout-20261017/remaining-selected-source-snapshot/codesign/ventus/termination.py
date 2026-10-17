"""Preserve concrete ENDPRG instruction occupancy without inventing a block fence."""

import re
from collections import defaultdict
from dataclasses import replace

from .ir import Operation


def append_endprg(workload, words):
    """Append the actual final ENDPRG to each same-program warp stream.

    Existing body operations must cover every preceding word exactly once. This
    helper does not infer global visibility, wait for all warps, or assign latency.
    ENDPRG travels through scalar collection/issue into warp_schedule, which is
    distinct from the collective `barrier` IR node.
    """
    if not words or words[-1] != 0x400B:
        raise ValueError("expected concrete final ENDPRG word 0x0000400b")
    streams = defaultdict(list)
    for op in workload.operations:
        streams[op.block, op.warp].append(op)
    terminals = []
    for (block, warp), stream in sorted(streams.items()):
        if len(stream) != len(words) - 1 or not stream:
            raise ValueError("body must cover all words preceding ENDPRG")
        matches = [re.fullmatch(r"(.*?pc)([0-9a-fA-F]{8})", op.name) for op in stream]
        if any(match is None for match in matches):
            raise ValueError("ENDPRG append requires concrete PC names")
        base = int(matches[0][2], 16)
        if any(int(match[2], 16) != base + 4 * i for i, match in enumerate(matches)):
            raise ValueError("body PC stream is not contiguous")
        pc = base + 4 * len(stream)
        if pc >= 2**32:
            raise ValueError("ENDPRG PC exceeds 32-bit address space")
        terminals.append(Operation(f"{matches[-1][1]}{pc:08x}", "endprg", block, warp))
    return replace(workload, operations=workload.operations + tuple(terminals))
