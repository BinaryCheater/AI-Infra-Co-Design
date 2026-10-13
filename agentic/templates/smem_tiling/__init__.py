"""The shared-memory tiling template (paper Listing 1), vendored from the RQ1 prototype.

Source: MLSYS_homework_1/studies/progressive_verification/smem_tiling (2026-10), files
smem_template.py, smem_search.py, smem_verify.py, unchanged except for relative imports.
Original sha256:
  smem_template.py 4450cdccfe19306ff8d4fcbae50f20fa3119bc9d33679a4e9c961d8d44c56b7d
  smem_search.py   5bdc7cf7efb96b09b8abdfea59c92745b8f157f0d10907606056933d030a5a87
  smem_verify.py   611c113293b508284123c9097ef1ed152ada88eb0f866e298c026ecd96fa2edb
"""
from .smem_template import (BROKEN_VARIANTS, CORRECTNESS_FACE, K_TILE_MENU, PREFETCH_MENU, PRECONDITIONS,
                            STAGE_MENU, TILE_MENU, HardwareContract, TilingDesign, Workload, baseline_cost,
                            broken, ceil_div, cost_model, lifetime_precondition, search_face_violations)
from .smem_search import enumerate_menu, solve_milp
from .smem_verify import EventModel, Outcome, index_obligations, schedule_obligations

__all__ = [n for n in dir() if not n.startswith("_")]
