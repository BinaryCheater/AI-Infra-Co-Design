# Real compiled workgroup programs

`exchange.elf` and `gemm.elf` are fresh official-compiler outputs from the maintained
OpenCL sources in `examples/kernels/`, using the explicit R1 work-item profile.
`build.json` and `gemm-build.json` bind source/tool/ELF hashes and resource metadata.
The first program needs 16 private spill bytes/thread; GEMM needs 12. Both need
512 static LDS bytes plus their per-warp scalar stacks. Register usage is separate
from bank-aligned launch allocation.

Full RTL numerical evidence, native checks, retained failures and reproduction:
[synchronization report](../../../docs/evidence/compiled-sync-20261017/README.md).
Unknown/divergent synchronization remains unsupported; these artifacts are not
a general correctness or timing-accuracy certificate.
