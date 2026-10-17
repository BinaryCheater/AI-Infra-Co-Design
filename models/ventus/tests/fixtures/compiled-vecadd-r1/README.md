# Compatible compiled FP32 artifacts

These ELF files were freshly compiled using the existing official Ventus LLVM
97df137e687669f9acf4bfd734779f52f2878614 build and explicit `rtl-r1` work-item
profile. The older official assembly instructions are unchanged; conservative
resource summaries are appended. `single-build.json` and `pairs-build.json`
record source/compiler/linker hashes and final resource metadata.

Original newer-CSR fixtures remain in `../compiled-vecadd/`. Numerical RTL evidence
and reproduction: [compiled workflow](../../../docs/evidence/compiled-functional-20261017/README.md).
Source licences are preserved alongside these generated artifacts.
