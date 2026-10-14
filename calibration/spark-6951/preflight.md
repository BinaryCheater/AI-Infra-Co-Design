# Pre-flight: spark-6951 (DGX Spark, GB10)

Recorded by Claude Code over SSH on 2026-10-14 (UTC), per `notes/hardware_calibration_design.md` §1.
The host is reached through an alias in the Mac's SSH config; no address or key is recorded here.

## Summary

| Item | Result |
|---|---|
| 1. Driver / GPU / clocks | driver 580.159.03, CUDA 13.0 runtime; **NVIDIA GB10**, 1 GPU; idle P8 at 208 MHz SM clock; applications clock 2418 MHz; max 3003 MHz; memory clock not reported by `nvidia-smi` (unified LPDDR5X). **Clocks cannot be locked**: `nvidia-smi -lgc` is refused for this user and `sudo` needs a password, so the suite must report clock variance (B11) and F1's intervals widen to 4 % as §9.1 anticipates. |
| 2. Toolchain | `nvcc` 13.0 (V13.0.88) at `/usr/local/cuda` (= cuda-13.0); compute capability **12.1 → `sm_121`**, confirmed by `cudaGetDeviceProperties` and by the cubin arch in the built binary. `sm_120` also compiles and runs. Note: CUDA 13 removed `cudaDeviceProp::clockRate`; use `cudaDeviceGetAttribute`. |
| 3. PyTorch | not in the system Python 3.12.3; **present in `~/.venv`**: torch 2.11.0+cu128, `torch.cuda.is_available()` = True, device "NVIDIA GB10", capability (12, 1); numpy 2.4.4, scipy 1.17.1 in the same venv. The suite's driver uses `~/.venv/bin/python`. |
| 4. CUTLASS / cuBLAS | CUTLASS: not under `/usr/local/cuda/include`; a clone exists in another user's tree and is not relied on (F3 uses cuBLAS for the reference, not CUTLASS). cuBLAS: `/usr/local/cuda/lib64/libcublas.so.13`, headers present, version 13.1.1. |
| 5. Declared device properties | see table below; these are the declared values, the suite measures the effective ones |
| 6. Disk, permissions, unattended runs | root disk 3.7 TB, 20 % used; home `/home/lishang`; `rsync`, `git`, `cmake`, `make`, `gcc/g++`, `tmux`, `nohup` present; no CPU-time ulimit; no other GPU processes running at pre-flight; the suite will live under `~/work/research/AI-Infra-Co-Design-calib/` (rsync target created). |

Hello-world kernel: compiled with `-arch=sm_121`, ran, returned the expected values ("no error", `h[5] == 10`).

## Declared device properties (`cudaGetDeviceProperties`, CUDA 13.0)

| Property | Value |
|---|---|
| name | NVIDIA GB10 |
| compute capability | 12.1 |
| SMs | 48 |
| warp size | 32 |
| shared memory per block (default / opt-in) | 49 152 / 101 376 bytes |
| shared memory per SM | 102 400 bytes |
| registers per SM / per block | 65 536 / 65 536 |
| L2 cache | 25 165 824 bytes (24 MiB) |
| max threads per block | 1024 |
| max blocks per SM | 24 |
| SM clock (attribute) | 2 418 000 kHz |
| memory clock (attribute) | 8 533 000 kHz, bus width 256 bits (LPDDR5X; datasheet ≈ 273 GB/s) |
| total memory | 128 520 015 872 bytes (unified with the CPU) |
| managed memory / pageable access / concurrent managed / integrated | 1 / 1 / 1 / 1 |

## System

- DGX OS on Ubuntu 24.04.4 LTS, kernel 6.17.0-1026-nvidia, aarch64; 20 CPUs; 119 GiB visible to the OS.
- Firmware: `fwupdmgr get-upgrades` reports **no pending updates** (Embedded Controller 0x02004203, update state Success), so there is no firmware upgrade to apply before the suite; the versions above are the ones to record in provenance.
- Power at idle 4–11 W; `power.limit` reported N/A; "SW Power Cap" is the active clocks-event reason at idle.

## Consequences for the suite

- Report clock variance (B11) with every run; acceptance intervals 4 % rather than 2 % (§9.1).
- Compile with `-arch=sm_121`; use `cudaDeviceGetAttribute` for clocks.
- Python driver: `~/.venv/bin/python` on the host; benchmarks themselves are plain CUDA C++ built with `nvcc`.
- Shared-memory tiles above 48 KiB need the opt-in attribute (`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`), up to 101 376 bytes.
- B12 (unified memory) applies: the GPU is integrated and pageable access is on.

## P0 (2026-10-15): profiler counters, and the tools the B13–B26 suite needs

| Item | Result |
|---|---|
| Profiler counters for non-admin users | **Done (2026-10-15, second pass).** The account was given passwordless sudo; `/etc/modprobe.d/nvidia-profiling.conf` now holds `options nvidia NVreg_RestrictProfilingToAdminUsers=0`, `update-initramfs -u` was run and the host rebooted. After the reboot `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 0` and `ncu --metrics lts__t_bytes.sum` reads counters as `lishang` (195,232 bytes for the clock-probe kernel). The GB10 exposes no `dram__*` metrics (unified LPDDR5X behind the L2); DRAM traffic is read as L2 lookup misses (`lts__t_sectors_*lookup_miss`), see the runner. Earlier (first pass, before the sudo grant) `ncu` failed with `ERR_NVGPUCTRPERM`. |
| `cuobjdump` / `nvdisasm` | present (CUDA 13.0, compiler build 36400806) |
| `cublasLt` | present: `libcublasLt.so.13`, `cublasLt.h` (B20.1 FP8 can use it) |
| Thread-block clusters + distributed shared memory | **supported**: `cudaDevAttrClusterLaunch = 1`; a `__cluster_dims__(2,1,1)` kernel launched and read a peer block's shared memory through `map_shared_rank` correctly |
| `wgmma.fence` | not supported on `sm_121` (ptxas error) |
| `setmaxnreg` | not supported on `sm_121` (ptxas error) |
| `tcgen05.alloc` | not supported on `sm_121` (ptxas error) |
