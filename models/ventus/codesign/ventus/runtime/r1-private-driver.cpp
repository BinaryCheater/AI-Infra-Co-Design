// Explicit host-driver correction for the unchanged R1 RTL library.
// LSU.scala interleaves private words over the whole workgroup and includes TID.
// Every warp must therefore receive the same workgroup private base.
#include "Vdut.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

static std::atomic<unsigned long long> applications{0};
extern "C" void private_driver_eval(Vdut* dut) asm("_ZN4Vdut9eval_stepEv");
extern "C" void private_driver_eval(Vdut* dut) {
    static auto next = reinterpret_cast<void (*)(Vdut*)>(
        dlsym(RTLD_NEXT, "_ZN4Vdut9eval_stepEv"));
    if (!next) std::abort();
    if (!dut->reset && dut->io_host_req_valid) {
        if (dut->io_host_req_bits_host_pds_size_per_wf != 0) ++applications;
        dut->io_host_req_bits_host_pds_size_per_wf = 0;
    }
    next(dut);
}

__attribute__((destructor)) static void report_private_driver() {
    std::fprintf(stderr, "ventus-private-driver: workgroup-private-v1 applications=%llu\n",
                 applications.load());
}
