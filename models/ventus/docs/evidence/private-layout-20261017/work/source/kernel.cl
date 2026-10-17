// Diagnostic: every lane owns three volatile private words across a block barrier.
// No cross-lane private communication is part of the source program.
__kernel void private_spill_u32(__global uint *out, uint count) {
    uint gid = get_global_id(0);
    volatile uint slots[3];
    slots[0] = gid * 16u;
    slots[1] = gid * 16u + 1u;
    slots[2] = gid * 16u + 2u;
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (gid < count) {
        out[gid * 3u] = slots[0];
        out[gid * 3u + 1u] = slots[1];
        out[gid * 3u + 2u] = slots[2];
    }
}
