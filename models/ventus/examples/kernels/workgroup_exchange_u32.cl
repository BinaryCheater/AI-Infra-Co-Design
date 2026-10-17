// Repeated cross-warp LDS exchange. Every workitem reaches both barriers,
// including padded tail lanes. The second barrier protects scratch reuse.
kernel void workgroup_exchange_u32(__global const uint *input,
                                   __global uint *output,
                                   uint count, uint rounds) {
  __local uint tile[128];
  uint lid = get_local_id(0);
  uint width = get_local_size(0);
  uint gid = get_global_id(0);
  for (uint round = 0; round < rounds; ++round) {
    uint value = gid < count ? input[gid] + round : 0;
    tile[lid] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    uint received = tile[(lid + 32) % width];
    if (gid < count) output[round * count + gid] = received;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
