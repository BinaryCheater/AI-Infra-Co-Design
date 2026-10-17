// A concrete cooperative FP32 GEMM: 8x8 outputs/workgroup, 8-element K stages.
// Padded lanes participate in both barriers; only valid outputs are written.
kernel void workgroup_gemm_f32(__global const float *a,
                              __global const float *b,
                              __global float *c, uint m, uint n, uint k) {
  __local float staged_a[64];
  __local float staged_b[64];
  uint lid = get_local_id(0);
  uint group = get_group_id(0);
  uint column_groups = (n + 7) / 8;
  uint lr = lid / 8, lc = lid % 8;
  uint row = (group / column_groups) * 8 + lr;
  uint col = (group % column_groups) * 8 + lc;
  float sum = 0;
  for (uint at = 0; at < k; at += 8) {
    staged_a[lid] = row < m && at + lc < k ? a[row * k + at + lc] : 0;
    staged_b[lid] = at + lr < k && col < n ? b[(at + lr) * n + col] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 0; i < 8; ++i)
      sum = __builtin_fmaf(staged_a[lr * 8 + i], staged_b[i * 8 + lc], sum);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (row < m && col < n) c[row * n + col] = sum;
}
