// ventus-cooperative-gemm-lowering-v1
#define TM 4u
#define TN 16u
#define TK 8u
kernel void cooperative_gemm_f32(__global const float *a,
    __global const float *b, __global float *c, uint m, uint n, uint k) {
  __local float sa[TM*TK];
  __local float sb[TK*TN];
  uint lid=get_local_id(0), group=get_group_id(0);
  uint groups_n=(n+TN-1)/TN, lr=lid/TN, lc=lid%TN;
  uint row=(group/groups_n)*TM+lr, col=(group%groups_n)*TN+lc;
  float sum=0;
  for(uint at=0;at<k;at+=TK) {
    for(uint i=lid;i<TM*TK;i+=TM*TN) {
      uint r=(group/groups_n)*TM+i/TK, t=at+i%TK;
      sa[i]=(r<m && t<k) ? a[r*k+t] : 0;
    }
    for(uint i=lid;i<TK*TN;i+=TM*TN) {
      uint t=at+i/TN, q=(group%groups_n)*TN+i%TN;
      sb[i]=(t<k && q<n) ? b[t*n+q] : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint i=0;i<TK && at+i<k;++i) sum=__builtin_fmaf(sa[lr*TK+i],sb[i*TN+lc],sum);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(row<m && col<n) c[row*n+col]=sum;
}
