// Diagnostic: signed byte/halfword loads, private byte masks, packed output
// stores, and a partial workgroup. This is not an optimization candidate.
__kernel void subword_u32(__global const char *bytes,
                         __global const short *halves,
                         __global uchar *out_bytes,
                         __global ushort *out_halves,
                         __global uint *out_words, uint n) {
  uint i = get_global_id(0);
  if (i < n) {
    volatile char b[2];
    volatile short h[2];
    b[0] = bytes[i];
    b[1] = (char)(i * 3 + 7);
    h[0] = halves[i];
    h[1] = (short)(i * 257 + 123);
    out_bytes[i] = (uchar)(b[0] ^ b[1]);
    out_halves[i] = (ushort)(h[0] ^ h[1]);
    out_words[2 * i] = (uint)((int)b[0] / (int)((i & 7) + 1));
    out_words[2 * i + 1] = (uint)((int)h[0] % (int)((i & 3) + 1));
  }
}
