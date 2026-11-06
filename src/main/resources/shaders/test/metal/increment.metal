/* increment.metal - Metal twin of shaders/test/increment.comp. Adds one to every element. */
#include <metal_stdlib>
using namespace metal;

kernel void increment(device float* data [[buffer(0)]],
                      constant uint& n [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        data[gid] = data[gid] + 1.0f;
    }
}
