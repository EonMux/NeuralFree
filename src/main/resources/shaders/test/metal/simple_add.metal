/* simple_add.metal - Metal twin of shaders/test/simple_add.comp. c = a + b. */
#include <metal_stdlib>
using namespace metal;

kernel void simple_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}
