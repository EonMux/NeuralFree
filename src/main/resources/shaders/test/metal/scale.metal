/* scale.metal - Metal twin of shaders/test/scale.comp. Multiplies every element by a factor.
 * The push-constant scalars arrive packed in one constant buffer at buffer(1). */
#include <metal_stdlib>
using namespace metal;

struct ScaleParams {
    float factor;
    uint n;
};

kernel void scale(device float* data [[buffer(0)]],
                  constant ScaleParams& params [[buffer(1)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid < params.n) {
        data[gid] = data[gid] * params.factor;
    }
}
