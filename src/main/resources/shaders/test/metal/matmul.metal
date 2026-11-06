/* matmul.metal - Metal twin of shaders/test/matmul.comp. C[M, N] = A[M, K] * B[K, N].
 * The push-constant scalars arrive packed in one constant buffer at buffer(3). */
#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
    uint M;
    uint K;
    uint N;
};

kernel void matmul(device const float* A [[buffer(0)]],
                   device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]],
                   constant MatmulParams& params [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.M || col >= params.N) {
        return;
    }
    float sum = 0.0f;
    for (uint k = 0; k < params.K; k++) {
        sum += A[row * params.K + k] * B[k * params.N + col];
    }
    C[row * params.N + col] = sum;
}
