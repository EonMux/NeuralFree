package com.eonmux.neuralfree.acceleration.backend.metal;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Detection of the SIMD-group operations of a generated MSL kernel, which decide whether a
 * threadgroup has to be a whole number of SIMD groups.
 */
public class MetalKernelHandleTest {

    @Test
    public void theSimdGroupBuiltinsAndReductionsAreDetected() {
        assertTrue(MetalKernelHandle.usesSubgroups(
                "kernel void k(uint gid [[thread_index_in_simdgroup]]) { }"));
        assertTrue(MetalKernelHandle.usesSubgroups("float s = simd_sum(x);"));
        assertTrue(MetalKernelHandle.usesSubgroups(
                "kernel void k(uint n [[simdgroups_per_threadgroup]]) { }"));
    }

    @Test
    public void aPlainKernelIsNotASubgroupKernel() {
        assertFalse(MetalKernelHandle.usesSubgroups("#include <metal_stdlib>\n#include <simd/simd.h>\n"
                + "using namespace metal;\n"
                + "kernel void increment(device float* data [[buffer(0)]], uint gid [[thread_position_in_grid]])"
                + " { data[gid] = data[gid] + 1.0f; }\n"));
    }
}
