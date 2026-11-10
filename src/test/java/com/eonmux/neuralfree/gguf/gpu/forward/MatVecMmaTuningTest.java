package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ops.gemm.MatVecMmaPlan;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The measured tensor-core matvec rules ({@link MatVecMmaTuning}) and the launch geometry of {@link MatVecMmaPlan}. */
public class MatVecMmaTuningTest {

    private static final DeviceKind CUDA_TENSOR = DeviceKind.of(BackendType.CUDA, DeviceKind.Vendor.NVIDIA,
            DeviceClass.DISCRETE, DeviceKind.CudaBand.TENSOR);

    @Test
    public void q4kServesFromTwoTokensOnEveryShape() {
        for (int[] s : new int[][]{{9728, 2560}, {2560, 9728}, {1024, 2560}, {151936, 2560}, {1024, 1024}}) {
            for (int n = 2; n <= 16; n++) {
                assertEquals(4, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q4_K, s[0], s[1], n));
            }
        }
    }

    @Test
    public void q5kServesFromTwoTokensExceptNarrowShortProjectionsAndWidensTheNarrowSixteenTokenTile() {
        for (int[] s : new int[][]{{9728, 2560}, {2560, 9728}, {3072, 1024}, {1024, 2560}, {151936, 1024}}) {
            for (int n = 2; n <= 8; n++) {
                assertEquals(4, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q5_K, s[0], s[1], n));
            }
        }
        for (int[] s : new int[][]{{2048, 1024}, {1024, 1024}}) {
            assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q5_K, s[0], s[1], 2));
            assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q5_K, s[0], s[1], 3));
            assertEquals(4, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q5_K, s[0], s[1], 4));
            assertEquals(4, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q5_K, s[0], s[1], 8));
        }
        for (int n = 9; n <= 16; n++) {
            assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q5_K, 1024, 3072, n));
            assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q5_K, 1024, 1024, n));
            assertEquals(4, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q5_K, 2048, 1024, n));
            assertEquals(4, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q5_K, 9728, 2560, n));
        }
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q5_K, 9728, 2304, 8));
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q5_K, 9728, 2560, 17));
    }

    @Test
    public void q6kServesTheEightTokenTileByRowClassAndTheSixteenTokenTileFromNine() {
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q6_K, 2560, 9728, 3));
        for (int n = 4; n <= 8; n++) {
            assertEquals(4, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q6_K, 2560, 9728, n));
            assertEquals(4, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q6_K, 2048, 1024, n));
        }
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q6_K, 151936, 2560, 2));
        assertEquals(4, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q6_K, 151936, 2560, 3));
        assertEquals(4, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q6_K, 151936, 1024, 8));
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q6_K, 1024, 2560, 5));
        assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q6_K, 1024, 2560, 6));
        assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q6_K, 1024, 3072, 8));
        for (int n = 9; n <= 16; n++) {
            assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q6_K, 2560, 9728, n));
            assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q6_K, 1024, 2560, n));
            assertEquals(2, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q6_K, 151936, 2560, n));
        }
    }

    @Test
    public void q80EightTokenTileServesPerShapeClass() {
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 151936, 1024, 3));
        assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 151936, 1024, 4));
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 151936, 2560, 4));
        assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 151936, 2560, 5));
        for (int[] s : new int[][]{{3072, 1024}, {2048, 1024}, {1024, 1024}}) {
            assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, s[0], s[1], 5));
            assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, s[0], s[1], 6));
        }
        for (int[] s : new int[][]{{9728, 2560}, {4096, 2560}, {1024, 2560}, {1024, 3072}, {1024, 2048}}) {
            assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, s[0], s[1], 7));
            assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, s[0], s[1], 8));
        }
        for (int n = 2; n <= 8; n++) {
            assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 2560, 4096, n));
            assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 2560, 9728, n));
        }
    }

    @Test
    public void q80SixteenTokenTileCrossesOverPerShapeClass() {
        assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 9728, 2560, 9));
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 2560, 9728, 10));
        assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 2560, 9728, 11));
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 1024, 2560, 9));
        assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 1024, 2560, 10));
        assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 151936, 2560, 9));
        assertEquals(8, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 151936, 1024, 16));
    }

    @Test
    public void otherKindsTypesAndShapesKeepTheCountedForms() {
        DeviceKind openCl = DeviceKind.of(BackendType.OPENCL, DeviceKind.Vendor.NVIDIA, DeviceClass.DISCRETE,
                DeviceKind.CudaBand.NONE);
        DeviceKind packed = DeviceKind.of(BackendType.CUDA, DeviceKind.Vendor.NVIDIA, DeviceClass.DISCRETE,
                DeviceKind.CudaBand.PACKED_DOT);
        assertEquals(0, MatVecMmaTuning.warps(openCl, GGMLType.Q4_K, 9728, 2560, 16));
        assertEquals(0, MatVecMmaTuning.warps(packed, GGMLType.Q4_K, 9728, 2560, 16));
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q4_0, 9728, 2560, 16));
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q4_K, 9728, 2304, 16));
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q8_0, 9728, 2592, 16));
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q4_K, 9728, 2560, 17));
        assertEquals(0, MatVecMmaTuning.warps(CUDA_TENSOR, GGMLType.Q4_K, 9728, 2560, 1));
    }

    @Test
    public void planGeometryFollowsTheKernelInstantiations() {
        MatVecMmaPlan q8 = MatVecMmaPlan.of(GGMLType.Q8_0, 77, 1024, 9, 8);
        assertEquals("Q8_0_MatVecMma16_Q8", q8.kernel());
        assertEquals(16, q8.tileRows());
        assertArrayEquals(new long[]{5 * 256L}, q8.globalWorkSize());
        assertArrayEquals(new long[]{256L}, q8.localWorkSize());
        MatVecMmaPlan q4 = MatVecMmaPlan.of(GGMLType.Q4_K, 77, 1024, 8, 4);
        assertEquals("Q4_K_MatVecMma8_Q8", q4.kernel());
        assertEquals(32, q4.tileRows());
        assertArrayEquals(new long[]{3 * 128L}, q4.globalWorkSize());
        assertEquals(32, MatVecMmaPlan.tileRows(GGMLType.Q8_0, 8));
        assertEquals(32, MatVecMmaPlan.tileRows(GGMLType.Q6_K, 16));
        assertTrue(MatVecMmaPlan.serves(GGMLType.Q6_K, 1024));
        assertFalse(MatVecMmaPlan.serves(GGMLType.Q6_K, 768));
        assertTrue(MatVecMmaPlan.serves(GGMLType.Q5_K, 1024));
        assertFalse(MatVecMmaPlan.serves(GGMLType.Q5_K, 768));
        assertFalse(MatVecMmaPlan.serves(GGMLType.Q5_0, 1024));
        MatVecMmaPlan q5 = MatVecMmaPlan.of(GGMLType.Q5_K, 77, 1024, 12, 4);
        assertEquals("Q5_K_MatVecMma16_Q8", q5.kernel());
        assertEquals(32, q5.tileRows());
        assertThrows(IllegalArgumentException.class, () -> MatVecMmaPlan.of(GGMLType.Q4_K, 77, 1024, 17, 4));
        assertThrows(IllegalArgumentException.class, () -> MatVecMmaPlan.of(GGMLType.Q4_K, 77, 1024, 4, 9));
        assertThrows(IllegalArgumentException.class, () -> MatVecMmaPlan.of(GGMLType.Q5_0, 77, 1024, 4, 4));
    }
}
