package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclKernelHandle;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;

/**
 * The RDNA3, CDNA and int8-tensor MMQ tiles on the OpenCL device, forced with
 * {@link MmqConfig#TILE_PROPERTY}: the OpenCL library compiles the kernels with
 * {@code -DNF_MMQ_TM/-DNF_MMQ_TN/-DNF_MMQ_TK} of the forced tile ({@link MmqConfig#kernelDefines}), and
 * the MMQ GEMM of each tile must match the dequantizing tiled GEMM and its split-K twin on the same
 * device. The forced tiles exercise the packed-dot datapath with several token groups per work-item
 * (RDNA3's 128-token tile, the int8-tensor tile's 32-deep stages) and the 128-row CDNA tile.
 */
@RunWith(Parameterized.class)
public class OpenClMmqTileParityTest {

    private static final int M = 70;   /* row tiles of 64 (two, the second partial) and 128 (one partial) */
    private static final int K = 256;
    private static final int N = 141;  /* token tiles of 128 (two, the second partial) and 64 (three) */

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> tiles() {
        Collection<Object[]> params = new ArrayList<>();
        params.add(new Object[]{MmqConfig.RDNA3.name(), MmqConfig.RDNA3});
        params.add(new Object[]{MmqConfig.CDNA.name(), MmqConfig.CDNA});
        params.add(new Object[]{MmqConfig.INT8_TENSOR.name(), MmqConfig.INT8_TENSOR});
        return params;
    }

    private final MmqConfig config;
    private HWAcceleration hwAccel;
    private OclComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;

    public OpenClMmqTileParityTest(String name, MmqConfig config) {
        this.config = config;
    }

    @Before
    public void setUp() {
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no OpenCL device", !hw.getCLDevices().isEmpty());
            CLDevice device = hw.getCLDevices().get(0);
            System.setProperty(MmqConfig.TILE_PROPERTY, config.name());
            OclComputeBackend b = new OclComputeBackend(device);
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable on " + b.capabilities().getDeviceName() + ": "
                    + availability.reason(), availability.available());
            hwAccel = hw;
            backend = b;
            queue = b.computeQueue();
            kernels = LlmKernels.open(b);
            OclKernelHandle probe = (OclKernelHandle) kernels.get("Q4_K_MatMul_MMQ");
            Assume.assumeTrue("the " + config.name() + " tile needs " + config.workGroupSize()
                    + "-item workgroups, more than the kernel's "
                    + probe.maxWorkGroupSize() + " on " + b.capabilities().getDeviceName(),
                    probe.maxWorkGroupSize() >= config.workGroupSize());
        } catch (Throwable t) {
            if (hw != null) {
                try {
                    hw.shutdown();
                } catch (Throwable ignored) {
                }
            }
            Assume.assumeTrue("OpenCL init failed: " + t.getMessage(), false);
        }
    }

    @After
    public void tearDown() {
        System.clearProperty(MmqConfig.TILE_PROPERTY);
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    /** The forced tile is the one the backend plans with. */
    @Test
    public void theBackendPlansWithTheForcedTile() {
        assertEquals(config, MmqConfig.forDevice(backend.backendType(),
                backend.capabilities().getComputeCapability()));
    }

    /** The MMQ GEMM of a legacy and a K-quant weight under the forced tile against the tiled GEMM. */
    @Test
    public void mmqGemmMatchesTheTiledGemmUnderTheForcedTile() throws Exception {
        FastGemmParity parity = new FastGemmParity(backend, kernels);
        for (GGMLType type : new GGMLType[]{GGMLType.Q4_K, GGMLType.IQ2_XS}) {
            parity.mmqMatchesTiledWithConfig(type, M, K, N, config.name(), config);
        }
    }

    /** The split-K twins carry the forced tile's row and token tiles and work-group size. */
    @Test
    public void mmqSplitKTwinMatchesThePlainKernelUnderTheForcedTile() throws Exception {
        FastGemmParity parity = new FastGemmParity(backend, kernels);
        parity.mmqSplitKMatchesPlain(GGMLType.Q4_K, M, 768, N, 3);
    }

    private static void assertEquals(Object expected, Object actual) {
        org.junit.Assert.assertEquals(expected, actual);
    }
}
