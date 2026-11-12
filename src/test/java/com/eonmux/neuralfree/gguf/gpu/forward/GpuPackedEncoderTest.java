package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.BatchSegments;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.OutputKind;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.EnumSet;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Packed memoryless encoder batches on a device ({@link BatchSegments}, {@code GpuFlashAttention.prefillRuns}):
 * several BERT sequences in one memoryless step, each attending only its own rows, produce every sequence's hidden
 * rows of its own single-sequence step, and the packed {@code encodeBatch} pools the rows {@code encode} pools. The
 * runs cover a one-row sequence and runs longer and shorter than the first, so a launch's query positions must be
 * its own run's. Batch invariance pins the attention to the per-row decode launches and the projections to the
 * single-row reduction order, so there the packed rows equal the single-sequence rows bit for bit. The subgroup
 * cases run on a device whose kernels include the 16-lane subgroup forms ({@link SubgroupFormBackends}, the Intel
 * GPUs), where the step's F32 keys at head dim 16 take the subgroup prefill kernel ({@code LLM_FlashAttnSg_F32})
 * with each run's own key start and query offset, and under batch invariance the subgroup decode kernel per row.
 */
public class GpuPackedEncoderTest {

    private static final int UBATCH = 96;
    private static final float TOL = 2e-3f;
    private static final int[] LENGTHS = {9, 1, 23, 4, 40};

    @After
    public void clearInvariance() {
        BatchInvariance.clearOverride();
    }

    private static int[][] sequences() {
        int[][] s = new int[LENGTHS.length][];
        for (int i = 0; i < LENGTHS.length; i++) {
            s[i] = new int[LENGTHS[i]];
            for (int t = 0; t < LENGTHS[i]; t++) {
                s[i][t] = 5 + (i * 31 + t * 7) % 200;
            }
        }
        return s;
    }

    @Test
    public void cudaPackedRowsEqualEachSequencesOwnStep() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        assertPackedRowsMatch(EngineConfig.BackendMode.CUDA, false);
    }

    @Test
    public void openclPackedRowsEqualEachSequencesOwnStep() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        assertPackedRowsMatch(EngineConfig.BackendMode.OPENCL, false);
    }

    @Test
    public void vulkanPackedRowsEqualEachSequencesOwnStep() throws IOException {
        Assume.assumeTrue("Vulkan not available", GpuTestUtil.isVulkanAvailable());
        assertPackedRowsMatch(EngineConfig.BackendMode.VULKAN, false);
    }

    @Test
    public void cudaBatchInvariantPackedRowsAreTheSequencesOwnBits() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        assertPackedRowsMatch(EngineConfig.BackendMode.CUDA, true);
    }

    @Test
    public void vulkanBatchInvariantPackedRowsAreTheSequencesOwnBits() throws IOException {
        Assume.assumeTrue("Vulkan not available", GpuTestUtil.isVulkanAvailable());
        assertPackedRowsMatch(EngineConfig.BackendMode.VULKAN, true);
    }

    @Test
    public void openclSubgroupPackedRowsEqualEachSequencesOwnStep() throws IOException {
        assertPackedRowsMatch(EngineConfig.BackendMode.OPENCL, false, subgroupDevice(BackendType.OPENCL));
    }

    @Test
    public void vulkanSubgroupPackedRowsEqualEachSequencesOwnStep() throws IOException {
        assertPackedRowsMatch(EngineConfig.BackendMode.VULKAN, false, subgroupDevice(BackendType.VULKAN));
    }

    @Test
    public void vulkanSubgroupBatchInvariantPackedRowsAreTheSequencesOwnBits() throws IOException {
        assertPackedRowsMatch(EngineConfig.BackendMode.VULKAN, true, subgroupDevice(BackendType.VULKAN));
    }

    /** The name of the first device of {@code api} whose kernels include the subgroup forms; skips without one. */
    private static String subgroupDevice(BackendType api) {
        try (SubgroupFormBackends devices = SubgroupFormBackends.open(api)) {
            return devices.backends().get(0).capabilities().getDeviceName();
        }
    }

    private static void assertPackedRowsMatch(EngineConfig.BackendMode mode, boolean invariant) throws IOException {
        assertPackedRowsMatch(mode, invariant, null);
    }

    /** {@code device} is a device-name substring to load on, or null for the backend's default device. */
    private static void assertPackedRowsMatch(EngineConfig.BackendMode mode, boolean invariant, String device)
            throws IOException {
        BatchInvariance.setEnabled(invariant);
        Path model = SyntheticModels.bert().writeTempFile();
        int[][] seqs = sequences();
        MemoryConfig memory = MemoryConfig.builder().nCtx(UBATCH).ubatch(UBATCH).build();
        try (EngineModel gpu = LLMModelFactory.open(model.toString(), EngineConfig.builder().backend(mode)
                .deviceFilter(device).gpuLayers(EngineConfig.GPU_LAYERS_ALL).memory(memory).build())) {
            Assume.assumeTrue(mode + " did not open the GPU engine", gpu instanceof GGUFModelGPU);
            ForwardBackend b = gpu.forwardBackend();
            String at = mode + (device != null ? " on " + device : "") + (invariant ? " batch-invariant" : "");
            assertEquals(at + " packs memoryless sequences", BatchSegments.MAX_PACKED, b.memorylessSequences());
            float[][][] own = new float[seqs.length][][];
            for (int s = 0; s < seqs.length; s++) {
                own[s] = b.hiddenRows(seqs[s], null);
            }
            float[][] packed = b.hiddenBatch(batch(seqs));
            float tol = invariant ? 0f : TOL;
            int row = 0;
            for (int s = 0; s < seqs.length; s++) {
                for (int t = 0; t < seqs[s].length; t++) {
                    assertArrayEquals(at + " sequence " + s + " row " + t, own[s][t], packed[row++], tol);
                }
            }
            float[][] pooled = gpu.encodeBatch(seqs);
            for (int s = 0; s < seqs.length; s++) {
                assertArrayEquals(at + " pooled sequence " + s, gpu.encode(seqs[s]), pooled[s], tol);
            }
        }
    }

    private static ForwardBatch batch(int[][] seqs) {
        int total = 0;
        for (int[] s : seqs) {
            total += s.length;
        }
        ForwardBatch.Builder b = ForwardBatch.builder(total);
        for (int s = 0; s < seqs.length; s++) {
            for (int t = 0; t < seqs[s].length; t++) {
                b.add(seqs[s][t], t, s, true);
            }
        }
        b.outputKinds(EnumSet.of(OutputKind.HIDDEN));
        return b.build();
    }
}
