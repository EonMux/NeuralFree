package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch traces of the paged flash launches of {@link GpuAttentionMixer} ({@link GpuPagedFlashAttention}) over a
 * paged F16 cache of two sequences, on the host-only fake backend with strict LLM argument rules: a backend with
 * tensor cores whose paged mma prefill kernel is missing runs every prefill segment through the
 * {@link PagedRowFallback} (one paged decode launch per row over the focus table), and the next decode step's single
 * launch over both sequences reads the step's own page table from Q and output offset 0.
 */
public class GpuPagedFlashAttentionTest {

    private static final int WIDTH = 256;
    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int HEAD_DIM = 64;
    private static final int Q_DIM = N_HEAD * HEAD_DIM;
    private static final int KV_DIM = N_HEAD_KV * HEAD_DIM;
    private static final int N_CTX = 64;
    private static final int UBATCH = 8;
    private static final int PARALLEL = 2;
    private static final int PAGE = 16;

    private FakeComputeBackend backend;
    private DeviceModelMemory memory;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private LlmKernels kernels;
    private GpuOps ops;
    private GpuMixer mixer;
    private final Map<String, GpuWeight> tensors = new HashMap<>();

    @After
    public void tearDown() {
        if (mixer != null) {
            mixer.close();
        }
        if (ops != null) {
            ops.close();
        }
        if (placeholders != null) {
            placeholders.close();
        }
        if (scratch != null) {
            scratch.close();
        }
        if (memory != null) {
            memory.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    // ================================================================== fixtures

    private static DeviceCapabilities tensorCoreCaps() {
        return new DeviceCapabilities.Builder()
                .deviceName("FakeTensorCores")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(32)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .supportsAsyncTransfer(true)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024)
                .maxWorkGroupCount(new long[]{65535, 65535, 65535})
                .computeCapability(89)
                .feature(DeviceFeature.COOPERATIVE_MATRIX)
                .build();
    }

    private static ModelSpec llama() {
        AttentionSpec attention = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD_KV).headDimK(HEAD_DIM)
                .headDimV(HEAD_DIM).nRot(HEAD_DIM).kqScale(0.125f).kvOwnerLayer(0).build();
        LayerSpec l = LayerSpec.builder(0, MixerKind.ATTENTION, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).qkNorm(attention.qkNorm()).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, WIDTH).build())
                .attention(attention)
                .rope(RopeSpec.builder(RopeType.NORM).nDims(HEAD_DIM).freqBase(10000f).build())
                .build();
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.LLAMA)).layerCounts(1, 0)
                .nCtxTrain(512).nEmbd(WIDTH).headKFull(HEAD_DIM).headVFull(HEAD_DIM).nRotFull(HEAD_DIM)
                .normEps(1e-5f).normRmsEps(1e-6f);
        hb.nHead(0, N_HEAD).nHeadKv(0, N_HEAD_KV);
        return ModelSpec.builder(hb.build(), TensorCatalog.of(Arch.LLAMA.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, WIDTH).build())
                .head(HeadSpec.builder().build())
                .layer(l)
                .build();
    }

    private void matrix(LayerSpec l, String base, int cols, int rows) {
        String n = TensorNames.layerName(l.stack(), l.weightLayer(), TensorNames.weight(base));
        tensors.put(n, GpuWeight.of(n, GGMLType.F32, new long[]{cols, rows},
                backend.allocateBuffer((long) cols * rows * Float.BYTES, true)));
    }

    /** Binds the llama mixer over a paged F16 cache of {@link #PARALLEL} sequences on a tensor-core backend. */
    private void bindPaged() {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).backendType(ComputeBackend.BackendType.OPENCL)
                .capabilities(tensorCoreCaps()).unavailableKernels(AttnKernels.flashAttnMmaPaged(KvDType.F16))
                .build();
        ModelSpec spec = llama();
        LayerSpec l = spec.layers().get(0);
        matrix(l, TensorNames.ATTN_Q, WIDTH, Q_DIM);
        matrix(l, TensorNames.ATTN_K, WIDTH, KV_DIM);
        matrix(l, TensorNames.ATTN_V, WIDTH, KV_DIM);
        matrix(l, TensorNames.ATTN_OUT, Q_DIM, WIDTH);
        MemoryConfig cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(N_CTX).parallel(PARALLEL).kvPageSize(PAGE).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
        memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan, MropeTextPositions.PPP0, null);
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * WIDTH)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * WIDTH)
                .slot(ScratchSlots.Q, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.K, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.V, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.ATTN_OUT, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.POS, (long) UBATCH * 4)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        kernels = LlmKernels.open(backend);
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        GpuForwardContext ctx = GpuForwardContext.builder(kernels, ops, tensors::get, spec).memory(memory).build();
        for (GpuMixerFactory f : AttnBlocks.gpuMixers()) {
            if (f.kind() == MixerKind.ATTENTION) {
                mixer = f.create(l);
            }
        }
        mixer.bind(ctx, l);
    }

    /**
     * Runs one batched step of one segment per sequence ({@code lens[s]} rows from {@code pos0s[s]}) through the
     * mixer and commits it, returning the paged decode launches it dispatched.
     */
    private List<FakeDispatch> runStep(int[] lens, int[] pos0s) {
        PagedSequences paged = memory.pagedSequences();
        int n = 0;
        int[] starts = new int[lens.length];
        int[] seqs = new int[lens.length];
        for (int s = 0; s < lens.length; s++) {
            seqs[s] = s;
            starts[s] = n;
            n += lens[s];
        }
        StepLayout layout = StepLayout.of(n, seqs, starts, lens, pos0s, new int[0], true);
        PagedStep step = paged.prepare(layout);
        int[] rows = new int[n];
        for (int s = 0; s < lens.length; s++) {
            for (int t = 0; t < lens[s]; t++) {
                rows[starts[s] + t] = pos0s[s] + t;
            }
        }
        memory.beginPagedStep(step, rows, null);
        backend.clearLog();
        mixer.batch(layout, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        List<FakeDispatch> launches = named(backend.dispatches(), AttnKernels.flashDecodePaged(KvDType.F16, KvDType.F16));
        memory.endPagedStep();
        paged.commit(step);
        for (int s = 0; s < lens.length; s++) {
            paged.appendTokens(s, new int[lens[s]]);
        }
        return launches;
    }

    private static List<FakeDispatch> named(List<FakeDispatch> ds, String kernel) {
        List<FakeDispatch> out = new ArrayList<>();
        for (FakeDispatch d : ds) {
            if (d.kernelName().equals(kernel)) {
                out.add(d);
            }
        }
        return out;
    }

    // ================================================================== traces

    @Test
    public void aPrefillWithoutThePagedMmaKernelRunsEveryRowThroughTheFocusTable() {
        bindPaged();
        GpuStepInputs steps = memory.stepInputs();
        GpuKvLayer kv = memory.kv(0);
        List<FakeDispatch> rows = runStep(new int[]{3, 2}, new int[]{0, 0});
        assertEquals("one paged decode launch per prefill row", 5, rows.size());
        for (int row = 0; row < rows.size(); row++) {
            FakeDispatch d = rows.get(row);
            assertEquals(1, d.intArg(AttnKernels.ARG_FDP_N_SEQ));
            assertEquals(row * Q_DIM, d.intArg(AttnKernels.ARG_FD_Q_OFF));
            assertEquals(row * Q_DIM, d.intArg(AttnKernels.ARG_FD_OUT_OFF));
            assertEquals(0, d.intArg(AttnKernels.ARG_FDP_Q_SEQ_STRIDE));
            assertEquals(0, d.intArg(AttnKernels.ARG_FDP_OUT_SEQ_STRIDE));
            assertNotSame(steps.pageTable(kv.group()), d.bufferArg(AttnKernels.ARG_FDP_BUF_PAGE_TABLE));
            assertNotSame(steps.seqKvLen(kv.group()), d.bufferArg(AttnKernels.ARG_FDP_BUF_SEQ_KV_LEN));
            assertNotSame(steps.qPosSeq(), d.bufferArg(AttnKernels.ARG_FDP_BUF_Q_POS_SEQ));
        }
    }

    @Test
    public void aDecodeAfterPerRowPrefillsReadsTheStepPageTableFromOffsetZero() {
        bindPaged();
        GpuStepInputs steps = memory.stepInputs();
        GpuKvLayer kv = memory.kv(0);
        runStep(new int[]{3, 2}, new int[]{0, 0});
        List<FakeDispatch> decode = runStep(new int[]{1, 1}, new int[]{3, 2});
        assertEquals("one paged decode launch over both sequences", 1, decode.size());
        FakeDispatch d = decode.get(0);
        assertEquals(2, d.intArg(AttnKernels.ARG_FDP_N_SEQ));
        assertEquals(0, d.intArg(AttnKernels.ARG_FD_Q_OFF));
        assertEquals(0, d.intArg(AttnKernels.ARG_FD_OUT_OFF));
        assertEquals(Q_DIM, d.intArg(AttnKernels.ARG_FDP_Q_SEQ_STRIDE));
        assertEquals(Q_DIM, d.intArg(AttnKernels.ARG_FDP_OUT_SEQ_STRIDE));
        assertSame(steps.pageTable(kv.group()), d.bufferArg(AttnKernels.ARG_FDP_BUF_PAGE_TABLE));
        assertSame(steps.seqKvLen(kv.group()), d.bufferArg(AttnKernels.ARG_FDP_BUF_SEQ_KV_LEN));
        assertSame(steps.qPosSeq(), d.bufferArg(AttnKernels.ARG_FDP_BUF_Q_POS_SEQ));
        assertTrue((d.intArg(AttnKernels.ARG_FD_FLAGS) & AttnKernels.FLAG_CAUSAL) != 0);
    }
}
