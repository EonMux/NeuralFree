package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Locale;

/**
 * Measures a long prompt's prefill latency on one device against the same prompt through the prefill
 * context-parallel split ({@code prefill_context_parallel_size} 2), reporting both numbers. Both ranks of the
 * split run as two contexts of the SAME physical OpenCL device, so the ranks serialize on one GPU: the measured
 * difference is the split's exchange and packing overhead (plus the protocol's synchronizations), not
 * multi-device scaling. The test asserts no ordering between the two numbers — on one device the split is
 * expected to be slower; across physical devices the trunk work halves per rank.
 */
public class PcpPrefillLatencyTest {

    private static final int UBATCH = 64;
    private static final int N_CTX = 512;
    private static final int PROMPT = 384;
    private static final int REPS = 3;

    private Path file;
    private HWAcceleration hw;
    private OclComputeBackend main;
    private OclComputeBackend peer;
    private LlmKernels kernels;
    private ComputeQueue queue;
    private GpuPlaceholders placeholders;
    private GpuScratch scratch;
    private GpuOps ops;
    private GGUFFileParser parser;
    private WeightStore store;
    private GpuWeightUploader uploader;
    private ModelSpec spec;
    private MemoryConfig mem;
    private DeviceModelMemory mainMemory;
    private GpuForwardContext mainContext;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .nLayer(8).nEmbd(256).nHead(8).nHeadKv(8).nFf(512).nCtxTrain(N_CTX).build()).writeTempFile();
        parser = new GGUFFileParser(file.toString());
        hw = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hw.getCLDevices().isEmpty());
        main = new OclComputeBackend(hw.getCLDevices().get(0));
        peer = new OclComputeBackend(hw.getCLDevices().get(0));
        kernels = LlmKernels.open(main);
        queue = main.computeQueue();
        placeholders = GpuPlaceholders.allocate(main, queue);
        mem = MemoryConfig.defaults().toBuilder().ubatch(UBATCH)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), mem);
        scratch = GpuScratch.forModel(main, spec, UBATCH);
        ops = new GpuOps(queue, kernels, scratch, placeholders);
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        uploader = new GpuWeightUploader(main, queue, parser.getTensors(), store, "llama",
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels)).build());
        uploader.uploadResident(t -> false);
        MemoryPlan mainPlan = MemoryPlanner.planForContext(spec, mem, N_CTX);
        mainMemory = DeviceModelMemory.allocate(main, queue, mainPlan, spec.hparams().mropeTextPositions(), null);
        mainContext = GpuForwardContext.builder(kernels, ops, uploader, spec).memory(mainMemory).build();
    }

    @After
    public void close() throws IOException {
        try {
            if (mainMemory != null) {
                mainMemory.close();
            }
            uploader.close();
            ops.close();
            scratch.close();
            store.close();
            placeholders.close();
            main.close();
            peer.close();
            parser.close();
        } finally {
            if (hw != null) {
                hw.shutdown();
            }
            Files.deleteIfExists(file);
        }
    }

    /** One full prompt of PREFILL steps, reading back the last row's logits of every step. */
    private void runPrompt(GpuForwardBackend backend) {
        int[] tokens = new int[PROMPT];
        for (int t = 0; t < PROMPT; t++) {
            tokens[t] = t % spec.hparams().nVocab();
        }
        for (int pos0 = 0; pos0 < PROMPT; pos0 += UBATCH) {
            backend.forwardRows(Arrays.copyOfRange(tokens, pos0, pos0 + UBATCH), pos0, new int[]{UBATCH - 1});
        }
    }

    /** The per-rep prefill latency of one backend in milliseconds, resetting the cache between reps. */
    private double[] timedReps(String tag, GpuForwardBackend backend) {
        double[] ms = new double[REPS];
        for (int rep = 0; rep < REPS; rep++) {
            mainMemory.reset();
            long start = System.nanoTime();
            runPrompt(backend);
            ms[rep] = (System.nanoTime() - start) / 1e6;
            System.out.println(String.format(Locale.ROOT, "%s rep %d: %.2f ms", tag, rep + 1, ms[rep]));
        }
        return ms;
    }

    private double median(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    @Test
    public void thePrefillLatencyOfTheQuerySplitIsReportedAgainstTheSingleDevice() throws IOException {
        TensorParallelGroup group = null;
        GpuForwardBackend whole = new GpuForwardBackend(mainContext, GpuFactoryCatalog.defaults(),
                GpuFactoryCatalog.headPreprocessors());
        try {
            runPrompt(whole);
            double[] wholeMs = timedReps("single device", whole);
            Arrays.sort(wholeMs);

            group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                    new GpuDevices.Selected(peer, "peer of the same device")));
            group.prefillContextParallelSize(2);
            TpAttentionPlan plan = TpAttentionPlan.context(spec, 2, uploader);
            group.init(spec, UBATCH, kernels, queue, placeholders, null, plan, mem, N_CTX);
            group.upload(store, uploader);
            GpuBlockRegistry registry = GpuFactoryCatalog.defaults().withMixerDecorator(group.attentionDecorator());
            GpuForwardBackend split = new GpuForwardBackend(mainContext, registry,
                    GpuFactoryCatalog.headPreprocessors());
            try {
                split.trunkRunner(group.pcpTrunkRunner(mainContext, split.layerExecutor()));
                runPrompt(split);
                double[] splitMs = timedReps("prefill context-parallel (2 ranks)", split);

                double wholeMedian = median(wholeMs);
                double splitMedian = median(splitMs);
                System.out.println(String.format(Locale.ROOT,
                        "prefill of %d tokens, ubatch %d, 8 layers x 256 embd: single device %.2f ms,"
                                + " prefill context-parallel 2 ranks %.2f ms (%.2fx). Both ranks run as two"
                                + " contexts of ONE physical OpenCL device, so the ranks serialize and the"
                                + " difference measures the split's exchanges and packing overhead, not"
                                + " multi-device scaling.",
                        PROMPT, UBATCH, wholeMedian, splitMedian, splitMedian / wholeMedian));
            } finally {
                split.close();
            }
        } finally {
            whole.close();
            if (group != null) {
                group.close();
            }
        }
    }
}
