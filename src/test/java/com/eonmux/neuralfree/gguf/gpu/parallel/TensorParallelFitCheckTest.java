package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;

import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The per-rank VRAM fit check of a tensor split runs before any rank allocates and refuses a rank that cannot hold its
 * shard, naming the rank, the bytes it needs and the shortfall. The test device reports 8 MiB of total VRAM, below
 * the check's fixed reserve, so the upload refuses instead of allocating a shard that would not fit.
 */
public class TensorParallelFitCheckTest {

    private static final int UBATCH = 4;
    private static final int N_CTX = 32;

    private Path file;
    private GGUFFileParser parser;
    private FakeComputeBackend main;
    private FakeComputeBackend peer;
    private WeightStore store;
    private GpuWeightUploader uploader;
    private ModelSpec spec;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).build()).writeTempFile();
        parser = new GGUFFileParser(file.toString());
        DeviceCapabilities tiny = new DeviceCapabilities.Builder()
                .deviceName("TinyFakeDevice")
                .totalVramBytes(8L << 20)
                .maxAllocSizeBytes(4L << 20)
                .computeUnits(1)
                .maxWorkGroupSize(256)
                .build();
        main = FakeComputeBackend.builder().capabilities(tiny).build();
        peer = FakeComputeBackend.builder().capabilities(tiny).build();
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        uploader = new GpuWeightUploader(main, main.computeQueue(), parser.getTensors(), store, "llama",
                GpuWeightUploader.config((type, use) -> true).build());
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), MemoryConfig.defaults());
    }

    @After
    public void close() throws IOException {
        try {
            uploader.close();
            store.close();
            main.close();
            peer.close();
            parser.close();
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void aRankThatCannotHoldItsShardIsRefusedWithTheShortfall() {
        FfnRowSplitPlan ffn = FfnRowSplitPlan.of(spec, 2, uploader);
        TpAttentionPlan attention = TpAttentionPlan.of(spec, 2, uploader);
        TensorParallelGroup group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same tiny device")));
        try {
            group.init(spec, UBATCH, LlmKernels.open(main), main.computeQueue(),
                    GpuPlaceholders.allocate(main, main.computeQueue()), ffn, attention,
                    MemoryConfig.defaults().toBuilder().ubatch(UBATCH).build(), N_CTX);
            GpuOutOfMemoryException e = assertThrows(GpuOutOfMemoryException.class,
                    () -> group.upload(store, uploader));
            assertTrue(e.getMessage(), e.getMessage().contains("rank0 (the model device)"));
            assertTrue(e.getMessage(), e.getMessage().contains("MiB"));
            assertTrue(e.getMessage(), e.getMessage().contains("short by"));
        } finally {
            try {
                group.close();
            } catch (IOException ignored) {
                // the failed initialization left nothing to release
            }
        }
    }
}
