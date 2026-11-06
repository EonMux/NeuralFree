package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
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
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The per-rank byte estimate of the VRAM fit check against the shards the uploaders actually allocate: for a
 * synthetic llama and two ranks, {@link ShardBytes#ffn} and {@link ShardBytes#attention} equal the summed device bytes
 * of the rank's uploaded weight tensors, so a fit check that passes does not hide a larger upload. The GPU cases are
 * skipped when no OpenCL device is available.
 */
public class ShardBytesTest {

    private static final int RANKS = 2;

    private Path file;
    private HWAcceleration hw;
    private OclComputeBackend backend;
    private GGUFFileParser parser;
    private WeightStore store;
    private GpuWeightUploader uploader;
    private ModelSpec spec;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).build()).writeTempFile();
        parser = new GGUFFileParser(file.toString());
        hw = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hw.getCLDevices().isEmpty());
        backend = new OclComputeBackend(hw.getCLDevices().get(0));
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        uploader = new GpuWeightUploader(backend, backend.computeQueue(), parser.getTensors(), store, "llama",
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(LlmKernels.open(backend))).build());
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), MemoryConfig.defaults());
    }

    @After
    public void close() throws IOException {
        try {
            uploader.close();
            store.close();
            backend.close();
            parser.close();
        } finally {
            if (hw != null) {
                hw.shutdown();
            }
            Files.deleteIfExists(file);
        }
    }

    private static long bytes(Map<String, GpuWeight> shard) {
        long total = 0;
        for (GpuWeight weight : shard.values()) {
            total += weight.sizeInBytes();
        }
        return total;
    }

    @Test
    public void theAttentionEstimateMatchesTheUploadedShards() {
        TpAttentionPlan plan = TpAttentionPlan.of(spec, RANKS, uploader);
        TpAttentionUploader shardUploader = new TpAttentionUploader(plan, uploader, store);
        try {
            for (int rank = 0; rank < RANKS; rank++) {
                Map<String, GpuWeight> shard = shardUploader.uploadRank(rank, backend, backend.computeQueue());
                assertEquals("rank " + rank + " attention bytes", bytes(shard),
                        ShardBytes.attention(plan, rank, uploader));
            }
        } finally {
            shardUploader.close();
        }
    }

    @Test
    public void theFeedForwardEstimateMatchesTheUploadedShards() {
        FfnRowSplitPlan plan = FfnRowSplitPlan.of(spec, RANKS, uploader);
        FfnRowSplitUploader shardUploader = new FfnRowSplitUploader(plan, uploader, store);
        try {
            for (int rank = 0; rank < RANKS; rank++) {
                Map<String, GpuWeight> shard = shardUploader.uploadRank(rank, backend, backend.computeQueue());
                assertEquals("rank " + rank + " feed-forward bytes", bytes(shard),
                        ShardBytes.ffn(plan, rank, uploader));
            }
        } finally {
            shardUploader.close();
        }
    }

    @Test
    public void theFeedForwardEstimateMatchesUnevenTensorSplitShards() {
        FfnRowSplitPlan plan = FfnRowSplitPlan.of(spec, RANKS, uploader, new double[]{3, 1});
        FfnRowSplitUploader shardUploader = new FfnRowSplitUploader(plan, uploader, store);
        try {
            for (int rank = 0; rank < RANKS; rank++) {
                Map<String, GpuWeight> shard = shardUploader.uploadRank(rank, backend, backend.computeQueue());
                assertEquals("rank " + rank + " uneven feed-forward bytes", bytes(shard),
                        ShardBytes.ffn(plan, rank, uploader));
            }
        } finally {
            shardUploader.close();
        }
    }

    @Test
    public void thePerTensorBytesMatchTheUploadedShards() {
        TpAttentionPlan attention = TpAttentionPlan.of(spec, RANKS, uploader);
        FfnRowSplitPlan ffn = FfnRowSplitPlan.of(spec, RANKS, uploader, new double[]{3, 1});
        TpAttentionUploader attentionUploader = new TpAttentionUploader(attention, uploader, store);
        FfnRowSplitUploader ffnUploader = new FfnRowSplitUploader(ffn, uploader, store);
        try {
            for (int rank = 0; rank < RANKS; rank++) {
                Map<String, Long> exact = ShardBytes.splitShards(ffn, attention, rank, uploader);
                Map<String, GpuWeight> uploaded = new java.util.LinkedHashMap<>(
                        attentionUploader.uploadRank(rank, backend, backend.computeQueue()));
                uploaded.putAll(ffnUploader.uploadRank(rank, backend, backend.computeQueue()));
                for (String name : exact.keySet()) {
                    GpuWeight weight = uploaded.get(name);
                    assertEquals("rank " + rank + " bytes of " + name,
                            weight == null ? 0L : weight.sizeInBytes(), exact.get(name).longValue());
                }
                for (String name : uploaded.keySet()) {
                    assertTrue("rank " + rank + " accounts " + name, exact.containsKey(name));
                }
            }
        } finally {
            attentionUploader.close();
            ffnUploader.close();
        }
    }

    @Test
    public void thePlacementAccountsTheExactMainDeviceShard() {
        TpAttentionPlan attention = TpAttentionPlan.of(spec, RANKS, uploader);
        FfnRowSplitPlan ffn = FfnRowSplitPlan.of(spec, RANKS, uploader, new double[]{3, 1});
        Map<String, Long> exact = ShardBytes.splitShards(ffn, attention, 0, uploader);
        long whole = uploader.plannedBytes(t -> true);
        long splitTensors = 0;
        long shards = 0;
        for (java.util.Map.Entry<String, Long> entry : exact.entrySet()) {
            splitTensors += uploader.plan(entry.getKey()).deviceBytes();
            shards += entry.getValue();
        }
        uploader.splitShards(exact);
        assertEquals("the placement counts each split tensor's rank-0 shard instead of the whole tensor",
                whole - splitTensors + shards, uploader.plannedBytes(t -> true));
    }
}
