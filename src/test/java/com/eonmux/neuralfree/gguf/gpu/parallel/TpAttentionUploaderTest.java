package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGMLType;
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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The shard content of a tensor-parallel attention upload on the OpenCL backend: rank {@code r}'s query, key and value
 * shards hold the stored rows of its heads, its output shard holds the matching column block of every stored row, and
 * the shard map keys are the full tensor names of the attention projections. The GPU cases are skipped when no OpenCL
 * device is available.
 */
public class TpAttentionUploaderTest {

    private static final int LAYERS = 2;
    private static final int RANKS = 2;
    private static final int N_EMBD = 64;
    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int HEAD_DIM = 16;
    private static final int Q_SHARD = N_HEAD / RANKS * HEAD_DIM;
    private static final int KV_SHARD = N_HEAD_KV / RANKS * HEAD_DIM;

    private Path file;
    private HWAcceleration hw;
    private OclComputeBackend backend;
    private GGUFFileParser parser;
    private WeightStore store;
    private GpuWeightUploader uploader;
    private ModelSpec spec;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(LAYERS).build()).writeTempFile();
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

    private static float[] read(ComputeQueue queue, GpuBuffer buffer, int floats) {
        float[] out = new float[floats];
        try (GpuEvent e = queue.enqueueReadBuffer(buffer, 0, floats * (long) Float.BYTES, out)) {
            queue.finish();
        }
        return out;
    }

    private float[] hostRows(String name, long firstRow, long nRows) {
        GpuWeightUploader.TensorPlan p = uploader.plan(name);
        float[] out = new float[(int) (nRows * p.cols())];
        ByteBuffer staged = ByteBuffer.allocateDirect((int) p.sourceRowBytes()).order(ByteOrder.LITTLE_ENDIAN);
        float[] row = new float[(int) p.cols()];
        for (long r = 0; r < nRows; r++) {
            staged.clear();
            try {
                store.readTensorDirect(p.info().offset() + (firstRow + r) * p.sourceRowBytes(),
                        (int) p.sourceRowBytes(), staged);
            } catch (IOException e) {
                throw new IllegalStateException("cannot read " + name, e);
            }
            staged.flip();
            staged.asFloatBuffer().get(row);
            System.arraycopy(row, 0, out, (int) (r * p.cols()), (int) p.cols());
        }
        return out;
    }

    private float hostAt(String name, long row, long col) {
        GpuWeightUploader.TensorPlan p = uploader.plan(name);
        ByteBuffer staged = ByteBuffer.allocateDirect((int) p.sourceRowBytes()).order(ByteOrder.LITTLE_ENDIAN);
        try {
            store.readTensorDirect(p.info().offset() + row * p.sourceRowBytes(), (int) p.sourceRowBytes(), staged);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read " + name, e);
        }
        return staged.getFloat((int) (col * Float.BYTES));
    }

    @Test
    public void theAttentionShardsHoldTheirHeadRowsAndOutputColumns() {
        TpAttentionPlan plan = TpAttentionPlan.of(spec, RANKS, uploader);
        TpAttentionUploader shardUploader = new TpAttentionUploader(plan, uploader, store);
        try {
            for (int rank = 0; rank < RANKS; rank++) {
                Map<String, GpuWeight> shards = shardUploader.uploadRank(rank, backend, backend.computeQueue());
                int firstHead = rank * (N_HEAD / RANKS);
                int firstKvHead = rank * (N_HEAD_KV / RANKS);
                for (int il = 0; il < LAYERS; il++) {
                    GpuWeight q = shards.get("blk." + il + ".attn_q.weight");
                    assertArrayEquals(new long[]{N_EMBD, Q_SHARD}, q.ne());
                    assertArrayEquals("blk." + il + ".attn_q rank " + rank,
                            hostRows("blk." + il + ".attn_q.weight", (long) firstHead * HEAD_DIM, Q_SHARD),
                            read(backend.computeQueue(), q.shard(0), Q_SHARD * N_EMBD), 0f);

                    GpuWeight k = shards.get("blk." + il + ".attn_k.weight");
                    assertArrayEquals(new long[]{N_EMBD, KV_SHARD}, k.ne());
                    assertArrayEquals("blk." + il + ".attn_k rank " + rank,
                            hostRows("blk." + il + ".attn_k.weight", (long) firstKvHead * HEAD_DIM, KV_SHARD),
                            read(backend.computeQueue(), k.shard(0), KV_SHARD * N_EMBD), 0f);

                    GpuWeight v = shards.get("blk." + il + ".attn_v.weight");
                    assertArrayEquals(new long[]{N_EMBD, KV_SHARD}, v.ne());
                    assertArrayEquals("blk." + il + ".attn_v rank " + rank,
                            hostRows("blk." + il + ".attn_v.weight", (long) firstKvHead * HEAD_DIM, KV_SHARD),
                            read(backend.computeQueue(), v.shard(0), KV_SHARD * N_EMBD), 0f);

                    GpuWeight out = shards.get("blk." + il + ".attn_output.weight");
                    assertEquals(GGMLType.F32, out.type());
                    assertArrayEquals(new long[]{Q_SHARD, N_EMBD}, out.ne());
                    float[] got = read(backend.computeQueue(), out.shard(0), Q_SHARD * N_EMBD);
                    for (int r = 0; r < N_EMBD; r++) {
                        for (int c = 0; c < Q_SHARD; c++) {
                            assertEquals("blk." + il + ".attn_output[" + r + "][" + (firstHead * HEAD_DIM + c)
                                            + "] rank " + rank,
                                    hostAt("blk." + il + ".attn_output.weight", r, firstHead * HEAD_DIM + c),
                                    got[r * Q_SHARD + c], 0f);
                        }
                    }
                }
            }
        } finally {
            shardUploader.close();
        }
    }

    @Test
    public void theShardMapHoldsOnlyTheAttentionProjections() {
        TpAttentionPlan plan = TpAttentionPlan.of(spec, RANKS, uploader);
        TpAttentionUploader shardUploader = new TpAttentionUploader(plan, uploader, store);
        try {
            Map<String, GpuWeight> shards = shardUploader.uploadRank(0, backend, backend.computeQueue());
            assertEquals(4 * LAYERS, shards.size());
            for (int il = 0; il < LAYERS; il++) {
                for (String suffix : new String[]{"attn_q.weight", "attn_k.weight", "attn_v.weight",
                        "attn_output.weight"}) {
                    assertTrue(suffix, shards.containsKey("blk." + il + "." + suffix));
                }
            }
            assertFalse(shards.containsKey("blk.0.ffn_up.weight"));
            assertNull(shards.get("blk.0.attn_q.bias"));
        } finally {
            shardUploader.close();
        }
    }
}
