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
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
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
import java.util.Arrays;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The shard content of a feed-forward row split upload on the OpenCL backend: rank {@code r}'s gate and up shards
 * hold rows {@code [r*s, (r+1)*s)} of the stored tensors, its down shard holds the matching column slice of every
 * stored row, and the shard map keys are the full tensor names of the projections only. The GPU cases are skipped
 * when no OpenCL device is available.
 */
public class FfnRowSplitUploaderTest {

    private static final int LAYERS = 2;
    private static final int RANKS = 2;
    private static final int N_FF = 128;
    private static final int N_EMBD = 64;
    private static final int SHARD = N_FF / RANKS;

    private Path file;
    private HWAcceleration hw;
    private OclComputeBackend backend;
    private GGUFFileParser parser;
    private WeightStore store;
    private GpuWeightUploader uploader;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(LAYERS).nFf(N_FF).build())
                .writeTempFile();
        parser = new GGUFFileParser(file.toString());
        hw = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hw.getCLDevices().isEmpty());
        backend = new OclComputeBackend(hw.getCLDevices().get(0));
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        uploader = new GpuWeightUploader(backend, backend.computeQueue(), parser.getTensors(), store, "llama",
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(LlmKernels.open(backend))).build());
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

    private ModelSpec spec() {
        return ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), MemoryConfig.defaults());
    }

    private FfnRowSplitPlan plan() {
        return FfnRowSplitPlan.of(spec(), RANKS, uploader);
    }

    private static float[] read(ComputeQueue queue, GpuBuffer buffer, int floats) {
        float[] out = new float[floats];
        try (GpuEvent e = queue.enqueueReadBuffer(buffer, 0, floats * (long) Float.BYTES, out)) {
            queue.finish();
        }
        return out;
    }

    /** The stored floats of one tensor's row range {@code [firstRow, firstRow + nRows)}. */
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

    /** One stored float of a tensor. */
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

    /** The stored floats of an F32 vector tensor. */
    private static float[] hostVector(GpuWeightUploader weights, WeightStore weightStore, String name)
            throws IOException {
        GpuWeightUploader.TensorPlan p = weights.plan(name);
        ByteBuffer staged = ByteBuffer.allocateDirect((int) p.sourceRowBytes()).order(ByteOrder.LITTLE_ENDIAN);
        weightStore.readTensorDirect(p.info().offset(), (int) p.sourceRowBytes(), staged);
        staged.flip();
        float[] out = new float[(int) p.cols()];
        staged.asFloatBuffer().get(out);
        return out;
    }

    @Test
    public void theRankShardsHoldTheirRowAndColumnRanges() {
        FfnRowSplitUploader shardUploader = new FfnRowSplitUploader(plan(), uploader, store);
        try {
            for (int rank = 0; rank < RANKS; rank++) {
                Map<String, GpuWeight> shards = shardUploader.uploadRank(rank, backend, backend.computeQueue());
                int firstRow = rank * SHARD;
                for (int il = 0; il < LAYERS; il++) {
                    assertArrayEquals("blk." + il + ".ffn_up rank " + rank,
                            hostRows("blk." + il + ".ffn_up.weight", firstRow, SHARD),
                            read(backend.computeQueue(), shards.get("blk." + il + ".ffn_up.weight").shard(0),
                                    SHARD * N_EMBD), 0f);
                    assertArrayEquals("blk." + il + ".ffn_gate rank " + rank,
                            hostRows("blk." + il + ".ffn_gate.weight", firstRow, SHARD),
                            read(backend.computeQueue(), shards.get("blk." + il + ".ffn_gate.weight").shard(0),
                                    SHARD * N_EMBD), 0f);
                    GpuWeight down = shards.get("blk." + il + ".ffn_down.weight");
                    assertEquals(GGMLType.F32, down.type());
                    assertArrayEquals(new long[]{SHARD, N_EMBD}, down.ne());
                    float[] got = read(backend.computeQueue(), down.shard(0), SHARD * N_EMBD);
                    for (int r = 0; r < N_EMBD; r++) {
                        for (int c = 0; c < SHARD; c++) {
                            assertEquals("blk." + il + ".ffn_down[" + r + "][" + (firstRow + c) + "] rank " + rank,
                                    hostAt("blk." + il + ".ffn_down.weight", r, firstRow + c), got[r * SHARD + c],
                                    0f);
                        }
                    }
                }
            }
        } finally {
            shardUploader.close();
        }
    }

    @Test
    public void theTensorSplitShardsHoldTheirProportionalRanges() {
        FfnRowSplitPlan uneven = FfnRowSplitPlan.of(spec(), RANKS, uploader, new double[]{3, 1});
        int[] starts = {0, 96};
        int[] sizes = {96, 32};
        FfnRowSplitUploader shardUploader = new FfnRowSplitUploader(uneven, uploader, store);
        try {
            for (int rank = 0; rank < RANKS; rank++) {
                Map<String, GpuWeight> shards = shardUploader.uploadRank(rank, backend, backend.computeQueue());
                int firstRow = starts[rank];
                int rows = sizes[rank];
                for (int il = 0; il < LAYERS; il++) {
                    assertArrayEquals("blk." + il + ".ffn_up rank " + rank,
                            hostRows("blk." + il + ".ffn_up.weight", firstRow, rows),
                            read(backend.computeQueue(), shards.get("blk." + il + ".ffn_up.weight").shard(0),
                                    rows * N_EMBD), 0f);
                    assertArrayEquals("blk." + il + ".ffn_gate rank " + rank,
                            hostRows("blk." + il + ".ffn_gate.weight", firstRow, rows),
                            read(backend.computeQueue(), shards.get("blk." + il + ".ffn_gate.weight").shard(0),
                                    rows * N_EMBD), 0f);
                    GpuWeight down = shards.get("blk." + il + ".ffn_down.weight");
                    assertArrayEquals(new long[]{rows, N_EMBD}, down.ne());
                    float[] got = read(backend.computeQueue(), down.shard(0), rows * N_EMBD);
                    for (int r = 0; r < N_EMBD; r++) {
                        for (int c = 0; c < rows; c++) {
                            assertEquals("blk." + il + ".ffn_down[" + r + "][" + (firstRow + c) + "] rank " + rank,
                                    hostAt("blk." + il + ".ffn_down.weight", r, firstRow + c), got[r * rows + c],
                                    0f);
                        }
                    }
                }
            }
        } finally {
            shardUploader.close();
        }
    }

    @Test
    public void theShardMapHoldsOnlyTheProjectionsUnderTheirFullNames() {
        FfnRowSplitUploader shardUploader = new FfnRowSplitUploader(plan(), uploader, store);
        try {
            Map<String, GpuWeight> shards = shardUploader.uploadRank(0, backend, backend.computeQueue());
            assertEquals(3 * LAYERS, shards.size());
            for (int il = 0; il < LAYERS; il++) {
                for (String suffix : new String[]{"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
                    GpuWeight w = shards.get("blk." + il + "." + suffix);
                    assertEquals("blk." + il + "." + suffix, w.name());
                    assertArrayEquals(w.name(), new long[]{N_EMBD, SHARD}, w.ne());
                }
            }
            assertFalse(shards.containsKey("blk.0.attn_q.weight"));
            assertNull(shards.get("blk.0.attn_output.weight"));
            assertTrue(shards.containsKey("blk.1.ffn_down.weight"));
        } finally {
            shardUploader.close();
        }
    }

    /**
     * An FFN with biases keeps the up bias sharded over the ranks (each rank adds its own hidden rows) and the
     * output-width down bias whole on rank 0 only: the all-reduce broadcasts rank 0's sum, so adding it on every
     * rank would count it once per device.
     */
    @Test
    public void theUpBiasIsShardedAndTheDownBiasIsAddedOnce() throws IOException {
        Path bertFile = SyntheticModels.bert(SyntheticModels.Config.bert().toBuilder().nLayer(1).build())
                .writeTempFile();
        try (GGUFFileParser bertParser = new GGUFFileParser(bertFile.toString());
             WeightStore bertStore = bertParser.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            GpuWeightUploader bertUploader = new GpuWeightUploader(backend, backend.computeQueue(),
                    bertParser.getTensors(), bertStore, "bert",
                    GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(LlmKernels.open(backend)))
                            .build());
            try {
                ModelSpec bertSpec = ModelSpecBuilder.build(GgufKv.of(bertParser),
                        TensorCatalog.of(bertParser), MemoryConfig.defaults());
                FfnRowSplitPlan bertPlan = FfnRowSplitPlan.of(bertSpec, RANKS, bertUploader);
                assertEquals("the bert FFN splits", 1, bertPlan.layers().size());
                FfnRowSplitUploader shardUploader = new FfnRowSplitUploader(bertPlan, bertUploader, bertStore);
                try {
                    Map<String, GpuWeight> rank0 = shardUploader.uploadRank(0, backend, backend.computeQueue());
                    Map<String, GpuWeight> rank1 = shardUploader.uploadRank(1, backend, backend.computeQueue());
                    String downBias = "blk.0.ffn_down.bias";
                    assertTrue("rank 0 carries the output-width bias", rank0.containsKey(downBias));
                    assertFalse("rank 1 must not add the output-width bias again", rank1.containsKey(downBias));
                    GpuWeight wholeDown = rank0.get(downBias);
                    assertArrayEquals(hostVector(bertUploader, bertStore, downBias),
                            read(backend.computeQueue(), wholeDown.shard(0), (int) wholeDown.cols()), 0f);

                    String upBias = "blk.0.ffn_up.bias";
                    float[] hostUp = hostVector(bertUploader, bertStore, upBias);
                    assertEquals(2 * SHARD, hostUp.length);
                    assertArrayEquals(Arrays.copyOfRange(hostUp, 0, SHARD),
                            read(backend.computeQueue(), rank0.get(upBias).shard(0), SHARD), 0f);
                    assertArrayEquals(Arrays.copyOfRange(hostUp, SHARD, 2 * SHARD),
                            read(backend.computeQueue(), rank1.get(upBias).shard(0), SHARD), 0f);
                } finally {
                    shardUploader.close();
                }
            } finally {
                bertUploader.close();
            }
        } finally {
            Files.deleteIfExists(bertFile);
        }
    }
}
