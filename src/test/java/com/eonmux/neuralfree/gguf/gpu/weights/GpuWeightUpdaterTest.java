package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.WeightUpdateTensor;
import com.eonmux.neuralfree.gguf.engine.WeightUpdater;
import com.eonmux.neuralfree.gguf.quant.RowQuantizers;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Live weight updates over the resident device tensors of a {@link GpuWeightUploader}: an update writes its bytes
 * into the existing shard buffers (raw when the dtype is the device type, decoded to F32 when the device copy is
 * dequantized), refuses a wrong tensor, shape or dtype by name, and finishes the queue before it returns.
 */
public class GpuWeightUpdaterTest {

    private static final int COLS = 64;
    private static final int ROWS = 32;

    private FakeComputeBackend backend;
    private GpuWeightUploader uploader;

    @After
    public void tearDown() {
        if (uploader != null) {
            uploader.close();
            uploader = null;
        }
        if (backend != null) {
            backend.close();
            backend = null;
        }
    }

    /** Five tensors over one blob: F32 matmul, native Q8_0, Q4_0 decoded on upload, an F32 norm, the output head. */
    private GpuWeightUploader openUploadedListing() {
        return openListing((type, use) -> type == GGMLType.F32 || type == GGMLType.Q8_0, t -> false);
    }

    private GpuWeightUploader openListing(GpuWeightUploader.TypePolicy policy,
                                          java.util.function.Predicate<GGUFTensorInfo> deferred) {
        backend = FakeComputeBackend.create();
        List<GGUFTensorInfo> tensors = Arrays.asList(
                new GGUFTensorInfo("blk.0.attn_q.weight", new long[]{COLS, ROWS}, GGMLType.F32, 0L),
                new GGUFTensorInfo("blk.0.ffn_down.weight", new long[]{COLS, ROWS}, GGMLType.Q8_0, offset(8192)),
                new GGUFTensorInfo("blk.0.ffn_gate.weight", new long[]{COLS, ROWS}, GGMLType.Q4_0, offset(8192 + 2176)),
                new GGUFTensorInfo("blk.0.attn_norm.weight", new long[]{COLS}, GGMLType.F32, offset(8192 + 2176 + 1152)),
                new GGUFTensorInfo("lm_head.weight", new long[]{COLS, 8}, GGMLType.F32,
                        offset(8192 + 2176 + 1152 + 256)));
        uploader = new GpuWeightUploader(backend, backend.computeQueue(), tensors, new PatternStore(),
                "llama", GpuWeightUploader.config(policy).build());
        uploader.uploadResident(deferred);
        return uploader;
    }

    private static long offset(long v) {
        return v;
    }

    /** Byte count of a whole tensor of {@code rows} rows of {@code cols} elements. */
    private static long bytes(GGMLType type, int rows) {
        return type.getBytesForElements((long) rows * COLS);
    }

    private static float[] updateFloats() {
        float[] fs = new float[ROWS * COLS];
        for (int i = 0; i < fs.length; i++) {
            fs[i] = ((i % 16) - 8) * 0.04f;
        }
        return fs;
    }

    /** The rows of {@code values} encoded in {@code type}, row-major over {@code ne[1..]}. */
    private static byte[] quantizedRows(GGMLType type, float[] values, int rows) {
        int rowBytes = RowQuantizers.bytesPerRow(type, COLS);
        byte[] out = new byte[rowBytes * rows];
        for (int r = 0; r < rows; r++) {
            RowQuantizers.quantizeRow(type, values, r * COLS, out, r * rowBytes, COLS, null);
        }
        return out;
    }

    private static ByteBuffer le(float[] values) {
        return WeightUpdateTensor.f32("scratch", new long[]{values.length}, values).dataBuffer();
    }

    private static FakeGpuBuffer buffer(GpuWeightUploader up, String name) {
        return (FakeGpuBuffer) up.find(name).shard(0);
    }

    @Test
    public void anF32UpdateOfAnF32TensorLandsInItsBuffer() {
        GpuWeightUploader up = openUploadedListing();
        float[] values = new float[ROWS * COLS];
        for (int i = 0; i < values.length; i++) {
            values[i] = i * 0.5f;
        }
        up.updateTensor("blk.0.attn_q.weight", GGMLType.F32, new long[]{COLS, ROWS}, le(values));
        assertArrayEquals(values, buffer(up, "blk.0.attn_q.weight").getFloats(0, values.length), 0f);
    }

    @Test
    public void aStoredTypeUpdateOfADecodedTensorDecodesIntoItsBuffer() {
        GpuWeightUploader up = openUploadedListing();
        assertEquals("the fixture tensor is decoded on upload", GGMLType.F32,
                up.deviceType("blk.0.ffn_gate.weight"));
        float[] values = updateFloats();
        byte[] raw = quantizedRows(GGMLType.Q4_0, values, ROWS);
        up.updateTensor("blk.0.ffn_gate.weight", GGMLType.Q4_0, new long[]{COLS, ROWS},
                ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN));
        assertArrayEquals("the roundtrip of the independent encoder and decoder", values,
                buffer(up, "blk.0.ffn_gate.weight").getFloats(0, values.length), 0.02f);
    }

    @Test
    public void anF32UpdateOfADecodedTensorWritesTheFloatsRaw() {
        GpuWeightUploader up = openUploadedListing();
        float[] values = updateFloats();
        up.updateTensor("blk.0.ffn_gate.weight", GGMLType.F32, new long[]{COLS, ROWS}, le(values));
        assertArrayEquals(values, buffer(up, "blk.0.ffn_gate.weight").getFloats(0, values.length), 0f);
    }

    @Test
    public void aStoredTypeUpdateOfANativeQuantTensorWritesItsBytesRaw() {
        GpuWeightUploader up = openUploadedListing();
        assertEquals(GGMLType.Q8_0, up.deviceType("blk.0.ffn_down.weight"));
        byte[] raw = quantizedRows(GGMLType.Q8_0, updateFloats(), ROWS);
        up.updateTensor("blk.0.ffn_down.weight", GGMLType.Q8_0, new long[]{COLS, ROWS},
                ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN));
        assertArrayEquals("the raw bytes land unchanged", raw, buffer(up, "blk.0.ffn_down.weight").toByteArray());
    }

    @Test
    public void anF32UpdateForANativeQuantTensorIsRefused() {
        GpuWeightUploader up = openUploadedListing();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> up.updateTensor(
                "blk.0.ffn_down.weight", GGMLType.F32, new long[]{COLS, ROWS},
                le(new float[ROWS * COLS])));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.0.ffn_down.weight")
                && e.getMessage().contains("would requantize") && e.getMessage().contains("Q8_0"));
    }

    @Test
    public void anUnknownDtypeUpdateIsRefused() {
        GpuWeightUploader up = openUploadedListing();
        byte[] bf16 = new byte[(int) bytes(GGMLType.BF16, ROWS)];
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> up.updateTensor(
                "blk.0.attn_q.weight", GGMLType.BF16, new long[]{COLS, ROWS},
                ByteBuffer.wrap(bf16).order(ByteOrder.LITTLE_ENDIAN)));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.0.attn_q.weight") && e.getMessage().contains("BF16"));
    }

    @Test
    public void aWrongShapeIsRefused() {
        GpuWeightUploader up = openUploadedListing();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> up.updateTensor(
                "blk.0.attn_q.weight", GGMLType.F32, new long[]{COLS, 16}, le(new float[16 * COLS])));
        assertTrue(e.getMessage(), e.getMessage().contains("[64, 32]") && e.getMessage().contains("[64, 16]"));
    }

    @Test
    public void anUnknownTensorIsRefusedByName() {
        GpuWeightUploader up = openUploadedListing();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> up.updateTensor(
                "blk.9.attn_q.weight", GGMLType.F32, new long[]{COLS, ROWS}, le(new float[ROWS * COLS])));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.9.attn_q.weight"));
    }

    @Test
    public void aNonResidentTensorIsRefusedByName() {
        GpuWeightUploader up = openListing((type, use) -> type == GGMLType.F32 || type == GGMLType.Q8_0,
                t -> t.name().equals("blk.0.ffn_gate.weight"));
        byte[] raw = quantizedRows(GGMLType.Q4_0, updateFloats(), ROWS);
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class, () -> up.updateTensor(
                "blk.0.ffn_gate.weight", GGMLType.Q4_0, new long[]{COLS, ROWS},
                ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.0.ffn_gate.weight")
                && e.getMessage().contains("not resident"));
    }

    @Test
    public void anUpdateFinishesTheQueueBeforeItReturns() {
        GpuWeightUploader up = openUploadedListing();
        int opsBefore = backend.ops().size();
        up.updateTensor("blk.0.attn_norm.weight", GGMLType.F32, new long[]{COLS}, le(fill(COLS, 2f)));
        List<FakeComputeBackend.Op> ops = backend.ops();
        assertEquals("the last op is the finish", FakeComputeBackend.OpKind.FINISH, ops.get(ops.size() - 1).kind());
        assertTrue(ops.size() > opsBefore);
        assertArrayEquals(fill(COLS, 2f), buffer(up, "blk.0.attn_norm.weight").getFloats(0, COLS), 0f);
    }

    @Test
    public void updatesAddressModelAliasesOfTheOutputHead() {
        GpuWeightUploader up = openUploadedListing();
        GpuWeightUpdater g = new GpuWeightUpdater(up, "llama");
        float[] values = fill(8 * COLS, 0.25f);
        g.apply(WeightUpdateTensor.f32("output.weight", new long[]{COLS, 8}, values));
        assertArrayEquals("output.weight resolves to the stored lm_head.weight",
                values, buffer(up, "lm_head.weight").getFloats(0, values.length), 0f);
    }

    @Test
    public void theUpdaterFollowsTheTransferLifecycle() {
        GpuWeightUploader up = openUploadedListing();
        GpuWeightUpdater g = new GpuWeightUpdater(up, "llama");
        assertEquals(WeightUpdater.DEFAULT_VERSION, g.version());
        assertFalse(g.updating());

        g.start();
        assertTrue(g.updating());
        g.apply(WeightUpdateTensor.f32("blk.0.attn_norm.weight", new long[]{COLS}, fill(COLS, 4f)));
        g.finish("rl-step-9");
        assertFalse(g.updating());
        assertEquals("rl-step-9", g.version());

        g.setVersion("v12");
        assertEquals("v12", g.version());

        g.shutdown();
        IllegalStateException e = assertThrows(IllegalStateException.class, g::start);
        assertTrue(e.getMessage(), e.getMessage().contains("shut down"));
        e = assertThrows(IllegalStateException.class, () -> g.apply(WeightUpdateTensor.f32(
                "blk.0.attn_norm.weight", new long[]{COLS}, fill(COLS, 4f))));
        assertTrue(e.getMessage(), e.getMessage().contains("shut down"));
    }

    @Test
    public void anUpdateAfterTheUploaderClosedIsRefused() throws IOException {
        GpuWeightUploader up = openUploadedListing();
        GpuWeightUpdater g = new GpuWeightUpdater(up, "llama");
        up.close();
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> g.apply(WeightUpdateTensor.f32(
                "blk.0.attn_norm.weight", new long[]{COLS}, fill(COLS, 4f))));
        assertTrue(e.getMessage(), e.getMessage().contains("closed"));
    }

    private static float[] fill(int n, float v) {
        float[] fs = new float[n];
        Arrays.fill(fs, v);
        return fs;
    }

    /** A store of one blob whose every byte is a distinct pattern, so stale contents never pass for an update. */
    private static final class PatternStore implements WeightStore {

        private static final byte PATTERN = 0x11;

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) {
            readTensor(tensorOffset, size, dst, 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) {
            for (int i = 0; i < size; i++) {
                dst[dstOffset + i] = PATTERN;
            }
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) {
            for (int i = 0; i < size; i++) {
                dst.put(PATTERN);
            }
        }

        @Override
        public void close() {
        }
    }
}
