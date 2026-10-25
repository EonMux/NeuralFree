package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.FileTensorRegions;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;

import static com.eonmux.neuralfree.io.safetensors.SafetensorsTestFiles.ramp;
import static com.eonmux.neuralfree.io.safetensors.SafetensorsTestFiles.writeRaw;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link StRowSource} and {@link ElementFormat}: rows of tensors at unaligned data offsets (begin 1 and 3) read
 * through the row source and through {@code FileTensorRegions} over the checkpoint's segment in both read modes, row
 * geometry of scalars, block and sub-byte formats, and bounds errors.
 */
public class StRowSourceTest {

    /** {@code a} U8 [1] at [0, 1), {@code b} F16 [1] at [1, 3), {@code c} F32 [2, 3] at [3, 27). */
    private static final String HEADER = "{"
            + "\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
            + "\"b\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[1,3]},"
            + "\"c\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[3,27]},"
            + "\"s\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[27,31]},"
            + "\"q\":{\"dtype\":\"F4\",\"shape\":[2,3],\"data_offsets\":[31,34]}}";
    private static final byte[] DATA = ramp(10, 34);

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path unaligned() throws IOException {
        Path p = tmp.getRoot().toPath().resolve("model.safetensors");
        writeRaw(p, HEADER, DATA);
        return p;
    }

    @Test
    public void readsRowsAtUnalignedOffsets() throws IOException {
        Path p = unaligned();
        try (SafetensorsCheckpoint ckpt = SafetensorsCheckpoint.open(p, p.getParent(), false)) {
            RowSource b = ckpt.source("b");
            assertEquals(1, b.directVirtualOffset());
            assertEquals(1, b.rows());
            assertEquals(2, b.rowBytes());
            assertArrayEquals(Arrays.copyOfRange(DATA, 1, 3), read(b, 0, 1));

            RowSource c = ckpt.source("c");
            assertEquals(3, c.directVirtualOffset());
            assertEquals(ElementFormat.of(StDType.F32), c.format());
            assertArrayEquals(new long[]{2, 3}, c.shape());
            assertEquals(2, c.rows());
            assertEquals(12, c.rowBytes());
            assertTrue(c.rowLocal());
            assertArrayEquals(Arrays.copyOfRange(DATA, 3, 27), read(c, 0, 2));
            assertArrayEquals(Arrays.copyOfRange(DATA, 15, 27), read(c, 1, 1));
            assertNull(c.outputScale());
            assertNull(c.actQuant());
            assertThrows(UnsupportedOperationException.class, c::foldInputPerm);
            assertThrows(UnsupportedOperationException.class, c::foldOutputScale);
        }
    }

    @Test
    public void readsTheSameBytesThroughRegionsOverTheSegment() throws IOException {
        Path p = unaligned();
        try (SafetensorsCheckpoint ckpt = SafetensorsCheckpoint.open(p, p.getParent(), false)) {
            assertEquals(1, ckpt.segments().size());
            for (TensorReadMode mode : TensorReadMode.values()) {
                try (FileTensorRegions regions = FileTensorRegions.open(ckpt.segments(), mode, "unaligned")) {
                    GGUFTensorInfo b = new GGUFTensorInfo("b", new long[]{1}, GGMLType.F16,
                            ckpt.directVirtualOffset("b"));
                    assertArrayEquals(Arrays.copyOfRange(DATA, 1, 3), bytes(regions.region(b, 0, 2)));
                    GGUFTensorInfo c = new GGUFTensorInfo("c", new long[]{3, 2}, GGMLType.F32,
                            ckpt.directVirtualOffset("c"));
                    ByteBuffer row1 = regions.region(c, 12, 12);
                    assertEquals(ByteOrder.LITTLE_ENDIAN, row1.order());
                    assertArrayEquals(Arrays.copyOfRange(DATA, 15, 27), bytes(row1));
                    float expected = ByteBuffer.wrap(DATA, 3, 4).order(ByteOrder.LITTLE_ENDIAN).getFloat();
                    assertEquals(expected, regions.region(c, 0, 4).getFloat(0), 0f);
                }
            }
        }
    }

    @Test
    public void scalarIsOneRowOfOneElement() throws IOException {
        Path p = unaligned();
        try (SafetensorsCheckpoint ckpt = SafetensorsCheckpoint.open(p, p.getParent(), false)) {
            RowSource s = ckpt.source("s");
            assertEquals(0, s.shape().length);
            assertEquals(1, s.rows());
            assertEquals(4, s.rowBytes());
            assertArrayEquals(Arrays.copyOfRange(DATA, 27, 31), read(s, 0, 1));
        }
    }

    @Test
    public void subByteRowsThatAreNotWholeBytesAreRefusedWhenRead() throws IOException {
        Path p = unaligned();
        try (SafetensorsCheckpoint ckpt = SafetensorsCheckpoint.open(p, p.getParent(), false)) {
            RowSource q = ckpt.source("q");
            assertEquals(2, q.rows());
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, q::rowBytes);
            assertTrue(e.getMessage(), e.getMessage().contains("tensor 'q'"));
            assertThrows(IllegalArgumentException.class, () -> q.readRows(0, 1, ByteBuffer.allocate(8)));
        }
    }

    @Test
    public void rejectsRowsOutsideTheTensorAndShortBuffers() throws IOException {
        Path p = unaligned();
        try (SafetensorsCheckpoint ckpt = SafetensorsCheckpoint.open(p, p.getParent(), false)) {
            RowSource c = ckpt.source("c");
            assertThrows(IllegalArgumentException.class, () -> c.readRows(1, 2, ByteBuffer.allocate(24)));
            assertThrows(IllegalArgumentException.class, () -> c.readRows(-1, 1, ByteBuffer.allocate(12)));
            assertThrows(IllegalArgumentException.class, () -> c.readRows(0, 1, ByteBuffer.allocate(11)));
            ByteBuffer dst = ByteBuffer.allocate(20);
            dst.position(4);
            c.readRows(1, 1, dst);
            assertEquals(16, dst.position());
            assertEquals(20, dst.limit());
        }
    }

    @Test
    public void sourceOutsideACheckpointIsNotDirect() throws IOException {
        Path p = unaligned();
        try (SafetensorsFile f = SafetensorsFile.open(p)) {
            StTensor t = f.tensor("c");
            StRowSource s = StRowSource.of(f, t);
            assertEquals(-1, s.directVirtualOffset());
            assertSame(t, s.tensor());
            assertTrue(s.describe().contains("tensor 'c' (F32 [2, 3])"));
            assertArrayEquals(Arrays.copyOfRange(DATA, 3, 15), read(s, 0, 1));
            StTensor foreign = new StTensor("c", StDType.F32, new long[]{6}, 3, 24);
            assertThrows(IllegalArgumentException.class, () -> StRowSource.of(f, foreign));
        }
    }

    @Test
    public void elementFormatsDescribeRowsAndTypes() {
        ElementFormat f32 = ElementFormat.of(StDType.F32);
        assertEquals(ElementFormat.of(GGMLType.F32), f32);
        assertSame(StDType.F32, ElementFormat.of(GGMLType.F32).stDType());
        assertTrue(f32.isFloat());
        assertFalse(f32.isBlock());
        assertEquals(16, f32.rowBytes(4));
        assertEquals("F32", f32.toString());

        ElementFormat q8 = ElementFormat.of(GGMLType.Q8_0);
        assertNull(q8.stDType());
        assertTrue(q8.isBlock());
        assertFalse(q8.isFloat());
        assertEquals(68, q8.rowBytes(64));
        assertThrows(IllegalArgumentException.class, () -> q8.rowBytes(48));
        assertEquals("Q8_0", q8.toString());

        ElementFormat fp8 = ElementFormat.of(StDType.F8_E4M3);
        assertNull(fp8.ggmlType());
        assertFalse(fp8.isFloat());
        assertEquals(5, fp8.rowBytes(5));

        ElementFormat f4 = ElementFormat.of(StDType.F4);
        assertEquals(2, f4.rowBytes(4));
        assertThrows(IllegalArgumentException.class, () -> f4.rowBytes(3));
        assertThrows(IllegalArgumentException.class, () -> ElementFormat.of(GGMLType.UNKNOWN));
        assertThrows(IllegalArgumentException.class, () -> f32.rowBytes(-1));
        assertThrows(IllegalArgumentException.class, () -> f32.rowBytes(Long.MAX_VALUE / 2));
    }

    private static byte[] read(RowSource s, long first, int count) throws IOException {
        ByteBuffer dst = ByteBuffer.allocate((int) (s.rowBytes() * count));
        s.readRows(first, count, dst);
        assertFalse(dst.hasRemaining());
        return dst.array();
    }

    private static byte[] bytes(ByteBuffer b) {
        byte[] out = new byte[b.remaining()];
        b.get(out);
        return out;
    }
}
