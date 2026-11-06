package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.StateTransfer;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * A cached block of a per-token-head cache carries the per-(token, KV head) scales of its rows beside the codes: the
 * host image keeps them, a shared head cuts them with the rows, and they install back into a fresh cache.
 */
public class BlockImageSidesTest {

    private static final int ROWS = 8;
    private static final int HEADS = 2;
    private static final int WIDTH = 4;
    private static final int SCALE_ROW = HEADS * Float.BYTES;

    private static KvLayout layout(KvDType k, KvDType v) {
        return KvLayout.of(ROWS, false, WIDTH, WIDTH, k, v, KvWindow.FULL_CAUSAL).withHeads(HEADS);
    }

    /** A cache's stored bytes without a backend: K/V rows and the one scale plane of a per-token-head type. */
    private static final class FakeCache implements StateTransfer.Cache {
        private final KvLayout layout;
        private final byte[] k;
        private final byte[] v;
        private final byte[] kScales;
        private final byte[] vScales;

        FakeCache(KvLayout layout) {
            this.layout = layout;
            this.k = new byte[Math.toIntExact(ROWS * layout.rowBytesK())];
            this.v = layout.vAliasesK() ? null : new byte[Math.toIntExact(ROWS * layout.rowBytesV())];
            this.kScales = layout.dtypeK().perTokenHead() ? new byte[ROWS * SCALE_ROW] : new byte[0];
            this.vScales = layout.dtypeV().perTokenHead() && !layout.vAliasesK() ? new byte[ROWS * SCALE_ROW]
                    : new byte[0];
        }

        @Override
        public int layer() {
            return 0;
        }

        @Override
        public int group() {
            return 0;
        }

        @Override
        public KvLayout layout() {
            return layout;
        }

        @Override
        public void read(boolean v, int row0, int rows, byte[] dst) {
            System.arraycopy(v ? this.v : k, row0 * rowBytes(v), dst, 0, rows * rowBytes(v));
        }

        @Override
        public void write(boolean v, int row0, int rows, byte[] src) {
            System.arraycopy(src, 0, v ? this.v : k, row0 * rowBytes(v), rows * rowBytes(v));
        }

        @Override
        public List<KvDType.SidePlane> sidePlanes(boolean v) {
            return (v ? layout.dtypeV() : layout.dtypeK()).sidePlanes();
        }

        @Override
        public void readSide(boolean v, int plane, int row0, int rows, byte[] dst) {
            System.arraycopy(v ? vScales : kScales, row0 * SCALE_ROW, dst, 0, rows * SCALE_ROW);
        }

        @Override
        public void writeSide(boolean v, int plane, int row0, int rows, byte[] src) {
            System.arraycopy(src, 0, v ? vScales : kScales, row0 * SCALE_ROW, rows * SCALE_ROW);
        }

        private int rowBytes(boolean v) {
            return Math.toIntExact(v ? layout.rowBytesV() : layout.rowBytesK());
        }

        byte[] scaleRow(int row) {
            return row(kScales, row);
        }

        byte[] vScaleRow(int row) {
            return row(vScales, row);
        }

        private static byte[] row(byte[] scales, int row) {
            byte[] out = new byte[SCALE_ROW];
            System.arraycopy(scales, row * SCALE_ROW, out, 0, SCALE_ROW);
            return out;
        }

        void setScale(int row, float kScale, float vScale) {
            ByteBuffer kb = ByteBuffer.wrap(kScales).order(ByteOrder.LITTLE_ENDIAN);
            ByteBuffer vb = ByteBuffer.wrap(vScales).order(ByteOrder.LITTLE_ENDIAN);
            for (int h = 0; h < HEADS; h++) {
                kb.putFloat(row * SCALE_ROW + h * Float.BYTES, kScale + h);
                vb.putFloat(row * SCALE_ROW + h * Float.BYTES, vScale + h);
            }
        }
    }

    private static BlockImage image(FakeCache cache, int rows) {
        int[] tokens = new int[rows];
        int[] positions = new int[rows];
        for (int i = 0; i < rows; i++) {
            tokens[i] = i + 1;
            positions[i] = i;
        }
        byte[] k = new byte[Math.toIntExact(rows * cache.layout().rowBytesK())];
        cache.read(false, 0, rows, k);
        byte[] v = new byte[Math.toIntExact(rows * cache.layout().rowBytesV())];
        cache.read(true, 0, rows, v);
        return new BlockImage(tokens, positions, true, new byte[][]{k}, new byte[][]{v},
                new byte[][][]{BlockImage.readSides(cache, false, 0, rows)},
                new byte[][][]{BlockImage.readSides(cache, true, 0, rows)});
    }

    @Test
    public void theImageCarriesTheScaleRowsWithTheCodes() {
        FakeCache cache = new FakeCache(layout(KvDType.INT4_PER_TOKEN_HEAD, KvDType.FP8_PER_TOKEN_HEAD));
        for (int r = 0; r < ROWS; r++) {
            cache.setScale(r, 0.25f * (r + 1), 0.5f * (r + 1));
        }
        BlockImage image = image(cache, 4);
        assertEquals(8L * 4 + 4 * cache.layout().rowBytesK() + 4 * cache.layout().rowBytesV() + 2 * 4 * SCALE_ROW,
                image.bytes());
        assertArrayEquals(cache.scaleRow(1), java.util.Arrays.copyOfRange(image.kSides(0)[0], SCALE_ROW,
                2 * SCALE_ROW));
        assertArrayEquals(cache.vScaleRow(0), java.util.Arrays.copyOf(image.vSides(0)[0], SCALE_ROW));

        // a shared head keeps the scales of the shared rows only
        BlockImage head = image.head(2, new BlockLayout(null, Collections.singletonList(cache)));
        assertEquals(2 * SCALE_ROW, head.kSides(0)[0].length);
        assertEquals(2 * SCALE_ROW, head.vSides(0)[0].length);
        assertArrayEquals(cache.scaleRow(0), java.util.Arrays.copyOf(head.kSides(0)[0], SCALE_ROW));
        assertArrayEquals(cache.vScaleRow(0), java.util.Arrays.copyOf(head.vSides(0)[0], SCALE_ROW));

        // the head installs into fresh rows, scales included
        FakeCache fresh = new FakeCache(layout(KvDType.INT4_PER_TOKEN_HEAD, KvDType.FP8_PER_TOKEN_HEAD));
        BlockImage.writeSides(fresh, false, 3, 2, head.kSides(0));
        BlockImage.writeSides(fresh, true, 5, 2, head.vSides(0));
        assertArrayEquals(cache.scaleRow(0), fresh.scaleRow(3));
        assertArrayEquals(cache.scaleRow(1), fresh.scaleRow(4));
        assertArrayEquals(cache.vScaleRow(0), fresh.vScaleRow(5));
        assertArrayEquals(cache.vScaleRow(1), fresh.vScaleRow(6));
    }

    @Test
    public void cachesWithoutScalePlanesKeepAnEmptySideList() {
        FakeCache cache = new FakeCache(layout(KvDType.F16, KvDType.F16));
        BlockImage image = image(cache, 4);
        assertEquals(0, image.kSides(0).length);
        assertEquals(0, image.vSides(0).length);
        assertEquals(8L * 4 + 4 * cache.layout().rowBytesK() + 4 * cache.layout().rowBytesV(), image.bytes());
        // a legacy image (no side arrays) installs into a cache without planes
        BlockImage legacy = new BlockImage(new int[]{1, 2}, new int[]{0, 1}, true,
                new byte[][]{new byte[2 * Math.toIntExact(cache.layout().rowBytesK())]},
                new byte[][]{new byte[2 * Math.toIntExact(cache.layout().rowBytesV())]});
        BlockImage.writeSides(cache, false, 0, 2, legacy.kSides(0));
        BlockImage.writeSides(cache, true, 0, 2, legacy.vSides(0));
    }
}
