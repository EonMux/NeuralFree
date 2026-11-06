package com.eonmux.neuralfree.gguf.hf.tensor;

import org.junit.Test;

import java.io.IOException;

import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.seq;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

/**
 * {@link QkvLayouts}: each fused QKV layout split and regrouped against hand-computed row orders, Qwen3-Next's QKVZ
 * regrouping, and DeepSeek's MLA {@code kv_b} split.
 */
public class QkvLayoutsTest {

    @Test
    public void thirdsAndSizes() throws IOException {
        RowSource[] t = QkvLayouts.splitQkv(seq(6, 2), QkvLayouts.Layout.THIRDS, 0, 0, 0);
        assertArrayEquals(new float[]{4, 5, 6, 7}, values(t[1]), 0f);
        RowSource fused = seq(8, 1);
        RowSource[] s = QkvLayouts.splitQkv(fused, QkvLayouts.Layout.SIZES, 2, 1, 2);
        assertArrayEquals(new float[]{0, 1, 2, 3}, values(s[0]), 0f);
        assertArrayEquals(new float[]{4, 5}, values(s[1]), 0f);
        assertArrayEquals(new float[]{6, 7}, values(s[2]), 0f);
        assertSame(fused, QkvLayouts.regroupQkv(fused, QkvLayouts.Layout.SIZES, 2, 1, 2));
        assertThrows(IllegalArgumentException.class,
                () -> QkvLayouts.splitQkv(seq(9, 1), QkvLayouts.Layout.SIZES, 2, 1, 2));
    }

    @Test
    public void perHeadInterleavedWeightAndBias() throws IOException {
        RowSource w = seq(12, 1);
        RowSource[] p = QkvLayouts.splitQkv(w, QkvLayouts.Layout.PER_HEAD_INTERLEAVED, 2, 2, 2);
        assertArrayEquals(new float[]{0, 1, 6, 7}, values(p[0]), 0f);
        assertArrayEquals(new float[]{2, 3, 8, 9}, values(p[1]), 0f);
        assertArrayEquals(new float[]{4, 5, 10, 11}, values(p[2]), 0f);
        RowSource bias = QkvLayouts.regroupQkv(seq(12), QkvLayouts.Layout.PER_HEAD_INTERLEAVED, 2, 2, 2);
        assertArrayEquals(new long[]{12}, bias.shape());
        assertArrayEquals(new float[]{0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11}, values(bias), 0f);
    }

    @Test
    public void groupedFalconLayout() throws IOException {
        RowSource w = seq(8, 2);
        RowSource regrouped = QkvLayouts.regroupQkv(w, QkvLayouts.Layout.GROUPED, 4, 2, 1);
        float[] v = values(regrouped);
        int[] rows = {0, 1, 4, 5, 2, 6, 3, 7};
        for (int i = 0; i < rows.length; i++) {
            assertArrayEquals(new float[]{2 * rows[i], 2 * rows[i] + 1}, new float[]{v[2 * i], v[2 * i + 1]}, 0f);
        }
    }

    @Test
    public void qwen3NextQkvzRegroup() throws IOException {
        RowSource src = seq(12, 1);
        RowSource[] qz = QkvLayouts.splitQkvz(src, 2, 4, 1, 1);
        assertArrayEquals(new float[]{0, 6, 1, 7, 2, 3, 8, 9}, values(qz[0]), 0f);
        assertArrayEquals(new float[]{4, 5, 10, 11}, values(qz[1]), 0f);
        for (RowSource part : qz) {
            assertSame("the parts trace the stored per-key-head rows", src, RowTransforms.storedSource(part));
            float[] v = values(part);
            for (int r = 0; r < v.length; r++) {
                assertEquals(v[r], RowTransforms.storedRow(part, r), 0f);
            }
        }
        assertThrows(IllegalArgumentException.class, () -> QkvLayouts.splitQkvz(seq(10, 1), 2, 4, 1, 1));
    }

    @Test
    public void mlaKvBSplitsAndTransposesPerHead() throws IOException {
        RowSource[] kv = QkvLayouts.mlaKvB(seq(6, 2), 2, 2, 1, 2);
        assertArrayEquals(new long[]{2, 2, 2}, kv[0].shape());
        assertArrayEquals(new float[]{0, 2, 1, 3, 6, 8, 7, 9}, values(kv[0]), 0f);
        assertFalse(kv[0].rowLocal());
        assertArrayEquals(new long[]{2, 1, 2}, kv[1].shape());
        assertArrayEquals(new float[]{4, 5, 10, 11}, values(kv[1]), 0f);
        assertThrows(IllegalArgumentException.class, () -> QkvLayouts.mlaKvB(seq(6, 2), 2, 2, 1, 3));
    }
}
