package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EngramRows}: the E4M3 rows scaled by their E8M0 block exponents and concatenated in layer order, the
 * exactness guard of the converter's {@code _e8m0_check}, and the Engram gate product.
 */
public class EngramRowsTest {

    private static final int DIM = 32;

    @Test
    public void theTableDequantisesAndConcatenatesTheLayers() throws IOException {
        EngramRows.Part first = new EngramRows.Part(e4m3(2, 0x38, 0x3C), scales(2, 127, 128), 2, DIM);
        EngramRows.Part second = new EngramRows.Part(e4m3(1, 0x40), scales(1, 126), 1, DIM);
        RowSource table = EngramRows.table(Arrays.asList(first, second), GGMLType.F32);
        assertArrayEquals(new long[]{3, DIM}, table.shape());
        assertEquals(StDType.F32, table.format().stDType());
        float[] values = values(table);
        for (int i = 0; i < DIM; i++) {
            assertEquals(1f, values[i], 0f);
            assertEquals(3f, values[DIM + i], 0f);
            assertEquals(1f, values[2 * DIM + i], 0f);
        }
        RowSource bf16 = EngramRows.table(Arrays.asList(first, second), GGMLType.BF16);
        assertEquals(StDType.BF16, bf16.format().stDType());
        assertEquals(2 * DIM, bf16.rowBytes());
        assertArrayEquals(values, values(bf16), 0f);
    }

    @Test
    public void anOutOfRangeBlockExponentIsRefused() {
        RowSource table = EngramRows.table(
                java.util.Collections.singletonList(new EngramRows.Part(e4m3(1, 0x38), scales(1, 2), 1, DIM)),
                GGMLType.BF16);
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> values(table));
        assertTrue(e.getMessage(), e.getMessage().contains("E8M0 exponent"));
    }

    @Test
    public void aScaleGridThatDoesNotFitTheRowsIsRefused() {
        RowSource wrong = RopeTensors.fromBytes(ElementFormat.of(StDType.U8), new long[]{1, 2},
                new byte[]{127, 127});
        RowSource table = EngramRows.table(
                java.util.Collections.singletonList(new EngramRows.Part(e4m3(1, 0x38), wrong, 1, DIM)),
                GGMLType.BF16);
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> values(table));
        assertTrue(e.getMessage(), e.getMessage().contains("one per 32 elements"));
    }

    @Test
    public void theProductMultipliesTwoMatricesInF32() throws IOException {
        RowSource a = RopeTensors.fromFloats(new long[]{2, 3}, new float[]{1, 2, 3, 4, 5, 6});
        RowSource b = RopeTensors.fromFloats(new long[]{2, 3}, new float[]{2, 2, 2, 0.5f, 0.5f, 0.5f});
        RowSource product = EngramRows.product(a, b);
        assertArrayEquals(new long[]{2, 3}, product.shape());
        assertEquals(StDType.F32, product.format().stDType());
        assertArrayEquals(new float[]{2, 4, 6, 2, 2.5f, 3}, values(product), 0f);
        assertThrows(IllegalArgumentException.class,
                () -> EngramRows.product(a, RopeTensors.fromFloats(new long[]{3, 2}, new float[6])));
    }

    // ------------------------------------------------------------------ helpers

    /** E4M3 rows of {@code DIM} codes: row {@code r} is filled with {@code codes[r]}. */
    private static RowSource e4m3(int rows, int... codes) {
        byte[] data = new byte[rows * DIM];
        for (int r = 0; r < rows; r++) {
            Arrays.fill(data, r * DIM, (r + 1) * DIM, (byte) codes[r]);
        }
        return RopeTensors.fromBytes(ElementFormat.of(StDType.F8_E4M3), new long[]{rows, DIM}, data);
    }

    /** One E8M0 scale per row. */
    private static RowSource scales(int rows, int... exponents) {
        byte[] data = new byte[rows];
        for (int r = 0; r < rows; r++) {
            data[r] = (byte) exponents[r];
        }
        return RopeTensors.fromBytes(ElementFormat.of(StDType.U8), new long[]{rows, 1}, data);
    }

    private static float[] values(RowSource src) throws IOException {
        long inner = src.shape()[src.shape().length - 1];
        int total = Math.multiplyExact((int) src.rows(), (int) inner);
        ByteBuffer bytes = ByteBuffer.allocate((int) Math.multiplyExact(src.rowBytes(), src.rows()))
                .order(ByteOrder.LITTLE_ENDIAN);
        src.readRows(0, (int) src.rows(), bytes);
        bytes.flip();
        float[] out = new float[total];
        for (int r = 0; r < src.rows(); r++) {
            DtypePolicy.decode(src.format(), bytes, (int) (r * src.rowBytes()), out, (int) (r * inner), (int) inner);
        }
        return out;
    }
}
