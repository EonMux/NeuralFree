package com.eonmux.neuralfree.gguf;

import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link GGUFTensorInfo#validate}: the checks every model source applies to its tensor table, with the messages the
 * GGUF parser has always reported under its {@code Invalid GGUF file '<path>'} prefix.
 */
public class GGUFTensorInfoValidateTest {

    private static final String PREFIX = "Invalid GGUF file 'm.gguf'";

    private static String rejection(String name, GGMLType type, long... dims) {
        GGUFTensorInfo info = new GGUFTensorInfo(name, dims, type, 0);
        return assertThrows(IOException.class, () -> GGUFTensorInfo.validate(PREFIX, info)).getMessage();
    }

    @Test
    public void acceptsWellFormedTensors() throws IOException {
        GGUFTensorInfo.validate(PREFIX, new GGUFTensorInfo("a", new long[]{64, 3}, GGMLType.Q8_0, 0));
        GGUFTensorInfo.validate(PREFIX, new GGUFTensorInfo("b", new long[]{5}, GGMLType.F32, 32));
        GGUFTensorInfo.validate(PREFIX, new GGUFTensorInfo("c", new long[0], GGMLType.F32, 64));
        StringBuilder name = new StringBuilder();
        for (int i = 0; i < GGUFTensorInfo.MAX_NAME_BYTES - 1; i++) {
            name.append('n');
        }
        GGUFTensorInfo.validate(PREFIX, new GGUFTensorInfo(name.toString(), new long[]{1, 1, 1, 1}, GGMLType.F16, 0));
    }

    @Test
    public void rejectsDimensionsBeyondSigned64Bits() {
        assertEquals(PREFIX + ": tensor 't' dimension 1 has 18446744073709551615 elements, more than a signed 64-bit"
                + " count", rejection("t", GGMLType.F32, 4, -1));
    }

    @Test
    public void rejectsElementCountOverflow() {
        assertEquals(PREFIX + ": tensor 't' has shape [4294967296, 4294967296], whose element count overflows a"
                + " signed 64-bit value", rejection("t", GGMLType.F32, 1L << 32, 1L << 32));
    }

    @Test
    public void rejectsUnknownAndRemovedTypes() {
        assertEquals(PREFIX + ": tensor 't' has ggml type Q4_2, which is unknown or was removed from ggml",
                rejection("t", GGMLType.Q4_2, 32));
        assertEquals(PREFIX + ": tensor 't' has ggml type UNKNOWN, which is unknown or was removed from ggml",
                rejection("t", GGMLType.UNKNOWN, 32));
    }

    @Test
    public void rejectsRowsThatAreNotWholeBlocks() {
        assertEquals(PREFIX + ": tensor 't' of type Q8_0 has 40 elements per row, not a multiple of the block size 32",
                rejection("t", GGMLType.Q8_0, 40, 2));
    }

    @Test
    public void rejectsByteSizeOverflow() {
        assertEquals(PREFIX + ": tensor 't' of type F32 with shape [4611686018427387904] has a byte size that"
                + " overflows a signed 64-bit value", rejection("t", GGMLType.F32, 1L << 62));
    }

    @Test
    public void rejectsLongNamesAndHighRanks() {
        StringBuilder name = new StringBuilder();
        for (int i = 0; i < GGUFTensorInfo.MAX_NAME_BYTES; i++) {
            name.append('n');
        }
        assertEquals(PREFIX + ": tensor '" + name + "' has a name of 64 bytes, more than the 63 allowed",
                rejection(name.toString(), GGMLType.F32, 4));
        assertEquals(PREFIX + ": tensor 't' has 5 dimensions, more than the maximum of 4",
                rejection("t", GGMLType.F32, 1, 1, 1, 1, 1));
    }
}
