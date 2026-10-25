package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.util.EnumSet;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for {@link ActQuantScheme}: the stable codes, their round trip, the run lengths and the checkpoint
 * parameters each scheme reads.
 */
public class ActQuantSchemeTest {

    @Test
    public void codesAreStableAndRoundTrip() {
        ActQuantScheme[] byCode = {
            ActQuantScheme.INT8_TOKEN, ActQuantScheme.INT8_STATIC, ActQuantScheme.INT8_STATIC_ASYM,
            ActQuantScheme.INT8_GROUP128, ActQuantScheme.FP8_TOKEN, ActQuantScheme.FP8_STATIC,
            ActQuantScheme.FP8_GROUP128, ActQuantScheme.NVFP4, ActQuantScheme.MXFP4, ActQuantScheme.MXFP8
        };
        assertEquals(byCode.length, ActQuantScheme.values().length);
        for (int code = 0; code < byCode.length; code++) {
            assertEquals(byCode[code] + " code", code, byCode[code].code());
            assertSame(byCode[code], ActQuantScheme.fromCode(code));
        }
        for (ActQuantScheme s : ActQuantScheme.values()) {
            assertSame(s, ActQuantScheme.fromCode(s.code()));
        }
    }

    @Test
    public void unknownCodesAreRejected() {
        for (int code : new int[]{-1, 10, 255, Integer.MIN_VALUE}) {
            try {
                ActQuantScheme.fromCode(code);
                fail("code " + code);
            } catch (GgufFormatException expected) {
                assertTrue(expected.getMessage().contains(Integer.toString(code)));
            }
        }
    }

    @Test
    public void runLengthsFollowTheSchemes() {
        int row = 4096;
        for (ActQuantScheme s : new ActQuantScheme[]{ActQuantScheme.INT8_TOKEN, ActQuantScheme.INT8_STATIC,
                ActQuantScheme.INT8_STATIC_ASYM, ActQuantScheme.FP8_TOKEN, ActQuantScheme.FP8_STATIC}) {
            assertEquals(s.toString(), row, s.runLength(row));
            assertEquals(s.toString(), 100, s.runLength(100));
        }
        assertEquals(128, ActQuantScheme.INT8_GROUP128.runLength(row));
        assertEquals(128, ActQuantScheme.FP8_GROUP128.runLength(row));
        assertEquals(16, ActQuantScheme.NVFP4.runLength(row));
        assertEquals(32, ActQuantScheme.MXFP4.runLength(row));
        assertEquals(32, ActQuantScheme.MXFP8.runLength(row));
    }

    @Test
    public void runLengthRejectsPartialRunsAndEmptyRows() {
        expectBadRow(ActQuantScheme.INT8_GROUP128, 100);
        expectBadRow(ActQuantScheme.NVFP4, 40);
        expectBadRow(ActQuantScheme.MXFP8, 48);
        expectBadRow(ActQuantScheme.INT8_TOKEN, 0);
        expectBadRow(ActQuantScheme.MXFP4, -32);
    }

    private static void expectBadRow(ActQuantScheme s, int rowLen) {
        try {
            s.runLength(rowLen);
            fail(s + " row of " + rowLen);
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains(Integer.toString(rowLen)));
        }
    }

    @Test
    public void checkpointParametersPerScheme() {
        Set<ActQuantScheme> statics = EnumSet.of(ActQuantScheme.INT8_STATIC, ActQuantScheme.INT8_STATIC_ASYM,
                ActQuantScheme.FP8_STATIC);
        for (ActQuantScheme s : ActQuantScheme.values()) {
            assertEquals(s + " static", statics.contains(s), s.isStatic());
            assertEquals(s + " zero point", s == ActQuantScheme.INT8_STATIC_ASYM, s.needsZeroPoint());
            assertEquals(s + " global scale", s == ActQuantScheme.NVFP4, s.needsGlobalScale());
        }
    }
}
