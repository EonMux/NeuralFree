package com.eonmux.neuralfree.gguf.ops.flow;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/** The stream layout rules of {@link MtpHead}'s stream operations: part widths and stream-norm weight forms. */
public class MtpHeadStreamsTest {

    private static final class TestRows extends MtpHead.Rows {
        TestRows(int width) {
            super("test", width);
        }
    }

    @Test
    public void streamPartSplitsTheRowEvenly() {
        assertEquals(8, MtpHead.streamPart(new TestRows(32), 4));
        assertEquals(32, MtpHead.streamPart(new TestRows(32), 1));
    }

    @Test
    public void streamPartRejectsUnevenAndEmptySplits() {
        assertThrows(IllegalArgumentException.class, () -> MtpHead.streamPart(new TestRows(30), 4));
        assertThrows(IllegalArgumentException.class, () -> MtpHead.streamPart(new TestRows(32), 0));
    }

    /** deepseek4 shares one {@code hnorm} vector across streams; qwen4exp holds one per stream. */
    @Test
    public void streamWeightRowsAcceptsSharedAndPerStreamWeights() {
        assertEquals(1, MtpHead.streamWeightRows("nextn.hnorm", 8, 8, 4));
        assertEquals(4, MtpHead.streamWeightRows("nextn.hnorm", 32, 8, 4));
        assertEquals(1, MtpHead.streamWeightRows("nextn.hnorm", 8, 8, 1));
    }

    @Test
    public void streamWeightRowsRejectsOtherSizes() {
        assertThrows(IllegalArgumentException.class, () -> MtpHead.streamWeightRows("nextn.hnorm", 16, 8, 4));
        assertThrows(IllegalArgumentException.class, () -> MtpHead.streamWeightRows("nextn.hnorm", 0, 8, 4));
    }
}
