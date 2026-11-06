package com.eonmux.neuralfree.gguf.ops.heads;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The late-interaction MaxSim over two per-token row sets: the sum over the query rows of the maximum dot product
 * over the document rows, with a refusal naming the side that breaks.
 */
public class LateInteractionTest {

    @Test
    public void maxSimSumsTheMaximalDots() {
        float[][] query = {{1.0f, 0.0f}, {0.0f, 1.0f}};
        float[][] document = {{0.6f, 0.8f}, {1.0f, 0.0f}};
        // query row 0 best matches {1, 0} (1.0); query row 1 best matches {0.6, 0.8} (0.8)
        assertEquals(1.0f + 0.8f, LateInteraction.maxSim(query, document), 1e-6f);
    }

    @Test
    public void everyQueryRowTakesItsOwnBestDocument() {
        float[][] query = {{1.0f, 0.0f}, {0.0f, -2.0f}};
        float[][] document = {{1.0f, 0.0f}, {0.0f, -3.0f}};
        // query row 0 best matches {1, 0} (1.0); query row 1 best matches {0, -3} (6.0)
        assertEquals(1.0f + 6.0f, LateInteraction.maxSim(query, document), 1e-6f);
    }

    @Test
    public void aSingleRowSetAnswersOneDot() {
        assertEquals(3.5f, LateInteraction.maxSim(new float[][]{{1.0f, 2.0f}}, new float[][]{{1.0f, 1.25f}}), 1e-6f);
    }

    @Test
    public void emptyAndNullSidesRefuseByName() {
        IllegalArgumentException empty = assertThrows(IllegalArgumentException.class,
                () -> LateInteraction.maxSim(new float[][]{{1.0f}}, new float[0][]));
        assertTrue(empty.getMessage(), empty.getMessage().contains("document"));
        IllegalArgumentException nullQuery = assertThrows(IllegalArgumentException.class,
                () -> LateInteraction.maxSim(null, new float[][]{{1.0f}}));
        assertTrue(nullQuery.getMessage(), nullQuery.getMessage().contains("query"));
    }

    @Test
    public void mismatchedWidthsRefuse() {
        assertThrows(IllegalArgumentException.class,
                () -> LateInteraction.maxSim(new float[][]{{1.0f, 2.0f}}, new float[][]{{1.0f, 2.0f, 3.0f}}));
    }
}
