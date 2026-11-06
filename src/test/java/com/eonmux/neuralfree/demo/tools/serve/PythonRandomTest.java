package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * The CPython-compatible generator: the values the reference's {@code random.Random} produces for
 * the same seeds, so a dataset shuffle is reproducible across the two implementations.
 */
public class PythonRandomTest {

    @Test
    public void theShuffleMatchesCpython() {
        assertEquals(Arrays.asList(7, 8, 1, 5, 3, 4, 2, 0, 9, 6), shuffled(0));
        assertEquals(Arrays.asList(6, 8, 9, 7, 5, 3, 0, 4, 1, 2), shuffled(1));
        assertEquals(Arrays.asList(8, 3, 1, 4, 7, 0, 9, 6, 2, 5), shuffled(7));
        assertEquals(Arrays.asList(8, 7, 3, 5, 1, 2, 9, 4, 0, 6), shuffled(12345));
    }

    @Test
    public void theRandomDrawsMatchCpython() {
        PythonRandom random = new PythonRandom(0);
        assertEquals(844, (int) (random.nextDouble() * 1000));
        assertEquals(757, (int) (random.nextDouble() * 1000));
        assertEquals(420, (int) (random.nextDouble() * 1000));
        PythonRandom bits = new PythonRandom(3);
        assertEquals(60, bits.getRandBits(8));
        assertEquals(151, bits.getRandBits(8));
        assertEquals(139, bits.getRandBits(8));
        assertEquals(33, bits.getRandBits(8));
        assertEquals(94, bits.getRandBits(8));
    }

    private static List<Integer> shuffled(long seed) {
        List<Integer> values = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            values.add(i);
        }
        new PythonRandom(seed).shuffle(values);
        return values;
    }
}
