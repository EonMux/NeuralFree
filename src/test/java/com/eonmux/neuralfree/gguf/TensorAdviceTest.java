package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaTopology;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link TensorAdvice}: the default advice follows the NUMA placement (random with it, the prefetch without it,
 * no advice for data larger than RAM), every named advice maps to its {@link MemoryLock.Advice}, {@code none}
 * disables it and anything else is refused by name.
 */
public class TensorAdviceTest {

    /** A model of 2 GiB on a host of 16 GiB. */
    private static final long MODEL = 2L << 30;
    private static final long RAM = 16L << 30;

    private static Properties with(String advice) {
        Properties p = new Properties();
        if (advice != null) {
            p.setProperty(TensorAdvice.PROP_ADVICE, advice);
        }
        return p;
    }

    @Test
    public void theDefaultPrefetchesWithoutNumaPlacement() {
        assertEquals(MemoryLock.Advice.WILLNEED, TensorAdvice.forMapping(with(null), MODEL, RAM));
        assertEquals(MemoryLock.Advice.WILLNEED, TensorAdvice.forMapping(with(" Auto "), MODEL, RAM));
    }

    @Test
    public void theDefaultSkipsThePrefetchOfDataLargerThanRam() {
        assertNull("a 20 GiB model on 16 GiB", TensorAdvice.forMapping(with(null), 20L << 30, RAM));
        assertEquals("exactly RAM still fits", MemoryLock.Advice.WILLNEED,
                TensorAdvice.forMapping(with("auto"), RAM, RAM));
        assertEquals("an unknown RAM size prefetches", MemoryLock.Advice.WILLNEED,
                TensorAdvice.forMapping(with(null), 20L << 30, -1));
        assertEquals("a named advice applies whatever the size", MemoryLock.Advice.WILLNEED,
                TensorAdvice.forMapping(with("willneed"), 20L << 30, RAM));
        assertEquals("the host's own RAM", HostMemory.totalBytes() > 0 ? null : MemoryLock.Advice.WILLNEED,
                TensorAdvice.forMapping(with(null), Long.MAX_VALUE));
    }

    @Test
    public void theDefaultReadsAtRandomUnderActiveNumaPlacement() {
        Properties p = with("auto");
        p.setProperty(NumaStrategy.PROPERTY, "distribute");
        MemoryLock.Advice expected = NumaTopology.host().isNuma() ? MemoryLock.Advice.RANDOM
                : MemoryLock.Advice.WILLNEED;
        assertEquals("placement is active only on a NUMA host", expected, TensorAdvice.forMapping(p, MODEL, RAM));
    }

    @Test
    public void namedAdviceIsTakenAsGivenAndNoneDisablesIt() {
        assertEquals(MemoryLock.Advice.WILLNEED, TensorAdvice.forMapping(with("willneed"), MODEL, RAM));
        assertEquals(MemoryLock.Advice.SEQUENTIAL, TensorAdvice.forMapping(with("SEQUENTIAL"), MODEL, RAM));
        assertEquals(MemoryLock.Advice.RANDOM, TensorAdvice.forMapping(with("random"), MODEL, RAM));
        assertEquals(MemoryLock.Advice.HUGEPAGE, TensorAdvice.forMapping(with("hugepage"), MODEL, RAM));
        assertNull(TensorAdvice.forMapping(with("none"), MODEL, RAM));
    }

    @Test
    public void unknownAdviceIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> TensorAdvice.forMapping(with("dontneed"), MODEL, RAM));
        assertTrue(e.getMessage(), e.getMessage().contains(TensorAdvice.PROP_ADVICE)
                && e.getMessage().contains("dontneed"));
        assertThrows(NullPointerException.class, () -> TensorAdvice.forMapping(null, MODEL, RAM));
        assertThrows(IllegalArgumentException.class, () -> TensorAdvice.forMapping(with("bogus"), 20L << 30, RAM));
    }
}
