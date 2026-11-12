package com.eonmux.neuralfree.gguf.gpu.parallel;

import org.junit.Test;

import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The host staging of the rank exchanges: one array per thread, reused and grown on demand up to a bound; a larger
 * exchange gets an array the thread does not keep.
 */
public class RankExchangeStagingTest {

    @Test
    public void aThreadReusesItsStagingAndGrowsItOnDemand() throws Exception {
        float[] first = RankExchange.staging(64);
        assertTrue(first.length >= 64);
        assertSame(first, RankExchange.staging(16));
        float[] grown = RankExchange.staging(first.length + 1L);
        assertTrue(grown.length > first.length);
        assertSame(grown, RankExchange.staging(1));

        AtomicReference<float[]> other = new AtomicReference<>();
        Thread t = new Thread(() -> other.set(RankExchange.staging(16)));
        t.start();
        t.join();
        assertNotSame("each thread has its own staging", grown, other.get());
    }

    @Test
    public void anExchangeBeyondTheKeptBoundIsNotKept() throws Exception {
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Thread t = new Thread(() -> {
            try {
                float[] kept = RankExchange.staging(RankExchange.KEPT_STAGING_FLOATS);
                assertSame("the bound itself is kept", kept, RankExchange.staging(1));
                float[] large = RankExchange.staging(RankExchange.KEPT_STAGING_FLOATS + 1L);
                assertTrue(large.length > RankExchange.KEPT_STAGING_FLOATS);
                assertNotSame("a larger exchange takes its own array", large,
                        RankExchange.staging(RankExchange.KEPT_STAGING_FLOATS + 1L));
                assertSame("and leaves the kept one", kept, RankExchange.staging(1));
            } catch (Throwable e) {
                failure.set(e);
            }
        });
        t.start();
        t.join();
        if (failure.get() != null) {
            throw new AssertionError(failure.get());
        }
    }

    @Test
    public void aStagingBeyondOneArrayIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> RankExchange.staging(Integer.MAX_VALUE));
    }
}
