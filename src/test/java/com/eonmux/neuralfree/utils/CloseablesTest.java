package com.eonmux.neuralfree.utils;

import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/** First-failure-wins aggregation of {@link Closeables}. */
public class CloseablesTest {

    @Test
    public void chainKeepsTheFirstFailureAndSuppressesTheRest() {
        RuntimeException a = new IllegalStateException("a");
        RuntimeException b = new IllegalArgumentException("b");
        assertSame(a, Closeables.chain(null, a));
        assertSame(a, Closeables.chain(a, b));
        assertArrayEquals(new Throwable[]{b}, a.getSuppressed());
    }

    @Test
    public void chainNeverSuppressesAFailureIntoItself() {
        RuntimeException a = new IllegalStateException("a");
        assertSame(a, Closeables.chain(a, a));
        assertEquals(0, a.getSuppressed().length);
    }

    @Test
    public void closeSkipsNullAndKeepsTheRunningFailure() {
        RuntimeException a = new IllegalStateException("a");
        assertNull(Closeables.close(null, null));
        assertSame(a, Closeables.close(null, a));
        assertSame(a, Closeables.close(() -> { }, a));
    }

    @Test
    public void closeReportsARuntimeFailureAsThrown() {
        RuntimeException boom = new IllegalStateException("boom");
        assertSame(boom, Closeables.close(() -> {
            throw boom;
        }, null));
    }

    @Test
    public void closeWrapsACheckedFailureOnlyWhenItBecomesTheReportedOne() {
        IOException io = new IOException("io");
        AutoCloseable failing = () -> {
            throw io;
        };
        RuntimeException reported = Closeables.close(failing, null);
        assertTrue(reported instanceof IllegalStateException);
        assertSame(io, reported.getCause());

        RuntimeException first = new IllegalArgumentException("first");
        assertSame(first, Closeables.close(failing, first));
        assertArrayEquals(new Throwable[]{io}, first.getSuppressed());
    }

    @Test
    public void closeAllClosesEveryElementInOrderDespiteFailures() {
        List<String> closed = new ArrayList<>();
        RuntimeException x = new IllegalStateException("x");
        RuntimeException y = new IllegalStateException("y");
        List<AutoCloseable> all = Arrays.asList(
                () -> closed.add("a"),
                () -> {
                    closed.add("b");
                    throw x;
                },
                null,
                () -> {
                    closed.add("c");
                    throw y;
                },
                () -> closed.add("d"));
        RuntimeException failure = Closeables.closeAll(all, null);
        assertEquals(Arrays.asList("a", "b", "c", "d"), closed);
        assertSame(x, failure);
        assertArrayEquals(new Throwable[]{y}, x.getSuppressed());
    }
}
