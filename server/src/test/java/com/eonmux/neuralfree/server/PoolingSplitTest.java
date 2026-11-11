package com.eonmux.neuralfree.server;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** The part bounds of a pooling call spread over replicas, and the failure the parts fold into. */
class PoolingSplitTest {

    private static int[][] lengths(int... n) {
        int[][] out = new int[n.length][];
        for (int i = 0; i < n.length; i++) {
            out[i] = new int[n[i]];
        }
        return out;
    }

    @Test
    void equalInputsSplitEvenly() {
        assertArrayEquals(new int[]{0, 2, 4}, PoolingSplit.bounds(lengths(10, 10, 10, 10), 2));
        assertArrayEquals(new int[]{0, 1, 2, 3}, PoolingSplit.bounds(lengths(5, 5, 5), 3));
    }

    @Test
    void partsBalanceTokensNotInputs() {
        // one long input carries half the tokens: it is a part of its own
        assertArrayEquals(new int[]{0, 1, 5}, PoolingSplit.bounds(lengths(40, 10, 10, 10, 10), 2));
        assertArrayEquals(new int[]{0, 4, 5}, PoolingSplit.bounds(lengths(10, 10, 10, 10, 40), 2));
    }

    @Test
    void everyPartKeepsAtLeastOneInput() {
        assertArrayEquals(new int[]{0, 1, 2, 3}, PoolingSplit.bounds(lengths(100, 1, 1), 3));
        assertArrayEquals(new int[]{0, 1, 2, 3}, PoolingSplit.bounds(lengths(1, 1, 100), 3));
        assertArrayEquals(new int[]{0, 3}, PoolingSplit.bounds(lengths(1, 2, 3), 1));
    }

    @Test
    void refusesMorePartsThanInputs() {
        assertThrows(IllegalArgumentException.class, () -> PoolingSplit.bounds(lengths(1, 1), 3));
        assertThrows(IllegalArgumentException.class, () -> PoolingSplit.bounds(lengths(1, 1), 0));
    }

    @Test
    void theFirstFailureStandsAndLaterOnesAreSuppressed() {
        IOException first = new IOException("first");
        IllegalStateException second = new IllegalStateException("second");
        CompletableFuture<Void> ok = CompletableFuture.completedFuture(null);
        CompletableFuture<Void> failed = CompletableFuture.failedFuture(second);
        assertNull(PoolingSplit.await(ok, null));
        assertSame(second, PoolingSplit.await(failed, null));
        assertSame(first, PoolingSplit.await(failed, first));
        assertArrayEquals(new Throwable[]{second}, first.getSuppressed());
        assertSame(first, assertThrows(IOException.class, () -> PoolingSplit.rethrow(first)));
        assertSame(second, assertThrows(IllegalStateException.class, () -> PoolingSplit.rethrow(second)));
    }
}
