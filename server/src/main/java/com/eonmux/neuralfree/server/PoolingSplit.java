package com.eonmux.neuralfree.server;

import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/**
 * The split of one multi-input pooling call over several replicas ({@link EnginePooling}): the contiguous part
 * bounds of about equal token counts, and the collection of the parts' failures into the one the call raises.
 * Stateless.
 */
final class PoolingSplit {

    private PoolingSplit() {
    }

    /**
     * The bounds of {@code parts} contiguous, non-empty parts of {@code inputs}: part {@code p} holds inputs
     * {@code bounds[p] .. bounds[p + 1] - 1}. A part closes once its running token count reaches its share of the
     * total, and closes early when the parts after it would otherwise run out of inputs.
     *
     * @throws IllegalArgumentException when {@code parts} is below one or above the input count
     */
    static int[] bounds(int[][] inputs, int parts) {
        if (parts < 1 || parts > inputs.length) {
            throw new IllegalArgumentException("cannot split " + inputs.length + " inputs into " + parts + " parts");
        }
        long total = 0;
        for (int[] input : inputs) {
            total += input.length;
        }
        int[] bounds = new int[parts + 1];
        bounds[parts] = inputs.length;
        long seen = 0;
        int i = 0;
        for (int p = 1; p < parts; p++) {
            long target = total * p / parts;
            int last = inputs.length - (parts - p);
            seen += inputs[i++].length;
            while (i < last && seen < target) {
                seen += inputs[i++].length;
            }
            bounds[p] = i;
        }
        return bounds;
    }

    /**
     * Waits for {@code part} and folds its failure into {@code failure}: the first failure stands, later ones join
     * it as suppressed exceptions.
     *
     * @return the call's failure so far, null while every part succeeded
     */
    static Throwable await(CompletableFuture<Void> part, Throwable failure) {
        Throwable cause;
        try {
            part.get();
            return failure;
        } catch (ExecutionException e) {
            cause = e.getCause() != null ? e.getCause() : e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            cause = new IOException("interrupted while waiting for the pooling parts on the other replicas", e);
        }
        if (failure == null) {
            return cause;
        }
        failure.addSuppressed(cause);
        return failure;
    }

    /** Raises {@code failure} as it is when it is unchecked or an {@code IOException}; does nothing for null. */
    static void rethrow(Throwable failure) throws IOException {
        if (failure == null) {
            return;
        }
        if (failure instanceof IOException io) {
            throw io;
        }
        if (failure instanceof RuntimeException re) {
            throw re;
        }
        if (failure instanceof Error err) {
            throw err;
        }
        throw new IOException(failure.getMessage(), failure);
    }
}
