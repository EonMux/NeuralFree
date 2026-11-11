package com.eonmux.neuralfree.gguf.mm;

import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * One call submitted to a rank thread of a {@link DataParallelVisionEncoder} that its caller can withdraw before it
 * starts, or else wait out. A caller that is interrupted never returns while its call still runs on the rank: the
 * encoder, and the media lock its caller holds around it, are never released under a running encode, so a LoRA
 * session, a release or a close that takes the lock next never meets a rank that is still computing.
 *
 * @param <T> the call's result
 */
final class RankTask<T> implements Callable<T> {

    private final Callable<T> body;
    /** Taken by whichever comes first: the rank thread starting the call, or the caller withdrawing it. */
    private final AtomicBoolean claimed = new AtomicBoolean();
    private final CountDownLatch finished = new CountDownLatch(1);

    RankTask(Callable<T> body) {
        this.body = Objects.requireNonNull(body, "body");
    }

    /** Runs the call unless its caller withdrew it first; a withdrawn call returns null without running. */
    @Override
    public T call() throws Exception {
        if (!claimed.compareAndSet(false, true)) {
            return null;
        }
        try {
            return body.call();
        } finally {
            finished.countDown();
        }
    }

    /**
     * Withdraws the call when the rank thread has not started it, else waits until it ended. The wait ignores
     * interrupts and leaves the caller's interrupt status set when it was set on entry or during the wait.
     */
    void withdrawOrAwait() {
        if (claimed.compareAndSet(false, true)) {
            return;
        }
        boolean interrupted = Thread.interrupted();
        while (true) {
            try {
                finished.await();
                break;
            } catch (InterruptedException e) {
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }
}
