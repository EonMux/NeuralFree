package com.eonmux.neuralfree.gguf.gpu.pipeline;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Runs one task per split stage at the same time, each stage's work on its own device: the first task runs on the
 * calling thread and every other one on a thread of its own named after its stage. The call returns once every task
 * has ended, waiting through interrupts (the tasks hold the stages' devices, which the caller releases only after
 * them) and restoring the interrupt status afterwards. A failure is thrown after every task has ended: the first
 * stage's failure in stage order, with the later ones attached as suppressed.
 */
final class ConcurrentStageWork {

    private ConcurrentStageWork() {
    }

    /**
     * Runs {@code tasks[i]} for stage {@code names[i]}, all at the same time.
     *
     * @throws IllegalArgumentException when the lists differ in length
     * @throws RuntimeException         the first failing stage's failure; an {@link Error} is rethrown as is
     */
    static void runAll(List<String> names, List<Runnable> tasks) {
        Objects.requireNonNull(names, "names");
        Objects.requireNonNull(tasks, "tasks");
        if (names.size() != tasks.size()) {
            throw new IllegalArgumentException(names.size() + " stage names for " + tasks.size() + " tasks");
        }
        int n = tasks.size();
        Throwable[] failures = new Throwable[n];
        List<Thread> threads = new ArrayList<>(Math.max(0, n - 1));
        try {
            for (int i = 1; i < n; i++) {
                int stage = i;
                Runnable task = Objects.requireNonNull(tasks.get(i), "task " + i);
                Thread t = new Thread(() -> failures[stage] = runCapturing(task), "nf-stage-" + names.get(i));
                t.setDaemon(true);
                t.start();
                threads.add(t);
            }
        } catch (RuntimeException | Error e) {
            joinAll(threads);
            throw e;
        }
        if (n > 0) {
            failures[0] = runCapturing(Objects.requireNonNull(tasks.get(0), "task 0"));
        }
        joinAll(threads);
        Throwable first = null;
        for (Throwable f : failures) {
            if (f == null) {
                continue;
            }
            if (first == null) {
                first = f;
            } else {
                first.addSuppressed(f);
            }
        }
        if (first instanceof Error) {
            throw (Error) first;
        }
        if (first != null) {
            throw (RuntimeException) first;
        }
    }

    private static Throwable runCapturing(Runnable task) {
        try {
            task.run();
            return null;
        } catch (RuntimeException | Error e) {
            return e;
        }
    }

    private static void joinAll(List<Thread> threads) {
        boolean interrupted = false;
        for (Thread t : threads) {
            while (true) {
                try {
                    t.join();
                    break;
                } catch (InterruptedException e) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }
}
