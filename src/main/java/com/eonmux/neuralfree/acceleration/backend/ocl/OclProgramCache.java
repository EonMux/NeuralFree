package com.eonmux.neuralfree.acceleration.backend.ocl;

import org.jocl.cl_program;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * The built programs of one {@link OclComputeBackend}, by cache key. Each key is built once: the first caller runs the
 * build on its own thread without holding any lock, and callers of the same key that arrive meanwhile wait for that
 * build and share its program, while builds of other keys and the backend's other work (allocations, event
 * bookkeeping) proceed. A failed build is not kept, so the next caller builds again. {@link #releaseAll} waits for the
 * builds still running and releases every program; a build started after it is refused.
 */
final class OclProgramCache {

    private final Map<String, FutureTask<cl_program>> programs = new ConcurrentHashMap<>();
    private volatile boolean closed;

    /**
     * The program of {@code key}, building it with {@code build} when no caller has.
     *
     * <p>A caller waiting for another caller's build keeps waiting when interrupted and returns with its interrupt
     * status set.</p>
     *
     * @throws IllegalStateException when the cache was released before the build ran
     * @throws RuntimeException      the build's own failure, unwrapped
     */
    cl_program getOrBuild(String key, Supplier<cl_program> build) {
        FutureTask<cl_program> task = programs.get(key);
        if (task == null) {
            FutureTask<cl_program> fresh = new FutureTask<>(build::get);
            task = programs.putIfAbsent(key, fresh);
            if (task == null) {
                task = fresh;
                if (closed) {
                    programs.remove(key, fresh);
                    fresh.cancel(false);
                    throw closedFailure(key);
                }
                fresh.run();
            }
        }
        return await(key, task);
    }

    /** Number of programs built or being built. */
    int size() {
        return programs.size();
    }

    /**
     * Refuses further builds, waits for the running ones and hands every built program to {@code release}; the
     * cache is empty afterwards. A release failure is thrown after every program was handed over, with the later
     * failures attached.
     */
    void releaseAll(Consumer<cl_program> release) {
        closed = true;
        List<FutureTask<cl_program>> tasks = new ArrayList<>(programs.values());
        programs.clear();
        RuntimeException failure = null;
        for (FutureTask<cl_program> task : tasks) {
            cl_program program = settled(task);
            if (program == null) {
                continue;
            }
            try {
                release.accept(program);
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    private cl_program await(String key, FutureTask<cl_program> task) {
        boolean interrupted = false;
        try {
            while (true) {
                try {
                    return task.get();
                } catch (InterruptedException e) {
                    interrupted = true;
                } catch (ExecutionException e) {
                    programs.remove(key, task);
                    throw unwrap(e);
                } catch (CancellationException e) {
                    throw closedFailure(key);
                }
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /** The program of a finished or running build, waiting for it; null when the build failed. */
    private static cl_program settled(FutureTask<cl_program> task) {
        boolean interrupted = false;
        try {
            while (true) {
                try {
                    return task.get();
                } catch (InterruptedException e) {
                    interrupted = true;
                } catch (ExecutionException | CancellationException e) {
                    return null;
                }
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private static IllegalStateException closedFailure(String key) {
        return new IllegalStateException("OpenCL backend is closed; program " + key + " is not built");
    }

    private static RuntimeException unwrap(ExecutionException e) {
        Throwable cause = e.getCause();
        if (cause instanceof RuntimeException) {
            return (RuntimeException) cause;
        }
        if (cause instanceof Error) {
            throw (Error) cause;
        }
        return new IllegalStateException("OpenCL program build failed: " + cause, cause);
    }
}
