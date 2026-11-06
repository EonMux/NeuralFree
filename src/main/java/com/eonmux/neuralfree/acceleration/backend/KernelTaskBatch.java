package com.eonmux.neuralfree.acceleration.backend;

/**
 * A batch of independent small kernel launches submitted as one device-side submission, so a
 * sequence of short kernels costs one launch instead of one launch per kernel. The decode
 * analogue of a multi-task kernel: the launcher records the tasks it would launch one by one
 * and flushes them together.
 *
 * <p>The tasks run in the order they were added, with the ordering of the queue's in-order
 * execution between them preserved, so the results are the same as launching each task
 * separately. A batch carries its own private copy of every task's arguments; it never changes
 * the arguments of the kernel handles the tasks were added with. Work sizes may differ between
 * tasks. Tasks are recorded with {@link #add} and consumed by {@link #launch}, which clears
 * them; the next launch records the next batch of tasks.</p>
 *
 * <p>A backend may keep the device-side batch structure of the last launch and reuse it when a
 * launch has the same shape (task count, kernels and work sizes per task), updating only the
 * arguments; {@link #graphReuses()} counts those launches. Not thread-safe: add, launch and
 * close come from one thread. Tasks that read or write the same memory must still be ordered
 * against each other like plain launches would be. Every buffer a task references must stay
 * open until the batch is closed, because a retained device-side structure keeps its raw
 * addresses.</p>
 */
public interface KernelTaskBatch extends AutoCloseable {

    /**
     * Records one kernel launch as a task of the batch. The full argument list is given here,
     * replacing any earlier one ({@code enqueueKernel} semantics, snapshot semantics per value).
     *
     * @throws IllegalArgumentException when the kernel, an argument, a buffer or a work size is
     *                                  invalid for this backend
     * @throws IllegalStateException    when the batch or its backend is closed
     */
    void add(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args);

    /** The tasks recorded since the last {@link #launch} or {@link #clear}. */
    int taskCount();

    /**
     * Submits every recorded task as one device-side submission, in add order, and clears the
     * batch. An empty batch is rejected: a caller launches nothing instead.
     *
     * @param queue an open queue of the batch's backend
     * @return an event that completes once the whole batch has finished
     * @throws IllegalArgumentException when the batch has no tasks, the queue is null or belongs
     *                                  to another backend
     * @throws IllegalStateException    when the batch or its backend is closed
     * @throws RuntimeException         the translated backend failure
     */
    GpuEvent launch(ComputeQueue queue, GpuEvent... waitOnEvents);

    /** Discards the recorded tasks without launching them. */
    void clear();

    /** Batches submitted so far. */
    long launches();

    /** Tasks every submitted batch ran. */
    long tasksLaunched();

    /** Submissions that reused the previous batch's device-side structure. */
    long graphReuses();

    /** True once {@link #close()} has run. */
    boolean isClosed();

    /**
     * Releases the batch's device-side resources. Tasks still recorded are discarded.
     * Idempotent.
     *
     * @throws RuntimeException the translated backend failure
     */
    @Override
    void close();
}
