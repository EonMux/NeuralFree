package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_event;

import static org.jocl.CL.*;

/**
 * An OpenCL event. Read events also carry the {@link OclReadCompletion} of the host-side copy, so
 * {@link #awaitCompletion()} returns only after the destination array holds the data, and throws
 * when the read terminated abnormally.
 */
public final class OclGpuEvent implements GpuEvent {
    private cl_event event;
    private boolean closed = false;
    private final OclReadCompletion readCompletion;
    private final boolean profiled;

    public OclGpuEvent(cl_event event) {
        this(event, null, false);
    }

    /**
     * @param event          the OpenCL event; owned by this object
     * @param readCompletion completion of a read's host copy, or null for other commands
     * @param profiled       true when the issuing queue was created with {@code CL_QUEUE_PROFILING_ENABLE}
     */
    OclGpuEvent(cl_event event, OclReadCompletion readCompletion, boolean profiled) {
        if (event == null) {
            throw new IllegalArgumentException("cl_event cannot be null");
        }
        this.event = event;
        this.readCompletion = readCompletion;
        this.profiled = profiled;
    }

    /**
     * Waits for the command and, for a read, for its host copy. When the wait fails because the
     * command itself terminated abnormally, the command's own status is reported (so an
     * out-of-memory termination stays a {@code GpuOutOfMemoryException}); otherwise the wait's.
     *
     * @throws com.eonmux.neuralfree.acceleration.GpuBackendException     when the command or a command
     *                                                                     it waited on failed
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when it failed for lack of memory
     */
    @Override
    public void awaitCompletion() throws InterruptedException {
        if (closed) {
            throw new IllegalStateException("Event already closed");
        }
        cl_event[] waitList = {event};
        try {
            OclErrors.check("clWaitForEvents", readCompletion != null ? "read" : null,
                    () -> clWaitForEvents(1, waitList));
        } catch (RuntimeException e) {
            if (readCompletion != null) {
                readCompletion.markReported();
            }
            int own = queryExecutionStatus(false);
            if (own < 0) {
                RuntimeException failure = OclErrors.translate(own, readCompletion != null ? "clEnqueueReadBuffer"
                        : "OpenCL command", "terminated abnormally", -1, null);
                failure.addSuppressed(e);
                throw failure;
            }
            throw e;
        }
        if (readCompletion != null) {
            readCompletion.await();
            readCompletion.throwIfFailed();
        }
    }

    /**
     * True once the command has completed and, for a read, its host copy has run.
     *
     * @throws com.eonmux.neuralfree.acceleration.GpuBackendException when the command terminated
     *                                                                 abnormally or the query failed
     */
    @Override
    public boolean isComplete() {
        if (closed) {
            return true;
        }
        int status = queryExecutionStatus(true);
        if (status < 0) {
            throw OclErrors.translate(status, "OpenCL command", "terminated abnormally", -1, null);
        }
        if (status != CL_COMPLETE) {
            return false;
        }
        return readCompletion == null || readCompletion.isDone();
    }

    /**
     * {@code CL_EVENT_COMMAND_EXECUTION_STATUS}: a command state, or the negative status of an
     * abnormal termination.
     *
     * @param strict true to throw when the query fails; false to return {@code CL_COMPLETE} instead
     */
    private int queryExecutionStatus(boolean strict) {
        int[] status = new int[1];
        try {
            OclErrors.check("clGetEventInfo", "CL_EVENT_COMMAND_EXECUTION_STATUS",
                    () -> clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, Sizeof.cl_int,
                            Pointer.to(status), null));
        } catch (RuntimeException e) {
            if (strict) {
                throw e;
            }
            return CL_COMPLETE;
        }
        return status[0];
    }

    @Override
    public void close() {
        if (!closed) {
            clReleaseEvent(event);
            event = null;
            closed = true;
        }
    }

    /** True once {@link #close()} has released the event. */
    public boolean isClosed() {
        return closed;
    }

    /**
     * The OpenCL event, for use as a wait by queues of the same backend.
     *
     * @throws IllegalStateException when the event has been closed
     */
    public cl_event getCLEvent() {
        if (closed) {
            throw new IllegalStateException("OpenCL event already closed; it cannot be used as a wait");
        }
        return event;
    }

    /**
     * Returns GPU-side elapsed time in nanoseconds for this event, or -1 only when the issuing
     * queue was created without profiling ({@code -Dneuralfree.gpu.profile=true} enables it).
     * Must be called after the event has completed (after finish() or awaitCompletion()).
     *
     * @return elapsed nanoseconds (CL_PROFILING_COMMAND_END - CL_PROFILING_COMMAND_START), or -1
     * @throws IllegalStateException                                       when the event has been closed
     * @throws com.eonmux.neuralfree.acceleration.GpuBackendException     when a profiling query fails,
     *                                                                     including when the event has
     *                                                                     not completed
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when a query fails for lack
     *                                                                     of memory
     */
    @Override
    public long getElapsedNanos() {
        if (closed || event == null) {
            throw new IllegalStateException("Event already closed");
        }
        if (!profiled) {
            return -1;
        }
        long[] start = new long[1];
        long[] end = new long[1];
        OclErrors.check("clGetEventProfilingInfo", "CL_PROFILING_COMMAND_START",
                () -> clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, Sizeof.cl_ulong,
                        Pointer.to(start), null));
        OclErrors.check("clGetEventProfilingInfo", "CL_PROFILING_COMMAND_END",
                () -> clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, Sizeof.cl_ulong,
                        Pointer.to(end), null));
        return end[0] - start[0];
    }
}
