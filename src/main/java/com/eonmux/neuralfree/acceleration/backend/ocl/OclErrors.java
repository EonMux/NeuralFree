package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import org.jocl.CLException;

import java.util.function.IntSupplier;

import static org.jocl.CL.CL_MEM_OBJECT_ALLOCATION_FAILURE;
import static org.jocl.CL.CL_OUT_OF_HOST_MEMORY;
import static org.jocl.CL.CL_OUT_OF_RESOURCES;
import static org.jocl.CL.CL_SUCCESS;
import static org.jocl.CL.stringFor_errorCode;

/**
 * Maps OpenCL status codes, returned or thrown, to the backend's exception types:
 * {@code CL_MEM_OBJECT_ALLOCATION_FAILURE}, {@code CL_OUT_OF_RESOURCES} and
 * {@code CL_OUT_OF_HOST_MEMORY} to {@link GpuOutOfMemoryException}, every other failure to
 * {@link GpuBackendException} carrying the status.
 *
 * <p>OpenCL allocates buffer storage lazily, so an out-of-memory condition usually surfaces at the
 * first enqueue that touches a buffer rather than at {@code clCreateBuffer}; every queue call is
 * therefore checked through this class.</p>
 *
 * <p>A GPU hang is reported through these statuses too. When the kernel driver resets a context whose
 * work hung, the Intel compute runtime ends every command queued on it with {@link #EXECUTION_ABORTED}
 * and fails the queue's synchronizations with {@code CL_OUT_OF_RESOURCES}; {@link #checkWait} names
 * both readings of that status, since the same one also reports a lazily allocated buffer the device
 * could not hold.</p>
 *
 * <p>JOCL either returns the status (exceptions disabled, its default) or throws a
 * {@link CLException} carrying it (exceptions enabled process-wide through
 * {@code CL.setExceptionsEnabled(true)}). {@link #check(String, String, IntSupplier)} handles both
 * modes, so the mapping holds whichever mode another component left JOCL in.</p>
 */
public final class OclErrors {

    /** Backend name carried by {@link GpuBackendException} and {@link GpuOutOfMemoryException}. */
    public static final String BACKEND = "OpenCL";
    /**
     * Execution status the Intel compute runtime gives a command that a GPU hang aborted (outside the
     * Khronos status range): the kernel driver reset the context the command was queued on.
     */
    public static final int EXECUTION_ABORTED = -777;

    private static final String WAIT_FAILURE_NOTE = "the device could not run the queued work: it ran out of"
            + " resources, or the kernel driver reset its context after a GPU hang";

    private OclErrors() {
    }

    /** True for the statuses that report exhausted device, host or driver memory. */
    public static boolean isOutOfMemory(int status) {
        return status == CL_MEM_OBJECT_ALLOCATION_FAILURE
                || status == CL_OUT_OF_RESOURCES
                || status == CL_OUT_OF_HOST_MEMORY;
    }

    /** The symbolic name of a status, for messages. */
    public static String nameOf(int status) {
        if (status == EXECUTION_ABORTED) {
            return "EXECUTION_ABORTED (aborted by a GPU hang that reset the context)";
        }
        return stringFor_errorCode(status);
    }

    /**
     * Runs one OpenCL call and translates its failure in either JOCL exception mode.
     *
     * @param op      the API call, for example {@code "clEnqueueNDRangeKernel"}
     * @param subject what the call acted on (a kernel name, a byte count), or null
     * @throws GpuOutOfMemoryException when the status reports exhausted memory
     * @throws GpuBackendException     for every other failure
     */
    public static void check(String op, String subject, IntSupplier call) {
        int status;
        try {
            status = call.getAsInt();
        } catch (CLException e) {
            throw translate(e.getStatus(), op, subject, -1, e);
        }
        if (status != CL_SUCCESS) {
            throw translate(status, op, subject, -1, null);
        }
    }

    /**
     * Runs one call that waits for queued work ({@code clFinish}, {@code clWaitForEvents}) and translates its
     * failure as {@link #check(String, String, IntSupplier)} does; a {@code CL_OUT_OF_RESOURCES} failure names
     * both of its causes at a wait, an exhausted device or a context the kernel driver reset after a GPU hang.
     *
     * @throws GpuOutOfMemoryException when the status reports exhausted memory
     * @throws GpuBackendException     for every other failure
     */
    public static void checkWait(String op, String subject, IntSupplier call) {
        int status;
        CLException thrown = null;
        try {
            status = call.getAsInt();
        } catch (CLException e) {
            status = e.getStatus();
            thrown = e;
        }
        if (status != CL_SUCCESS) {
            String note = status == CL_OUT_OF_RESOURCES ? WAIT_FAILURE_NOTE : null;
            String what = subject == null || subject.isEmpty() ? note
                    : note == null ? subject : subject + "; " + note;
            throw translate(status, op, what, -1, thrown);
        }
    }

    /**
     * Checks a status returned by a call made with exceptions disabled.
     *
     * @throws GpuOutOfMemoryException when the status reports exhausted memory
     * @throws GpuBackendException     for every other failure
     */
    public static void check(int status, String op, String subject) {
        if (status != CL_SUCCESS) {
            throw translate(status, op, subject, -1, null);
        }
    }

    /**
     * Translates a failed status.
     *
     * @param status         the OpenCL status; must not be {@code CL_SUCCESS}
     * @param op             the API call or the operation that reported the status
     * @param subject        what the call acted on, or null
     * @param requestedBytes bytes the call allocates, reported on out-of-memory; -1 when not an allocation
     * @param cause          the JOCL exception that carried the status, or null
     */
    public static RuntimeException translate(int status, String op, String subject, long requestedBytes,
                                             Throwable cause) {
        if (status == CL_SUCCESS) {
            throw new IllegalArgumentException("CL_SUCCESS is not a failure of " + op);
        }
        String what = op + (subject == null || subject.isEmpty() ? "" : " (" + subject + ")") + " failed: "
                + nameOf(status);
        if (isOutOfMemory(status)) {
            GpuOutOfMemoryException oom = requestedBytes >= 0
                    ? new GpuOutOfMemoryException(what, requestedBytes, BACKEND)
                    : new GpuOutOfMemoryException(what + " (backend=" + BACKEND + ")");
            if (cause != null) {
                oom.initCause(cause);
            }
            return oom;
        }
        return cause == null
                ? new GpuBackendException(BACKEND, status, what)
                : new GpuBackendException(BACKEND, status, what, cause);
    }
}
