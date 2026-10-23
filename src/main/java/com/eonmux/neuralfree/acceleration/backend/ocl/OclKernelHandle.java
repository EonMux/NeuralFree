package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import org.jocl.cl_kernel;
import org.jocl.cl_program;

import static org.jocl.CL.*;

/**
 * An OpenCL kernel object together with the limits the driver reports for it once it is compiled:
 * its argument count ({@code CL_KERNEL_NUM_ARGS}), its largest work-group size
 * ({@code CL_KERNEL_WORK_GROUP_SIZE}, which register or local-memory pressure can push below the
 * device maximum) and its static local memory ({@code CL_KERNEL_LOCAL_MEM_SIZE}). A limit the driver
 * did not report is {@link #UNKNOWN} and is not enforced.
 */
public final class OclKernelHandle implements KernelHandle {

    /** Value of a limit the driver did not report. */
    public static final long UNKNOWN = -1;

    private cl_program program;
    private cl_kernel kernel;
    private final String kernelName;
    private final boolean ownsProgram;
    private final int numArgs;
    private final long maxWorkGroupSize;
    private final long localMemBytes;
    private final String deviceName;
    private boolean closed = false;

    /** A handle that owns its program and carries no driver-reported limits. */
    public OclKernelHandle(cl_program program, cl_kernel kernel, String kernelName) {
        this(program, kernel, kernelName, true, (int) UNKNOWN, UNKNOWN, UNKNOWN, "");
    }

    /**
     * @param numArgs          {@code CL_KERNEL_NUM_ARGS}, or {@link #UNKNOWN}
     * @param maxWorkGroupSize {@code CL_KERNEL_WORK_GROUP_SIZE}, or {@link #UNKNOWN}
     * @param localMemBytes    {@code CL_KERNEL_LOCAL_MEM_SIZE}, or {@link #UNKNOWN}
     * @param deviceName       the device the kernel was built for, for messages
     */
    OclKernelHandle(cl_program program, cl_kernel kernel, String kernelName, boolean ownsProgram,
                    int numArgs, long maxWorkGroupSize, long localMemBytes, String deviceName) {
        if (program == null) {
            throw new IllegalArgumentException("cl_program cannot be null");
        }
        if (kernel == null) {
            throw new IllegalArgumentException("cl_kernel cannot be null");
        }
        if (kernelName == null || kernelName.isEmpty()) {
            throw new IllegalArgumentException("kernelName cannot be null or empty");
        }
        this.program = program;
        this.kernel = kernel;
        this.kernelName = kernelName;
        this.ownsProgram = ownsProgram;
        this.numArgs = numArgs < 0 ? (int) UNKNOWN : numArgs;
        this.maxWorkGroupSize = maxWorkGroupSize <= 0 ? UNKNOWN : maxWorkGroupSize;
        this.localMemBytes = localMemBytes < 0 ? UNKNOWN : localMemBytes;
        this.deviceName = deviceName == null ? "" : deviceName;
    }

    @Override
    public String kernelName() {
        return kernelName;
    }

    /** {@code CL_KERNEL_NUM_ARGS}, or {@link #UNKNOWN}. */
    public int numArgs() {
        return numArgs;
    }

    /** {@code CL_KERNEL_WORK_GROUP_SIZE}, or {@link #UNKNOWN}. */
    public long maxWorkGroupSize() {
        return maxWorkGroupSize;
    }

    /** {@code CL_KERNEL_LOCAL_MEM_SIZE} in bytes, or {@link #UNKNOWN}. */
    public long localMemBytes() {
        return localMemBytes;
    }

    /**
     * Checks a full argument list against the kernel's argument count.
     *
     * @throws IllegalArgumentException when the count is known and differs
     */
    void checkArgCount(int count) {
        if (numArgs != UNKNOWN && count != numArgs) {
            throw new IllegalArgumentException("OpenCL kernel " + kernelName + " takes " + numArgs
                    + " arguments but " + count + " were given; enqueueKernel(..., args) replaces every argument");
        }
    }

    /**
     * Checks a work-group size against the kernel's compiled limit.
     *
     * @param localWorkSize the work-group size per dimension; null lets the driver choose and passes
     * @throws UnsupportedKernelException when the limit is known and the work-group is larger
     */
    void checkLocalSize(long[] localWorkSize) {
        if (localWorkSize == null || maxWorkGroupSize == UNKNOWN) {
            return;
        }
        long items = 1;
        for (long l : localWorkSize) {
            items = l > 0 && items > Long.MAX_VALUE / l ? Long.MAX_VALUE : items * l;
        }
        if (items > maxWorkGroupSize) {
            throw new UnsupportedKernelException("OpenCL kernel " + kernelName
                    + (deviceName.isEmpty() ? "" : " on " + deviceName) + " supports work-groups of at most "
                    + maxWorkGroupSize + " work-items (CL_KERNEL_WORK_GROUP_SIZE), but a dispatch asked for "
                    + java.util.Arrays.toString(localWorkSize) + " = " + items);
        }
    }

    @Override
    public void close() {
        if (!closed) {
            clReleaseKernel(kernel);
            if (ownsProgram) {
                clReleaseProgram(program);
            }
            kernel = null;
            program = null;
            closed = true;
        }
    }

    public cl_program getCLProgram() {
        return program;
    }

    public cl_kernel getCLKernel() {
        return kernel;
    }
}
