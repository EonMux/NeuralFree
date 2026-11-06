package com.eonmux.neuralfree.acceleration.hip;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;

/**
 * Maps HIP runtime status codes to the backend's exception types: out of memory to
 * {@code GpuOutOfMemoryException}, a code object the device cannot run or a missing function to
 * {@code UnsupportedKernelException}, anything else to {@code GpuBackendException}.
 *
 * <p>The codes mirror the {@code hipError_t} enum of the ROCm runtime; the shim returns them as
 * plain ints. {@link #check(String, IntSupplier)} runs one call and translates its status.</p>
 */
public final class HipErrors {

    /** {@code hipSuccess}. */
    public static final int SUCCESS = 0;
    /** {@code hipErrorInvalidValue}. */
    public static final int INVALID_VALUE = 1;
    /** {@code hipErrorOutOfMemory}. */
    public static final int OUT_OF_MEMORY = 2;
    /** {@code hipErrorInvalidConfiguration}. */
    public static final int INVALID_CONFIGURATION = 9;
    /** {@code hipErrorInvalidDevice}. */
    public static final int INVALID_DEVICE = 101;
    /** {@code hipErrorInvalidImage}: a corrupt or unsupported code object. */
    public static final int INVALID_IMAGE = 200;
    /** {@code hipErrorInvalidContext}. */
    public static final int INVALID_CONTEXT = 201;
    /** {@code hipErrorNoBinaryForGpu}: the code object holds no code for this processor. */
    public static final int NO_BINARY_FOR_GPU = 209;
    /** {@code hipErrorInvalidKernelFile}: not a code object at all. */
    public static final int INVALID_KERNEL_FILE = 218;
    /** {@code hipErrorInvalidHandle}: an event, stream or module that no longer exists. */
    public static final int INVALID_HANDLE = 400;
    /** {@code hipErrorNotReady}: an asynchronous operation has not finished ({@code hipEventQuery}). */
    public static final int NOT_READY = 600;
    /** {@code hipErrorLaunchOutOfResources}: too many registers or threads for the function. */
    public static final int LAUNCH_OUT_OF_RESOURCES = 701;
    /** {@code hipErrorUnknown}. */
    public static final int UNKNOWN = 999;

    private static final String BACKEND = "HIP";

    private HipErrors() {
    }

    /**
     * Throws the translated failure when {@code code} is not {@code hipSuccess}.
     *
     * @return {@code code} when it is {@code hipSuccess}
     */
    public static int call(int code, String op) {
        if (code == SUCCESS) {
            return code;
        }
        throw translate(code, op, -1);
    }

    /**
     * Runs one driver call and translates its status.
     *
     * @throws RuntimeException the translated failure
     */
    public static void check(String op, IntSupplier call) {
        check(op, -1, call);
    }

    /**
     * Runs one driver call that allocates {@code requestedBytes} and translates its status.
     *
     * @param requestedBytes bytes the call allocates, reported on out-of-memory; -1 when not an allocation
     * @throws RuntimeException the translated failure
     */
    public static void check(String op, long requestedBytes, IntSupplier call) {
        int code = call.getAsInt();
        if (code != SUCCESS) {
            throw translate(code, op, requestedBytes);
        }
    }

    /** Translates a status code returned while performing {@code op}. */
    public static RuntimeException translate(int code, String op, long requestedBytes) {
        String what = op + " failed: " + nameOf(code);
        if (code == OUT_OF_MEMORY) {
            return requestedBytes >= 0
                    ? new GpuOutOfMemoryException(what, requestedBytes, BACKEND)
                    : new GpuOutOfMemoryException(what + " (backend=" + BACKEND + ")");
        }
        if (code == NO_BINARY_FOR_GPU || code == INVALID_IMAGE || code == INVALID_KERNEL_FILE) {
            return new UnsupportedKernelException(what + " (the code object does not support this device)", null);
        }
        if (code == INVALID_HANDLE) {
            return new GpuBackendException(BACKEND, code, what + " (the stream, event or module no longer exists)");
        }
        return new GpuBackendException(BACKEND, code, what);
    }

    /** The driver's symbolic name of a status code, for messages. */
    public static String nameOf(int code) {
        if (code == SUCCESS) {
            return "hipSuccess";
        }
        String symbolic = HipNative.isAvailable() ? HipNative.errorString(code) : null;
        return symbolic == null || symbolic.isEmpty() ? "hipError " + code : symbolic;
    }

    /**
     * True for the module-load statuses that mean this device cannot run the image: no code for
     * the processor, a corrupt or foreign image.
     */
    public static boolean isImageIncompatible(int code) {
        return code == NO_BINARY_FOR_GPU || code == INVALID_IMAGE || code == INVALID_KERNEL_FILE;
    }

    /** A supplier of an {@code int} status, the shape every HIP call returns. */
    @FunctionalInterface
    public interface IntSupplier {
        /** Runs the call and returns its status. */
        int getAsInt();
    }
}
