package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import jcuda.CudaException;
import jcuda.driver.CUresult;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.IntSupplier;

/**
 * Maps CUDA driver results, thrown or returned, to the backend's exception types:
 * out of memory to {@code GpuOutOfMemoryException}, a missing symbol or a module
 * image the device cannot run to {@code UnsupportedKernelException}, anything else
 * to {@code GpuBackendException}.
 *
 * <p>JCuda either throws a {@link CudaException} carrying
 * {@link CUresult#stringFor(int)} as its message (exceptions enabled, which
 * {@code HWAcceleration} turns on) or returns the result code (exceptions
 * disabled). {@link #check(String, IntSupplier)} handles both modes.</p>
 */
public final class CudaErrors {

    /** {@code CUDA_SUCCESS}. */
    public static final int SUCCESS = CUresult.CUDA_SUCCESS;
    /** {@code CUDA_ERROR_OUT_OF_MEMORY}. */
    public static final int OUT_OF_MEMORY = CUresult.CUDA_ERROR_OUT_OF_MEMORY;
    /** {@code CUDA_ERROR_NOT_FOUND}. */
    public static final int NOT_FOUND = CUresult.CUDA_ERROR_NOT_FOUND;
    /** {@code CUDA_ERROR_NOT_READY}. */
    public static final int NOT_READY = CUresult.CUDA_ERROR_NOT_READY;

    private static final String BACKEND = "CUDA";
    private static final String UNKNOWN_PREFIX = "INVALID CUresult: ";
    /** Upper bound of the driver's result codes ({@code CUDA_ERROR_UNKNOWN} is 999). */
    private static final int MAX_CODE = 1000;
    private static final Map<String, Integer> CODES_BY_NAME = buildCodeTable();

    private CudaErrors() {
    }

    /**
     * Checks the result of a driver call made with exceptions disabled.
     *
     * @return {@code result} when it is {@code CUDA_SUCCESS}
     * @throws RuntimeException the translated failure otherwise
     */
    public static int call(int result, String op) {
        if (result == SUCCESS) {
            return result;
        }
        throw translate(result, op, -1);
    }

    /**
     * Runs one driver call and translates its failure in either JCuda exception mode.
     *
     * @throws RuntimeException the translated failure
     */
    public static void check(String op, IntSupplier call) {
        check(op, -1, call);
    }

    /**
     * Runs one driver call that allocates {@code requestedBytes} and translates its
     * failure in either JCuda exception mode.
     *
     * @param requestedBytes bytes the call allocates, reported on out-of-memory; -1 when not an allocation
     * @throws RuntimeException the translated failure
     */
    public static void check(String op, long requestedBytes, IntSupplier call) {
        int result;
        try {
            result = call.getAsInt();
        } catch (CudaException e) {
            throw translate(e, op, requestedBytes);
        }
        if (result != SUCCESS) {
            throw translate(result, op, requestedBytes);
        }
    }

    /** Translates a driver exception thrown while performing {@code op}. */
    public static RuntimeException translate(CudaException e, String op, long requestedBytes) {
        return translate(codeOf(e), op, requestedBytes, e);
    }

    /** Translates a driver result code returned while performing {@code op}. */
    public static RuntimeException translate(int code, String op, long requestedBytes) {
        return translate(code, op, requestedBytes, null);
    }

    /** True when the exception reports {@code CUDA_ERROR_NOT_READY}. */
    public static boolean isNotReady(CudaException e) {
        return codeOf(e) == NOT_READY;
    }

    /** The driver result code carried by the exception message, or -1 when it cannot be recovered. */
    public static int codeOf(CudaException e) {
        if (e == null || e.getMessage() == null) {
            return -1;
        }
        String message = e.getMessage().trim();
        Integer code = CODES_BY_NAME.get(message);
        if (code != null) {
            return code;
        }
        if (message.startsWith(UNKNOWN_PREFIX)) {
            try {
                return Integer.parseInt(message.substring(UNKNOWN_PREFIX.length()).trim());
            } catch (NumberFormatException ignored) {
                return -1;
            }
        }
        return -1;
    }

    /** The driver's symbolic name of a result code, for messages. */
    public static String nameOf(int code) {
        return code < 0 ? "unknown CUDA error" : CUresult.stringFor(code);
    }

    /**
     * True for the module-load results that mean this device or driver cannot run the
     * image: no SASS or PTX for the architecture, a PTX version newer than the driver,
     * a corrupt image, or no JIT available.
     */
    public static boolean isImageIncompatible(int code) {
        return code == CUresult.CUDA_ERROR_INVALID_IMAGE
                || code == CUresult.CUDA_ERROR_NO_BINARY_FOR_GPU
                || code == CUresult.CUDA_ERROR_INVALID_PTX
                || code == CUresult.CUDA_ERROR_JIT_COMPILER_NOT_FOUND
                || code == CUresult.CUDA_ERROR_UNSUPPORTED_PTX_VERSION
                || code == CUresult.CUDA_ERROR_JIT_COMPILATION_DISABLED;
    }

    private static RuntimeException translate(int code, String op, long requestedBytes, Throwable cause) {
        String what = op + " failed: " + nameOf(code);
        if (code == OUT_OF_MEMORY) {
            GpuOutOfMemoryException oom = requestedBytes >= 0
                    ? new GpuOutOfMemoryException(what, requestedBytes, BACKEND)
                    : new GpuOutOfMemoryException(what + " (backend=" + BACKEND + ")");
            if (cause != null) {
                oom.initCause(cause);
            }
            return oom;
        }
        if (code == NOT_FOUND) {
            return new UnsupportedKernelException(what, cause);
        }
        if (isImageIncompatible(code)) {
            return new UnsupportedKernelException(what + " (the module image does not support this device or driver)",
                    cause);
        }
        return cause == null
                ? new GpuBackendException(BACKEND, code, what)
                : new GpuBackendException(BACKEND, code, what, cause);
    }

    private static Map<String, Integer> buildCodeTable() {
        Map<String, Integer> table = new HashMap<>();
        for (int code = 0; code <= MAX_CODE; code++) {
            String name = CUresult.stringFor(code);
            if (!name.startsWith(UNKNOWN_PREFIX)) {
                table.put(name, code);
            }
        }
        return Collections.unmodifiableMap(table);
    }
}
