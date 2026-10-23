package com.eonmux.neuralfree.acceleration.backend.cuda;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;

import java.util.concurrent.atomic.AtomicBoolean;

import static jcuda.driver.JCudaDriver.cuCtxSetCurrent;
import static jcuda.driver.JCudaDriver.cuDevicePrimaryCtxRelease;
import static jcuda.driver.JCudaDriver.cuDevicePrimaryCtxRetain;

/**
 * Reference-counted handle on a device's CUDA primary context, made current on
 * the calling thread at every public entry point of the CUDA backend.
 *
 * <p>Each instance holds exactly one retain of the primary context, so several
 * backends (and any future cudart or cuBLAS user) on one device share a single
 * context. {@link #makeCurrent()} keeps the last context it set per thread and
 * calls {@code cuCtxSetCurrent} only when the thread last ran another context.
 * Code that switches contexts on a backend thread without going through this
 * class (for example {@code cuCtxCreate}) must not interleave with backend calls
 * on that thread.</p>
 */
public final class CudaContext {

    /** The context each thread last made current through any {@code CudaContext}. */
    private static final ThreadLocal<CUcontext> CURRENT = new ThreadLocal<>();

    private final CUdevice device;
    private final CUcontext context;
    private final AtomicBoolean released = new AtomicBoolean(false);

    private CudaContext(CUdevice device, CUcontext context) {
        this.device = device;
        this.context = context;
    }

    /**
     * Retains the primary context of {@code device} ({@code cuDevicePrimaryCtxRetain}).
     * The driver must already be initialised ({@code cuInit}).
     *
     * @throws IllegalArgumentException when {@code device} is null
     * @throws RuntimeException         the translated driver failure
     */
    public static CudaContext retainPrimary(CUdevice device) {
        if (device == null) {
            throw new IllegalArgumentException("CUDA device must not be null");
        }
        CUcontext ctx = new CUcontext();
        CudaErrors.check("cuDevicePrimaryCtxRetain", () -> cuDevicePrimaryCtxRetain(ctx, device));
        return new CudaContext(device, ctx);
    }

    /** The retained primary context. */
    public CUcontext context() {
        return context;
    }

    /** The device the context belongs to. */
    public CUdevice device() {
        return device;
    }

    /**
     * Makes the context current on this thread; a thread-local identity check skips redundant calls.
     *
     * @throws IllegalStateException when the context has been released
     * @throws RuntimeException      the translated driver failure
     */
    public void makeCurrent() {
        if (released.get()) {
            throw new IllegalStateException("CUDA primary context of " + device + " has been released");
        }
        if (CURRENT.get() == context) {
            return;
        }
        CudaErrors.check("cuCtxSetCurrent", () -> cuCtxSetCurrent(context));
        CURRENT.set(context);
    }

    /** True once {@link #release()} has run. */
    public boolean isReleased() {
        return released.get();
    }

    /**
     * Releases the retained primary context ({@code cuDevicePrimaryCtxRelease}); idempotent.
     * When this was the last retain, the driver destroys the context and every allocation in it.
     *
     * @throws RuntimeException the translated driver failure of the first release
     */
    public void release() {
        if (!released.compareAndSet(false, true)) {
            return;
        }
        if (CURRENT.get() == context) {
            CURRENT.remove();
        }
        CudaErrors.check("cuDevicePrimaryCtxRelease", () -> cuDevicePrimaryCtxRelease(device));
    }

    @Override
    public String toString() {
        return "CudaContext{" + device + (released.get() ? ", released" : "") + "}";
    }
}
