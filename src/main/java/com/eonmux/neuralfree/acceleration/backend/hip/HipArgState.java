package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.backend.StickyArgs;

import java.util.BitSet;

/**
 * Sticky launch arguments of one HIP kernel handle, materialized per launch into the
 * {@code Object[]} the shim turns into {@code hipModuleLaunchKernel}'s parameter table.
 *
 * <p>Parameter values (read by the runtime at launch):</p>
 * <ul>
 *   <li>{@link GpuBuffer}: the buffer's device pointer, validated to be an open buffer of the
 *       handle's backend at every launch;</li>
 *   <li>{@code Integer}, {@code Boolean} (0/1), {@code int[1]}: a 4-byte parameter;</li>
 *   <li>{@code Float}, {@code float[1]}: a 4-byte parameter;</li>
 *   <li>{@code Long}, {@code long[1]}, {@code Double}, {@code double[1]}: an 8-byte parameter
 *       (legacy ML callers);</li>
 *   <li>other primitive arrays: passed by reference for the launch, from the snapshot the
 *       {@link StickyArgs} store holds (legacy ML callers).</li>
 * </ul>
 * Values are captured when they are set (snapshot semantics); later mutation of the caller's
 * holder has no effect until the argument is set again.
 *
 * <p><b>Signature check.</b> A handle with a build-time {@link KernelSignature} (every kernel of
 * a kernel library) has its argument list checked before the parameters are handed to the
 * launch: the argument count, the number of buffers and of scalars, and each scalar's kind in
 * scalar order. The runtime reads exactly as many parameters of exactly the sizes as the kernel
 * declares, so a list that disagrees would make it read past the parameter table or reinterpret
 * values. The check runs only when an argument's kind or the argument count changed.</p>
 */
@NotThreadSafe("Mutated by setKernelArg/enqueueKernel; callers serialise access per handle.")
public final class HipArgState {

    private final HipComputeBackend backend;
    private final KernelSignature signature;
    private final StickyArgs sticky;
    private long checkedLayoutVersion = -1;
    /** The last parameter table and the argument version it was built at. */
    private Object[] params;
    private long paramsVersion = -1;

    /**
     * @param backend    the backend whose buffers the arguments may reference
     * @param kernelName kernel name for messages
     * @param signature  build-time argument signature of the kernel, or null to launch unchecked
     * @throws IllegalArgumentException when the backend is null or the kernel name is null or empty
     */
    public HipArgState(HipComputeBackend backend, String kernelName, KernelSignature signature) {
        if (backend == null) {
            throw new IllegalArgumentException("HIP backend must not be null");
        }
        this.backend = backend;
        this.signature = signature;
        this.sticky = new StickyArgs(kernelName);
    }

    /**
     * Sets one argument with snapshot semantics.
     *
     * @throws IllegalArgumentException when the index is negative, the argument is null or of an
     *                                  unsupported type, or a buffer is not an open HIP buffer of
     *                                  this backend; the stored arguments are then unchanged
     */
    public void set(int index, Object arg) {
        if (arg instanceof GpuBuffer) {
            devicePointerOf((GpuBuffer) arg, index);
        }
        sticky.set(index, arg);
    }

    /**
     * Replaces every argument, as {@code enqueueKernel(..., args)} does.
     *
     * @throws IllegalArgumentException when the array or an element is null or unsupported, or a
     *                                  buffer is not an open HIP buffer of this backend; the stored
     *                                  arguments are then unchanged
     */
    public void setAll(Object[] args) {
        if (args == null) {
            throw new IllegalArgumentException("argument array of " + sticky.kernelName() + " is null");
        }
        for (int i = 0; i < args.length; i++) {
            if (args[i] instanceof GpuBuffer) {
                devicePointerOf((GpuBuffer) args[i], i);
            }
        }
        sticky.setAll(args);
    }

    /**
     * The parameter table for {@code hipModuleLaunchKernel}: one element per argument, buffers as
     * their device pointer, everything else as its snapshot. The table is built when an argument
     * changed since the last call and shared otherwise, so a launch that repeats its arguments (every
     * decode step) allocates nothing; the buffers are checked to be open on every call. The shim
     * copies the values during the launch call, and callers only read the table.
     *
     * @throws IllegalStateException    naming the first unset argument when the list is incomplete,
     *                                  or the first buffer argument closed since it was set
     * @throws IllegalArgumentException naming the kernel and the argument when the list does not match
     *                                  the kernel's signature
     */
    public Object[] launchParams() {
        sticky.requireComplete();
        checkSignature();
        long version = sticky.version();
        if (params != null && paramsVersion == version) {
            for (int i = 0; i < params.length; i++) {
                requireOpen(sticky.get(i), i);
            }
            return params;
        }
        int count = sticky.count();
        Object[] built = new Object[count];
        for (int i = 0; i < count; i++) {
            Object value = sticky.get(i);
            requireOpen(value, i);
            built[i] = value instanceof HipGpuBuffer ? (Object) ((HipGpuBuffer) value).getDevicePointer() : value;
        }
        params = built;
        paramsVersion = version;
        return built;
    }

    private void requireOpen(Object value, int i) {
        if (value instanceof HipGpuBuffer && ((HipGpuBuffer) value).isClosed()) {
            throw new IllegalStateException("arg " + i + " of " + sticky.kernelName()
                    + " is a HIP buffer that was closed after it was set");
        }
    }

    /** The underlying argument snapshots. */
    public StickyArgs sticky() {
        return sticky;
    }

    /**
     * Deep-cloned copy of the current argument values, in argument order: buffers by identity, scalars
     * boxed, arrays cloned, unset indices null. A captured command graph node keeps this as the record
     * of what its parameters hold.
     */
    public Object[] snapshotValues() {
        int n = sticky.count();
        Object[] out = new Object[n];
        for (int i = 0; i < n; i++) {
            out[i] = sticky.isSet(i) ? KernelArgs.snapshot(sticky.get(i), null) : null;
        }
        return out;
    }

    /** The build-time signature launches are checked against, or null. */
    public KernelSignature signature() {
        return signature;
    }

    public int count() {
        return sticky.count();
    }

    public boolean isComplete() {
        return sticky.isComplete();
    }

    /**
     * Checks the complete argument list against the signature, once per argument layout: buffers
     * and scalars are counted in declared order, each argument must be a buffer exactly where the
     * signature's parameter layout declares one (when it records a layout), and each scalar's
     * kind must be the one its position in the scalar order declares.
     */
    private void checkSignature() {
        if (signature == null || checkedLayoutVersion == sticky.layoutVersion()) {
            return;
        }
        String kernelName = sticky.kernelName();
        int count = sticky.count();
        int buffers = 0;
        int scalars = 0;
        for (int i = 0; i < count; i++) {
            KernelArgs.ArgKind kind = sticky.kind(i);
            signature.checkPosition(kernelName, i, kind, KernelArgs.describe(sticky.get(i)));
            if (kind == KernelArgs.ArgKind.BUFFER) {
                buffers++;
            } else {
                signature.checkScalar(kernelName, i, scalars, kind, KernelArgs.describe(sticky.get(i)));
                scalars++;
            }
        }
        signature.checkCounts(kernelName, buffers, scalars);
        checkedLayoutVersion = sticky.layoutVersion();
    }

    private long devicePointerOf(GpuBuffer buffer, int index) {
        if (!(buffer instanceof HipGpuBuffer)) {
            throw new IllegalArgumentException("arg " + index + " of " + sticky.kernelName() + " is a "
                    + buffer.getClass().getSimpleName() + ", not a HIP buffer");
        }
        HipGpuBuffer hip = (HipGpuBuffer) buffer;
        if (hip.backend() != backend) {
            throw new IllegalArgumentException("arg " + index + " of " + sticky.kernelName()
                    + " is a buffer of another HIP backend");
        }
        Long ptr = hip.getDevicePointer();
        if (ptr == null) {
            throw new IllegalArgumentException("arg " + index + " of " + sticky.kernelName()
                    + " is a closed HIP buffer");
        }
        return ptr;
    }
}
