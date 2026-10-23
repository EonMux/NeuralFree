package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.backend.StickyArgs;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;

import java.util.Arrays;
import java.util.BitSet;

/**
 * Sticky launch arguments of one CUDA kernel handle, kept as a reusable
 * {@code kernelParams} pointer array. Scalar values live in handle-owned arrays,
 * so a value change needs no rebuild; the pointer array is rebuilt only when an
 * argument's kind or buffer changes.
 *
 * <p>Parameter slots (each a pointer to the bytes of one kernel parameter, read by
 * {@code cuLaunchKernel} at launch):</p>
 * <ul>
 *   <li>{@link GpuBuffer}: a pointer to the buffer's {@link CUdeviceptr}, so the kernel
 *       receives the 8-byte device address;</li>
 *   <li>{@code Integer}, {@code Boolean} (0/1), {@code int[1]}: a handle-owned {@code int[1]};</li>
 *   <li>{@code Float}, {@code float[1]}: a handle-owned {@code float[1]};</li>
 *   <li>{@code Long}, {@code long[1]}, {@code Double}, {@code double[1]}: a handle-owned
 *       one-element {@code long[]}/{@code double[]} (legacy ML callers);</li>
 *   <li>other primitive arrays: a handle-owned copy passed by value (legacy ML callers).</li>
 * </ul>
 * Values are captured when they are set (snapshot semantics); later mutation of the
 * caller's holder has no effect until the argument is set again. Buffer arguments must
 * be open buffers of the handle's backend when they are set, and still open when the
 * parameters are taken for a launch.
 *
 * <p><b>Signature check.</b> A handle with a build-time {@link KernelSignature} (every kernel of a
 * kernel library) has its argument list checked before the parameters are handed to
 * {@code cuLaunchKernel}: the argument count, the number of buffers and of scalars, and each
 * scalar's kind in scalar order (an int32/uint32 parameter takes {@code Integer}, {@code Boolean} or
 * {@code int[1]}; a float32 parameter {@code Float} or {@code float[1]}; 8-byte and array values are
 * rejected). The driver reads exactly as many parameter slots, of exactly the sizes, as the kernel
 * declares, so a list that disagrees would make it read past the slot array or reinterpret values.
 * The check runs only when an argument's kind or the argument count changed.</p>
 */
@NotThreadSafe("Mutated by setKernelArg/enqueueKernel; callers serialise access per handle.")
public final class CudaArgState {

    private static final int INITIAL_CAPACITY = 16;

    private final CudaComputeBackend backend;
    private final KernelSignature signature;
    private final StickyArgs sticky;
    /** Per index: the CUdeviceptr of a buffer arg, or the handle-owned value array of a by-value arg. */
    private Object[] holders = new Object[INITIAL_CAPACITY];
    /** Per index: the parameter pointer handed to the driver; replaced only when the holder object changes. */
    private NativePointerObject[] slots = new NativePointerObject[INITIAL_CAPACITY];
    private Pointer kernelParams;
    private int builtCount = -1;
    private boolean stale = true;
    private long checkedLayoutVersion = -1;

    /**
     * @param backend    the backend whose buffers the arguments may reference
     * @param kernelName kernel name for messages
     * @param signature  build-time argument signature of the kernel, or null to launch unchecked
     * @throws IllegalArgumentException when the backend is null or the kernel name is null or empty
     */
    public CudaArgState(CudaComputeBackend backend, String kernelName, KernelSignature signature) {
        if (backend == null) {
            throw new IllegalArgumentException("CUDA backend must not be null");
        }
        this.backend = backend;
        this.signature = signature;
        this.sticky = new StickyArgs(kernelName);
    }

    /**
     * Sets one argument with snapshot semantics.
     *
     * @throws IllegalArgumentException when the index is negative, the argument is null or of an
     *                                  unsupported type, or a buffer is not an open CUDA
     *                                  buffer of this backend; the stored arguments are then unchanged
     */
    public void set(int index, Object arg) {
        if (arg instanceof GpuBuffer) {
            devicePointerOf((GpuBuffer) arg, index);
        }
        sticky.set(index, arg);
        sync();
    }

    /**
     * Replaces every argument, as {@code enqueueKernel(..., args)} does.
     *
     * @throws IllegalArgumentException when the array or an element is null or unsupported, or a
     *                                  buffer is not an open CUDA buffer of this backend; the stored
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
        sync();
    }

    /**
     * The {@code kernelParams} pointer for {@code cuLaunchKernel}. The same object is returned
     * until an argument's kind, buffer or the argument count changes.
     *
     * @throws IllegalStateException    naming the first unset argument when the list is incomplete,
     *                                  or the first buffer argument closed since it was set
     * @throws IllegalArgumentException naming the kernel and the argument when the list does not match
     *                                  the kernel's signature
     */
    public Pointer kernelParams() {
        sticky.requireComplete();
        checkSignature();
        int count = sticky.count();
        for (int i = 0; i < count; i++) {
            Object value = sticky.get(i);
            if (value instanceof CudaGpuBuffer && ((CudaGpuBuffer) value).isClosed()) {
                throw new IllegalStateException("arg " + i + " of " + sticky.kernelName()
                        + " is a CUDA buffer that was closed after it was set");
            }
        }
        if (stale || builtCount != count) {
            kernelParams = Pointer.to(Arrays.copyOf(slots, count));
            builtCount = count;
            stale = false;
        }
        return kernelParams;
    }

    /** The underlying argument snapshots. */
    public StickyArgs sticky() {
        return sticky;
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
     * signature's parameter layout declares one (when it records a layout), and each scalar's kind
     * must be the one its position in the scalar order declares.
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

    /** Copies every changed snapshot into its parameter holder. */
    private void sync() {
        BitSet dirty = sticky.dirty();
        for (int i = dirty.nextSetBit(0); i >= 0; i = dirty.nextSetBit(i + 1)) {
            if (sticky.isSet(i)) {
                bind(i, sticky.get(i));
            }
        }
        sticky.clearDirty();
    }

    private void bind(int i, Object value) {
        ensureCapacity(i + 1);
        if (value instanceof GpuBuffer) {
            CUdeviceptr ptr = devicePointerOf((GpuBuffer) value, i);
            if (holders[i] != ptr) {
                replace(i, ptr, Pointer.to(ptr));
            }
            return;
        }
        if (value instanceof Integer) {
            putInt(i, (Integer) value);
        } else if (value instanceof Boolean) {
            putInt(i, ((Boolean) value) ? 1 : 0);
        } else if (value instanceof Float) {
            putFloat(i, (Float) value);
        } else if (value instanceof Long) {
            putLong(i, (Long) value);
        } else if (value instanceof Double) {
            putDouble(i, (Double) value);
        } else {
            putArray(i, value);
        }
    }

    private void putInt(int i, int v) {
        if (holders[i] instanceof int[] && ((int[]) holders[i]).length == 1) {
            ((int[]) holders[i])[0] = v;
        } else {
            int[] h = {v};
            replace(i, h, Pointer.to(h));
        }
    }

    private void putFloat(int i, float v) {
        if (holders[i] instanceof float[] && ((float[]) holders[i]).length == 1) {
            ((float[]) holders[i])[0] = v;
        } else {
            float[] h = {v};
            replace(i, h, Pointer.to(h));
        }
    }

    private void putLong(int i, long v) {
        if (holders[i] instanceof long[] && ((long[]) holders[i]).length == 1) {
            ((long[]) holders[i])[0] = v;
        } else {
            long[] h = {v};
            replace(i, h, Pointer.to(h));
        }
    }

    private void putDouble(int i, double v) {
        if (holders[i] instanceof double[] && ((double[]) holders[i]).length == 1) {
            ((double[]) holders[i])[0] = v;
        } else {
            double[] h = {v};
            replace(i, h, Pointer.to(h));
        }
    }

    /** A primitive-array snapshot: copied into a same-shaped holder, or cloned into a new one. */
    private void putArray(int i, Object array) {
        Object holder = holders[i];
        if (holder != null && holder.getClass() == array.getClass()
                && java.lang.reflect.Array.getLength(holder) == java.lang.reflect.Array.getLength(array)) {
            System.arraycopy(array, 0, holder, 0, java.lang.reflect.Array.getLength(array));
            return;
        }
        if (array instanceof int[]) {
            int[] h = ((int[]) array).clone();
            replace(i, h, Pointer.to(h));
        } else if (array instanceof float[]) {
            float[] h = ((float[]) array).clone();
            replace(i, h, Pointer.to(h));
        } else if (array instanceof long[]) {
            long[] h = ((long[]) array).clone();
            replace(i, h, Pointer.to(h));
        } else if (array instanceof double[]) {
            double[] h = ((double[]) array).clone();
            replace(i, h, Pointer.to(h));
        } else if (array instanceof byte[]) {
            byte[] h = ((byte[]) array).clone();
            replace(i, h, Pointer.to(h));
        } else if (array instanceof short[]) {
            short[] h = ((short[]) array).clone();
            replace(i, h, Pointer.to(h));
        } else {
            throw new IllegalArgumentException("arg " + i + " of " + sticky.kernelName()
                    + ": unsupported kernel argument type " + KernelArgs.describe(array));
        }
    }

    private void replace(int i, Object holder, NativePointerObject slot) {
        holders[i] = holder;
        slots[i] = slot;
        stale = true;
    }

    private CUdeviceptr devicePointerOf(GpuBuffer buffer, int index) {
        if (!(buffer instanceof CudaGpuBuffer)) {
            throw new IllegalArgumentException("arg " + index + " of " + sticky.kernelName() + " is a "
                    + buffer.getClass().getSimpleName() + ", not a CUDA buffer");
        }
        CudaGpuBuffer cuda = (CudaGpuBuffer) buffer;
        if (cuda.backend() != backend) {
            throw new IllegalArgumentException("arg " + index + " of " + sticky.kernelName()
                    + " is a buffer of another CUDA backend");
        }
        CUdeviceptr ptr = cuda.getCUDevicePtr();
        if (ptr == null) {
            throw new IllegalArgumentException("arg " + index + " of " + sticky.kernelName()
                    + " is a closed CUDA buffer");
        }
        return ptr;
    }

    private void ensureCapacity(int needed) {
        if (needed <= holders.length) {
            return;
        }
        int capacity = Math.max(needed, holders.length * 2);
        holders = Arrays.copyOf(holders, capacity);
        slots = Arrays.copyOf(slots, capacity);
    }
}
