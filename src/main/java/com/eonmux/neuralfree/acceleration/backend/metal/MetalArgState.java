package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.backend.StickyArgs;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Sticky arguments of one Metal kernel handle, materialised at dispatch into buffer bindings and
 * by-value scalar bytes.
 *
 * <p>Mapping: the k-th buffer argument binds at MSL buffer index {@code k} and the scalar
 * arguments are packed, in argument order, into one little-endian constant buffer bound at index
 * {@code n} ({@code n} buffers). Generated MSL declares that block as the kernel's push-constant
 * struct, so the packed words are exactly its members. The layout is validated against the
 * build-time {@link KernelSignature} when the handle has one (a kernel-library artifact): the
 * buffer count, the scalar count and each scalar's kind must match. A handle compiled from raw MSL
 * text ({@link MetalComputeBackend#compileKernel}) has no signature and binds by position.</p>
 */
@NotThreadSafe("Owned by one kernel handle; callers serialise dispatches of a handle.")
public final class MetalArgState {

    private final String kernelName;
    private final KernelSignature signature;
    private final StickyArgs sticky;

    /**
     * @param signature build-time signature, or null for a kernel compiled from raw MSL text
     * @throws IllegalArgumentException when the kernel name is empty
     */
    public MetalArgState(String kernelName, KernelSignature signature) {
        if (kernelName == null || kernelName.isEmpty()) {
            throw new IllegalArgumentException("kernelName must not be null or empty");
        }
        this.kernelName = kernelName;
        this.signature = signature;
        this.sticky = new StickyArgs(kernelName);
    }

    public String kernelName() {
        return kernelName;
    }

    /** The build-time signature, or null when the kernel's arguments are not checked. */
    public KernelSignature signature() {
        return signature;
    }

    /** Sets one argument with snapshot semantics. */
    public void set(int index, Object arg) {
        sticky.set(index, arg);
    }

    /** Replaces every argument, as {@code enqueueKernel(..., args)} does. */
    public void setAll(Object[] args) {
        sticky.setAll(args);
    }

    /** The underlying argument snapshots. */
    public StickyArgs sticky() {
        return sticky;
    }

    /** True when every argument the kernel takes has been set. */
    public boolean isComplete() {
        if (!sticky.isComplete()) {
            return false;
        }
        return signature == null || sticky.count() == signature.argCount();
    }

    /**
     * Classifies and validates the stored arguments.
     *
     * @throws IllegalStateException    when an argument was never set
     * @throws IllegalArgumentException when the arguments disagree with the signature or a bound
     *                                  buffer is closed or belongs to another backend
     */
    public Bindings materialize() {
        sticky.requireComplete();
        int count = sticky.count();
        int bufferCount = 0;
        int scalarCount = 0;
        for (int i = 0; i < count; i++) {
            if (sticky.kind(i) == KernelArgs.ArgKind.BUFFER) {
                bufferCount++;
            } else {
                scalarCount++;
            }
        }
        MetalBuffer[] buffers = new MetalBuffer[bufferCount];
        byte[][] scalars = new byte[scalarCount][];
        int b = 0;
        int s = 0;
        for (int i = 0; i < count; i++) {
            Object value = sticky.get(i);
            KernelArgs.ArgKind kind = sticky.kind(i);
            if (kind == KernelArgs.ArgKind.BUFFER) {
                if (!(value instanceof MetalBuffer)) {
                    throw new IllegalArgumentException(kernelName + ": arg " + i + " is a "
                            + value.getClass().getSimpleName() + ", not a Metal buffer of this backend");
                }
                MetalBuffer buffer = (MetalBuffer) value;
                if (buffer.isClosed()) {
                    throw new IllegalArgumentException(kernelName + ": arg " + i + " (" + buffer
                            + ") is closed");
                }
                buffers[b++] = buffer;
                continue;
            }
            if (signature != null) {
                signature.checkScalar(kernelName, i, s, kind, KernelArgs.describe(value));
            }
            scalars[s++] = bytesOf(i, value);
        }
        if (signature != null) {
            signature.checkCounts(kernelName, bufferCount, scalarCount);
        }
        sticky.clearDirty();
        return new Bindings(buffers, scalars);
    }

    private byte[] bytesOf(int index, Object value) {
        int size = KernelArgs.byteSize(value);
        ByteBuffer bytes = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
        KernelArgs.writeLE(bytes, 0, value);
        return bytes.array();
    }

    /** The materialised argument list of one dispatch. Immutable. */
    public static final class Bindings {
        private final MetalBuffer[] buffers;
        private final byte[][] scalars;

        Bindings(MetalBuffer[] buffers, byte[][] scalars) {
            this.buffers = buffers;
            this.scalars = scalars;
        }

        /** Buffers in MSL buffer-argument order (indices {@code 0..n-1}). */
        public MetalBuffer[] buffers() {
            return buffers.clone();
        }

        /** Scalar argument bytes, little-endian, in scalar order. */
        public byte[][] scalars() {
            byte[][] copy = new byte[scalars.length][];
            for (int i = 0; i < scalars.length; i++) {
                copy[i] = scalars[i].clone();
            }
            return copy;
        }

        /**
         * The scalar arguments packed, in order, into one little-endian constant buffer: the
         * kernel's push-constant struct. Empty when the kernel takes no scalars.
         */
        public byte[] packedScalars() {
            int size = 0;
            for (byte[] scalar : scalars) {
                size += scalar.length;
            }
            byte[] packed = new byte[size];
            int offset = 0;
            for (byte[] scalar : scalars) {
                System.arraycopy(scalar, 0, packed, offset, scalar.length);
                offset += scalar.length;
            }
            return packed;
        }

        public int bufferCount() {
            return buffers.length;
        }

        public int scalarCount() {
            return scalars.length;
        }

        @Override
        public String toString() {
            return "MetalArgState.Bindings{buffers=" + buffers.length + ", scalars=" + scalars.length + "}";
        }
    }
}
