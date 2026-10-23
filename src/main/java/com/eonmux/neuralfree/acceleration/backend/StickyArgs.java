package com.eonmux.neuralfree.acceleration.backend;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;

import java.util.Arrays;
import java.util.BitSet;

/**
 * Per-kernel-handle argument storage with OpenCL {@code clSetKernelArg} semantics,
 * used by the CUDA and Vulkan handles.
 *
 * <p>Each argument is snapshotted when it is set ({@link KernelArgs#snapshot}).
 * The arguments stay set across dispatches until they are overwritten, so a
 * handle can be pre-bound once and dispatched many times.</p>
 *
 * <p>Change tracking:</p>
 * <ul>
 *   <li>{@link #version()} increases whenever any stored value changes;</li>
 *   <li>{@link #layoutVersion()} increases whenever the kind of an index or the
 *       argument count changes, which is when a backend must re-derive its
 *       parameter layout;</li>
 *   <li>{@link #dirty()} holds the indices changed since {@link #clearDirty()}.</li>
 * </ul>
 * Setting an argument to a value equal to the stored one changes nothing.
 */
@NotThreadSafe("Mutated by setKernelArg/enqueueKernel; callers serialise access per handle.")
public final class StickyArgs {

    private static final int INITIAL_CAPACITY = 16;

    private final String kernelName;
    private Object[] values = new Object[INITIAL_CAPACITY];
    private KernelArgs.ArgKind[] kinds = new KernelArgs.ArgKind[INITIAL_CAPACITY];
    private final BitSet set = new BitSet(INITIAL_CAPACITY);
    private final BitSet dirty = new BitSet(INITIAL_CAPACITY);
    private int count;
    private long version;
    private long layoutVersion;

    public StickyArgs(String kernelName) {
        if (kernelName == null || kernelName.isEmpty()) {
            throw new IllegalArgumentException("kernelName must not be null or empty");
        }
        this.kernelName = kernelName;
    }

    public String kernelName() {
        return kernelName;
    }

    /**
     * Sets one argument.
     *
     * @throws IllegalArgumentException when {@code index < 0}, the argument is
     *                                  null, or its type is unsupported
     */
    public void set(int index, Object arg) {
        if (index < 0) {
            throw new IllegalArgumentException("arg index " + index + " of " + kernelName + " is negative");
        }
        if (arg == null) {
            throw new IllegalArgumentException("arg " + index + " of " + kernelName + " is null");
        }
        KernelArgs.ArgKind kind;
        try {
            kind = KernelArgs.kindOf(arg);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("arg " + index + " of " + kernelName + ": " + e.getMessage(), e);
        }
        ensureCapacity(index + 1);
        if (set.get(index) && kinds[index] == kind && KernelArgs.sameValue(values[index], arg)) {
            return;
        }
        boolean layoutChanged = !set.get(index) || kinds[index] != kind;
        values[index] = KernelArgs.snapshot(arg, values[index]);
        kinds[index] = kind;
        set.set(index);
        dirty.set(index);
        if (index >= count) {
            count = index + 1;
        }
        if (layoutChanged) {
            layoutVersion++;
        }
        version++;
    }

    /**
     * Replaces the whole argument list, as {@code enqueueKernel(..., args)} does.
     * Indices at or beyond {@code args.length} become unset.
     *
     * @throws IllegalArgumentException when {@code args} or any element is null or unsupported;
     *                                  in that case the stored arguments are left unchanged
     */
    public void setAll(Object[] args) {
        if (args == null) {
            throw new IllegalArgumentException("argument array of " + kernelName + " is null");
        }
        for (int i = 0; i < args.length; i++) {
            if (args[i] == null) {
                throw new IllegalArgumentException("arg " + i + " of " + kernelName + " is null");
            }
            try {
                KernelArgs.kindOf(args[i]);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("arg " + i + " of " + kernelName + ": " + e.getMessage(), e);
            }
        }
        if (args.length < count) {
            for (int i = args.length; i < count; i++) {
                values[i] = null;
                kinds[i] = null;
            }
            set.clear(args.length, count);
            dirty.clear(args.length, count);
            count = args.length;
            layoutVersion++;
            version++;
        }
        for (int i = 0; i < args.length; i++) {
            set(i, args[i]);
        }
    }

    /**
     * The stored snapshot of argument {@code i}.
     *
     * @throws IllegalStateException when the argument has not been set
     */
    public Object get(int i) {
        requireSet(i);
        return values[i];
    }

    /**
     * The kind of argument {@code i}.
     *
     * @throws IllegalStateException when the argument has not been set
     */
    public KernelArgs.ArgKind kind(int i) {
        requireSet(i);
        return kinds[i];
    }

    public boolean isSet(int i) {
        return i >= 0 && i < count && set.get(i);
    }

    /** One past the highest index ever set (and not cleared by {@link #setAll}). */
    public int count() {
        return count;
    }

    /** True when every index in {@code [0, count())} is set. */
    public boolean isComplete() {
        return set.nextClearBit(0) >= count;
    }

    /**
     * Throws when an argument in {@code [0, count())} is unset, naming the first missing index.
     *
     * @throws IllegalStateException when the list is incomplete
     */
    public void requireComplete() {
        int missing = set.nextClearBit(0);
        if (missing < count) {
            throw new IllegalStateException("arg " + missing + " of " + kernelName
                    + " was never set (" + count + " args expected)");
        }
    }

    public long version() {
        return version;
    }

    public long layoutVersion() {
        return layoutVersion;
    }

    /** Live view of the indices changed since the last {@link #clearDirty()}; callers must not modify it. */
    public BitSet dirty() {
        return dirty;
    }

    public void clearDirty() {
        dirty.clear();
    }

    private void requireSet(int i) {
        if (!isSet(i)) {
            throw new IllegalStateException("arg " + i + " of " + kernelName + " was never set");
        }
    }

    private void ensureCapacity(int needed) {
        if (needed <= values.length) {
            return;
        }
        int capacity = Math.max(needed, values.length * 2);
        values = Arrays.copyOf(values, capacity);
        kinds = Arrays.copyOf(kinds, capacity);
    }
}
