package com.eonmux.neuralfree.acceleration.backend;

import java.util.Arrays;
import java.util.Objects;

/**
 * The state of one kernel dispatch captured in a {@link GpuGraph}: its position among the graph's
 * kernel dispatches (buffers and other captured commands do not take a position), the kernel's name,
 * the launch geometry and the argument values bound at capture time — buffers by identity, scalars
 * boxed, arrays cloned. A patcher matches captured dispatches on these values
 * ({@link GpuStepPatcher#accepts(GpuGraphDispatch)}), a graph reports its current state through
 * {@link GpuGraph#kernelDispatch(int)}, and the index addresses the dispatch in
 * {@link GpuGraphPatchSink#patchKernel(int, long[], int[], Object[])}. Immutable: the array getters copy.
 */
public final class GpuGraphDispatch {

    private final int index;
    private final String kernelName;
    private final long[] grid;
    private final long[] block;
    private final Object[] args;

    /**
     * @param index      the dispatch's position among the graph's kernel dispatches, in capture order
     * @param kernelName the compiled kernel's name
     * @param grid       the global work size, never null
     * @param block      the local work size, or null when the kernel launched with an implementation-chosen one
     * @param args       the bound argument values, one entry per argument, never null
     */
    public GpuGraphDispatch(int index, String kernelName, long[] grid, long[] block, Object[] args) {
        this.index = index;
        this.kernelName = Objects.requireNonNull(kernelName, "kernelName");
        this.grid = Objects.requireNonNull(grid, "grid").clone();
        this.block = block != null ? block.clone() : null;
        this.args = Objects.requireNonNull(args, "args").clone();
    }

    /** The dispatch's position among the graph's kernel dispatches, in capture order. */
    public int index() {
        return index;
    }

    /** The compiled kernel's name. */
    public String kernelName() {
        return kernelName;
    }

    /** The global work size (copy). */
    public long[] grid() {
        return grid.clone();
    }

    /** The local work size (copy), or null when the kernel launched with an implementation-chosen one. */
    public long[] block() {
        return block != null ? block.clone() : null;
    }

    /** The bound argument values (copy of the array; buffers by identity, arrays shared). */
    public Object[] args() {
        return args.clone();
    }

    /** The bound value of argument {@code i}. */
    public Object arg(int i) {
        return args[i];
    }

    @Override
    public String toString() {
        return "GpuGraphDispatch{" + index + ": " + kernelName
                + ", grid=" + Arrays.toString(grid) + ", " + args.length + " args}";
    }
}
