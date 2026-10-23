package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuBlockRegistry;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.ops.hc.HcSubBlock;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The compute surface every draft head runs on: the target model's device, scratch and kernels, with tensors read
 * from the target ({@link Source#TARGET}) or from a separate drafter file ({@link Source#DRAFT}).
 *
 * <p>A drafter must run where its target runs: gemma4-assistant attends over the target's KV caches, and every
 * drafter consumes the target's hidden rows. {@link #gpu} binds to a target
 * {@link GpuForwardContext} (its queue, kernel library, dispatch helpers and scratch), {@link #cpu} to a target
 * {@link CpuForwardContext}. Forward passes of the target and its drafter alternate on one thread, so the drafter
 * shares the target's scratch slots; its own row buffers are named temporaries that no block reuses.
 *
 * <p>Operations work on {@link Rows}: {@link #ubatch()} rows of a fixed width, filled from row 0. Transformer layers
 * run through the target's block registry, either as a {@link Stack} of whole layers (the layer executor) or as the
 * separate mixer and FFN {@link Blocks} of one layer (the eagle3 decoder, whose attention input and residual differ
 * from a standard block). Both step a {@link Memory}: the drafter's own caches, or a view of the target's caches.
 * The stream operations ({@link #normStreams}, {@link #concatStreams}, {@link #projectStreams}) treat each row as
 * {@code streams} equal consecutive parts, the hyper-connection streams of an MTP head, and a {@link Collapse}
 * reduces such rows to one stream. Not thread-safe.
 */
public abstract class MtpHead implements AutoCloseable {

    /** Which model's tensors an operation reads. */
    public enum Source {
        /** The target model (NextN/MTP blocks, shared embeddings and LM head). */
        TARGET,
        /** The drafter file (eagle3, gemma4-assistant, an MTP draft file of the target's model). */
        DRAFT
    }

    /** Device rows: {@link MtpHead#ubatch()} rows of {@link #width()} floats. */
    public abstract static class Rows {
        private final String name;
        private final int width;

        Rows(String name, int width) {
            this.name = name;
            this.width = width;
        }

        public String name() { return name; }

        /** Floats per row. */
        public int width() { return width; }

        @Override
        public String toString() {
            return "Rows{" + name + ", width=" + width + '}';
        }
    }

    /** Layers run in order over a residual stream held in {@link Rows}, stepping one {@link Memory}. */
    public interface Stack extends AutoCloseable {

        /**
         * Runs every layer over {@code n} tokens at positions {@code pos0 ..}; {@code x} holds the stack input on entry
         * and its output afterwards.
         */
        void run(int pos0, int n, int[] tokens, Rows x);

        @Override
        void close();
    }

    /** The mixer and FFN block of one layer, run separately by a drafter that wires its own residuals. */
    public interface Blocks extends AutoCloseable {

        /** Mixer over {@code n} rows of {@code in} at positions {@code pos0 ..} into {@code out}; steps the memory. */
        void mixer(int pos0, int n, int[] tokens, Rows in, Rows out);

        /** FFN over {@code n} rows of {@code in} into {@code out}. */
        void ffn(int n, Rows in, Rows out);

        @Override
        void close();
    }

    /** A hyper-connection collapse site ({@link HcSubBlock}) bound to the target's runtime and tensors. */
    public interface Collapse {

        /**
         * Collapses {@code n} rows of {@code streams} (the site's {@code hc * n_embd} floats each) into {@code n} rows
         * of {@code out} ({@code n_embd} floats each).
         */
        void run(Rows streams, Rows out, int n);
    }

    /**
     * Memory a drafter's layers step: caches of its own ({@link #ownsCaches()}), whose positions follow the target's,
     * or a view that reads the target's caches without writing them.
     */
    public abstract static class Memory implements AutoCloseable {
        private final boolean owns;

        Memory(boolean owns) {
            this.owns = owns;
        }

        /** The memory the layers bind to. */
        public abstract ModelMemory memory();

        /** True when the drafter writes these caches (NextN, eagle3); false for a view of the target's caches. */
        public boolean ownsCaches() {
            return owns;
        }

        /**
         * Prepares own caches for a step at {@code pos0}: positions from {@code pos0} on are dropped, so a step
         * rewrites them. A view needs nothing.
         *
         * @throws IllegalStateException when positions below {@code pos0} were never written, or the stored ones
         *                               cannot be dropped
         */
        public void align(int pos0) {
            if (!owns) {
                return;
            }
            ModelMemory m = memory();
            int nPast = m.sequence().nPast();
            if (nPast < pos0) {
                throw new IllegalStateException("draft caches hold " + nPast + " positions; a step at " + pos0
                        + " would leave a gap");
            }
            int drop = nPast - pos0;
            if (drop > 0) {
                if (!m.canRollback(drop)) {
                    throw new IllegalStateException("draft caches cannot drop their last " + drop + " positions");
                }
                m.rollback(drop);
            }
        }

        /** Empties own caches; a view has nothing to empty. */
        public void reset() {
            if (owns) {
                memory().reset();
            }
        }

        /** Writes the step's positions and token ids where the blocks read them. */
        abstract void begin(int pos0, int n, int[] tokens);

        /** Records the step's tokens in own caches' sequence. */
        final void end(int pos0, int n, int[] tokens) {
            if (owns) {
                memory().sequence().append(tokens, 0, n);
            }
        }

        @Override
        public void close() {
            memory().close();
        }
    }

    private final ModelSpec target;
    private final List<AutoCloseable> owned = new ArrayList<>();
    private boolean closed;

    MtpHead(ModelSpec target) {
        this.target = Objects.requireNonNull(target, "target");
    }

    /**
     * A head on the target's GPU runtime.
     *
     * @param target   the target's forward context (queue, kernels, dispatch helpers, scratch, weights, memory)
     * @param registry the block registry the target's layers were built from
     * @param draft    the drafter file's resident weights, or null when every tensor comes from the target
     */
    public static MtpHead gpu(GpuForwardContext target, GpuBlockRegistry registry, GpuWeights draft) {
        return new GpuMtpHead(target, registry, draft);
    }

    /**
     * A head on the target's CPU runtime.
     *
     * @param target   the target's forward context (scratch, weights, memory)
     * @param registry the block registry the target's layers were built from
     * @param draft    the drafter file's weights, or null when every tensor comes from the target
     */
    public static MtpHead cpu(CpuForwardContext target, CpuBlockRegistry registry, CpuWeights draft) {
        return new CpuMtpHead(target, registry, draft);
    }

    /** The target model's spec. */
    public ModelSpec target() {
        return target;
    }

    /** Maximum rows per operation. */
    public abstract int ubatch();

    /** Row buffer {@code name} of {@link #ubatch()} rows of {@code width}; repeated calls return the same rows. */
    public abstract Rows rows(String name, int width);

    /** True when the source holds tensor {@code name} (full GGUF name). */
    public abstract boolean has(Source s, String name);

    /** Dimensions {@code ne} of a tensor. */
    public abstract long[] shape(Source s, String name);

    /** A vector tensor copied to the host as floats. */
    public abstract float[] vector(Source s, String name);

    /** An I32 or I64 tensor copied to the host as ints. */
    public abstract int[] integers(Source s, String name);

    /** Writes {@code nRows} rows from {@code src[0 ..]}. */
    public abstract void write(Rows dst, float[] src, int nRows);

    /** Reads rows {@code [0, nRows)} into {@code dst[0 ..]}. */
    public abstract void read(Rows src, int nRows, float[] dst);

    /** Reads row {@code row} into {@code dst[0 ..]}. */
    public abstract void readRow(Rows src, int row, float[] dst);

    /** {@code out[t] = table[tokens[t]]} for {@code t < n}, with the table's inverse Hadamard when it has one. */
    public abstract void embed(Source s, String table, int[] tokens, int n, Rows out);

    /** Row norm of {@code n} rows, times {@code weight} when non-null; {@code out} may be {@code in}. */
    public abstract void norm(NormKind kind, Source s, String weight, float eps, Rows in, Rows out, int n);

    /** {@code out[t] = [a[t] | b[t]]}. */
    public abstract void concat(Rows a, Rows b, Rows out, int n);

    /** {@code out[t] = W . in[t]}. */
    public abstract void project(Source s, String weight, Rows in, Rows out, int n);

    /** {@code out = a + b} over {@code n} rows; {@code out} may be {@code a} or {@code b}. */
    public abstract void add(Rows a, Rows b, Rows out, int n);

    /** {@code x *= s} over {@code n} rows. */
    public abstract void scale(Rows x, float s, int n);

    /**
     * Row norm of every part of {@code n} rows split into {@code streams} parts of {@code width / streams} floats,
     * times {@code weight} when non-null: a vector of {@code width / streams} values shared by every part, or
     * {@code streams} such vectors back to back ({@code [width / streams, streams]}, or stored flat) with part
     * {@code s} scaled by vector {@code s}. {@code out} may be {@code in}.
     *
     * @throws IllegalArgumentException when the width does not split into {@code streams} parts or the weight fits
     *                                  neither form
     */
    public abstract void normStreams(NormKind kind, Source s, String weight, float eps, Rows in, Rows out, int n,
                                     int streams);

    /**
     * {@code out[t]} is {@code streams} parts {@code [a[t] | b[t, p]]}: each part repeats row {@code t} of {@code a}
     * and follows it with part {@code p} of row {@code t} of {@code b} ({@code b} rows of {@code streams * B} floats,
     * {@code out} rows of {@code streams * (a.width + B)}).
     */
    public abstract void concatStreams(Rows a, Rows b, Rows out, int n, int streams);

    /**
     * {@code out[t, p] = W . in[t, p]} for every part {@code p} of {@code n} rows split into {@code streams} parts
     * ({@code in} rows of {@code streams * W.cols}, {@code out} rows of {@code streams * W.rows}).
     */
    public abstract void projectStreams(Source s, String weight, Rows in, Rows out, int n, int streams);

    /**
     * The collapse of {@code site}, bound once over the tensors of {@code s} and closed with the head.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor of the site is absent
     * @throws IllegalArgumentException when a tensor does not match the site
     */
    public abstract Collapse collapse(Source s, HcSubBlock site);

    /** {@code dst = W . in[row]} on the host ({@code W} rows floats). */
    public abstract void logits(Source s, String weight, Rows in, int row, float[] dst);

    /** Caches of the drafter's own, allocated for {@code plan}. */
    public abstract Memory ownMemory(ModelSpec spec, MemoryPlan plan);

    /**
     * A view of the target's caches for a drafter whose plan reads target layers ({@code MemoryPlanner.planDrafter}):
     * the layers attend over the target's rows and visibility, never write them, and take their query positions from
     * the step.
     */
    public abstract Memory targetCaches(ModelSpec spec, MemoryPlan plan);

    /** Whole layers of {@code spec}, with weights from {@code s}, stepping {@code memory}. */
    public abstract Stack stack(ModelSpec spec, Source s, Memory memory, List<LayerSpec> layers);

    /** The separate mixer and FFN of one layer of {@code spec}, stepping {@code memory}. */
    public abstract Blocks blocks(ModelSpec spec, Source s, Memory memory, LayerSpec layer);

    /** Registers a resource closed with the head. */
    final <T extends AutoCloseable> T own(T resource) {
        checkOpen();
        owned.add(resource);
        return resource;
    }

    final void checkOpen() {
        if (closed) {
            throw new IllegalStateException("draft head is closed");
        }
    }

    static void checkRows(Rows r, int n, int ubatch) {
        Objects.requireNonNull(r, "rows");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException(r.name() + ": " + n + " rows outside [1, " + ubatch + "]");
        }
    }

    static void checkWidth(Rows r, long width, String what) {
        if (r.width() != width) {
            throw new IllegalArgumentException(what + " produces rows of " + width + " floats, " + r + " holds "
                    + r.width());
        }
    }

    /**
     * Floats per part of {@code r} split into {@code streams} parts.
     *
     * @throws IllegalArgumentException when {@code streams < 1} or the width does not split evenly
     */
    static int streamPart(Rows r, int streams) {
        if (streams < 1 || r.width() % streams != 0) {
            throw new IllegalArgumentException(r + " does not split into " + streams + " streams");
        }
        return r.width() / streams;
    }

    /**
     * Vectors of {@code part} values a stream-norm weight of {@code elements} values holds: 1 (shared by every part)
     * or {@code streams} (one per part).
     *
     * @throws IllegalArgumentException when the weight fits neither form
     */
    static int streamWeightRows(String weight, long elements, int part, int streams) {
        if (elements == part) {
            return 1;
        }
        if (elements == (long) part * streams) {
            return streams;
        }
        throw new IllegalArgumentException(weight + " holds " + elements + " values; a norm of " + streams
                + " streams of " + part + " needs " + part + " or " + (long) part * streams);
    }

    static int[] toInts(String name, GGMLType type, ByteBuffer le, long count) {
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(name + " has " + count + " elements");
        }
        int[] out = new int[(int) count];
        for (int i = 0; i < out.length; i++) {
            if (type == GGMLType.I32) {
                out[i] = le.getInt(4 * i);
            } else {
                long v = le.getLong(8 * i);
                if (v < Integer.MIN_VALUE || v > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException(name + "[" + i + "] = " + v + " does not fit an int");
                }
                out[i] = (int) v;
            }
        }
        return out;
    }

    static void requireIntegerType(String name, GGMLType type) {
        if (type != GGMLType.I32 && type != GGMLType.I64) {
            throw new IllegalArgumentException(name + " must be I32 or I64, got " + type);
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        List<AutoCloseable> reversed = new ArrayList<>(owned);
        Collections.reverse(reversed);
        for (AutoCloseable c : reversed) {
            try {
                c.close();
            } catch (Exception e) {
                if (failure == null) {
                    failure = new IllegalStateException("closing the draft head failed", e);
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        owned.clear();
        if (failure != null) {
            throw failure;
        }
    }
}
