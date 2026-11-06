package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.quant.TapProduct;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * The tap of the GPU matmul inputs, the device counterpart of {@code CpuMatVec.observeGlobally}: when a collection
 * tool installs an {@link Observer}, every product the forward pass runs on the device reads its input rows back to
 * the host — after the weight's input transform, exactly the {@code src1} llama.cpp's imatrix callback copies from
 * device memory — and hands them to the observer:
 * <ul>
 *   <li>the projections {@link GpuOps} runs ({@code matvec}, {@code matmul}, {@code matvecAdd}): one dense product
 *       ({@link #dense});</li>
 *   <li>the per-head and per-group products of a stacked 3-D weight that blocks run with their own id-matvec kernels
 *       (absorbed MLA {@code attn_k_b}/{@code attn_v_b}, the grouped DeepSeek-V4 {@code attn_output_a}): one dense
 *       product whose (token, matrix) rows are reported per matrix, as llama.cpp's 3-D {@code MUL_MAT} collection
 *       keeps one matrix per head ({@link #heads});</li>
 *   <li>the routed expert products: one indirect product whose (token, slot) pairs are reported with the expert
 *       they ran on, as llama.cpp's {@code MUL_MAT_ID} collection walks the routing ids ({@link #experts}).</li>
 * </ul>
 * Each product is first offered to {@link Observer#wants}, llama.cpp's ask, with its kind and its input row count;
 * an open {@link TapProduct} scope makes a call part of the scope's dense product.
 *
 * <p>The read-back synchronizes the queue at every tapped matmul, so an installed tap serializes the forward pass;
 * products the observer does not want are never read. With no observer installed the tap is one volatile read per
 * product.</p>
 */
public final class GpuMatmulTap {

    /** Receives the host copies of tapped matmul inputs; called on the forward thread. */
    public interface Observer {

        /**
         * llama.cpp's ask: true when the inputs of the weight stored as {@code weight} in a product of {@code rows}
         * input rows are collected; {@code indirect} for a routed expert product ({@code MUL_MAT_ID}), else a dense
         * one ({@code MUL_MAT}).
         */
        boolean wants(String weight, boolean indirect, int rows);

        /**
         * {@code nTokens} input rows of {@code cols} values at {@code x[xOff + t * xStride]} of the weight
         * {@code weight}, for matrix {@code matrix} of its {@code matrices} (0 and 0 for a 2-D weight). The array
         * is the observer's to keep.
         */
        void observe(String weight, int matrices, int matrix, float[] x, int xOff, int xStride, int cols,
                     int nTokens);
    }

    private static volatile Observer global;

    private GpuMatmulTap() {
    }

    /** Installs {@code observer} for every later GPU matmul of the process; {@code null} removes it. */
    public static void observeGlobally(Observer observer) {
        global = observer;
    }

    /** The installed observer, or null. */
    public static Observer global() {
        return global;
    }

    /** True when an observer is installed and wants the routed expert products of {@code w}. */
    public static boolean wantsExperts(GpuWeight w) {
        return asked(w, true, 0) != null;
    }

    /**
     * Taps the {@code nTokens} rows of {@code cols} floats at element {@code xOff} of {@code x}, the input of the
     * dense projection {@code w}, when the observer wants it.
     */
    static void dense(ComputeQueue queue, GpuWeight w, GpuBuffer x, int xOff, int nTokens, int cols) {
        Observer o = asked(w, false, nTokens);
        if (o == null) {
            return;
        }
        float[] rows = readFloats(queue, x, xOff, Math.multiplyExact(nTokens, cols));
        o.observe(w.name(), 0, 0, rows, 0, cols, cols, nTokens);
    }

    /**
     * Taps the dense product of the stacked 3-D weight {@code w} of {@code nMat} matrices over {@code nTokens}
     * tokens, when the observer wants it: the input of token {@code t}, matrix {@code m} is the row
     * {@code t * nMat + m} of {@code cols} floats at element {@code xOff + (t * nMat + m) * rowStride} of {@code x},
     * reported as matrix {@code m}.
     *
     * @throws IllegalArgumentException when the counts are not positive or a row is wider than its stride
     */
    public static void heads(ComputeQueue queue, GpuWeight w, int nMat, GpuBuffer x, int xOff, int rowStride,
                             int cols, int nTokens) {
        if (nMat < 1 || nTokens < 1 || cols < 1 || rowStride < cols) {
            throw new IllegalArgumentException("per-matrix rows of " + (w == null ? "?" : w.name()) + ": " + nMat
                    + " matrices x " + nTokens + " tokens of " + cols + " values at stride " + rowStride);
        }
        Observer o = asked(w, false, nTokens);
        if (o == null) {
            return;
        }
        int rows = Math.multiplyExact(nTokens, nMat);
        float[] in = readFloats(queue, x, xOff, Math.addExact(Math.multiplyExact(rows - 1, rowStride), cols));
        for (int r = 0; r < rows; r++) {
            o.observe(w.name(), nMat, r % nMat, in, r * rowStride, rowStride, cols, 1);
        }
    }

    /**
     * Taps the input rows of the expert product {@code w} of {@code nExpert} experts: {@code rows} rows of
     * {@code cols} floats at element {@code xOff} of {@code x}, one per routed pair when {@code perPair}, else one
     * per token shared by its {@code k} pairs; {@code ids} holds the expert of every pair, token-major.
     *
     * @throws IllegalStateException when a routed id is outside {@code [0, nExpert)}
     */
    public static void experts(ComputeQueue queue, GpuWeight w, int nExpert, int[] ids, int k, GpuBuffer x, int xOff,
                               int rows, int cols, boolean perPair) {
        Observer o = asked(w, true, rows);
        if (o == null) {
            return;
        }
        float[] in = readFloats(queue, x, xOff, Math.multiplyExact(rows, cols));
        for (int p = 0; p < ids.length; p++) {
            int ex = ids[p];
            if (ex < 0 || ex >= nExpert) {
                throw new IllegalStateException(w.name() + ": routed expert " + ex + " outside [0, " + nExpert + ")");
            }
            int row = perPair ? p : p / k;
            o.observe(w.name(), nExpert, ex, in, row * cols, cols, cols, 1);
        }
    }

    /**
     * The installed observer when it wants the product of {@code w} over {@code rows} input rows ({@code indirect}
     * for a routed one) as the open {@link TapProduct} scope makes it, else null.
     */
    private static Observer asked(GpuWeight w, boolean indirect, int rows) {
        Observer o = global;
        if (o == null || w == null || !o.wants(w.name(), TapProduct.indirect(indirect), TapProduct.rows(rows))) {
            return null;
        }
        return o;
    }

    /** {@code n} floats from element {@code off} of {@code buf}, after every enqueued command. */
    public static float[] readFloats(ComputeQueue queue, GpuBuffer buf, long off, int n) {
        float[] out = new float[n];
        await(Objects.requireNonNull(queue, "queue").enqueueReadBuffer(buf, off * Float.BYTES,
                (long) n * Float.BYTES, out));
        return out;
    }

    /** {@code n} little-endian ints from the 4-byte element {@code off} of {@code buf}, after every command. */
    public static int[] readInts(ComputeQueue queue, GpuBuffer buf, long off, int n) {
        byte[] raw = new byte[Math.multiplyExact(n, Integer.BYTES)];
        await(Objects.requireNonNull(queue, "queue").enqueueReadBuffer(buf, off * Integer.BYTES, raw.length, raw));
        int[] out = new int[n];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(out);
        return out;
    }

    private static void await(GpuEvent read) {
        try {
            read.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("the matmul input read-back was interrupted", e);
        } finally {
            read.close();
        }
    }
}
