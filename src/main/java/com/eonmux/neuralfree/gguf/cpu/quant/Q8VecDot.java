package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;
import java.nio.ByteOrder;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK8_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_QS;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * CPU vec-dot of Q8_0 weights with Q8_0 activations, {@code ggml_vec_dot_q8_0_q8_0_generic}
 * (llama.cpp ggml/src/ggml-cpu/quants.c:451-479).
 * <p>
 * The class also holds the int8 primitives that every block vec-dot of this package ends in: weight
 * quants are unpacked (with 64-bit SWAR on little-endian {@code getLong} reads) into a per-thread
 * {@code byte[]} block buffer, and each sub-block is then reduced by {@link #dotI8} against the activation
 * quants. The reduction is a branch-free counted loop over two {@code byte[]} arrays with an {@code int}
 * accumulator, the loop shape C2's superword pass targets for auto-vectorization. The block buffer is
 * created once per thread, so no call allocates.
 */
public final class Q8VecDot implements CpuVecDot {

    /** The Q8_0 vec-dot. */
    public static final Q8VecDot Q8_0 = new Q8VecDot();

    private static final Set<GGMLType> SUPPORTED = Collections.unmodifiableSet(EnumSet.of(GGMLType.Q8_0));

    /** {@code 0x01} in every byte. */
    static final long BYTE_ONES = 0x0101010101010101L;
    /** Low nibble of every byte. */
    static final long LOW_NIBBLES = 0x0F0F0F0F0F0F0F0FL;
    /** Bit 7 of every byte. */
    static final long BYTE_SIGNS = 0x8080808080808080L;

    private static final VarHandle LE_LONGS =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(Scratch::new);

    private Q8VecDot() {
    }

    /** True for the single type this class handles, Q8_0. */
    public static boolean supports(GGMLType type) {
        return type == GGMLType.Q8_0;
    }

    /** The weight types this class handles. */
    public static Set<GGMLType> supportedTypes() {
        return SUPPORTED;
    }

    /**
     * @throws IllegalArgumentException when {@code type} is not Q8_0
     */
    public static Q8VecDot forType(GGMLType type) {
        if (!supports(type)) {
            throw new IllegalArgumentException("Q8VecDot handles Q8_0 weights only, not " + type);
        }
        return Q8_0;
    }

    /** The weight type this vec-dot reads, Q8_0. */
    public GGMLType weightType() {
        return GGMLType.Q8_0;
    }

    /** The activation type this vec-dot consumes, Q8_0. */
    public GGMLType activationType() {
        return GGMLType.Q8_0;
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return dot(w.rowBuffer(row), w.rowOffset(row), x, scratch().quants);
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        final byte[] aux = scratch().quants;
        CpuVecDotRows.forEachRow(w, rowStart, rowCount, y, yOff, (b, off) -> dot(b, off, x, aux));
    }

    private static void check(CpuTensorRef w, QuantizedActivations x) {
        requireWeightType(w, GGMLType.Q8_0);
        x.requireCompatible(w, GGMLType.Q8_0);
    }

    /**
     * {@code ggml_vec_dot_q8_0_q8_0_generic} (quants.c:451-479): per block
     * {@code sumf += sumi * (d_x * d_y)} with {@code sumi = sum_j x.qs[j] * y.qs[j]} over 32 quants.
     */
    private static float dot(ByteBuffer b, int off, QuantizedActivations x, byte[] aux) {
        final byte[] yq = x.qs;
        final float[] yd = x.d;
        final int nb = x.n / QK8_0;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q8_0_BYTES) {
            final int qs = off + Q8_0_QS;
            putLong(aux, 0, b.getLong(qs));
            putLong(aux, 8, b.getLong(qs + 8));
            putLong(aux, 16, b.getLong(qs + 16));
            putLong(aux, 24, b.getLong(qs + 24));
            final int sumi = dotI8(aux, 0, yq, ib * QK8_0, QK8_0);
            sumf += sumi * (fp16At(b, off + Q8_0_D) * yd[ib]);
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "Q8VecDot{Q8_0 x Q8_0}";
    }

    // ------------------------------------------------------------------ shared int8 primitives

    /** Per-thread block buffers of the vec-dot kernels in this package. */
    static final class Scratch {
        /** Unpacked signed weight quants of one block or super-block. */
        final byte[] quants = new byte[QK_K];
        /** Unpacked sub-block scales of one super-block. */
        final byte[] scales = new byte[16];
    }

    /** The calling thread's block buffers. */
    static Scratch scratch() {
        return SCRATCH.get();
    }

    /** {@code sum_i a[ao + i] * b[bo + i]} for {@code i} in {@code [0, n)}; exact in {@code int} for every block size. */
    static int dotI8(byte[] a, int ao, byte[] b, int bo, int n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += a[ao + i] * b[bo + i];
        }
        return sum;
    }

    /** Stores the eight bytes of {@code v} at {@code a[i .. i + 8)}, least significant byte first. */
    static void putLong(byte[] a, int i, long v) {
        LE_LONGS.set(a, i, v);
    }

    /**
     * Per-byte {@code v - bias} for bytes {@code v} in {@code [0, 2 * bias)} with {@code bias} in
     * {@code {4, 8, 16, 32}}, as signed bytes: {@code (v + 0x80 - bias) ^ 0x80}. The addition never
     * carries into the next byte because {@code v + 0x80 - bias < 0x100}.
     *
     * @param biasComplement {@code 0x80 - bias} replicated into every byte
     */
    static long subtractBias(long v, long biasComplement) {
        return (v + biasComplement) ^ BYTE_SIGNS;
    }

    static void requireWeightType(CpuTensorRef w, GGMLType expected) {
        if (w == null) {
            throw new IllegalArgumentException("weight tensor is null");
        }
        if (w.type() != expected) {
            throw new IllegalArgumentException("the " + expected + " vec-dot cannot read " + w.type() + " weights");
        }
    }
}
