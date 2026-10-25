package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.IntPacking.Layout;
import com.eonmux.neuralfree.gguf.hf.quant.IntPacking.ParamTable;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * The tensors of one quantized module. A suffix names {@code <module>.<suffix>} when the module's primary tensor
 * is dotted, and {@code <module>_<suffix>} (the bare {@code <module>} for {@code weight}) for fused-expert
 * parameters; the rewritten weight takes the matching name. Records the tensors it hands out as consumed.
 */
final class IntModuleTensors {

    /** Elements read in one call when a tensor is read whole. */
    private static final long MAX_SMALL_TENSOR = 1L << 26;

    private final IntRewriteBatch batch;
    private final String name;
    private final boolean underscore;
    private final Set<String> consumed = new LinkedHashSet<>();

    private IntModuleTensors(IntRewriteBatch batch, String name, boolean underscore) {
        this.batch = batch;
        this.name = name;
        this.underscore = underscore;
    }

    static IntModuleTensors resolve(IntRewriteBatch b, String module, String... primaries) throws IOException {
        if (module == null || module.isEmpty()) {
            throw new IllegalArgumentException("empty module name");
        }
        b.checkOnlineTransforms(module);
        IntModuleTensors dotted = new IntModuleTensors(b, module, false);
        IntModuleTensors underscore = new IntModuleTensors(b, module, true);
        for (String p : primaries) {
            if (dotted.has(p)) {
                return dotted;
            }
            if (underscore.has(p)) {
                return underscore;
            }
        }
        throw b.invalid(module, "the checkpoint has no tensor " + module + "." + String.join(" or " + module + ".", primaries));
    }

    String name() {
        return name;
    }

    /** The quantization format of the rewrite this module belongs to. */
    QuantFormat format() {
        return batch.args().format();
    }

    String tensorName(String suffix) {
        if (!underscore) {
            return name + "." + suffix;
        }
        return suffix.equals("weight") ? name : name + "_" + suffix;
    }

    /** The name of the rewritten weight. */
    String outputName() {
        return tensorName("weight");
    }

    Set<String> consumed() {
        return Collections.unmodifiableSet(consumed);
    }

    boolean has(String suffix) {
        return batch.index().has(tensorName(suffix));
    }

    /** The tensor of {@code suffix} without consuming it; null when absent. */
    RowSource peek(String suffix) {
        String n = tensorName(suffix);
        return batch.index().has(n) ? batch.index().get(n) : null;
    }

    RowSource take(String suffix) throws IOException {
        RowSource s = takeOptional(suffix);
        if (s == null) {
            throw invalid("the checkpoint has no tensor " + tensorName(suffix));
        }
        return s;
    }

    RowSource takeOptional(String suffix) {
        String n = tensorName(suffix);
        if (!batch.index().has(n)) {
            return null;
        }
        consumed.add(n);
        return batch.index().get(n);
    }

    IOException invalid(String message) {
        return batch.invalid(name, message);
    }

    /** The expert count of a tensor of base rank {@code rank}: 0 without a leading expert dimension. */
    int experts(RowSource src, int rank, String suffix) throws IOException {
        long[] s = src.shape();
        if (s.length == rank) {
            return 0;
        }
        if (s.length == rank + 1 && s[0] > 0 && s[0] <= Integer.MAX_VALUE) {
            return (int) s[0];
        }
        throw invalid(tensorName(suffix) + " has shape " + Arrays.toString(s) + "; expected rank " + rank
                + ", or " + (rank + 1) + " with a leading expert dimension");
    }

    /** The dimensions after the expert dimension, checked for rank {@code rank} and {@code experts}. */
    long[] tail(RowSource src, int experts, int rank, String suffix) throws IOException {
        long[] s = src.shape();
        int lead = experts > 0 ? 1 : 0;
        if (s.length != rank + lead || (lead == 1 && s[0] != experts)) {
            throw invalid(tensorName(suffix) + " has shape " + Arrays.toString(s) + "; expected rank "
                    + (rank + lead) + (lead == 1 ? " with " + experts + " experts first" : ""));
        }
        return Arrays.copyOfRange(s, lead, s.length);
    }

    int width(long v, String suffix) throws IOException {
        if (v <= 0 || v > HfConfig.MAX_WIDTH) {
            throw invalid(tensorName(suffix) + " gives a width of " + v + ", outside [1, " + HfConfig.MAX_WIDTH
                    + "]");
        }
        return (int) v;
    }

    /** The layout of {@code words} words holding {@code count} values; the word count must fit one exactly. */
    Layout layout(long count, long words, int bits, String suffix, String axis) throws IOException {
        if (IntPacking.ambiguous(count, words, bits)) {
            throw invalid(tensorName(suffix) + " has " + words + " words along " + axis + ", which fit "
                    + count + " " + bits + "-bit values in both the per-word and the bit-stream layout");
        }
        Layout l = IntPacking.layoutOf(count, words, bits);
        if (l == null) {
            throw invalid(tensorName(suffix) + " has " + words + " words along " + axis + "; " + count + " "
                    + bits + "-bit values need " + IntPacking.wordsFor(count, bits, Layout.PER_WORD)
                    + " (per word) or " + IntPacking.wordsFor(count, bits, Layout.BIT_STREAM) + " (bit stream)");
        }
        return l;
    }

    void requireDim(long actual, long expected, String suffix, String what) throws IOException {
        if (actual != expected) {
            throw invalid(tensorName(suffix) + " has " + actual + " " + what + "; expected " + expected);
        }
    }

    void requireDtype(RowSource src, String suffix, StDType... allowed) throws IOException {
        StDType d = src.format().stDType();
        for (StDType a : allowed) {
            if (a == d) {
                return;
            }
        }
        throw invalid(tensorName(suffix) + " is " + src.format() + "; expected " + Arrays.toString(allowed));
    }

    StDType requireFloat(RowSource src, String suffix) throws IOException {
        StDType d = src.format().stDType();
        if (d == null || !IntPacking.isFloat(d)) {
            throw invalid(tensorName(suffix) + " is " + src.format() + "; expected F16, BF16, F32 or F64");
        }
        return d;
    }

    /** Every value of a small integer tensor. */
    long[] longs(RowSource src, String suffix) throws IOException {
        StDType d = src.format().stDType();
        if (d == null || !IntPacking.isInteger(d)) {
            throw invalid(tensorName(suffix) + " is " + src.format() + "; expected an integer dtype");
        }
        ByteBuffer b = readSmall(src, suffix);
        long[] out = new long[(int) (b.capacity() / (d.bits() / 8))];
        for (int i = 0; i < out.length; i++) {
            out[i] = IntPacking.longAt(b, i, d);
        }
        return out;
    }

    /** Every value of a small float tensor. */
    float[] floats(RowSource src, String suffix) throws IOException {
        StDType d = requireFloat(src, suffix);
        ByteBuffer b = readSmall(src, suffix);
        float[] out = new float[(int) (b.capacity() / (d.bits() / 8))];
        for (int i = 0; i < out.length; i++) {
            out[i] = IntPacking.floatAt(b, i, d);
        }
        return out;
    }

    private ByteBuffer readSmall(RowSource src, String suffix) throws IOException {
        long numel = 1;
        for (long v : src.shape()) {
            numel = Math.multiplyExact(numel, v);
        }
        if (numel > MAX_SMALL_TENSOR) {
            throw invalid(tensorName(suffix) + " has " + numel + " elements, more than a parameter tensor holds");
        }
        return IntPacking.rows(src, 0, Math.toIntExact(src.rows()));
    }

    /** One value, or one per expert, of a float tensor. */
    float[] perExpertFloats(RowSource src, int experts, String suffix) throws IOException {
        float[] v = floats(src, suffix);
        requirePerExpert(v.length, experts, suffix);
        return v;
    }

    /** One integer value, or one per expert. */
    int[] perExpertInts(RowSource src, int experts, String suffix) throws IOException {
        StDType d = src.format().stDType();
        long[] v;
        if (d != null && IntPacking.isFloat(d)) {
            float[] f = floats(src, suffix);
            v = new long[f.length];
            for (int i = 0; i < f.length; i++) {
                if (f[i] != Math.rint(f[i])) {
                    throw invalid(tensorName(suffix) + " holds " + f[i] + "; expected an integer");
                }
                v[i] = (long) f[i];
            }
        } else {
            v = longs(src, suffix);
        }
        requirePerExpert(v.length, experts, suffix);
        int[] out = new int[v.length];
        for (int i = 0; i < v.length; i++) {
            if (v[i] < Integer.MIN_VALUE || v[i] > Integer.MAX_VALUE) {
                throw invalid(tensorName(suffix) + " holds " + v[i] + ", outside the int range");
            }
            out[i] = (int) v[i];
        }
        return out;
    }

    private void requirePerExpert(int length, int experts, String suffix) throws IOException {
        if (length != 1 && (experts == 0 || length != experts)) {
            throw invalid(tensorName(suffix) + " has " + length + " values; expected 1"
                    + (experts > 0 ? " or " + experts : ""));
        }
    }

    /**
     * The group index of each input column, per expert, each checked to lie in {@code [0, groups)}; the
     * tensor is I32 or I64 {@code [E?, K]}.
     */
    int[][] groupIndex(RowSource src, int experts, int k, int groups, String suffix) throws IOException {
        long[] v = longs(src, suffix);
        int rows = Math.max(1, experts);
        if (v.length != (long) rows * k) {
            throw invalid(tensorName(suffix) + " has " + v.length + " values; expected " + (long) rows * k);
        }
        int[][] out = new int[rows][k];
        for (int e = 0; e < rows; e++) {
            for (int c = 0; c < k; c++) {
                long g = v[e * k + c];
                if (g < 0 || g >= groups) {
                    throw invalid(tensorName(suffix) + "[" + (experts > 0 ? e + ", " : "") + c + "] = " + g
                            + " is outside [0, " + groups + ")");
                }
                out[e][c] = (int) g;
            }
        }
        return out;
    }

    /**
     * Parameters stored row-major: {@code [E?, N, G]}, {@code [E?, N]} or {@code [E?, N, 1]} when there is one
     * group, one value per expert ({@code [E]}, {@code [E, 1]}), or one in all ({@code []}, {@code [1]}).
     */
    ParamTable rowMajor(RowSource src, int experts, long n, int groups, String suffix) throws IOException {
        long[] s = src.shape();
        long numel = 1;
        for (long v : s) {
            numel = Math.multiplyExact(numel, v);
        }
        if (numel == 1) {
            return IntPacking.broadcast(src, 1, groups);
        }
        int lead = experts > 0 ? 1 : 0;
        boolean expertsOk = lead == 0 || (s.length > 0 && s[0] == experts);
        if (lead == 1 && expertsOk && numel == experts) {
            return IntPacking.broadcast(src, experts, groups);
        }
        long[] t = expertsOk ? Arrays.copyOfRange(s, lead, s.length) : new long[0];
        boolean ok = expertsOk && ((t.length == 2 && t[0] == n && t[1] == groups)
                || (groups == 1 && t.length == 1 && t[0] == n));
        if (!ok) {
            throw invalid(tensorName(suffix) + " has shape " + Arrays.toString(s) + "; expected "
                    + (lead == 1 ? "[" + experts + ", " : "[") + n + ", " + groups + "]");
        }
        return IntPacking.rowMajor(src, n, groups);
    }

    /** Parameters stored group-major: {@code [E?, G, N]}, or {@code [E?, N]} when there is one group. */
    ParamTable groupMajor(RowSource src, int experts, long n, int groups, String suffix) throws IOException {
        long[] s = src.shape();
        int lead = experts > 0 ? 1 : 0;
        boolean expertsOk = lead == 0 || (s.length > 0 && s[0] == experts);
        long[] t = expertsOk ? Arrays.copyOfRange(s, lead, s.length) : new long[0];
        boolean ok = expertsOk && ((t.length == 2 && t[0] == groups && t[1] == n)
                || (groups == 1 && t.length == 1 && t[0] == n));
        if (!ok) {
            throw invalid(tensorName(suffix) + " has shape " + Arrays.toString(s) + "; expected "
                    + (lead == 1 ? "[" + experts + ", " : "[") + groups + ", " + n + "]");
        }
        return IntPacking.groupMajor(src, n, groups);
    }
}
