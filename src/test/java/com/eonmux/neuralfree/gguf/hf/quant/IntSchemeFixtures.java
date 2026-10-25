package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.st.StRowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import com.eonmux.neuralfree.tensor.HalfFloats;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Shared fixtures of the integer scheme tests: hand-built safetensors checkpoints ({@link IntFixture}), reference
 * weights ({@link Ref}), the GPTQ, AWQ and compressed-tensors tensor writers and the packing helpers. Every
 * reference value is exact in fp16, so decoded weights compare exactly.
 */
final class IntSchemeFixtures {

    /** The module every single-module fixture quantizes. */
    static final String M = "model.layers.0.self_attn.q_proj";
    /** A configuration without quantization settings. */
    static final String PLAIN = "{'architectures':['A']}";

    private IntSchemeFixtures() {
    }

    /** Asserts {@code decoded[n][j] == ref(n, perm[j])} and the inverse identity {@code decoded[n][inv[k]] == ref(n, k)}. */
    static void assertPermuted(Ref r, int[] perm, float[] decoded) {
        int[] inv = new int[perm.length];
        for (int j = 0; j < perm.length; j++) {
            inv[perm[j]] = j;
        }
        for (int n = 0; n < r.n; n++) {
            for (int j = 0; j < r.k; j++) {
                assertEquals("row " + n + " column " + j, r.value(n, perm[j]), decoded[n * r.k + j], 0f);
            }
            for (int k = 0; k < r.k; k++) {
                assertEquals("row " + n + " input " + k, r.value(n, k), decoded[n * r.k + inv[k]], 0f);
            }
        }
    }

    /** {@code g_idx[k] = k % 2}: two interleaved groups of {@code k/2} columns. */
    static int[] alternating(int k) {
        int[] g = new int[k];
        for (int i = 0; i < k; i++) {
            g[i] = i % 2;
        }
        return g;
    }

    /** Adds GPTQ tensors of {@code r}; the stored zero is {@code z - zeroOffset} (1 for GPTQ v1). */
    static void addGptq(IntFixture f, String m, Ref r, int zeroOffset, boolean withGidx) {
        int[] qw = gptqQweight(r, false);
        f.i32(m + ".qweight", new long[]{qw.length / r.n, r.n}, qw);
        int groups = r.groups();
        int[][] zg = new int[groups][r.n];
        for (int g = 0; g < groups; g++) {
            for (int n = 0; n < r.n; n++) {
                zg[g][n] = (int) r.zero[n][g] - zeroOffset;
            }
        }
        int[] qz = packAlongN(zg, r.bits);
        f.i32(m + ".qzeros", new long[]{groups, qz.length / groups}, qz);
        f.f16(m + ".scales", new long[]{groups, r.n}, r.groupMajorScales());
        if (withGidx) {
            f.i32(m + ".g_idx", new long[]{r.k}, r.gidx);
        }
    }

    /** Adds AWQ tensors of {@code r} ({@code qweight [K, N/8]}, {@code qzeros [G, N/8]}, {@code scales [G, N]}). */
    static void addAwq(IntFixture f, String m, Ref r, boolean withZeros) {
        int np = r.n / 8;
        int[] qw = new int[r.k * np];
        for (int k = 0; k < r.k; k++) {
            for (int n = 0; n < r.n; n++) {
                qw[k * np + n / 8] |= r.codes[n][k] << (4 * IntPacking.awqSlot(n % 8));
            }
        }
        f.i32(m + ".qweight", new long[]{r.k, np}, qw);
        int groups = r.groups();
        if (withZeros) {
            int[] qz = new int[groups * np];
            for (int g = 0; g < groups; g++) {
                for (int n = 0; n < r.n; n++) {
                    qz[g * np + n / 8] |= ((int) r.zero[n][g]) << (4 * IntPacking.awqSlot(n % 8));
                }
            }
            f.i32(m + ".qzeros", new long[]{groups, np}, qz);
        }
        f.f16(m + ".scales", new long[]{groups, r.n}, r.groupMajorScales());
    }

    /**
     * Adds compressed-tensors pack-quantized tensors of {@code r}: rows packed along K, {@code weight_scale}
     * {@code [N, G]}, {@code weight_shape}, and the zero points packed along N when {@code withZeros}.
     */
    static void addCt(IntFixture f, String m, Ref r, boolean withZeros) {
        int[] wp = ctPacked(r);
        f.i32(m + ".weight_packed", new long[]{r.n, wp.length / r.n}, wp);
        f.f16(m + ".weight_scale", new long[]{r.n, r.groups()}, r.rowMajorScales());
        f.i64(m + ".weight_shape", new long[]{2}, r.n, r.k);
        if (withZeros) {
            int per = 32 / r.bits;
            int groups = r.groups();
            int words = (r.n + per - 1) / per;
            int[] zp = new int[words * groups];
            for (int n = 0; n < r.n; n++) {
                for (int g = 0; g < groups; g++) {
                    zp[(n / per) * groups + g] |= ((int) r.zero[n][g]) << (r.bits * (n % per));
                }
            }
            f.i32(m + ".weight_zero_point", new long[]{words, groups}, zp);
        }
    }

    /** compressed-tensors {@code weight_packed}: each row's codes packed per word along K. */
    static int[] ctPacked(Ref r) {
        int cols = (r.k + 32 / r.bits - 1) / (32 / r.bits);
        int[] out = new int[r.n * cols];
        for (int n = 0; n < r.n; n++) {
            System.arraycopy(packPerWord(r.codes[n], r.bits), 0, out, n * cols, cols);
        }
        return out;
    }

    /** GPTQ {@code qweight [Kw, N]}: column {@code n} holds row n's codes packed along K. */
    static int[] gptqQweight(Ref r, boolean stream) {
        int[][] cols = new int[r.n][];
        for (int n = 0; n < r.n; n++) {
            cols[n] = stream ? packBitStream(r.codes[n], r.bits) : packPerWord(r.codes[n], r.bits);
        }
        int kw = cols[0].length;
        int[] out = new int[kw * r.n];
        for (int w = 0; w < kw; w++) {
            for (int n = 0; n < r.n; n++) {
                out[w * r.n + n] = cols[n][w];
            }
        }
        return out;
    }

    /** Rows of {@code [G][N]} values, each packed per word along N. */
    static int[] packAlongN(int[][] rows, int bits) {
        int[] first = packPerWord(rows[0], bits);
        int[] out = new int[rows.length * first.length];
        for (int g = 0; g < rows.length; g++) {
            System.arraycopy(packPerWord(rows[g], bits), 0, out, g * first.length, first.length);
        }
        return out;
    }

    /** {@code 32/b} values per word, least significant first. */
    static int[] packPerWord(int[] v, int bits) {
        int per = 32 / bits;
        int[] out = new int[(v.length + per - 1) / per];
        for (int i = 0; i < v.length; i++) {
            out[i / per] |= v[i] << (bits * (i % per));
        }
        return out;
    }

    /** Value {@code e} at bit {@code e*b} of the little-endian word stream. */
    static int[] packBitStream(int[] v, int bits) {
        int[] out = new int[(v.length * bits + 31) / 32];
        for (int e = 0; e < v.length; e++) {
            long bit = (long) e * bits;
            int w = (int) (bit >>> 5);
            int shift = (int) (bit & 31);
            out[w] |= v[e] << shift;
            if (shift + bits > 32) {
                out[w + 1] |= v[e] >>> (32 - shift);
            }
        }
        return out;
    }

    static float[] concat(float[] a, float[] b) {
        float[] out = new float[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    static int[] concat(int[] a, int[] b) {
        int[] out = new int[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    static float[] slice(float[] v, int from, int len) {
        float[] out = new float[len];
        System.arraycopy(v, from, out, 0, len);
        return out;
    }

    /** The decoded values of rows {@code [first, first + count)} read on their own. */
    static float[] rows(RowSource s, long first, int count) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(count * s.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        s.readRows(first, count, b);
        assertEquals(b.capacity(), b.position());
        long[] shape = s.shape();
        return IntFixture.decode(b.array(), s.format().ggmlType(), count * shape[shape.length - 1]);
    }

    /**
     * A reference integer weight {@code w[n][k] = (codes[n][k] - zero[n][g]) * scale[n][g]}, {@code g} from
     * {@code gidx} or {@code k / groupSize}. Scales are small multiples of 1/8 and zero points integers, so every
     * product, every {@code -z*s} and every value is exact in fp16.
     */
    static final class Ref {
        final int n;
        final int k;
        final int bits;
        final int groupSize;
        final int[] gidx;
        final int[][] codes;
        final float[][] scale;
        final float[][] zero;

        Ref(int n, int k, int bits, int groupSize, int seed, int[] gidx, boolean asymmetric) {
            this.n = n;
            this.k = k;
            this.bits = bits;
            this.groupSize = groupSize;
            this.gidx = gidx;
            int levels = 1 << bits;
            int groups = groups();
            codes = new int[n][k];
            scale = new float[n][groups];
            zero = new float[n][groups];
            for (int r = 0; r < n; r++) {
                for (int c = 0; c < k; c++) {
                    codes[r][c] = (r * 31 + c * 17 + seed * 7 + (c * c) % 5) % levels;
                }
                for (int g = 0; g < groups; g++) {
                    scale[r][g] = 0.125f * (1 + (r + 3 * g + seed) % 5);
                    zero[r][g] = asymmetric ? (r * 5 + g * 3 + seed) % levels : levels / 2;
                }
            }
        }

        int groups() {
            return (k + groupSize - 1) / groupSize;
        }

        int group(int col) {
            return gidx == null ? col / groupSize : gidx[col];
        }

        float value(int row, int col) {
            int g = group(col);
            return (codes[row][col] - zero[row][g]) * scale[row][g];
        }

        float[] values() {
            float[] out = new float[n * k];
            for (int r = 0; r < n; r++) {
                for (int c = 0; c < k; c++) {
                    out[r * k + c] = value(r, c);
                }
            }
            return out;
        }

        /** Scales as {@code [N, G]}. */
        float[] rowMajorScales() {
            int groups = groups();
            float[] out = new float[n * groups];
            for (int r = 0; r < n; r++) {
                System.arraycopy(scale[r], 0, out, r * groups, groups);
            }
            return out;
        }

        /** Scales as {@code [G, N]}. */
        float[] groupMajorScales() {
            int groups = groups();
            float[] out = new float[n * groups];
            for (int r = 0; r < n; r++) {
                for (int g = 0; g < groups; g++) {
                    out[g * n + r] = scale[r][g];
                }
            }
            return out;
        }
    }

    /**
     * Hand-built safetensors tensors written to a file and indexed. A name added again replaces the earlier
     * tensor. Close it to release the file.
     */
    static final class IntFixture implements Closeable {
        private final Path dir;
        private final Map<String, Object[]> tensors = new LinkedHashMap<>();
        private SafetensorsFile file;
        private int files;

        IntFixture(Path dir) {
            this.dir = dir;
        }

        IntFixture add(String name, StDType dtype, long[] shape, byte[] data) {
            tensors.put(name, new Object[]{dtype, shape, data});
            return this;
        }

        IntFixture i32(String name, long[] shape, int... v) {
            ByteBuffer b = le(v.length * 4);
            for (int x : v) {
                b.putInt(x);
            }
            return add(name, StDType.I32, shape, b.array());
        }

        IntFixture u32(String name, long[] shape, int... v) {
            ByteBuffer b = le(v.length * 4);
            for (int x : v) {
                b.putInt(x);
            }
            return add(name, StDType.U32, shape, b.array());
        }

        IntFixture i64(String name, long[] shape, long... v) {
            ByteBuffer b = le(v.length * 8);
            for (long x : v) {
                b.putLong(x);
            }
            return add(name, StDType.I64, shape, b.array());
        }

        IntFixture bytes(String name, StDType dtype, long[] shape, int... v) {
            byte[] b = new byte[v.length];
            for (int i = 0; i < v.length; i++) {
                b[i] = (byte) v[i];
            }
            return add(name, dtype, shape, b);
        }

        IntFixture f16(String name, long[] shape, float... v) {
            ByteBuffer b = le(v.length * 2);
            for (float x : v) {
                b.putShort((short) FpBlockDecoders.fp32ToFp16(x));
            }
            return add(name, StDType.F16, shape, b.array());
        }

        IntFixture bf16(String name, long[] shape, float... v) {
            ByteBuffer b = le(v.length * 2);
            for (float x : v) {
                b.putShort((short) HalfFloats.floatToBFloat16(x));
            }
            return add(name, StDType.BF16, shape, b.array());
        }

        IntFixture f32(String name, long[] shape, float... v) {
            ByteBuffer b = le(v.length * 4);
            for (float x : v) {
                b.putFloat(x);
            }
            return add(name, StDType.F32, shape, b.array());
        }

        /** Writes the tensors to a new file and indexes them; the previous file is closed. */
        HfTensorIndex index() throws IOException {
            close();
            SafetensorsWriter.Builder builder = SafetensorsWriter.builder();
            for (Map.Entry<String, Object[]> e : tensors.entrySet()) {
                Object[] t = e.getValue();
                builder.add(e.getKey(), (StDType) t[0], (long[]) t[1], (byte[]) t[2]);
            }
            Path p = dir.resolve("model-" + files++ + ".safetensors");
            builder.build().write(p);
            file = SafetensorsFile.open(p);
            Map<String, RowSource> m = new LinkedHashMap<>();
            for (StTensor t : file.tensors()) {
                m.put(t.name(), StRowSource.of(file, t));
            }
            return HfTensorIndex.of(m);
        }

        @Override
        public void close() throws IOException {
            if (file != null) {
                file.close();
                file = null;
            }
        }

        static HfConfig config(Path dir, String singleQuotedJson) {
            return HfConfigLoader.fromJson(dir, singleQuotedJson.replace('\'', '"'));
        }

        /** Every value of {@code s}, read in one call and decoded to f32. */
        static float[] values(RowSource s) throws IOException {
            ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(s.rows() * s.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
            s.readRows(0, Math.toIntExact(s.rows()), b);
            assertEquals(b.capacity(), b.position());
            long numel = 1;
            for (long d : s.shape()) {
                numel *= d;
            }
            return decode(b.array(), s.format().ggmlType(), numel);
        }

        static float[] decode(byte[] raw, GGMLType type, long numel) {
            ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            float[] out = new float[Math.toIntExact(numel)];
            switch (type) {
                case F32:
                    for (int i = 0; i < out.length; i++) {
                        out[i] = b.getFloat(i * 4);
                    }
                    return out;
                case F16:
                    for (int i = 0; i < out.length; i++) {
                        out[i] = FpBlockDecoders.fp16ToFp32(b.getShort(i * 2));
                    }
                    return out;
                case BF16:
                    for (int i = 0; i < out.length; i++) {
                        out[i] = FpBlockDecoders.bf16ToFp32(b.getShort(i * 2));
                    }
                    return out;
                default:
                    return CpuBlockDecoders.dequantize(raw, type, numel);
            }
        }

        private static ByteBuffer le(int bytes) {
            return ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
        }
    }
}
