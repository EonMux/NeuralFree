package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfRope;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

/**
 * Tensors the converter generates rather than reads: RoPE frequency factors and in-memory constants. All
 * are small, held in memory, row-local and F32 unless stated.
 */
public final class RopeTensors {

    private RopeTensors() {
    }

    /** Which Phi-3 LongRoPE factor list. */
    public enum Factors { LONG, SHORT }

    /**
     * llama.cpp {@code LlamaModel.generate_extra_tensors} (LC/conversion/llama.py:299-327): the {@code rope_freqs}
     * factors of a {@code rope_type: llama3} configuration, computed in float32 as the converter's tensors are.
     *
     * @return empty when the global RoPE type is not llama3
     */
    public static Optional<RowSource> llama3Freqs(HfConfig cfg) {
        HfRope.Params p = cfg.rope().global();
        String type = p.type().orElse("");
        if (!type.toLowerCase(Locale.ROOT).equals("llama3")) {
            return Optional.empty();
        }
        double base = p.theta().orElse(10000.0);
        int dim = cfg.findInt("head_dim").isPresent() ? cfg.findInt("head_dim").getAsInt()
                : cfg.requireInt("hidden_size") / cfg.requireInt("num_attention_heads");
        return Optional.of(llama3Freqs(base, dim, 0, p.factor().orElse(8.0), p.lowFreqFactor().orElse(1.0),
                p.highFreqFactor().orElse(4.0), p.originalMaxPositionEmbeddings().orElse(8192), false));
    }

    /**
     * The llama3 factor of each frequency {@code 1 / base^(2i / rotaryDim)}: 1 above the high-frequency wavelength
     * bound, {@code factor} below the low one, and the smooth interpolation between them.
     *
     * @param padTo         the length to pad to with 1.0 (Step35, LC/conversion/step3.py:287-341), or 0
     * @param f64Wavelength true to compute wavelengths in double precision, as Step35 does through {@code float(freq)};
     *                      false for the float32 tensor arithmetic of llama.py, where a Python number divided by a
     *                      tensor is {@code tensor.reciprocal() * number} ({@code Tensor.__rtruediv__})
     * @throws IllegalArgumentException when {@code rotaryDim} is not positive and even
     */
    public static RowSource llama3Freqs(double base, int rotaryDim, int padTo, double factor, double lowFreqFactor,
                                       double highFreqFactor, int oldContextLen, boolean f64Wavelength) {
        if (rotaryDim <= 0 || rotaryDim % 2 != 0) {
            throw new IllegalArgumentException("llama3 rope factors need a positive even rotary width, got "
                    + rotaryDim);
        }
        int n = rotaryDim / 2;
        float[] out = new float[Math.max(n, padTo)];
        Arrays.fill(out, 1f);
        double lowWavelen = oldContextLen / lowFreqFactor;
        double highWavelen = oldContextLen / highFreqFactor;
        for (int i = 0; i < n; i++) {
            float x = (float) (2 * i) / (float) rotaryDim;
            float freq = 1f / (float) Math.pow((float) base, x);
            if (f64Wavelength) {
                double wavelen = 2 * Math.PI / freq;
                if (wavelen < highWavelen) {
                    out[i] = 1f;
                } else if (wavelen > lowWavelen) {
                    out[i] = (float) factor;
                } else {
                    double smooth = (oldContextLen / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
                    out[i] = (float) (1.0 / ((1.0 - smooth) / factor + smooth));
                }
            } else {
                float wavelen = (1f / freq) * (float) (2 * Math.PI);
                if (wavelen < (float) highWavelen) {
                    out[i] = 1f;
                } else if (wavelen > (float) lowWavelen) {
                    out[i] = (float) factor;
                } else {
                    float smooth = ((1f / wavelen) * (float) oldContextLen - (float) lowFreqFactor)
                            / (float) (highFreqFactor - lowFreqFactor);
                    out[i] = 1f / ((1f - smooth) / (float) factor + smooth);
                }
            }
        }
        return fromFloats(new long[]{out.length}, out);
    }

    /**
     * Phi-3 LongRoPE factors (LC/conversion/phi.py:180-202): {@code long_factor} or {@code short_factor}, each
     * {@code rope_dims / 2} long with {@code rope_dims = int(partial_rotary_factor * hidden_size) / heads}.
     *
     * @return empty when the configuration has no {@code long_factor}
     * @throws GgufFormatException when the RoPE type is not longrope or yarn, a list is missing, or the lengths are
     *                             wrong
     */
    public static Optional<RowSource> longropeFactors(HfConfig cfg, Factors which) {
        HfRope.Params p = cfg.rope().global();
        Optional<List<Double>> longF = p.longFactor();
        if (longF.isEmpty() || longF.get().isEmpty()) {
            return Optional.empty();
        }
        String type = p.type().orElse("").toLowerCase(Locale.ROOT);
        if (type.isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": LongRoPE factors need rope_scaling.type");
        }
        if (!type.equals("su") && !type.equals("longrope") && !type.equals("yarn")) {
            throw new GgufFormatException(cfg.source() + ": rope scaling type '" + type + "' is not supported with"
                    + " long_factor");
        }
        Optional<List<Double>> shortF = p.shortFactor();
        if (shortF.isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": LongRoPE needs both long_factor and short_factor");
        }
        double rot = p.partialRotaryFactor().orElse(1.0);
        int ropeDims = (int) (rot * cfg.requireInt("hidden_size")) / cfg.requireInt("num_attention_heads");
        List<Double> l = longF.get();
        List<Double> s = shortF.get();
        if (l.size() != s.size() || 2 * l.size() != ropeDims) {
            throw new GgufFormatException(cfg.source() + ": the rope long and short factors must have "
                    + ropeDims / 2.0 + " entries; long_factor has " + l.size() + ", short_factor " + s.size());
        }
        List<Double> chosen = which == Factors.LONG ? l : s;
        float[] v = new float[chosen.size()];
        for (int i = 0; i < v.length; i++) {
            v[i] = chosen.get(i).floatValue();
        }
        return Optional.of(fromFloats(new long[]{v.length}, v));
    }

    /**
     * Gemma 4 proportional RoPE frequency factors (LC/conversion/gemma.py:735-762): 1.0 for the
     * {@code int(headDim * partialRotaryFactor / 2)} rotated pairs, 1e30 for the remaining {@code headDim / 2} slots.
     */
    public static RowSource proportionalFreqs(int headDim, double partialRotaryFactor) {
        if (headDim <= 0 || !(partialRotaryFactor >= 0) || partialRotaryFactor > 1) {
            throw new IllegalArgumentException("proportional rope needs a positive head width and a rotary factor in"
                    + " [0, 1], got " + headDim + ", " + partialRotaryFactor);
        }
        int nRot = (int) (headDim * partialRotaryFactor / 2);
        int half = headDim / 2;
        float[] v = new float[half];
        for (int i = 0; i < half; i++) {
            v[i] = i < nRot ? 1f : 1e30f;
        }
        return fromFloats(new long[]{half}, v);
    }

    /** An F32 tensor of {@code shape} filled with {@code value}. */
    public static RowSource constant(long[] shape, float value) {
        long n = count(shape);
        float[] v = new float[Math.toIntExact(n)];
        Arrays.fill(v, value);
        return fromFloats(shape, v);
    }

    /** An F32 tensor of {@code shape} holding {@code data} in row-major order. */
    public static RowSource fromFloats(long[] shape, float[] data) {
        checkCount(shape, data.length);
        ByteBuffer b = ByteBuffer.allocate(4 * data.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : data) {
            b.putFloat(f);
        }
        return new InMemory("F32 constant " + Arrays.toString(shape), ElementFormat.of(GGMLType.F32), shape, b.array());
    }

    /** An I32 tensor of {@code shape} holding {@code data} in row-major order. */
    public static RowSource fromInts(long[] shape, int[] data) {
        checkCount(shape, data.length);
        ByteBuffer b = ByteBuffer.allocate(4 * data.length).order(ByteOrder.LITTLE_ENDIAN);
        for (int x : data) {
            b.putInt(x);
        }
        return new InMemory("I32 constant " + Arrays.toString(shape), ElementFormat.of(GGMLType.I32), shape, b.array());
    }

    /** An I64 tensor of {@code shape} holding {@code data} in row-major order. */
    public static RowSource fromLongs(long[] shape, long[] data) {
        checkCount(shape, data.length);
        ByteBuffer b = ByteBuffer.allocate(8 * data.length).order(ByteOrder.LITTLE_ENDIAN);
        for (long x : data) {
            b.putLong(x);
        }
        return new InMemory("I64 constant " + Arrays.toString(shape), ElementFormat.of(GGMLType.I64), shape, b.array());
    }

    /**
     * A tensor of {@code shape} whose bytes in {@code format} are {@code data} (little-endian, row-major).
     *
     * @throws IllegalArgumentException when the byte count does not match the shape and format
     */
    public static RowSource fromBytes(ElementFormat format, long[] shape, byte[] data) {
        return new InMemory(format + " constant " + Arrays.toString(shape), format, shape, data.clone());
    }

    private static long count(long[] shape) {
        long n = 1;
        for (long d : shape) {
            if (d < 0) {
                throw new IllegalArgumentException("negative dimension in " + Arrays.toString(shape));
            }
            n = Math.multiplyExact(n, d);
        }
        if (n > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("an in-memory tensor of shape " + Arrays.toString(shape)
                    + " is too large");
        }
        return n;
    }

    private static void checkCount(long[] shape, int length) {
        if (count(shape) != length) {
            throw new IllegalArgumentException(length + " values do not fill the shape " + Arrays.toString(shape));
        }
    }

    /** Bytes held in memory. */
    private static final class InMemory extends RowTransforms.Derived {
        private final byte[] data;

        InMemory(String describe, ElementFormat format, long[] shape, byte[] data) {
            super(describe, format, shape, ExpertTransforms.Siblings.NONE);
            if (Math.multiplyExact(rows(), rowBytes()) != data.length) {
                throw new IllegalArgumentException(describe + ": " + data.length + " bytes do not match "
                        + rows() + " rows of " + rowBytes() + " bytes");
            }
            this.data = data;
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        void produce(long first, int count, ByteBuffer out) throws IOException {
            out.put(data, (int) (first * rowBytes()), (int) (count * rowBytes()));
        }
    }
}
