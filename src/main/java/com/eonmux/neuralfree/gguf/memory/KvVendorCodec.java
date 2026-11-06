package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.utils.numeric.Fp8Codec;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Reference codecs of the vendor KV cache formats, computed from the formulas of vLLM's kernels:
 * <ul>
 *   <li><b>NVFP4</b> ({@code --kv-cache-dtype nvfp4}, and {@code nvfp4_4over6}): every 16 consecutive values of a
 *       head share one E4M3 scale {@code sf = e4m3(vecMax / 6)} (the 4over6 variant searches {@code vecMax / 4}
 *       and keeps the smaller squared reconstruction error), the values are E2M1 codes of {@code x / float(sf)} packed
 *       two per byte (the even value in the low nibble), and the layer's checkpoint KV scale divides the values
 *       before the store and multiplies them back at read ({@link #nvfp4Quantize}, {@link #e2m1Value}). The formulas
 *       are {@code cvt_warp_fp16_to_fp4} / {@code cvt_warp_fp16_to_fp4_4over6} of vLLM
 *       csrc/quantization/fp4/nvfp4_utils.cuh and csrc/nvfp4_kv_cache_kernels.cu.</li>
 *   <li><b>TurboQuant</b> ({@code turboquant_k8v4}, {@code turboquant_4bit_nc}, {@code turboquant_k3v4_nc},
 *       {@code turboquant_3bit_nc}): the key stream is either E4M3 codes ({@code turboquant_k8v4}) or per-coordinate
 *       Lloyd-Max centroid indices of {@code Hadamard(x / ||x||)} with the vector norm in a per-head plane
 *       ({@link #turboQuantKeyCodes}); the value stream is a per-head uniform quantizer with an affine scale and
 *       zero ({@link #turboQuantValueCodes}). The reference is vLLM's Triton store/decode pair
 *       vllm/v1/attention/ops/triton_turboquant_store.py and triton_turboquant_decode.py with the centroids of
 *       model_executor/layers/quantization/turboquant/centroids.py and the normalized Sylvester Hadamard matrix of
 *       v1/attention/backends/turboquant_attn.py.</li>
 *   <li><b>DeepSeek MLA latent rows</b> ({@code fp8_ds_mla}, {@code nvfp4_ds_mla}): 512 NoPE values quantized per
 *       128-value tile (E4M3 with a power-of-two F32 tile scale) or per 16-value group (E2M1 with an E4M3 scale),
 *       followed by 64 RoPE values stored unscaled (BF16 or E4M3) ({@link #fp8DsMlaStore}, {@link #nvfp4DsMlaStore}).
 *       The reference is vLLM csrc/cache_kernels.cu {@code concat_and_cache_ds_mla_kernel} and
 *       csrc/nvfp4_ds_mla_cache_kernels.cu; the scale-factor byte permutation of the latter is a device layout
 *       detail and the side plane here keeps the group scales in element order.</li>
 * </ul>
 * Every method is stateless; the centroid tables are cached per {@code (headDim, bits)}. Immutable data only.
 */
public final class KvVendorCodec {

    /** Values per NVFP4 scale group ({@code CVT_FP4_SF_VEC_SIZE}). */
    public static final int NVFP4_GROUP = 16;

    /** Values per DeepSeek MLA NoPE tile ({@code fp8_ds_mla}). */
    public static final int DS_MLA_TILE = 128;

    /** E2M1 magnitudes of the codes {@code 0..7}: 0, 0.5, 1, 1.5, 2, 3, 4, 6. */
    private static final float[] E2M1_VALUES = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

    /** Smallest NVFP4 group scale of {@code nvfp4_ds_mla} ({@code 2^-9}). */
    private static final float NVFP4_DS_MLA_MIN_SF = 0.001953125f;

    /** Smallest tile scale of {@code fp8_ds_mla} ({@code kFp8ScaleDivisor}, vLLM csrc/cache_kernels.cu:678). */
    private static final float FP8_DS_MLA_EPS = 1e-4f;

    /** Largest code magnitude of {@code fp8_ds_mla} NoPE values ({@code kFp8ScaleDivisor} = 448). */
    private static final float FP8_DS_MLA_QMAX = 448.0f;

    /** Smallest TurboQuant value scale (vLLM clamps it at 1e-8). */
    private static final float TQ_VALUE_EPS = 1e-8f;

    /** Smallest TurboQuant key norm (vLLM divides by {@code ||k|| + 1e-8}). */
    private static final float TQ_NORM_EPS = 1e-8f;

    /** Lloyd-Max integration bound in standard deviations (vLLM {@code solve_lloyd_max}). */
    private static final double LL_LIMIT = 3.5;

    /** Trapezoid intervals of the Lloyd-Max integrals (vLLM {@code _trapz}). */
    private static final int LL_INTERVALS = 200;

    /** Lloyd-Max iterations and convergence tolerance (vLLM {@code solve_lloyd_max}). */
    private static final int LL_ITERATIONS = 200;
    private static final double LL_TOLERANCE = 1e-10;

    private static final ConcurrentMap<Integer, float[]> CENTROIDS = new ConcurrentHashMap<>();
    private static final ConcurrentMap<Integer, float[]> MIDPOINTS = new ConcurrentHashMap<>();

    private KvVendorCodec() {
    }

    // ------------------------------------------------------------------ E2M1 / NVFP4

    /** The value of an E2M1 code in {@code [0, 15]}: the code's magnitude from {@link #E2M1_VALUES}, signed by bit 3. */
    public static float e2m1Value(int code) {
        int c = code & 0xF;
        float magnitude = E2M1_VALUES[c & 0x7];
        return (c & 0x8) != 0 ? -magnitude : magnitude;
    }

    /**
     * The E2M1 code of {@code x}, round to nearest with ties to even and saturation to 6, as
     * {@code cvt.rn.satfinite.e2m1x2.f32} (vLLM {@code round_to_nearest_e2m1}, csrc/nvfp4_kv_cache_kernels.cu:56-75).
     */
    public static int e2m1Code(float x) {
        float ax = Math.abs(x);
        int magnitude;
        if (ax <= 0.25f) {
            magnitude = 0;
        } else if (ax < 0.75f) {
            magnitude = 1;
        } else if (ax <= 1.25f) {
            magnitude = 2;
        } else if (ax < 1.75f) {
            magnitude = 3;
        } else if (ax <= 2.5f) {
            magnitude = 4;
        } else if (ax < 3.5f) {
            magnitude = 5;
        } else if (ax <= 5.0f) {
            magnitude = 6;
        } else {
            magnitude = 7;
        }
        return (Float.floatToRawIntBits(x) & 0x80000000) != 0 ? magnitude | 0x8 : magnitude;
    }

    /** The E4M3 code of {@code value} (RNE, saturating), the NVFP4 and {@code fp8_ds_mla} scale encoding. */
    public static int e4m3Code(float value) {
        return Fp8Codec.E4M3FN.encode(value);
    }

    /** The value of an E4M3 code. */
    public static float e4m3Value(int code) {
        return Fp8Codec.E4M3FN.decode(code & 0xFF);
    }

    /**
     * The E4M3 scale of one NVFP4 group: {@code e4m3(vecMax / 6)}, or, with {@code fourOverSix}, the divisor (6 or 4)
     * whose 16 codes reconstruct the group with the smaller squared error (ties keep 6), as vLLM's 4over6 store.
     *
     * @throws IllegalArgumentException for an empty group
     */
    public static float nvfp4GroupScale(float[] x, int xOff, int n, boolean fourOverSix) {
        if (n <= 0) {
            throw new IllegalArgumentException("NVFP4 needs a positive group size, got " + n);
        }
        float vecMax = 0.0f;
        for (int i = 0; i < n; i++) {
            float a = Math.abs(x[xOff + i]);
            vecMax = vecMax > a ? vecMax : a;
        }
        float sf6 = e4m3Value(e4m3Code(vecMax / 6.0f));
        if (!fourOverSix) {
            return sf6;
        }
        float sf4 = e4m3Value(e4m3Code(vecMax / 4.0f));
        return reconstructionError(x, xOff, n, sf4) < reconstructionError(x, xOff, n, sf6) ? sf4 : sf6;
    }

    /** Squared error of a group quantized under {@code sf}; a zero scale reconstructs zeros (vLLM's error loop). */
    private static float reconstructionError(float[] x, int xOff, int n, float sf) {
        if (sf == 0.0f) {
            float error = 0.0f;
            for (int i = 0; i < n; i++) {
                error += x[xOff + i] * x[xOff + i];
            }
            return error;
        }
        float outputScale = 1.0f / sf;
        float dequantScale = 1.0f / outputScale;
        float error = 0.0f;
        for (int i = 0; i < n; i++) {
            float q = e2m1Value(e2m1Code(x[xOff + i] * outputScale)) * dequantScale;
            float d = q - x[xOff + i];
            error += d * d;
        }
        return error;
    }

    /**
     * Quantizes {@code n} values into NVFP4 codes under {@code sf} and packs them two per byte into
     * {@code dst[dstOff..]}, the even value in the low nibble; {@code n} must be even.
     */
    public static void nvfp4Pack(float[] x, int xOff, int n, float sf, byte[] dst, int dstOff) {
        if (n <= 0 || (n & 1) != 0) {
            throw new IllegalArgumentException("NVFP4 packs two values per byte, got " + n + " values");
        }
        float outputScale = sf != 0.0f ? 1.0f / sf : 0.0f;
        for (int i = 0; i < n; i += 2) {
            int lo = e2m1Code(x[xOff + i] * outputScale);
            int hi = e2m1Code(x[xOff + i + 1] * outputScale);
            dst[dstOff + (i >> 1)] = (byte) (lo | (hi << 4));
        }
    }

    /** The E2M1 code at {@code index} of two-values-per-byte storage. */
    public static int nvfp4Code(byte[] src, int byteOff, int index) {
        int b = src[byteOff + (index >> 1)] & 0xFF;
        return (index & 1) == 0 ? b & 0xF : b >>> 4;
    }

    // ------------------------------------------------------------------ TurboQuant values

    /**
     * Quantizes one TurboQuant value vector of {@code n} values into uniform codes of {@code levels} (15 for 4 bits,
     * 7 for 3 bits): {@code scale = max((max - min) / levels, 1e-8)}, {@code code = clamp(trunc((x - min)/scale +
     * 0.5), 0, levels)} (vLLM {@code _store_quantized_value}). The caller stores {@code scale} and {@code min} as the
     * row's F16 plane values and passes their F16-rounded forms back at read, as vLLM does.
     *
     * @param codes      receives the codes, one per value
     * @param scaleOut   receives {@code scale} (index 0)
     * @param zeroOut    receives {@code min} (index 0)
     * @throws IllegalArgumentException for an empty vector
     */
    public static void turboQuantValueCodes(float[] x, int xOff, int n, int levels, int[] codes, float[] scaleOut,
                                            float[] zeroOut) {
        if (n <= 0) {
            throw new IllegalArgumentException("TurboQuant values need a positive head width, got " + n);
        }
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            min = Math.min(min, x[xOff + i]);
            max = Math.max(max, x[xOff + i]);
        }
        float scale = (max - min) / levels;
        scale = scale > TQ_VALUE_EPS ? scale : TQ_VALUE_EPS;
        for (int i = 0; i < n; i++) {
            int q = (int) ((x[xOff + i] - min) / scale + 0.5f);
            codes[i] = Math.max(0, Math.min(levels, q));
        }
        scaleOut[0] = scale;
        zeroOut[0] = min;
    }

    /** Packs {@code n} codes of {@code bits} bits (3 or 4) little-endian into {@code dst[dstOff..]}. */
    public static void turboQuantPack(int[] codes, int n, int bits, byte[] dst, int dstOff) {
        if (bits != 3 && bits != 4) {
            throw new IllegalArgumentException("TurboQuant codes are 3 or 4 bits, got " + bits);
        }
        if (n <= 0 || n % 8 != 0) {
            throw new IllegalArgumentException("TurboQuant packs its codes in groups of 8, got " + n + " values");
        }
        if (bits == 4) {
            for (int i = 0; i < n; i += 2) {
                dst[dstOff + (i >> 1)] = (byte) ((codes[i] & 0xF) | ((codes[i + 1] & 0xF) << 4));
            }
            return;
        }
        for (int g = 0; g < n; g += 8) {
            int packed = 0;
            for (int i = 0; i < 8; i++) {
                packed |= (codes[g + i] & 0x7) << (3 * i);
            }
            dst[dstOff + (g >> 3) * 3] = (byte) packed;
            dst[dstOff + (g >> 3) * 3 + 1] = (byte) (packed >>> 8);
            dst[dstOff + (g >> 3) * 3 + 2] = (byte) (packed >>> 16);
        }
    }

    /** The code at {@code index} of a 3- or 4-bit packed TurboQuant stream. */
    public static int turboQuantCode(byte[] src, int byteOff, int index, int bits) {
        if (bits == 4) {
            int b = src[byteOff + (index >> 1)] & 0xFF;
            return (index & 1) == 0 ? b & 0xF : b >>> 4;
        }
        int bit = index * 3;
        int at = byteOff + (bit >> 3);
        int raw = (src[at] & 0xFF) | ((src[StrictMath.min(at + 1, src.length - 1)] & 0xFF) << 8);
        return (raw >>> (bit & 7)) & 0x7;
    }

    // ------------------------------------------------------------------ TurboQuant keys

    /**
     * The Lloyd-Max centroids of a {@code d}-dimensional unit vector rotated by a random orthogonal matrix: the optimal
     * quantizer of {@code N(0, 1/d)} (vLLM {@code centroids.solve_lloyd_max}, 200 trapezoid intervals, 200 iterations,
     * tolerance 1e-10, float32 values). Cached per {@code (d, bits)}.
     *
     * @throws IllegalArgumentException for a non-positive dimension or a bit width outside {@code [1, 8]}
     */
    public static float[] centroids(int d, int bits) {
        if (d <= 0 || bits <= 0 || bits > 8) {
            throw new IllegalArgumentException("Lloyd-Max centroids need d > 0 and bits in [1, 8], got " + d + ", "
                    + bits);
        }
        int key = d * 31 + bits;
        return CENTROIDS.computeIfAbsent(key, k -> computeCentroids(d, bits));
    }

    /** The decision boundaries between the centroids: the midpoints of consecutive sorted centroids. */
    public static float[] midpoints(int d, int bits) {
        int key = d * 31 + bits;
        return MIDPOINTS.computeIfAbsent(key, k -> {
            float[] c = centroids(d, bits).clone();
            java.util.Arrays.sort(c);
            float[] m = new float[c.length - 1];
            for (int i = 0; i < m.length; i++) {
                m[i] = (c[i] + c[i + 1]) / 2.0f;
            }
            return m;
        });
    }

    private static float[] computeCentroids(int d, int bits) {
        int levels = 1 << bits;
        double sigma2 = 1.0 / d;
        double sigma = Math.sqrt(sigma2);
        double lo = -LL_LIMIT * sigma;
        double hi = LL_LIMIT * sigma;
        double[] centroids = new double[levels];
        for (int i = 0; i < levels; i++) {
            centroids[i] = lo + (hi - lo) * (i + 0.5) / levels;
        }
        for (int iter = 0; iter < LL_ITERATIONS; iter++) {
            double[] edges = new double[levels + 1];
            edges[0] = lo * 3.0;
            for (int i = 0; i < levels - 1; i++) {
                edges[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0;
            }
            edges[levels] = hi * 3.0;
            double[] next = new double[levels];
            double delta = 0.0;
            for (int i = 0; i < levels; i++) {
                double den = trapz(edges[i], edges[i + 1], sigma2, false);
                next[i] = den > 1e-15 ? trapz(edges[i], edges[i + 1], sigma2, true) / den : centroids[i];
                delta = Math.max(delta, Math.abs(next[i] - centroids[i]));
            }
            if (delta < LL_TOLERANCE) {
                break;
            }
            centroids = next;
        }
        float[] out = new float[levels];
        for (int i = 0; i < levels; i++) {
            out[i] = (float) centroids[i];
        }
        return out;
    }

    /** {@code x * pdf(x)} over {@code [a, b]} with {@code pdf = N(0, sigma2)}, or {@code pdf} itself. */
    private static double trapz(double a, double b, double sigma2, boolean weighted) {
        double h = (b - a) / LL_INTERVALS;
        double sum = 0.5 * (f(a, sigma2, weighted) + f(b, sigma2, weighted));
        for (int i = 1; i < LL_INTERVALS; i++) {
            sum += f(a + i * h, sigma2, weighted);
        }
        return sum * h;
    }

    private static double f(double x, double sigma2, boolean weighted) {
        double pdf = (1.0 / Math.sqrt(2.0 * Math.PI * sigma2)) * Math.exp(-x * x / (2.0 * sigma2));
        return weighted ? x * pdf : pdf;
    }

    /** The centroid index vLLM's binary search returns for {@code y}: the count of midpoints not above {@code y}. */
    public static int centroidIndex(float y, float[] midpoints) {
        int lo = 0;
        int hi = midpoints.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (y >= midpoints[mid]) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return Math.min(lo, midpoints.length);
    }

    /**
     * The centroid vector of {@code indices} with vLLM's norm correction: when {@code normCorrection} is set the
     * vector is divided by its own norm, so the inverse rotation of a unit-norm centroid vector has unit length.
     *
     * @return the vector's norm, written to {@code normOut[0]}
     */
    public static float centroidsWithNorm(int[] indices, int n, float[] centroids, boolean normCorrection,
                                          float[] dst, int dstOff, float[] normOut) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            dst[dstOff + i] = centroids[indices[i]];
            sum += dst[dstOff + i] * dst[dstOff + i];
        }
        float norm = (float) Math.sqrt(sum + 1e-16f);
        if (normCorrection) {
            for (int i = 0; i < n; i++) {
                dst[dstOff + i] /= norm;
            }
        }
        normOut[0] = norm;
        return norm;
    }

    /**
     * Multiplies {@code x[0, n)} by the orthonormal Sylvester Hadamard matrix {@code H[i][j] = (-1)^popcount(i & j) /
     * sqrt(n)} (vLLM {@code _build_hadamard}); {@code n} must be a power of two.
     *
     * @throws IllegalArgumentException for a non-power-of-two width
     */
    public static void hadamard(float[] x, int xOff, int n, float[] dst, int dstOff) {
        if (n <= 0 || Integer.bitCount(n) != 1) {
            throw new IllegalArgumentException("the TurboQuant Hadamard rotation needs a power-of-two width, got " + n);
        }
        float inv = (float) (1.0 / Math.sqrt(n));
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += ((Integer.bitCount(i & j) & 1) == 0 ? x[xOff + j] : -x[xOff + j]);
            }
            dst[dstOff + i] = sum * inv;
        }
    }

    /** True when {@code n} is a power of two, the width the TurboQuant Hadamard rotation needs. */
    public static boolean hadamardWidth(int n) {
        return n > 0 && Integer.bitCount(n) == 1;
    }

    // ------------------------------------------------------------------ DeepSeek MLA latent rows

    /**
     * Stores one {@code fp8_ds_mla} latent row of {@link KvDType#DS_MLA_ROW_VALUES} values into
     * {@code dst[dstOff, dstOff + 640)} and its four tile scales into {@code scales[scaleOff, scaleOff + 4)}:
     * NoPE values {@code [0, 512)} in 4 tiles of 128 E4M3 codes under
     * {@code scale = 2^ceil(log2(max(amax / 448, 1e-4)))}; RoPE values {@code [512, 576)} as unscaled BF16
     * (vLLM {@code concat_and_cache_ds_mla_kernel}, csrc/cache_kernels.cu:597-720).
     */
    public static void fp8DsMlaStore(float[] x, int xOff, byte[] dst, int dstOff, float[] scales, int scaleOff) {
        int nope = KvDType.DS_MLA_NOPE_VALUES;
        int rope = KvDType.DS_MLA_ROPE_VALUES;
        for (int t = 0; t < nope / DS_MLA_TILE; t++) {
            int base = xOff + t * DS_MLA_TILE;
            float amax = 0.0f;
            for (int i = 0; i < DS_MLA_TILE; i++) {
                amax = Math.max(amax, Math.abs(x[base + i]));
            }
            float scale = Math.max(amax / FP8_DS_MLA_QMAX, FP8_DS_MLA_EPS);
            scale = (float) Math.pow(2.0, Math.ceil(Math.log(scale) / Math.log(2.0)));
            scales[scaleOff + t] = scale;
            for (int i = 0; i < DS_MLA_TILE; i++) {
                dst[dstOff + t * DS_MLA_TILE + i] = (byte) e4m3Code(x[base + i] / scale);
            }
        }
        int ropeDst = dstOff + nope;
        for (int i = 0; i < rope; i++) {
            int bits = KvBlockCodec.fp32ToBf16(x[xOff + nope + i]);
            dst[ropeDst + 2 * i] = (byte) bits;
            dst[ropeDst + 2 * i + 1] = (byte) (bits >>> 8);
        }
    }

    /**
     * Reads one {@code fp8_ds_mla} latent row of 576 values: NoPE values scaled by their tile scale, RoPE values
     * decoded from BF16.
     */
    public static void fp8DsMlaLoad(byte[] src, int srcOff, float[] scales, int scaleOff, float[] dst, int dstOff) {
        int nope = KvDType.DS_MLA_NOPE_VALUES;
        int rope = KvDType.DS_MLA_ROPE_VALUES;
        for (int t = 0; t < nope / DS_MLA_TILE; t++) {
            float scale = scales[scaleOff + t];
            for (int i = 0; i < DS_MLA_TILE; i++) {
                dst[dstOff + t * DS_MLA_TILE + i] = e4m3Value(src[srcOff + t * DS_MLA_TILE + i]) * scale;
            }
        }
        int ropeSrc = srcOff + nope;
        for (int i = 0; i < rope; i++) {
            int bits = (src[ropeSrc + 2 * i] & 0xFF) | ((src[ropeSrc + 2 * i + 1] & 0xFF) << 8);
            dst[dstOff + nope + i] = FpBlockDecoders.bf16ToFp32((short) bits);
        }
    }

    /**
     * Stores one {@code nvfp4_ds_mla} latent row: 512 NoPE values in 32 groups of 16 E2M1 codes under
     * {@code sf = e4m3(max(amax / 6, 2^-9))} (the plane receives {@code sf}'s value), then 64 RoPE values as unscaled
     * E4M3 (vLLM {@code concat_and_cache_nvfp4_ds_mla_kernel}, csrc/nvfp4_ds_mla_cache_kernels.cu).
     */
    public static void nvfp4DsMlaStore(float[] x, int xOff, byte[] dst, int dstOff, float[] sfOut, int sfOff) {
        int nope = KvDType.DS_MLA_NOPE_VALUES;
        int rope = KvDType.DS_MLA_ROPE_VALUES;
        for (int g = 0; g < nope / NVFP4_GROUP; g++) {
            int base = xOff + g * NVFP4_GROUP;
            float amax = 0.0f;
            for (int i = 0; i < NVFP4_GROUP; i++) {
                amax = Math.max(amax, Math.abs(x[base + i]));
            }
            float sf = e4m3Value(e4m3Code(Math.max(amax / 6.0f, NVFP4_DS_MLA_MIN_SF)));
            sfOut[sfOff + g] = sf;
            float inv = sf != 0.0f ? 1.0f / sf : 0.0f;
            int byteBase = dstOff + g * (NVFP4_GROUP / 2);
            for (int i = 0; i < NVFP4_GROUP; i += 2) {
                int lo = e2m1Code(x[base + i] * inv);
                int hi = e2m1Code(x[base + i + 1] * inv);
                dst[byteBase + (i >> 1)] = (byte) (lo | (hi << 4));
            }
        }
        int ropeDst = dstOff + nope / 2;
        for (int i = 0; i < rope; i++) {
            dst[ropeDst + i] = (byte) e4m3Code(x[xOff + nope + i]);
        }
    }

    /**
     * Reads one {@code nvfp4_ds_mla} latent row of 576 values: NoPE values as E2M1 times their E4M3 group scale,
     * RoPE values as unscaled E4M3.
     */
    public static void nvfp4DsMlaLoad(byte[] src, int srcOff, float[] sf, int sfOff, float[] dst, int dstOff) {
        int nope = KvDType.DS_MLA_NOPE_VALUES;
        int rope = KvDType.DS_MLA_ROPE_VALUES;
        for (int g = 0; g < nope / NVFP4_GROUP; g++) {
            float scale = sf[sfOff + g];
            int byteBase = srcOff + g * (NVFP4_GROUP / 2);
            for (int i = 0; i < NVFP4_GROUP; i++) {
                dst[dstOff + g * NVFP4_GROUP + i] = e2m1Value(nvfp4Code(src, byteBase, i)) * scale;
            }
        }
        int ropeSrc = srcOff + nope / 2;
        for (int i = 0; i < rope; i++) {
            dst[dstOff + nope + i] = e4m3Value(src[ropeSrc + i]);
        }
    }
}
