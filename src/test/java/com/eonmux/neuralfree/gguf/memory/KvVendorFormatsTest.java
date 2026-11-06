package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.memory.cpu.CpuKvLayer;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.utils.numeric.Fp8Codec;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Parity of the vendor KV formats with reference codecs written from vLLM's kernels: NVFP4 and its 4over6 scale
 * search ({@code cvt_warp_fp16_to_fp4} / {@code cvt_warp_fp16_to_fp4_4over6}, csrc/quantization/fp4/nvfp4_utils.cuh
 * and csrc/nvfp4_kv_cache_kernels.cu), the TurboQuant key and value streams
 * (vllm/v1/attention/ops/triton_turboquant_store.py / _decode.py with the Lloyd-Max centroids of centroids.py and the
 * normalized Sylvester Hadamard rotation of turboquant_attn.py), and the DeepSeek MLA latent rows
 * (csrc/cache_kernels.cu {@code concat_and_cache_ds_mla_kernel}, csrc/nvfp4_ds_mla_cache_kernels.cu); plus the name
 * parsing, the pair rules and the row/plane geometry of each format. The reference formulas are re-derived here
 * instead of calling {@link KvVendorCodec}, so a change of the engine codec fails the test.
 */
public class KvVendorFormatsTest {

    /** vLLM's Lloyd-Max centroids for d = 64 (centroids.solve_lloyd_max), 3 and 4 bits. */
    private static final float[] CENTROIDS_64_3 = {
            -0.268793693f, -0.167897501f, -0.0944556878f, -0.0306229492f,
            0.0306229492f, 0.0944556878f, 0.167897501f, 0.268793693f};
    private static final float[] CENTROIDS_64_4 = {
            -0.341365277f, -0.25855589f, -0.20223522f, -0.157032196f, -0.11780603f, -0.0821099898f,
            -0.0485172239f, -0.0160534576f, 0.0160534576f, 0.0485172239f, 0.0821099898f, 0.11780603f,
            0.157032196f, 0.20223522f, 0.25855589f, 0.341365277f};

    private static float[] random(int n, long seed, float scale) {
        Random r = new Random(seed);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) (r.nextGaussian() * scale);
        }
        return x;
    }

    private static CpuKvLayer layer(KvDType k, KvDType v, int nHeadKv, int headDimK, int headDimV) {
        KvLayout l = KvLayout.of(4, false, nHeadKv * headDimK, nHeadKv * headDimV, k, v, KvWindow.FULL_CAUSAL)
                .withHeads(nHeadKv);
        return CpuKvLayer.allocate(0, l, 0, nHeadKv, headDimK, headDimV);
    }

    private static CpuKvLayer latent(KvDType type) {
        KvLayout l = KvLayout.latent(4, false, KvDType.DS_MLA_ROW_VALUES, KvDType.DS_MLA_NOPE_VALUES, type,
                KvWindow.FULL_CAUSAL);
        return CpuKvLayer.allocate(0, l, 0, 1, KvDType.DS_MLA_ROW_VALUES, KvDType.DS_MLA_NOPE_VALUES);
    }

    // ------------------------------------------------------------------ names and pair rules

    @Test
    public void parsesTheVllmCacheDtypeNames() {
        assertEquals(KvDType.NVFP4, KvDType.parse("nvfp4", "p"));
        assertEquals(KvDType.NVFP4_4OVER6, KvDType.parse("NVFP4_4OVER6", "p"));
        assertEquals(KvDType.FP8_DS_MLA, KvDType.parse("fp8_ds_mla", "p"));
        assertEquals(KvDType.NVFP4_DS_MLA, KvDType.parse("nvfp4_ds_mla", "p"));
        assertEquals(KvDType.FP8_E4M3, KvDType.parse("fp8_inc", "p"));
        assertEquals(KvDType.parse("fp8_e4m3", "p"), KvDType.parse("fp8_inc", "p"));

        assertEquals(KvCacheTypes.of(KvDType.TQ_K8_KEYS, KvDType.TQ_4BIT_VALUES),
                KvCacheTypes.parse("turboquant_k8v4", "p"));
        assertEquals(KvCacheTypes.of(KvDType.TQ_4BIT_KEYS, KvDType.TQ_4BIT_VALUES),
                KvCacheTypes.parse("turboquant_4bit_nc", "p"));
        assertEquals(KvCacheTypes.of(KvDType.TQ_3BIT_KEYS, KvDType.TQ_4BIT_VALUES),
                KvCacheTypes.parse("turboquant_k3v4_nc", "p"));
        assertEquals(KvCacheTypes.of(KvDType.TQ_3BIT_KEYS, KvDType.TQ_3BIT_VALUES),
                KvCacheTypes.parse("turboquant_3bit_nc", "p"));
        assertEquals(KvCacheTypes.of(KvDType.NVFP4), KvCacheTypes.parse("nvfp4", "p"));

        IllegalArgumentException pair = assertThrows(IllegalArgumentException.class,
                () -> KvDType.parse("turboquant_k8v4", "--cache-type-k"));
        assertTrue(pair.getMessage(), pair.getMessage().contains(MemoryConfig.PROP_KV_TYPE));
        IllegalArgumentException split = assertThrows(IllegalArgumentException.class,
                () -> KvCacheTypes.checkPair(KvDType.TQ_4BIT_KEYS, KvDType.F16, "p"));
        assertTrue(split.getMessage(), split.getMessage().contains("do not form a cache format"));
        assertThrows(IllegalArgumentException.class,
                () -> KvCacheTypes.checkPair(KvDType.NVFP4, KvDType.F16, "p"));
    }

    @Test
    public void plansTheTurboQuantPresetOnBothCaches() {
        MemoryConfig cfg = MemoryConfig.fromProperties(properties(MemoryConfig.PROP_KV_TYPE, "turboquant_3bit_nc"));
        assertEquals(KvDType.TQ_3BIT_KEYS, cfg.typeK());
        assertEquals(KvDType.TQ_3BIT_VALUES, cfg.typeV());
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.builder().typeK(KvDType.TQ_3BIT_KEYS).typeV(KvDType.F16).build());
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.builder().typeK(KvDType.NVFP4).typeV(KvDType.F16).build());
    }

    private static java.util.Properties properties(String key, String value) {
        java.util.Properties p = new java.util.Properties();
        p.setProperty(key, value);
        return p;
    }

    // ------------------------------------------------------------------ geometry

    @Test
    public void vendorRowsSizeTheirPackedBytesAndPlanes() {
        assertEquals(32, KvDType.NVFP4.rowBytes(64));
        assertEquals(32, KvDType.NVFP4_4OVER6.rowBytes(64));
        assertEquals(32, KvDType.TQ_4BIT_KEYS.rowBytes(64));
        assertEquals(24, KvDType.TQ_3BIT_KEYS.rowBytes(64));
        assertEquals(32, KvDType.TQ_4BIT_VALUES.rowBytes(64));
        assertEquals(24, KvDType.TQ_3BIT_VALUES.rowBytes(64));
        assertEquals(640, KvDType.FP8_DS_MLA.rowBytes(576));
        assertEquals(320, KvDType.NVFP4_DS_MLA.rowBytes(576));

        KvDType.SidePlane sf = KvDType.NVFP4.sidePlanes().get(0);
        assertEquals(4, sf.widthFor(64, 2));
        assertEquals(4L, sf.rowBytesFor(64, 2));
        assertEquals(2, KvDType.TQ_4BIT_KEYS.sidePlanes().get(0).widthFor(128, 2));
        assertEquals(32, KvDType.NVFP4_DS_MLA.sidePlanes().get(0).widthFor(576, 1));
        assertEquals(4, KvDType.FP8_DS_MLA.sidePlanes().get(0).widthFor(576, 1));
        KvLayout latent = KvLayout.latent(2, false, 576, 512, KvDType.FP8_DS_MLA, KvWindow.FULL_CAUSAL);
        assertEquals(2 * 640 + 2 * 4 * Float.BYTES, latent.totalBytes());
    }

    // ------------------------------------------------------------------ NVFP4

    @Test
    public void nvfp4MatchesTheReferenceGroupQuantizer() {
        for (boolean fourOverSix : new boolean[]{false, true}) {
            KvDType type = fourOverSix ? KvDType.NVFP4_4OVER6 : KvDType.NVFP4;
            CpuKvLayer c = layer(type, type, 2, 64, 64);
            float[] x = random(128, 31 + type.code(), 2.0f);
            c.writeK(1, x, 0);
            c.writeV(2, x, 0);
            assertArrayEquals(referenceNvfp4(x, fourOverSix), readRow(c, 1, false), 0f);
            assertArrayEquals(referenceNvfp4(x, fourOverSix), readRow(c, 2, true), 0f);
            assertEquals(64, c.layout().rowBytesK());
            assertEquals(8, c.sidePlaneRowBytes(false, 0));
        }
    }

    @Test
    public void nvfp4AppliesTheCheckpointScaleAroundTheStore() {
        CpuKvLayer c = layer(KvDType.NVFP4, KvDType.NVFP4, 1, 32, 32);
        float[] x = random(32, 77, 2.0f);
        float kScale = 0.5f;
        float[] divided = x.clone();
        for (int i = 0; i < divided.length; i++) {
            divided[i] /= kScale;
        }
        c.writeK(0, divided, 0);
        float[] back = readRow(c, 0, false);
        for (int i = 0; i < back.length; i++) {
            back[i] *= kScale;
        }
        assertArrayEquals(referenceNvfp4(x, false), back, 1e-6f);
    }

    private static float[] referenceNvfp4(float[] x, boolean fourOverSix) {
        float[] out = new float[x.length];
        for (int g = 0; g < x.length; g += 16) {
            float vecMax = 0.0f;
            for (int i = 0; i < 16; i++) {
                vecMax = Math.max(vecMax, Math.abs(x[g + i]));
            }
            float best = refSf(vecMax / 6.0f);
            if (fourOverSix) {
                float alt = refSf(vecMax / 4.0f);
                if (refError(x, g, alt) < refError(x, g, best)) {
                    best = alt;
                }
            }
            for (int i = 0; i < 16; i++) {
                float scaled = best != 0.0f ? x[g + i] * (1.0f / best) : 0.0f;
                out[g + i] = refE2m1(scaled) * best;
            }
        }
        return out;
    }

    private static float refError(float[] x, int g, float sf) {
        float error = 0.0f;
        if (sf == 0.0f) {
            for (int i = 0; i < 16; i++) {
                error += x[g + i] * x[g + i];
            }
            return error;
        }
        for (int i = 0; i < 16; i++) {
            float q = refE2m1(x[g + i] * (sf != 0.0f ? 1.0f / sf : 0.0f)) * sf;
            error += (q - x[g + i]) * (q - x[g + i]);
        }
        return error;
    }

    private static float refSf(float value) {
        return Fp8Codec.E4M3FN.decode(Fp8Codec.E4M3FN.encode(value));
    }

    /** vLLM {@code round_to_nearest_e2m1}, including its ties-to-even boundaries and the $\pm$0 sign. */
    private static float refE2m1(float x) {
        float ax = Math.abs(x);
        float q;
        if (ax <= 0.25f) {
            q = 0.0f;
        } else if (ax < 0.75f) {
            q = 0.5f;
        } else if (ax <= 1.25f) {
            q = 1.0f;
        } else if (ax < 1.75f) {
            q = 1.5f;
        } else if (ax <= 2.5f) {
            q = 2.0f;
        } else if (ax < 3.5f) {
            q = 3.0f;
        } else if (ax <= 5.0f) {
            q = 4.0f;
        } else {
            q = 6.0f;
        }
        return Math.copySign(q, x);
    }

    @Test
    public void e2m1CodesMatchTheDeviceConversion() {
        assertEquals(0.0f, KvVendorCodec.e2m1Value(KvVendorCodec.e2m1Code(0.24f)), 0f);
        assertEquals(0.5f, KvVendorCodec.e2m1Value(KvVendorCodec.e2m1Code(0.26f)), 0f);
        assertEquals(1.0f, KvVendorCodec.e2m1Value(KvVendorCodec.e2m1Code(1.25f)), 0f);
        assertEquals(1.5f, KvVendorCodec.e2m1Value(KvVendorCodec.e2m1Code(1.3f)), 0f);
        assertEquals(2.0f, KvVendorCodec.e2m1Value(KvVendorCodec.e2m1Code(2.5f)), 0f);
        assertEquals(3.0f, KvVendorCodec.e2m1Value(KvVendorCodec.e2m1Code(2.6f)), 0f);
        assertEquals(4.0f, KvVendorCodec.e2m1Value(KvVendorCodec.e2m1Code(5.0f)), 0f);
        assertEquals(6.0f, KvVendorCodec.e2m1Value(KvVendorCodec.e2m1Code(7.0f)), 0f);
        assertEquals(-6.0f, KvVendorCodec.e2m1Value(KvVendorCodec.e2m1Code(-7.0f)), 0f);
    }

    // ------------------------------------------------------------------ TurboQuant

    @Test
    public void lloydMaxCentroidsMatchTheVllmTables() {
        assertCentroids(CENTROIDS_64_3, KvVendorCodec.centroids(64, 3), 1e-6f);
        assertCentroids(CENTROIDS_64_4, KvVendorCodec.centroids(64, 4), 1e-6f);
        for (float v : KvVendorCodec.centroids(128, 4)) {
            assertTrue("centroid " + v + " is not finite", Float.isFinite(v));
        }
    }

    private static void assertCentroids(float[] expected, float[] actual, float tolerance) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals("centroid " + i, expected[i], actual[i], tolerance);
        }
    }

    @Test
    public void turboQuantValuesMatchTheReferenceUniformQuantizer() {
        CpuKvLayer c = layer(KvDType.TQ_4BIT_VALUES, KvDType.TQ_4BIT_VALUES, 2, 64, 64);
        float[] x = random(128, 9, 2.0f);
        c.writeK(0, x, 0);
        assertArrayEquals(referenceTurboQuantValues(x, 15, 4), readRow(c, 0, false), 0f);
        KvDType three = KvDType.TQ_3BIT_VALUES;
        CpuKvLayer c3 = layer(three, three, 1, 64, 64);
        c3.writeV(3, x, 0);
        assertArrayEquals(referenceTurboQuantValues(Arrays.copyOf(x, 64), 7, 3), readRow(c3, 3, true), 0f);
        assertEquals(24, c3.layout().rowBytesV());
    }

    @Test
    public void turboQuantK8v4StoresFp8KeysAndFourBitValues() {
        CpuKvLayer c = layer(KvDType.TQ_K8_KEYS, KvDType.TQ_4BIT_VALUES, 2, 64, 64);
        float[] keys = random(128, 17, 1.5f);
        float[] values = random(128, 18, 2.0f);
        c.writeK(1, keys, 0);
        c.writeV(1, values, 0);
        float[] expectedKeys = new float[keys.length];
        for (int i = 0; i < keys.length; i++) {
            expectedKeys[i] = Fp8Codec.E4M3FN.decode(Fp8Codec.E4M3FN.encode(keys[i]));
        }
        assertArrayEquals(expectedKeys, readRow(c, 1, false), 0f);
        assertArrayEquals(referenceTurboQuantValues(values, 15, 4), readRow(c, 1, true), 0f);
        assertEquals(128, c.layout().rowBytesK());
        assertEquals(64, c.layout().rowBytesV());
    }

    private static float[] referenceTurboQuantValues(float[] x, int levels, int bits) {
        float[] out = new float[x.length];
        int head = 64;
        for (int h = 0; h < x.length / head; h++) {
            float min = Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < head; i++) {
                min = Math.min(min, x[h * head + i]);
                max = Math.max(max, x[h * head + i]);
            }
            float scale = (max - min) / levels;
            scale = scale > 1e-8f ? scale : 1e-8f;
            float storedScale = FpBlockDecoders.fp16ToFp32(FpBlockDecoders.fp32ToFp16(scale));
            float storedZero = FpBlockDecoders.fp16ToFp32(FpBlockDecoders.fp32ToFp16(min));
            for (int i = 0; i < head; i++) {
                int code = (int) ((x[h * head + i] - min) / scale + 0.5f);
                code = Math.max(0, Math.min(levels, code));
                out[h * head + i] = code * storedScale + storedZero;
            }
        }
        return out;
    }

    /**
     * TurboQuant key stream parity: the engine's unrotated decode of a key dot a plain query equals vLLM's rotated
     * query ({@code q @ H}) dot the norm-corrected centroid vector times the vector norm.
     */
    @Test
    public void turboQuantKeyScoresMatchTheRotatedQueryForm() {
        for (int bits : new int[]{3, 4}) {
            KvDType keyType = bits == 3 ? KvDType.TQ_3BIT_KEYS : KvDType.TQ_4BIT_KEYS;
            CpuKvLayer c = layer(keyType, keyType, 1, 64, 64);
            float[] key = random(64, 100 + bits, 1.5f);
            c.writeK(2, key, 0);
            float[] decoded = readRow(c, 2, false);
            float[] reference = referenceTurboQuantKey(key, bits == 3 ? CENTROIDS_64_3 : CENTROIDS_64_4);
            assertArrayEquals(reference, decoded, 1e-5f);

            float[] q = random(64, 200 + bits, 1.0f);
            float engineScore = dot(q, decoded);
            float vllmScore = dot(hadamard(q), centroidVector(key, bits == 3 ? CENTROIDS_64_3 : CENTROIDS_64_4,
                    true)) * FpBlockDecoders.fp16ToFp32(FpBlockDecoders.fp32ToFp16(vectorNorm(key)));
            assertEquals("key score at " + bits + " bits", vllmScore, engineScore, 1e-4f * Math.max(1f,
                    Math.abs(vllmScore)));
        }
    }

    @Test
    public void turboQuantKeyStreamUsesThePowerOfTwoRotation() {
        CpuKvLayer c = layer(KvDType.TQ_3BIT_KEYS, KvDType.TQ_3BIT_KEYS, 1, 96, 96);
        assertThrows(IllegalArgumentException.class, () -> c.writeK(0, random(96, 1, 1f), 0));
        assertTrue(KvVendorCodec.hadamardWidth(128));
        assertFalse(KvVendorCodec.hadamardWidth(96));
    }

    private static float[] referenceTurboQuantKey(float[] key, float[] centroids) {
        float norm = vectorNorm(key);
        float[] unit = new float[key.length];
        for (int i = 0; i < key.length; i++) {
            unit[i] = key[i] / (norm + 1e-8f);
        }
        float[] rotated = hadamard(unit);
        float[] mid = midpoints(centroids);
        float[] c = new float[key.length];
        float sum = 0.0f;
        for (int i = 0; i < key.length; i++) {
            c[i] = centroids[upperBound(rotated[i], mid)];
            sum += c[i] * c[i];
        }
        float cNorm = (float) Math.sqrt(sum + 1e-16f);
        for (int i = 0; i < c.length; i++) {
            c[i] /= cNorm;
        }
        float[] decoded = hadamard(c);
        for (int i = 0; i < decoded.length; i++) {
            decoded[i] *= FpBlockDecoders.fp16ToFp32(FpBlockDecoders.fp32ToFp16(norm));
        }
        return decoded;
    }

    private static float[] centroidVector(float[] key, float[] centroids, boolean normCorrect) {
        float norm = vectorNorm(key);
        float[] unit = new float[key.length];
        for (int i = 0; i < key.length; i++) {
            unit[i] = key[i] / (norm + 1e-8f);
        }
        float[] rotated = hadamard(unit);
        float[] mid = midpoints(centroids);
        float[] c = new float[key.length];
        float sum = 0.0f;
        for (int i = 0; i < key.length; i++) {
            c[i] = centroids[upperBound(rotated[i], mid)];
            sum += c[i] * c[i];
        }
        if (normCorrect) {
            float cNorm = (float) Math.sqrt(sum + 1e-16f);
            for (int i = 0; i < c.length; i++) {
                c[i] /= cNorm;
            }
        }
        return c;
    }

    private static float vectorNorm(float[] x) {
        float sum = 0.0f;
        for (float v : x) {
            sum += v * v;
        }
        return (float) Math.sqrt(sum);
    }

    private static int upperBound(float y, float[] mid) {
        int lo = 0;
        int hi = mid.length;
        while (lo < hi) {
            int middle = (lo + hi) >>> 1;
            if (y >= mid[middle]) {
                lo = middle + 1;
            } else {
                hi = middle;
            }
        }
        return Math.min(lo, mid.length);
    }

    private static float[] midpoints(float[] centroids) {
        float[] c = centroids.clone();
        Arrays.sort(c);
        float[] m = new float[c.length - 1];
        for (int i = 0; i < m.length; i++) {
            m[i] = (c[i] + c[i + 1]) / 2.0f;
        }
        return m;
    }

    private static float[] hadamard(float[] x) {
        int n = x.length;
        float[] out = new float[n];
        double inv = 1.0 / Math.sqrt(n);
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += (Integer.bitCount(i & j) & 1) == 0 ? x[j] : -x[j];
            }
            out[i] = (float) (sum * inv);
        }
        return out;
    }

    private static float dot(float[] a, float[] b) {
        float s = 0.0f;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    // ------------------------------------------------------------------ DeepSeek MLA latent rows

    @Test
    public void fp8DsMlaMatchesTheReferenceLatentRow() {
        CpuKvLayer c = latent(KvDType.FP8_DS_MLA);
        float[] x = random(KvDType.DS_MLA_ROW_VALUES, 3, 2.0f);
        c.writeK(0, x, 0);
        // vLLM's 656-byte record keeps the four tile scales inline; here they live in the side plane.
        assertEquals(640, c.layout().rowBytesK());
        assertEquals(656, c.layout().totalBytes() / c.layout().rows());
        assertArrayEquals(referenceFp8DsMla(x), readRow(c, 0, false), 0f);
        float[] v = new float[KvDType.DS_MLA_NOPE_VALUES];
        c.readV(0, 0, v, 0);
        float[] full = readRow(c, 0, false);
        assertArrayEquals(Arrays.copyOf(full, KvDType.DS_MLA_NOPE_VALUES), v, 0f);
        assertEquals(1, c.sidePlaneCount(false));
        assertEquals(16, c.sidePlaneRowBytes(false, 0));
    }

    @Test
    public void nvfp4DsMlaMatchesTheReferenceLatentRow() {
        CpuKvLayer c = latent(KvDType.NVFP4_DS_MLA);
        float[] x = random(KvDType.DS_MLA_ROW_VALUES, 4, 2.0f);
        c.writeK(0, x, 0);
        assertEquals(320, c.layout().rowBytesK());
        assertEquals(352, c.layout().totalBytes() / c.layout().rows());
        assertArrayEquals(referenceNvfp4DsMla(x), readRow(c, 0, false), 0f);
        assertEquals(32, c.sidePlaneRowBytes(false, 0));
    }

    private static float[] referenceFp8DsMla(float[] x) {
        float[] out = x.clone();
        for (int t = 0; t < 4; t++) {
            float amax = 0.0f;
            for (int i = 0; i < 128; i++) {
                amax = Math.max(amax, Math.abs(x[t * 128 + i]));
            }
            float scale = Math.max(amax / 448.0f, 1e-4f);
            scale = (float) Math.pow(2.0, Math.ceil(Math.log(scale) / Math.log(2.0)));
            for (int i = 0; i < 128; i++) {
                out[t * 128 + i] = Fp8Codec.E4M3FN.decode(Fp8Codec.E4M3FN.encode(x[t * 128 + i] / scale)) * scale;
            }
        }
        for (int i = 512; i < 576; i++) {
            out[i] = FpBlockDecoders.bf16ToFp32(KvBlockCodec.fp32ToBf16(x[i]));
        }
        return out;
    }

    private static float[] referenceNvfp4DsMla(float[] x) {
        float[] out = x.clone();
        for (int g = 0; g < 32; g++) {
            float amax = 0.0f;
            for (int i = 0; i < 16; i++) {
                amax = Math.max(amax, Math.abs(x[g * 16 + i]));
            }
            float sf = refSf(Math.max(amax / 6.0f, 0.001953125f));
            for (int i = 0; i < 16; i++) {
                out[g * 16 + i] = refE2m1(sf != 0.0f ? x[g * 16 + i] * (1.0f / sf) : 0.0f) * sf;
            }
        }
        for (int i = 512; i < 576; i++) {
            out[i] = Fp8Codec.E4M3FN.decode(Fp8Codec.E4M3FN.encode(x[i]));
        }
        return out;
    }

    // ------------------------------------------------------------------ planes travel with rows

    @Test
    public void copyRowsCarriesTheVendorSidePlanes() {
        CpuKvLayer c = layer(KvDType.NVFP4, KvDType.NVFP4, 2, 64, 64);
        float[] x = random(128, 55, 2.0f);
        c.writeK(0, x, 0);
        c.writeV(0, x, 0);
        c.copyRows(0, 3, 1);
        assertArrayEquals(readRow(c, 0, false), readRow(c, 3, false), 0f);
        assertArrayEquals(readRow(c, 0, true), readRow(c, 3, true), 0f);
        byte[] plane0 = new byte[8];
        byte[] plane3 = new byte[8];
        c.readSideBytes(false, 0, 0, 1, plane0);
        c.readSideBytes(false, 0, 3, 1, plane3);
        assertArrayEquals(plane0, plane3);

        CpuKvLayer tq = layer(KvDType.TQ_4BIT_KEYS, KvDType.TQ_4BIT_VALUES, 1, 64, 64);
        tq.writeK(0, x.clone(), 0);
        tq.writeV(0, Arrays.copyOf(x, 64), 0);
        tq.copyRows(0, 2, 1);
        assertArrayEquals(readRow(tq, 0, false), readRow(tq, 2, false), 0f);
        assertArrayEquals(readRow(tq, 0, true), readRow(tq, 2, true), 0f);
    }

    /** The latent formats fill one K row and its V alias, so a layout for them rejects a different V width. */
    @Test
    public void latentFormatsKeepTheirFixedRowShape() {
        assertThrows(IllegalArgumentException.class,
                () -> KvDType.FP8_DS_MLA.rowBytes(512));
        assertTrue(KvDType.NVFP4_DS_MLA.latentFormat());
        assertTrue(KvDType.FP8_DS_MLA.latentFormat());
        assertFalse(KvDType.NVFP4.latentFormat());
    }

    // ------------------------------------------------------------------ helpers

    private static float[] readRow(CpuKvLayer c, int row, boolean v) {
        int hd = v ? c.headDimV() : c.headDimK();
        float[] out = new float[c.nHeadKv() * hd];
        for (int h = 0; h < c.nHeadKv(); h++) {
            if (v) {
                c.readV(row, h, out, h * hd);
            } else {
                c.readK(row, h, out, h * hd);
            }
        }
        return out;
    }

}
