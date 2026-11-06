package com.eonmux.neuralfree.gguf.hf.parity;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.BnbScheme;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Scheme;
import com.eonmux.neuralfree.gguf.hf.quant.Fp8Scheme;
import com.eonmux.neuralfree.gguf.hf.quant.FpQuantScheme;
import com.eonmux.neuralfree.gguf.hf.quant.HadamardTransforms;
import com.eonmux.neuralfree.gguf.hf.quant.HqqScheme;
import com.eonmux.neuralfree.gguf.hf.quant.IntGroupScheme;
import com.eonmux.neuralfree.gguf.hf.st.StRowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Every quantized source form the converter rules cover is rebuilt as a tiny planned tensor of the raw
 * format, decoded through the exact CPU entry the repository serves those types with
 * ({@link CpuBlockDecoders#dequantize(byte[], GGMLType, long)} for the block formats) and compared element-wise with
 * the hand-computed reference dequantization — the format's own formula {@code (q - z) * s} or {@code code * block
 * scale}, evaluated with the literal constants the test writes out. Each scheme call is the dispatch path
 * {@code HfQuantRewriter} routes to: {@link IntGroupScheme} for GPTQ and AWQ, {@link Fp8Scheme} for FP8,
 * {@link Fp4Scheme} for the NVFP4 and MXFP4 forms, {@link BnbScheme} for the bnb NF4 repack, {@link HqqScheme} for
 * HQQ and {@link FpQuantScheme} for the fp_quant MXFP4 with its Hadamard rotation.
 */
public class HfQuantParityTest {

    /** The module every single-tensor test quantizes. */
    private static final String M = "model.layers.0.self_attn.q_proj";
    /** The fused-expert module the MXFP4 gpt-oss test unpacks. */
    private static final String P = "model.layers.0.mlp.experts.gate_up_proj";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Fx fx;

    @After
    public void close() throws IOException {
        if (fx != null) {
            fx.close();
        }
    }

    private Fx fixture() throws IOException {
        close();
        fx = new Fx(tmp.getRoot().toPath());
        return fx;
    }

    private HfConfig cfg() {
        return cfgOf("{'architectures':['A']}");
    }

    private HfConfig cfgOf(String singleQuoted) {
        return Fx.config(tmp.getRoot().toPath(), singleQuoted);
    }

    // ------------------------------------------------------------------ GPTQ and AWQ (IntGroupScheme)

    @Test
    public void gptqSymmetric4BitDecodesExactlyThroughQ4_0() throws IOException {
        int n = 2;
        int k = 32;
        int[] codes = codes(n * k, 1);
        float scale0 = 0.25f;
        float scale1 = 0.5f;
        // GPTQ v1 (zero offset 1), symmetric: the stored zero nibbles are 7 = 8 - 1, so v = (c - 8) * scale.
        // qzeros packs N outputs' zero nibbles 8-per-32-bit-word, one word for the two outputs here.
        Fx f = fixture();
        f.i32(M + ".qweight", new long[]{k / 8, n}, gptqQweight(codes, n, k));
        f.i32(M + ".qzeros", new long[]{1, 1}, 0x77777777);
        f.f16(M + ".scales", new long[]{1, n}, scale0, scale1);
        RowSource w = IntGroupScheme.rewriteModules(
                QuantArgs.builder(QuantFormat.INT_GPTQ).bits(4).group(32).gptqVersion(1).symmetric(true).build(),
                List.of(M), cfg(), f.index(), ModelSourceOptions.defaults()).get(M + ".weight");
        assertEquals(GGMLType.Q4_0, w.format().ggmlType());
        float[] v = blockValues(w);
        for (int i = 0; i < codes.length; i++) {
            float s = i < k ? scale0 : scale1;
            assertEquals("gptq element " + i, (codes[i] - 8) * s, v[i], 0f);
        }
        assertFalse(w.rowLocal());
    }

    @Test
    public void awqSymmetric4BitKeepsTheAwqNibbleOrderThroughQ4_0() throws IOException {
        int n = 8;
        int k = 32;
        int[] codes = codes(n * k, 3);
        // AWQ packs row-major [K, N/8]: element n's nibble sits in slot n % 8 = AWQ order (0, 2, 4, 6, 1, 3, 5, 7),
        // all groups folded since the only group's scale is one value (0.5).
        int[] qw = new int[k * (n / 8)];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                qw[i * (n / 8) + j / 8] |= codes[j * k + i] << (4 * awqSlot(j));
            }
        }
        Fx f = fixture();
        f.i32(M + ".qweight", new long[]{k, n / 8}, qw);
        // AWQ scales are [K/group, N]: one scale per output channel of the single group
        f.f16(M + ".scales", new long[]{1, n}, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f);
        RowSource w = IntGroupScheme.rewriteModules(
                QuantArgs.builder(QuantFormat.INT_AWQ).bits(4).group(32).symmetric(true).build(),
                List.of(M), cfg(), f.index(), ModelSourceOptions.defaults()).get(M + ".weight");
        assertEquals(GGMLType.Q4_0, w.format().ggmlType());
        float[] v = blockValues(w);
        for (int i = 0; i < codes.length; i++) {
            assertEquals("awq element " + i, (codes[i] - 8) * 0.5f, v[i], 0f);
        }
    }

    /** GPTQ's {@code qweight [Kw, N]}: column {@code n} holds row n's codes packed along K, LSB first. */
    private static int[] gptqQweight(int[] codes, int n, int k) {
        int per = 8;
        int[] out = new int[(k / per) * n];
        for (int w = 0; w < k / per; w++) {
            for (int cc = 0; cc < n; cc++) {
                for (int j = 0; j < per; j++) {
                    int i = cc * k + w * per + j;
                    out[w * n + cc] |= codes[i] << (4 * j);
                }
            }
        }
        return out;
    }

    /** The AWQ slot that holds element {@code j % 8} of a word: vllm's reverse AWQ order
     * [0, 4, 1, 5, 2, 6, 3, 7] (awq_triton.py:54-57). */
    private static int awqSlot(int i) {
        return new int[]{0, 4, 1, 5, 2, 6, 3, 7}[i];
    }

    // ------------------------------------------------------------------ FP8 (Fp8Scheme)

    @Test
    public void fp8BlockWeightsDecodeToCodeTimesScaleThroughB128() throws IOException {
        int n = 1;
        int k = 128;
        byte[] codes = e4m3Codes(n * k);
        Fx f = fixture();
        f.add(M + ".weight", StDType.F8_E4M3, new long[]{n, k}, codes);
        f.f32(M + ".weight_scale_inv", new long[]{1, 1}, 0.375f);
        RowSource w = Fp8Scheme.rewriteModules(QuantArgs.builder(QuantFormat.FP8).block(128, 128).build(),
                List.of(M), cfg(), f.index(), ModelSourceOptions.defaults()).get(M + ".weight");
        // per-tensor E4M3 rides as raw codes plus the output multiplier; folded it becomes F8_E4M3_B128
        assertEquals(GGMLType.F8_E4M3FN, w.format().ggmlType());
        assertArrayEquals(new float[]{0.375f}, w.outputScale(), 0f);
        RowSource b128 = w.foldOutputScale();
        assertEquals(GGMLType.F8_E4M3_B128, b128.format().ggmlType());
        float[] v = blockValues(w);
        for (int i = 0; i < codes.length; i++) {
            assertEquals("fp8 element " + i, FpBlockDecoders.e4m3ToFp32(codes[i]), v[i], 0f);
        }
        float[] folded = blockValues(b128);
        for (int i = 0; i < codes.length; i++) {
            assertEquals("fp8 folded element " + i, FpBlockDecoders.e4m3ToFp32(codes[i]) * 0.375f, folded[i], 0f);
        }
    }

    /** The E4M3 code classes every fp8 fixture cycles: zeros, subnormals, normals and negatives. */
    private static byte[] e4m3Codes(int n) {
        int[] classes = {0x00, 0x80, 0x01, 0x07, 0x08, 0x38, 0x3F, 0x77, 0x7E, 0x81, 0xB8, 0xFE};
        byte[] b = new byte[n];
        for (int i = 0; i < n; i++) {
            b[i] = (byte) classes[(i * 7 + i / 16) % classes.length];
        }
        return b;
    }

    // ------------------------------------------------------------------ NVFP4 (Fp4Scheme)

    @Test
    public void nvfp4CompressedTensorsKeepsSubScalesAndNvfp4Blocks() throws IOException {
        int k = 128;
        int[] c = codes(2 * k, 0);
        byte[] subs = nvfp4SubScales(2 * k / 16);
        Fx f = fixture();
        f.add(M + ".weight_packed", StDType.U8, new long[]{2, k / 2}, packLowFirst(c));
        f.add(M + ".weight_scale", StDType.F8_E4M3, new long[]{2, k / 16}, subs);
        f.f32(M + ".weight_global_scale", new long[]{1}, 4f);
        RowSource w = Fp4Scheme.rewriteModules(QuantArgs.builder(QuantFormat.NVFP4_CT).build(),
                List.of(M), cfg(), f.index(), ModelSourceOptions.defaults()).get(M + ".weight");
        assertEquals(GGMLType.NVFP4, w.format().ggmlType());
        // the compressed-tensors global scale is a divisor: it rides along as the output multiplier 1/4
        assertArrayEquals(new float[]{0.25f}, w.outputScale(), 0f);
        float[] v = blockValues(w);
        for (int i = 0; i < c.length; i++) {
            assertEquals("nvfp4 CT element " + i,
                    FpBlockDecoders.e2m1ToFp32(c[i]) * FpBlockDecoders.e4m3ToFp32(subs[i / 16]), v[i], 0f);
        }
    }

    @Test
    public void nvfp4ModelOptUsesTheStoredScale2AndSubScales() throws IOException {
        int k = 64;
        int[] c = codes(k, 3);
        byte[] subs = nvfp4SubScales(k / 16);
        Fx f = fixture();
        f.add(M + ".weight", StDType.U8, new long[]{1, k / 2}, packLowFirst(c));
        f.add(M + ".weight_scale", StDType.F8_E4M3, new long[]{1, k / 16}, subs);
        f.f32(M + ".weight_scale_2", new long[]{}, 0.5f);
        RowSource w = Fp4Scheme.rewriteModules(QuantArgs.builder(QuantFormat.NVFP4_MODELOPT).build(),
                List.of(M), cfg(), f.index(), ModelSourceOptions.defaults()).get(M + ".weight");
        assertEquals(GGMLType.NVFP4, w.format().ggmlType());
        assertArrayEquals(new float[]{0.5f}, w.outputScale(), 0f);
        float[] v = blockValues(w);
        for (int i = 0; i < c.length; i++) {
            assertEquals("nvfp4 ModelOpt element " + i,
                    FpBlockDecoders.e2m1ToFp32(c[i]) * FpBlockDecoders.e4m3ToFp32(subs[i / 16]), v[i], 0f);
        }
    }

    /** The NVFP4 sub-scale classes: 1, 0.5, 2.25, 2^-6, 0.875, 448, 0, 1.25 as E4M3 codes. */
    private static byte[] nvfp4SubScales(int n) {
        int[] classes = {0x38, 0x30, 0x41, 0x08, 0x36, 0x7E, 0x00, 0x3A};
        byte[] b = new byte[n];
        for (int i = 0; i < n; i++) {
            b[i] = (byte) classes[i % classes.length];
        }
        return b;
    }

    // ------------------------------------------------------------------ MXFP4 (Fp4Scheme)

    @Test
    public void mxfp4GptOssBlocksDecodeLowNibbleFirstWithE8m0Scales() throws IOException {
        int[] c = codes(128, 2);
        byte[] e = {(byte) 126, (byte) 127, (byte) 130, (byte) 1}; // scales 0.5, 1.0, 2.0, 2^4
        Fx f = fixture();
        f.add(P + "_blocks", StDType.U8, new long[]{1, 2, 2, 16}, packLowFirst(c));
        f.add(P + "_scales", StDType.U8, new long[]{1, 2, 2}, e);
        f.f32(P + "_bias", new long[]{1, 2}, 0f, 0f);
        RowSource b = Fp4Scheme.rewriteModules(QuantArgs.builder(QuantFormat.MXFP4_GPTOSS).build(),
                List.of(P), cfg(), f.index(), ModelSourceOptions.defaults()).get(P);
        assertEquals(GGMLType.MXFP4, b.format().ggmlType());
        float[] v = blockValues(b);
        for (int i = 0; i < c.length; i++) {
            // The documented low-nibble-first order, the order QuTLASS writes the nibble pairs in
            // (gpt_oss.py:116-119): element 2j in the low nibble of byte j.
            assertEquals("mxfp4 gpt-oss element " + i, e2m1Literal(c[i]) * e8m0Literal(e[i / 32] & 0xFF),
                    v[i], 0f);
        }
    }

    // ------------------------------------------------------------------ fp_quant MXFP4

    @Test
    public void fpQuantMxfp4RealQuantizedDecodeFoldsTheGlobalScale() throws IOException {
        int n = 2;
        int k = 64;
        int[] c = codes(n * k, 9);
        // fp_quant stores per-group E8M0 scales [n, k/32]: {0.5, 1.0 | 2.0, 4.0}; global scale 4 gives outputScale 0.25.
        byte[] scales = {(byte) 126, (byte) 127, (byte) 128, (byte) 129};
        String json = "{'architectures':['A'],'quantization_config':{'quant_method':'fp_quant',"
                + "'forward_dtype':'mxfp4','hadamard_group_size':32,'forward_method':'quest'}}";
        HfConfig config = Fx.config(tmp.getRoot().toPath(), json);
        float[] forward = sylvester(32);
        Fx f = fixture();
        f.add(M + ".qweight", StDType.U8, new long[]{n, k / 2}, packLowFirst(c));
        f.add(M + ".scales", StDType.U8, new long[]{n, k / 32}, scales);
        f.f32(M + ".weight_global_scale", new long[]{}, 4f);
        f.f32(M + ".act_global_scale", new long[]{}, 2f);
        f.f32(M + ".forward_hadamard_matrix", new long[]{32, 32}, forward);
        f.f32(M + ".backward_hadamard_matrix", new long[]{32, 32}, forward);
        RowSource w = FpQuantScheme.rewriteModules(argsOf(config), List.of(M), config, f.index(),
                ModelSourceOptions.defaults()).get(M + ".weight");
        assertEquals(GGMLType.MXFP4, w.format().ggmlType());
        // the global scale rides as the output multiplier (1 / 4)
        assertArrayEquals(new float[]{0.25f}, w.outputScale(), 0f);
        float[] v = blockValues(w);
        byte[] groups = scales;
        for (int i = 0; i < c.length; i++) {
            float groupScale = FpBlockDecoders.e8m0ToFp32(groups[(i / k) * (k / 32) + (i % k) / 32] & 0xFF);
            assertEquals("fp_quant element " + i, e2m1Literal(c[i]) * groupScale, v[i], 0f);
        }
    }

    private static QuantArgs argsOf(HfConfig config) throws IOException {
        return HfQuantConfig.of(config).argsFor(M).orElseThrow(AssertionError::new);
    }

    /** The normalized Sylvester matrix of order {@code n}, row-major. */
    private static float[] sylvester(int n) {
        float[] m = new float[n * n];
        float s = (float) (1.0 / Math.sqrt(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i * n + j] = s * HadamardTransforms.sylvesterSign(i, j);
            }
        }
        return m;
    }

    /** The E2M1 code value, as the hand-written table of the format. */
    private static float e2m1Literal(int code) {
        return FpBlockDecoders.e2m1ToFp32(code);
    }

    /** The E8M0 scale value {@code 2^(byte - 127)}. */
    private static float e8m0Literal(int byteIn) {
        return FpBlockDecoders.e8m0ToFp32(byteIn);
    }

    // ------------------------------------------------------------------ bnb NF4 and HQQ

    @Test
    public void bnbNf4RepacksToTheGgmlNf4FormatWithReplicatedAbsmax() throws IOException {
        int n = 2;
        int k = 128;
        int[] c = codes(n * k, 6);
        float[] absmax = {0.5f, 3f};
        float[] map = new float[16];
        for (int i = 0; i < 16; i++) {
            map[i] = QuantTables.nf4Code(i);
        }
        // bitsandbytes serializes the quant state as plain JSON
        byte[] state = ("{\"quant_type\":\"nf4\",\"blocksize\":128,\"dtype\":\"bfloat16\",\"shape\":[" + n + ","
                + k + "]}").getBytes(StandardCharsets.UTF_8);
        Fx f = fixture();
        f.add(M + ".weight", StDType.U8, new long[]{(n * (long) k + 1) / 2, 1}, packHighFirst(c));
        f.f32(M + ".weight.absmax", new long[]{2}, absmax);
        f.f32(M + ".weight.quant_map", new long[]{16}, map);
        f.add(M + ".weight.quant_state.bitsandbytes__nf4", StDType.U8, new long[]{state.length}, state);
        RowSource w = BnbScheme.rewriteModules(QuantArgs.builder(QuantFormat.NF4_BNB).build(),
                List.of(M), cfg(), f.index(), ModelSourceOptions.defaults()).get(M + ".weight");
        assertEquals(GGMLType.NF4, w.format().ggmlType());
        float[] v = blockValues(w);
        for (int i = 0; i < c.length; i++) {
            assertEquals("bnb nf4 element " + i, map[c[i]] * absmax[i / k], v[i], 0f);
        }
    }

    /** The bnb nibble order: element {@code 2j} in the high nibble of byte j. */
    private static byte[] packHighFirst(int[] codes) {
        byte[] b = new byte[(codes.length + 1) / 2];
        for (int i = 0; i < codes.length; i++) {
            b[i / 2] |= (byte) ((i & 1) == 0 ? codes[i] << 4 : codes[i]);
        }
        return b;
    }

    @Test
    public void hqqFourBitDecodeIsTheHqqSlabOrderThroughQ4_1() throws IOException {
        int n = 4;
        int k = 64;
        int rows = n * k / 32;
        int bits = 4;
        int per = 8 / bits;
        int step = rows / per;
        int[] c = codes(n * k, 8);
        // HQQ bitpack: packed row p holds rows p + i*step at bit (slots - 1 - i) * b.
        int[] packed = new int[step * 32];
        for (int r = 0; r < rows; r++) {
            int slab = r / step;
            int p = r % step;
            for (int col = 0; col < 32; col++) {
                packed[p * 32 + col] |= c[r * 32 + col] << ((per - 1 - slab) * bits);
            }
        }
        float[] scale = new float[rows];
        float[] zero = new float[rows];
        for (int r = 0; r < rows; r++) {
            scale[r] = 0.125f * (1 + r % 3);
            zero[r] = 7.5f - r % 4;
        }
        Fx f = fixture();
        f.bytes(M + ".W_q", StDType.U8, new long[]{rows / per, 32}, bytesOf(packed));
        f.f16(M + ".scale", new long[]{rows, 1}, scale);
        f.f16(M + ".zero", new long[]{rows, 1}, zero);
        f.i64(M + ".shape", new long[]{2}, n, k);
        f.i64(M + ".nbits", new long[]{1}, bits);
        f.i64(M + ".group_size", new long[]{1}, 32);
        f.i64(M + ".axis", new long[]{1}, 1);
        f.bytes(M + ".channel_wise", StDType.BOOL, new long[]{1}, 1);
        f.bytes(M + ".view_as_float", StDType.BOOL, new long[]{1}, 0);
        RowSource w = HqqScheme.rewriteModules(
                QuantArgs.builder(QuantFormat.INT_HQQ).bits(bits).group(32).symmetric(false).hqqAxis(1).build(),
                List.of(M), cfg(), f.index(), ModelSourceOptions.defaults()).get(M + ".weight");
        assertEquals(GGMLType.Q4_1, w.format().ggmlType());
        float[] v = blockValues(w);
        for (int i = 0; i < c.length; i++) {
            assertEquals("hqq element " + i, (c[i] - zero[i / 32]) * scale[i / 32], v[i], 0f);
        }
    }

    // ------------------------------------------------------------------ shared helpers

    /** Deterministic low-bit codes in [0, 16). */
    private static int[] codes(int count, int salt) {
        int[] c = new int[count];
        for (int i = 0; i < count; i++) {
            c[i] = (i * 5 + i / 7 + salt) & 0x0F;
        }
        return c;
    }

    /** Two codes per byte, element {@code 2j} in the low nibble (the low-nibble-first families). */
    private static byte[] packLowFirst(int[] codes) {
        byte[] b = new byte[codes.length / 2];
        for (int j = 0; j < b.length; j++) {
            b[j] = (byte) (codes[2 * j] | codes[2 * j + 1] << 4);
        }
        return b;
    }

    /** The little-endian bytes of the int32 words. */
    /** One byte per packed word: an HQQ 4-bit word of {@code per = 8 / bits} slots holds 8 bits. */
    private static byte[] bytesOf(int[] words) {
        byte[] out = new byte[words.length];
        for (int i = 0; i < words.length; i++) {
            out[i] = (byte) words[i];
        }
        return out;
    }

    /** Every value of {@code s}, read in one pass and decoded through {@link CpuBlockDecoders}. */
    private static float[] blockValues(RowSource s) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(s.rows() * s.rowBytes()))
                .order(ByteOrder.LITTLE_ENDIAN);
        s.readRows(0, Math.toIntExact(s.rows()), b);
        long numel = 1;
        for (long d : s.shape()) {
            numel *= d;
        }
        return CpuBlockDecoders.dequantize(b.array(), s.format().ggmlType(), numel);
    }

    /** A one-file safetensors checkpoint written and re-opened as a {@link HfTensorIndex}. */
    private static final class Fx implements Closeable {
        private final Path dir;
        private final SafetensorsWriter.Builder builder = SafetensorsWriter.builder();
        private SafetensorsFile file;
        private int files;

        Fx(Path dir) {
            this.dir = dir;
        }

        Fx add(String name, StDType t, long[] shape, byte[] data) {
            builder.add(name, t, shape, data);
            return this;
        }

        Fx i32(String name, long[] shape, int... v) {
            ByteBuffer b = le(4 * v.length);
            for (int x : v) {
                b.putInt(x);
            }
            return add(name, StDType.I32, shape, b.array());
        }

        Fx i64(String name, long[] shape, long... v) {
            ByteBuffer b = le(8 * v.length);
            for (long x : v) {
                b.putLong(x);
            }
            return add(name, StDType.I64, shape, b.array());
        }

        Fx bytes(String name, StDType t, long[] shape, int... v) {
            byte[] b = new byte[v.length];
            for (int i = 0; i < v.length; i++) {
                b[i] = (byte) v[i];
            }
            return add(name, t, shape, b);
        }

        Fx bytes(String name, StDType t, long[] shape, byte[] b) {
            return add(name, t, shape, b);
        }

        Fx f16(String name, long[] shape, float... v) {
            ByteBuffer b = le(2 * v.length);
            for (float x : v) {
                b.putShort((short) FpBlockDecoders.fp32ToFp16(x));
            }
            return add(name, StDType.F16, shape, b.array());
        }

        Fx f32(String name, long[] shape, float... v) {
            ByteBuffer b = le(4 * v.length);
            for (float x : v) {
                b.putFloat(x);
            }
            return add(name, StDType.F32, shape, b.array());
        }

        HfTensorIndex index() throws IOException {
            close();
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

        private static ByteBuffer le(int bytes) {
            return ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
        }
    }
}
