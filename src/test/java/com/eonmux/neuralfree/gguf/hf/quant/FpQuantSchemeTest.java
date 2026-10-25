package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.Fp8SchemeTest.Fixture;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link FpQuantScheme}: real MXFP4 and NVFP4 weights with the output multiplier {@code 1 / weight_global_scale},
 * pseudo-quantized {@code dqweight} passthrough, validation of the stored rotation against {@code transform_init},
 * and the {@code gsr} rejection. Configs go through {@link HfQuantConfig} so the arguments match a real checkpoint's.
 */
public class FpQuantSchemeTest {

    private static final String M = "model.layers.0.mlp.up_proj";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Fixture fx;

    @After
    public void close() throws IOException {
        if (fx != null) {
            fx.close();
        }
    }

    private Fixture fixture() throws IOException {
        close();
        fx = new Fixture(tmp.getRoot().toPath());
        return fx;
    }

    private HfConfig config(String quant) {
        return Fixture.config(tmp.getRoot().toPath(), "{'architectures':['A'],'quantization_config':"
                + "{'quant_method':'fp_quant'," + quant + "}}");
    }

    private static QuantArgs args(HfConfig cfg) throws IOException {
        return HfQuantConfig.of(cfg).argsFor(M).orElseThrow(AssertionError::new);
    }

    /** The normalized Sylvester matrix of order {@code n}, row-major. */
    static float[] hadamard(int n) {
        float[] m = new float[n * n];
        float s = (float) (1.0 / Math.sqrt(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i * n + j] = s * HadamardTransforms.sylvesterSign(i, j);
            }
        }
        return m;
    }

    static float[] identity(int n) {
        float[] m = new float[n * n];
        for (int i = 0; i < n; i++) {
            m[i * n + i] = 1f;
        }
        return m;
    }

    private Fixture real(int n, int k, StDType scaleType, byte[] scales, int group, float[] forward)
            throws IOException {
        int hg = (int) Math.round(Math.sqrt(forward.length));
        return fixture().add(M + ".qweight", StDType.U8, new long[]{n, k / 2},
                Fp4SchemeTest.packLowFirst(Fp4SchemeTest.codes(n * k, 4)))
                .add(M + ".scales", scaleType, new long[]{n, k / group}, scales)
                .f32(M + ".weight_global_scale", new long[]{}, 4f)
                .f32(M + ".act_global_scale", new long[]{}, 2f)
                .f32(M + ".forward_hadamard_matrix", new long[]{hg, hg}, forward)
                .f32(M + ".backward_hadamard_matrix", new long[]{hg, hg}, forward);
    }

    @Test
    public void realMxfp4KeepsBlocksAndScalesOutputByTheInverseWeightGlobalScale() throws IOException {
        HfConfig cfg = config("'forward_dtype':'mxfp4','hadamard_group_size':32,'forward_method':'quest'");
        QuantArgs a = args(cfg);
        assertEquals(QuantFormat.FPQUANT_MXFP4, a.format());
        int n = 2;
        int k = 64;
        byte[] e = {(byte) 126, (byte) 127, (byte) 129, (byte) 100};
        HfTensorIndex in = real(n, k, StDType.U8, e, 32, hadamard(32)).index();
        HfTensorIndex out = FpQuantScheme.rewriteModules(a, List.of(M), cfg, in, ModelSourceOptions.defaults());
        for (String gone : new String[]{".qweight", ".scales", ".weight_global_scale", ".act_global_scale",
            ".forward_hadamard_matrix", ".backward_hadamard_matrix"}) {
            assertFalse(gone, out.has(M + gone));
        }
        RowSource w = out.get(M + ".weight");
        assertEquals(GGMLType.MXFP4, w.format().ggmlType());
        assertArrayEquals(new long[]{n, k}, w.shape());
        assertArrayEquals(new float[]{0.25f}, w.outputScale(), 0f);
        assertNull(w.actQuant());
        int[] c = Fp4SchemeTest.codes(n * k, 4);
        float[] v = Fixture.values(w);
        for (int i = 0; i < c.length; i++) {
            assertEquals(FpBlockDecoders.e2m1ToFp32(c[i]) * FpBlockDecoders.e8m0ToFp32(e[i / 32] & 0xFF), v[i], 0f);
        }
        assertEquals(32, HadamardTransforms.fpQuant(a, List.of(M)).orElseThrow(AssertionError::new).blockSize());
    }

    @Test
    public void realNvfp4ScalesOutputByTheInverseWeightGlobalScale() throws IOException {
        HfConfig cfg = config("'forward_dtype':'nvfp4','hadamard_group_size':16");
        QuantArgs a = args(cfg);
        assertEquals(QuantFormat.FPQUANT_NVFP4, a.format());
        byte[] s = {0x38, 0x30, 0x41, 0x08};
        HfTensorIndex in = real(1, 64, StDType.F8_E4M3, s, 16, hadamard(16)).index();
        RowSource w = FpQuantScheme.rewriteModules(a, List.of(M), cfg, in, ModelSourceOptions.defaults())
                .get(M + ".weight");
        assertEquals(GGMLType.NVFP4, w.format().ggmlType());
        assertArrayEquals(new float[]{0.25f}, w.outputScale(), 0f);
        int[] c = Fp4SchemeTest.codes(64, 4);
        float[] v = Fixture.values(w.foldOutputScale());
        for (int i = 0; i < c.length; i++) {
            assertEquals(FpBlockDecoders.e2m1ToFp32(c[i]) * FpBlockDecoders.e4m3ToFp32(s[i / 16]) * 0.25f, v[i],
                    0f);
        }
    }

    @Test
    public void pseudoQuantizedWeightsLoadAsStored() throws IOException {
        HfConfig cfg = config("'forward_dtype':'mxfp4','hadamard_group_size':32,'pseudoquantization':true");
        QuantArgs a = args(cfg);
        assertEquals(QuantFormat.FPQUANT_PSEUDO, a.format());
        HfTensorIndex in = fixture().bf16(M + ".dqweight", new long[]{1, 32}, new float[32])
                .f32(M + ".forward_hadamard_matrix", new long[]{32, 32}, hadamard(32)).index();
        HfTensorIndex out = FpQuantScheme.rewriteModules(a, List.of(M), cfg, in, ModelSourceOptions.defaults());
        assertSame(in.get(M + ".dqweight"), out.get(M + ".weight"));
        assertFalse(out.has(M + ".dqweight"));
        assertFalse(out.has(M + ".forward_hadamard_matrix"));
    }

    @Test
    public void aStoredRotationOtherThanTheConfiguredOneIsNamed() throws IOException {
        HfConfig cfg = config("'forward_dtype':'mxfp4','hadamard_group_size':32");
        float[] h = hadamard(32);
        h[3 * 32 + 5] += 0.01f;
        HfTensorIndex in = real(1, 64, StDType.U8, new byte[]{127, 127}, 32, h).index();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> FpQuantScheme.rewriteModules(args(cfg),
                List.of(M), cfg, in, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains("entry (3, 5)"));
        HfTensorIndex wrongSize = real(1, 64, StDType.U8, new byte[]{127, 127}, 32, hadamard(64)).index();
        GgufFormatException s = assertThrows(GgufFormatException.class, () -> FpQuantScheme.rewriteModules(
                args(cfg), List.of(M), cfg, wrongSize, ModelSourceOptions.defaults()));
        assertTrue(s.getMessage(), s.getMessage().contains("expected [32, 32]"));
    }

    @Test
    public void identityTransformNeedsTheIdentityMatrixAndNoPlan() throws IOException {
        HfConfig cfg = config("'forward_dtype':'mxfp4','hadamard_group_size':32,'transform_init':'identity'");
        QuantArgs a = args(cfg);
        HfTensorIndex in = real(1, 64, StDType.U8, new byte[]{127, 127}, 32, identity(32)).index();
        RowSource w = FpQuantScheme.rewriteModules(a, List.of(M), cfg, in, ModelSourceOptions.defaults())
                .get(M + ".weight");
        assertEquals(GGMLType.MXFP4, w.format().ggmlType());
        assertFalse(HadamardTransforms.fpQuant(a, List.of(M)).isPresent());
        HfTensorIndex rotated = real(1, 64, StDType.U8, new byte[]{127, 127}, 32, hadamard(32)).index();
        assertThrows(GgufFormatException.class, () -> FpQuantScheme.rewriteModules(a, List.of(M), cfg, rotated,
                ModelSourceOptions.defaults()));
    }

    @Test
    public void gsrIsRejected() throws IOException {
        HfConfig cfg = config("'forward_dtype':'mxfp4','hadamard_group_size':32,'transform_init':'gsr'");
        HfTensorIndex in = real(1, 64, StDType.U8, new byte[]{127, 127}, 32, hadamard(32)).index();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> FpQuantScheme.rewriteModules(args(cfg),
                List.of(M), cfg, in, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains("'gsr'"));
    }

    @Test
    public void missingGlobalScaleIsNamed() throws IOException {
        HfConfig cfg = config("'forward_dtype':'mxfp4','hadamard_group_size':32");
        HfTensorIndex in = fixture().add(M + ".qweight", StDType.U8, new long[]{1, 32}, new byte[32])
                .add(M + ".scales", StDType.U8, new long[]{1, 2}, new byte[]{127, 127})
                .f32(M + ".weight_global_scale", new long[]{}, 1f)
                .f32(M + ".forward_hadamard_matrix", new long[]{32, 32}, hadamard(32)).index();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> FpQuantScheme.rewriteModules(args(cfg),
                List.of(M), cfg, in, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains("act_global_scale"));
    }
}
