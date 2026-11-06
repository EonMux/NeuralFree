package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.hf.config.HfTransformConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.Fp8SchemeTest.Fixture;
import com.eonmux.neuralfree.gguf.hf.quant.HadamardTransforms.HadamardPlan;
import com.eonmux.neuralfree.gguf.hf.quant.HadamardTransforms.Result;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HadamardTransforms}: the fp_quant plan, compressed-tensors {@code hadamard} (identity signs) and
 * {@code random-hadamard} (explicit signs recovered from the stored matrix and tiled per {@code head_dim}), the offline
 * locations that are consumed, and every rejected location, type and option.
 */
public class HadamardTransformsTest {

    private static final String Q = "model.layers.0.self_attn.q_proj";
    private static final String O = "model.layers.0.self_attn.o_proj";
    private static final String D = "model.layers.0.mlp.down_proj";

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

    private static HfTransformConfig transforms(String singleQuoted) {
        return HfTransformConfig.parse(Json.parse(singleQuoted.replace('\'', '"')), "transform_config");
    }

    private static String scheme(String name, String type, Integer headDim, String applies) {
        return "{'config_groups':{'" + name + "':{'type':'" + type + "'"
                + (headDim == null ? "" : ",'head_dim':" + headDim) + ",'apply':[" + applies + "]}}}";
    }

    private static String apply(String location, String target) {
        return "{'targets':['" + target + "'],'location':'" + location + "'}";
    }

    /** The unnormalized Sylvester matrix of order {@code n} with row {@code i} multiplied by {@code d[i]}. */
    private static float[] signedSylvester(float[] d) {
        int n = d.length;
        float[] m = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i * n + j] = d[i] * HadamardTransforms.sylvesterSign(i, j);
            }
        }
        return m;
    }

    private static float[] ones(int n) {
        float[] d = new float[n];
        Arrays.fill(d, 1f);
        return d;
    }

    private static float[] signs(int n, int salt) {
        float[] d = new float[n];
        for (int i = 0; i < n; i++) {
            d[i] = ((i * 3 + salt) % 5) < 2 ? -1f : 1f;
        }
        return d;
    }

    private Fixture weight(Fixture f, String module, int n, int k) {
        return f.bf16(module + ".weight", new long[]{n, k}, new float[n * k]);
    }

    @Test
    public void fpQuantPlanCoversEveryModuleWeight() {
        QuantArgs a = QuantArgs.builder(QuantFormat.FPQUANT_NVFP4).fpQuantHadamardGroup(16)
                .fpQuantTransform(QuantArgs.FpQuantTransform.HADAMARD).build();
        HadamardPlan p = HadamardTransforms.fpQuant(a, List.of(Q, D)).orElseThrow(AssertionError::new);
        assertEquals(16, p.blockSize());
        assertEquals("identity", p.signMode());
        assertEquals(List.of(Q + ".weight", D + ".weight"), p.weightNames());
        assertEquals(0, p.signWidths().length);
        HadamardPlan r = p.renamed(n -> n.replace("model.layers.", "blk."));
        assertEquals("blk.0.self_attn.q_proj.weight", r.weightNames().get(0));
        assertThrows(GgufFormatException.class, () -> p.renamed(n -> "same"));
        QuantArgs gsr = a.toBuilder().fpQuantTransform(QuantArgs.FpQuantTransform.GSR).build();
        assertThrows(GgufFormatException.class, () -> HadamardTransforms.fpQuant(gsr, List.of(Q)));
    }

    @Test
    public void deterministicHadamardGivesIdentitySignsAndConsumesTheMatrices() throws IOException {
        Fixture f = fixture();
        weight(f, Q, 2, 64);
        weight(f, D, 2, 64);
        f.f32(Q + ".u_input.weight", new long[]{64, 64}, signedSylvester(ones(64)));
        f.f32(D + ".u_input.weight", new long[]{64, 64}, signedSylvester(ones(64)));
        HfTensorIndex in = f.index();
        Result r = HadamardTransforms.compressedTensors(transforms(scheme("u", "hadamard", null,
                apply("input", "Linear"))), in);
        HadamardPlan p = r.plan().orElseThrow(AssertionError::new);
        assertEquals(64, p.blockSize());
        assertEquals("identity", p.signMode());
        assertEquals(Set.of(Q + ".weight", D + ".weight"), new HashSet<>(p.weightNames()));
        assertEquals(2, p.weightNames().size());
        assertFalse(r.index().has(Q + ".u_input.weight"));
        assertFalse(r.index().has(D + ".u_input.weight"));
        assertSame(in.get(Q + ".weight"), r.index().get(Q + ".weight"));
    }

    @Test
    public void randomHadamardSignsAreRecoveredAndTiledPerHeadDim() throws IOException {
        float[] dq = signs(32, 1);
        float[] dd = signs(32, 2);
        Fixture f = fixture();
        weight(f, Q, 2, 64);
        weight(f, D, 2, 96);
        f.f32(Q + ".v_input.weight", new long[]{32, 32}, signedSylvester(dq));
        f.f32(D + ".v_input.weight", new long[]{32, 32}, signedSylvester(dd));
        f.add(D + ".v_input.perm", StDType.I64, new long[]{32}, new byte[8 * 32]);
        HfTensorIndex in = f.index();
        Result r = HadamardTransforms.compressedTensors(transforms(scheme("v", "random-hadamard", 32,
                apply("input", "Linear"))), in);
        HadamardPlan p = r.plan().orElseThrow(AssertionError::new);
        assertEquals(32, p.blockSize());
        assertEquals("explicit", p.signMode());
        assertArrayEquals(new int[]{64, 96}, p.signWidths());
        float[] s64 = p.signs(64);
        float[] s96 = p.signs(96);
        for (int i = 0; i < 64; i++) {
            assertEquals(dq[i % 32], s64[i], 0f);
        }
        for (int i = 0; i < 96; i++) {
            assertEquals(dd[i % 32], s96[i], 0f);
        }
        assertEquals(160, p.signValues().length);
        assertNull(p.signs(128));
        assertFalse(r.index().has(D + ".v_input.perm"));
    }

    @Test
    public void aMatrixThatIsNotASignedSylvesterMatrixIsNamed() throws IOException {
        float[] m = signedSylvester(signs(32, 0));
        m[5 * 32 + 9] = -m[5 * 32 + 9];
        Fixture f = fixture();
        weight(f, Q, 1, 32);
        HfTensorIndex in = f.f32(Q + ".v_input.weight", new long[]{32, 32}, m).index();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HadamardTransforms.compressedTensors(
                transforms(scheme("v", "random-hadamard", null, apply("input", "Linear"))), in));
        assertTrue(e.getMessage(), e.getMessage().contains("entry (5, 9)"));
    }

    @Test
    public void offlineLocationsAreConsumedWithoutAPlan() throws IOException {
        Fixture f = fixture();
        weight(f, Q, 1, 32);
        HfTensorIndex in = f.f32(Q + ".w_weight_input.weight", new long[]{32, 32}, signedSylvester(ones(32)))
                .f32(Q + ".w_weight_output.weight", new long[]{32, 32}, signedSylvester(ones(32))).index();
        Result r = HadamardTransforms.compressedTensors(transforms(scheme("w", "random-matrix", null,
                apply("weight_input", "Linear") + "," + apply("weight_output", "Linear"))), in);
        assertFalse(r.plan().isPresent());
        assertEquals(List.of(Q + ".weight"), r.index().names());
    }

    private void assertRejected(String config, String part) throws IOException {
        Fixture f = fixture();
        weight(f, Q, 1, 32);
        HfTensorIndex in = f.f32(Q + ".t_input.weight", new long[]{32, 32}, signedSylvester(ones(32))).index();
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HadamardTransforms.compressedTensors(transforms(config), in));
        assertTrue(e.getMessage(), e.getMessage().contains(part));
    }

    @Test
    public void unsupportedLocationsTypesAndOptionsAreNamed() throws IOException {
        for (String loc : new String[]{"output", "k_cache", "q_attn"}) {
            assertRejected(scheme("t", "hadamard", null, apply(loc, "Linear")), "hadamard@" + loc
                    + " is not supported");
        }
        for (String type : new String[]{"matrix", "random-matrix"}) {
            assertRejected(scheme("t", type, null, apply("input", "Linear")), type + "@input is not supported");
        }
        assertRejected("{'config_groups':{'t':{'type':'random-hadamard','randomize':true,'apply':["
                + apply("input", "Linear") + "]}}}", "with randomize");
        assertRejected("{'config_groups':{'t':{'type':'random-hadamard','apply':[{'targets':['Linear'],"
                + "'location':'input','inverse':true}]}}}", "with inverse");
    }

    @Test
    public void twoBlockSizesAndConflictingSignsAreNamed() throws IOException {
        Fixture f = fixture();
        weight(f, Q, 1, 64);
        weight(f, D, 1, 32);
        HfTensorIndex in = f.f32(Q + ".t_input.weight", new long[]{64, 64}, signedSylvester(ones(64)))
                .f32(D + ".t_input.weight", new long[]{32, 32}, signedSylvester(ones(32))).index();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HadamardTransforms.compressedTensors(
                transforms(scheme("t", "hadamard", null, apply("input", "Linear"))), in));
        assertTrue(e.getMessage(), e.getMessage().contains("block sizes 64 and 32")
                || e.getMessage().contains("block sizes 32 and 64"));

        Fixture g = fixture();
        weight(g, Q, 1, 32);
        weight(g, O, 1, 32);
        HfTensorIndex in2 = g.f32(Q + ".t_input.weight", new long[]{32, 32}, signedSylvester(signs(32, 0)))
                .f32(O + ".t_input.weight", new long[]{32, 32}, signedSylvester(signs(32, 3))).index();
        GgufFormatException c = assertThrows(GgufFormatException.class, () -> HadamardTransforms.compressedTensors(
                transforms(scheme("t", "random-hadamard", null, apply("input", "Linear"))), in2));
        assertTrue(c.getMessage(), c.getMessage().contains("two different sign vectors"));
    }

    @Test
    public void blocksAbovePrismKernelLimitAreAcceptedUpToTheLoaderLimit() {
        HadamardPlan p = HadamardPlan.of(2048, Collections.emptyMap(), List.of(Q + ".weight"));
        assertEquals(2048, p.blockSize());
        assertEquals(1 << 30, HadamardPlan.of(HadamardTransforms.MAX_BLOCK, Collections.emptyMap(),
                List.of(Q + ".weight")).blockSize());
        QuantArgs a = QuantArgs.builder(QuantFormat.FPQUANT_MXFP4).fpQuantHadamardGroup(4096)
                .fpQuantTransform(QuantArgs.FpQuantTransform.HADAMARD).build();
        assertEquals(4096, HadamardTransforms.fpQuant(a, List.of(Q)).orElseThrow(AssertionError::new).blockSize());
        for (int bad : new int[]{0, -64, 48, 3}) {
            GgufFormatException e = assertThrows(GgufFormatException.class,
                    () -> HadamardPlan.of(bad, Collections.emptyMap(), List.of(Q + ".weight")));
            assertEquals("Hadamard block " + bad + " is not a power of two at most " + (1 << 30), e.getMessage());
        }
    }

    @Test
    public void weightNamesMustBeNonEmptyNonNullAndDistinct() {
        for (List<String> bad : List.of(Collections.<String>emptyList(), Arrays.asList(Q + ".weight", null),
                List.of(Q + ".weight", Q + ".weight"))) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> HadamardPlan.of(64, Collections.emptyMap(), bad));
            assertTrue(e.getMessage(), e.getMessage().startsWith("Hadamard weight names must be non-empty"));
        }
        assertEquals(List.of(Q + ".weight", D + ".weight"),
                HadamardPlan.of(64, Collections.emptyMap(), List.of(Q + ".weight", D + ".weight")).weightNames());
    }

    @Test
    public void anInputTransformWithoutItsWeightIsNamed() throws IOException {
        HfTensorIndex in = fixture().f32(Q + ".t_input.weight", new long[]{32, 32}, signedSylvester(ones(32)))
                .index();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HadamardTransforms.compressedTensors(
                transforms(scheme("t", "hadamard", null, apply("input", "Linear"))), in));
        assertTrue(e.getMessage(), e.getMessage().contains("has no weight " + Q + ".weight"));
    }
}
