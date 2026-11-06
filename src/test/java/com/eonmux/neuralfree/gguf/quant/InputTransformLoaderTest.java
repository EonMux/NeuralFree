package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link InputTransformLoader}: the PRISM-only path, the {@code .input_perm} tensors, the activation-quantization
 * carrier with its static siblings and {@code scale_ub} under {@link ActivationQuant#CHECKPOINT} only, and the
 * validation of every input; plus the reading of the siblings from a GGUF file.
 */
public class InputTransformLoaderTest {

    private static final String Q = "blk.0.attn_q.weight";
    private static final String DOWN = "blk.0.ffn_down.weight";
    private static final String EXPS = "blk.0.ffn_up_exps.weight";

    /** A model's tensors: shapes, I32 and F32 values by name. */
    private static final class Model {
        final Map<String, Object> kv = new HashMap<>();
        final Map<String, long[]> dims = new HashMap<>();
        final Map<String, int[]> ints = new HashMap<>();
        final Map<String, float[]> floats = new HashMap<>();

        Model() {
            kv.put("general.architecture", "llama");
            dims.put("token_embd.weight", new long[]{128, 1000});
            dims.put("output.weight", new long[]{128, 1000});
            dims.put(Q, new long[]{128, 128});
            dims.put(DOWN, new long[]{256, 128});
            dims.put(EXPS, new long[]{128, 64, 4});
        }

        Model ints(String name, int... v) {
            dims.put(name, new long[]{v.length});
            ints.put(name, v);
            return this;
        }

        Model floats(String name, float... v) {
            dims.put(name, new long[]{v.length});
            floats.put(name, v);
            return this;
        }

        Model prism() {
            kv.put("prism.hadamard.version", 1);
            kv.put("prism.hadamard.block_size", 64);
            kv.put("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
            kv.put("prism.hadamard.axis", "input-last-dimension");
            kv.put("prism.hadamard.sign_mode", "identity");
            kv.put("prism.hadamard.weight_names", new String[]{Q, DOWN});
            return this;
        }

        Model carrier(String[] names, ActQuantScheme... schemes) {
            int[] codes = new int[schemes.length];
            for (int i = 0; i < codes.length; i++) {
                codes[i] = schemes[i].code();
            }
            kv.put(GgufKeys.NF_ACT_QUANT_VERSION, 1);
            kv.put(GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES, names);
            kv.put(GgufKeys.NF_ACT_QUANT_SCHEMES, codes);
            return this;
        }

        InputTransformLoader.Result load(ActivationQuant mode) throws IOException {
            return InputTransformLoader.load(GgufKv.ofMap(kv), dims.keySet(), dims::get, 0, 0, mode,
                    ints::get, floats::get);
        }

        void expectFormat(ActivationQuant mode, String fragment) {
            GgufFormatException e = assertThrows(GgufFormatException.class, () -> load(mode));
            assertTrue(e.getMessage() + " should mention " + fragment, e.getMessage().contains(fragment));
        }
    }

    private static int[] reversed(int n) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = n - 1 - i;
        }
        return p;
    }

    @Test
    public void noTransformsGiveAnEmptyResult() throws IOException {
        InputTransformLoader.Result r = new Model().load(ActivationQuant.CHECKPOINT);
        assertTrue(r.isEmpty());
        assertTrue(r.hadamardOnly());
        assertFalse(r.tiedOutput());
        assertSame(InputTransform.IDENTITY, r.forward(Q));
        assertNull(r.hadamard(Q));
    }

    @Test
    public void prismOnlyPathMatchesThePrismLoader() throws IOException {
        Model m = new Model().prism();
        PrismHadamardLoader.Result prism = PrismHadamardLoader.load(GgufKv.ofMap(m.kv), m.dims::get, 0, 0);
        for (ActivationQuant mode : ActivationQuant.values()) {
            InputTransformLoader.Result r = m.load(mode);
            assertFalse(r.isEmpty());
            assertTrue(r.hadamardOnly());
            assertEquals(prism.forward().keySet(), r.forward().keySet());
            for (Map.Entry<String, HadamardSpec> e : prism.forward().entrySet()) {
                InputTransform t = r.forward(e.getKey());
                assertEquals(e.getValue(), t.hadamard());
                assertEquals(e.getValue(), r.hadamard(e.getKey()));
                assertNull(t.actQuant());
                assertFalse(t.hasPerm());
            }
            assertEquals(prism.inverse(), r.inverse());
            assertEquals(prism.tiedOutput(), r.tiedOutput());
        }
    }

    @Test
    public void bindsTheActivationPreScaleInEveryMode() throws IOException {
        float[] pre = new float[128];
        for (int i = 0; i < pre.length; i++) {
            pre[i] = i % 2 == 0 ? 0.5f : 2f;
        }
        Model m = new Model().floats("blk.0.attn_q.input_pre_scale", pre);
        for (ActivationQuant mode : ActivationQuant.values()) {
            InputTransformLoader.Result r = m.load(mode);
            InputTransform t = r.forward(Q);
            assertArrayEquals(pre, t.preScale(), 0f);
            assertTrue(t.hasPreScale());
            assertFalse(t.isIdentity());
            assertFalse(r.hadamardOnly());
        }
        assertTrue("an unrelated weight keeps the identity", r(m).forward(DOWN).isIdentity());
    }

    /** The pre-scale may hold one value per expert of a stacked tensor, in expert-major order. */
    @Test
    public void bindsAPerExpertActivationPreScale() throws IOException {
        float[] pre = new float[4 * 128];
        for (int i = 0; i < pre.length; i++) {
            pre[i] = 1f + i % 3;
        }
        Model m = new Model().floats("blk.0.ffn_up_exps.input_pre_scale", pre);
        InputTransform t = m.load(ActivationQuant.WEIGHT_ONLY).forward(EXPS);
        assertArrayEquals(pre, t.preScale(), 0f);
        assertEquals(4, t.preScaleSets(128));
    }

    @Test
    public void rejectsABadActivationPreScale() {
        new Model().floats("blk.0.attn_q.input_pre_scale", new float[64])
                .expectFormat(ActivationQuant.WEIGHT_ONLY, "blk.0.attn_q.input_pre_scale");
        new Model().floats("blk.0.ffn_up_exps.input_pre_scale", new float[100])
                .expectFormat(ActivationQuant.WEIGHT_ONLY, "blk.0.ffn_up_exps.input_pre_scale");
        Model shortRead = new Model().floats("blk.0.attn_q.input_pre_scale", new float[128]);
        shortRead.floats.put("blk.0.attn_q.input_pre_scale", new float[127]);
        shortRead.expectFormat(ActivationQuant.WEIGHT_ONLY, "blk.0.attn_q.input_pre_scale");
        new Model().floats("blk.0.attn_k.input_pre_scale", new float[128])
                .expectFormat(ActivationQuant.WEIGHT_ONLY, "blk.0.attn_k.weight");
    }

    private static InputTransformLoader.Result r(Model m) throws IOException {
        return m.load(ActivationQuant.WEIGHT_ONLY);
    }

    @Test
    public void bindsTheInputPermutationInEveryMode() throws IOException {
        int[] perm = reversed(128);
        Model m = new Model().prism().ints("blk.0.attn_q.input_perm", perm);
        for (ActivationQuant mode : ActivationQuant.values()) {
            InputTransformLoader.Result r = m.load(mode);
            InputTransform t = r.forward(Q);
            assertArrayEquals(perm, t.perm());
            assertTrue("the rotation stays", t.hadamard() != null);
            assertFalse(r.hadamardOnly());
            assertTrue(r.forward(DOWN).isHadamardOnly());
        }
    }

    @Test
    public void rejectsABadInputPermutation() {
        new Model().ints("blk.0.attn_q.input_perm", reversed(64))
                .expectFormat(ActivationQuant.WEIGHT_ONLY, "blk.0.attn_q.input_perm");
        int[] duplicate = reversed(128);
        duplicate[5] = duplicate[6];
        new Model().ints("blk.0.attn_q.input_perm", duplicate)
                .expectFormat(ActivationQuant.WEIGHT_ONLY, "blk.0.attn_q.input_perm");
        int[] outOfRange = reversed(128);
        outOfRange[0] = 128;
        new Model().ints("blk.0.attn_q.input_perm", outOfRange)
                .expectFormat(ActivationQuant.WEIGHT_ONLY, "blk.0.attn_q.input_perm");
        new Model().ints("blk.0.attn_k.input_perm", reversed(128))
                .expectFormat(ActivationQuant.WEIGHT_ONLY, "blk.0.attn_k.weight");
        Model shortRead = new Model().ints("blk.0.attn_q.input_perm", reversed(128));
        shortRead.ints.put("blk.0.attn_q.input_perm", reversed(127));
        shortRead.expectFormat(ActivationQuant.WEIGHT_ONLY, "blk.0.attn_q.input_perm");
    }

    @Test
    public void readsTheCarrierUnderCheckpointOnly() throws IOException {
        Model m = new Model().carrier(new String[]{Q}, ActQuantScheme.FP8_STATIC)
                .floats("blk.0.attn_q.input_scale", 0.5f);
        InputTransformLoader.Result weightOnly = m.load(ActivationQuant.WEIGHT_ONLY);
        assertTrue(weightOnly.isEmpty());
        assertSame(InputTransform.IDENTITY, weightOnly.forward(Q));
        InputTransformLoader.Result checkpoint = m.load(ActivationQuant.CHECKPOINT);
        InputTransform.ActQuant a = checkpoint.forward(Q).actQuant();
        assertEquals(ActQuantScheme.FP8_STATIC, a.scheme());
        assertEquals(0.5f, a.scale(0), 0f);
        assertFalse(checkpoint.hadamardOnly());
        assertTrue(checkpoint.inverse().isEmpty());
    }

    @Test
    public void weightOnlyIgnoresAMalformedCarrier() throws IOException {
        Model m = new Model();
        m.kv.put(GgufKeys.NF_ACT_QUANT_VERSION, 7);
        assertTrue(m.load(ActivationQuant.WEIGHT_ONLY).isEmpty());
        m.expectFormat(ActivationQuant.CHECKPOINT, GgufKeys.NF_ACT_QUANT_VERSION);
    }

    @Test
    public void rejectsAnUnknownCodeUnderCheckpoint() {
        Model m = new Model();
        m.kv.put(GgufKeys.NF_ACT_QUANT_VERSION, 1);
        m.kv.put(GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES, new String[]{Q});
        m.kv.put(GgufKeys.NF_ACT_QUANT_SCHEMES, new int[]{42});
        m.expectFormat(ActivationQuant.CHECKPOINT, Q);
    }

    @Test
    public void bindsTheStaticSiblingsPerExpert() throws IOException {
        Model m = new Model().carrier(new String[]{Q, EXPS}, ActQuantScheme.INT8_STATIC_ASYM, ActQuantScheme.NVFP4)
                .floats("blk.0.attn_q.input_scale", 0.25f)
                .ints("blk.0.attn_q.input_zero_point", -3)
                .floats("blk.0.ffn_up_exps.input_scale", 1f, 2f, 3f, 4f);
        InputTransformLoader.Result r = m.load(ActivationQuant.CHECKPOINT);
        InputTransform.ActQuant q = r.forward(Q).actQuant();
        assertEquals(0.25f, q.scale(0), 0f);
        assertEquals(-3, q.zeroPoint(0));
        assertEquals(1, q.entries());
        InputTransform.ActQuant e = r.forward(EXPS).actQuant();
        assertEquals(4, e.entries());
        for (int i = 0; i < 4; i++) {
            assertEquals(i + 1f, e.globalScale(i), 0f);
            assertEquals(0f, e.scale(i), 0f);
        }
        assertThrows(IllegalArgumentException.class, () -> e.globalScale(4));
    }

    @Test
    public void rejectsMissingOrMiscountedSiblings() {
        new Model().carrier(new String[]{Q}, ActQuantScheme.INT8_STATIC)
                .expectFormat(ActivationQuant.CHECKPOINT, "blk.0.attn_q.input_scale");
        new Model().carrier(new String[]{Q}, ActQuantScheme.INT8_STATIC_ASYM)
                .floats("blk.0.attn_q.input_scale", 0.25f)
                .expectFormat(ActivationQuant.CHECKPOINT, "blk.0.attn_q.input_zero_point");
        new Model().carrier(new String[]{EXPS}, ActQuantScheme.FP8_STATIC)
                .floats("blk.0.ffn_up_exps.input_scale", 1f, 2f, 3f)
                .expectFormat(ActivationQuant.CHECKPOINT, "blk.0.ffn_up_exps.input_scale");
        new Model().carrier(new String[]{Q}, ActQuantScheme.FP8_STATIC)
                .floats("blk.0.attn_q.input_scale", 0.5f, 0.5f)
                .expectFormat(ActivationQuant.CHECKPOINT, "blk.0.attn_q.input_scale");
        new Model().carrier(new String[]{Q}, ActQuantScheme.FP8_STATIC)
                .floats("blk.0.attn_q.input_scale", 0f)
                .expectFormat(ActivationQuant.CHECKPOINT, Q);
    }

    @Test
    public void rejectsACarrierEntryThatDoesNotFitItsWeight() {
        new Model().carrier(new String[]{"blk.0.attn_k.weight"}, ActQuantScheme.INT8_TOKEN)
                .expectFormat(ActivationQuant.CHECKPOINT, "blk.0.attn_k.weight");
        Model narrow = new Model().carrier(new String[]{Q}, ActQuantScheme.INT8_GROUP128);
        narrow.dims.put(Q, new long[]{64, 128});
        narrow.expectFormat(ActivationQuant.CHECKPOINT, Q);
    }

    @Test
    public void threadsTheScaleUpperBoundToFp8Token() throws IOException {
        Model m = new Model().carrier(new String[]{Q, DOWN}, ActQuantScheme.FP8_TOKEN, ActQuantScheme.INT8_TOKEN);
        InputTransformLoader.Result unbounded = m.load(ActivationQuant.CHECKPOINT);
        assertEquals(Float.POSITIVE_INFINITY, unbounded.forward(Q).actQuant().scale(0), 0f);
        m.kv.put(GgufKeys.NF_ACT_QUANT_SCALE_UB, 900f);
        InputTransformLoader.Result bounded = m.load(ActivationQuant.CHECKPOINT);
        assertEquals(900f, bounded.forward(Q).actQuant().scale(0), 0f);
        assertEquals(900f, bounded.forward(Q).actQuant().scaleUpperBound(), 0f);
        assertEquals(0f, bounded.forward(DOWN).actQuant().scale(0), 0f);
    }

    @Test
    public void combinesEveryPartOnOneWeight() throws IOException {
        int[] perm = reversed(128);
        InputTransformLoader.Result r = new Model().prism().ints("blk.0.attn_q.input_perm", perm)
                .carrier(new String[]{Q}, ActQuantScheme.MXFP4).load(ActivationQuant.CHECKPOINT);
        InputTransform t = r.forward(Q);
        assertTrue(t.hadamard() != null);
        assertEquals(ActQuantScheme.MXFP4, t.actQuant().scheme());
        assertArrayEquals(perm, t.perm());
    }

    // ------------------------------------------------------------------ GGUF file

    private static byte[] i32Bytes(int... v) {
        ByteBuffer b = ByteBuffer.allocate(v.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int x : v) {
            b.putInt(x);
        }
        return b.array();
    }

    private static GgufTestWriter fileModel() {
        return new GgufTestWriter()
                .architecture("llama")
                .tensorRandom("blk.0.attn_q.weight", GGMLType.F32, 5L, -1f, 1f, 8, 4)
                .tensor("blk.0.attn_q.input_perm", GGMLType.I32, new long[]{8}, i32Bytes(7, 6, 5, 4, 3, 2, 1, 0));
    }

    private interface SourceCheck {
        void run(ModelSource source, GgufKv kv, TensorCatalog catalog) throws IOException;
    }

    private static void withSource(GgufTestWriter w, SourceCheck check) throws IOException {
        Path path = w.writeTempFile();
        try (ModelSource source = ModelSources.open(path.toString(), ModelSourceOptions.defaults())) {
            check.run(source, GgufKv.of(source), TensorCatalog.of(source));
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void readsTheSiblingsOfAGgufFile() throws IOException {
        GgufTestWriter w = fileModel()
                .u32(GgufKeys.NF_ACT_QUANT_VERSION, 1)
                .strArray(GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES, "blk.0.attn_q.weight")
                .i32Array(GgufKeys.NF_ACT_QUANT_SCHEMES, ActQuantScheme.INT8_STATIC_ASYM.code())
                .tensorF32("blk.0.attn_q.input_scale", new long[]{1}, 0.125f)
                .tensor("blk.0.attn_q.input_zero_point", GGMLType.I32, new long[]{1}, i32Bytes(-7));
        withSource(w, (source, kv, catalog) -> {
            InputTransformLoader.Result r = InputTransformLoader.load(source, kv, catalog, ActivationQuant.CHECKPOINT);
            InputTransform t = r.forward("blk.0.attn_q.weight");
            assertArrayEquals(new int[]{7, 6, 5, 4, 3, 2, 1, 0}, t.perm());
            assertEquals(ActQuantScheme.INT8_STATIC_ASYM, t.actQuant().scheme());
            assertEquals(0.125f, t.actQuant().scale(0), 0f);
            assertEquals(-7, t.actQuant().zeroPoint(0));
        });
    }

    @Test
    public void rejectsASiblingOfTheWrongType() throws IOException {
        GgufTestWriter w = new GgufTestWriter()
                .architecture("llama")
                .tensorRandom("blk.0.attn_q.weight", GGMLType.F32, 5L, -1f, 1f, 8, 4)
                .tensorF32("blk.0.attn_q.input_perm", new long[]{8}, 7, 6, 5, 4, 3, 2, 1, 0);
        withSource(w, (source, kv, catalog) -> {
            GgufFormatException e = assertThrows(GgufFormatException.class,
                    () -> InputTransformLoader.load(source, kv, catalog, ActivationQuant.WEIGHT_ONLY));
            assertTrue(e.getMessage(), e.getMessage().contains("I32"));
        });
    }

    @Test
    public void cpuWeightsBindTheInputPermutation() throws IOException {
        withSource(fileModel(), (source, kv, catalog) -> {
            try (CpuWeights weights = CpuWeights.open(source)) {
                InputTransform t = weights.of(weights.get("blk.0.attn_q.weight")).transform();
                assertArrayEquals(new int[]{7, 6, 5, 4, 3, 2, 1, 0}, t.perm());
                assertNull(t.actQuant());
            }
        });
    }

    @Test
    public void cpuWeightsRefuseAnInputPermutationOfAnExpertTensor() throws IOException {
        GgufTestWriter w = new GgufTestWriter()
                .architecture("llama")
                .tensorRandom("blk.0.ffn_up_exps.weight", GGMLType.F32, 5L, -1f, 1f, 8, 4, 2)
                .tensor("blk.0.ffn_up_exps.input_perm", GGMLType.I32, new long[]{8},
                        i32Bytes(7, 6, 5, 4, 3, 2, 1, 0));
        withSource(w, (source, kv, catalog) -> {
            try (CpuWeights weights = CpuWeights.open(source)) {
                GgufFormatException e = assertThrows(GgufFormatException.class,
                        () -> weights.get("blk.0.ffn_up_exps.weight"));
                assertTrue(e.getMessage(), e.getMessage().contains("input permutation")
                        && e.getMessage().contains("blk.0.ffn_up_exps.weight"));
            }
        });
    }
}
