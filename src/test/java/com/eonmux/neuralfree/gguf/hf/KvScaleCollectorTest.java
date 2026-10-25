package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.hf.KvScaleCollector.Kind;
import com.eonmux.neuralfree.gguf.hf.KvScaleCollector.Result;
import com.eonmux.neuralfree.gguf.hf.KvScaleCollector.Scale;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link KvScaleCollector}: every vLLM KV-cache scale form, the head-count rules and the value checks. */
public class KvScaleCollectorTest {

    private static final String L0 = "model.layers.0.";
    private static final String L1 = "model.layers.1.";
    private static final HfConfig CFG = HfConfigLoader.fromJson(Paths.get("."),
            "{\"architectures\":[\"LlamaForCausalLM\"],\"num_attention_heads\":4,\"num_key_value_heads\":2}");

    /** An index of {@code nameAndValues} pairs (each value array becomes a 1-D F32 tensor) plus one weight. */
    private static HfTensorIndex index(Object... nameAndValues) {
        Map<String, RowSource> m = new LinkedHashMap<>();
        m.put(L0 + "self_attn.k_proj.weight", RopeTensors.fromFloats(new long[]{2, 2}, new float[]{1, 2, 3, 4}));
        for (int i = 0; i < nameAndValues.length; i += 2) {
            float[] v = (float[]) nameAndValues[i + 1];
            m.put((String) nameAndValues[i], RopeTensors.fromFloats(new long[]{v.length}, v));
        }
        return HfTensorIndex.of(m);
    }

    private static float[] f(float... v) {
        return v;
    }

    private static Result collect(Object... nameAndValues) throws IOException {
        return KvScaleCollector.collect(CFG, index(nameAndValues));
    }

    private static float[] read(RowSource s) throws IOException {
        ByteBuffer b = ByteBuffer.allocate((int) (s.rows() * s.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        s.readRows(0, (int) s.rows(), b);
        b.flip();
        float[] out = new float[b.remaining() / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getFloat();
        }
        return out;
    }

    private static Map<String, float[]> byName(Result r) throws IOException {
        Map<String, float[]> m = new LinkedHashMap<>();
        for (Scale s : r.scales()) {
            m.put(s.ggufName(), read(s.source()));
        }
        return m;
    }

    @Test
    public void ggufNamesFollowTheKvScaleClaims() {
        assertEquals("blk.3.attn_k.kv_scale", Kind.K.ggufName(3));
        assertEquals("blk.3.attn_v.kv_scale", Kind.V.ggufName(3));
        assertEquals("blk.3.attn_q.q_scale", Kind.Q.ggufName(3));
        assertEquals("blk.3.attn_prob_scale", Kind.PROB.ggufName(3));
    }

    @Test
    public void everyNameFormIsRecognized() {
        String[][] forms = {
                {"self_attn.k_scale", "K"}, {"self_attn.v_scale", "V"}, {"self_attn.q_scale", "Q"},
                {"self_attn.k_proj.k_scale", "K"}, {"self_attn.v_proj.v_scale", "V"},
                {"self_attn.qkv_proj.k_scale", "K"}, {"self_attn.qkv_proj.v_scale", "V"},
                {"self_attn.qkqkv_proj.k_scale", "K"}, {"self_attn.qkqkv_proj.v_scale", "V"},
                {"mixer.k_proj.k_scale", "K"}, {"mixer.v_proj.v_scale", "V"},
                {"self_attn.q.scale", "Q"},
                {"self_attn.k_cache.scale", "K"}, {"self_attn.v_cache.scale", "V"},
                {"self_attn.k_proj.output_scale", "K"}, {"self_attn.v_proj.output_scale", "V"},
                {"self_attn.q_proj.output_scale", "Q"},
                {"self_attn.prob_output_scale", "PROB"}, {"self_attn.prob_scale", "PROB"},
        };
        for (String[] form : forms) {
            assertEquals(form[0], Kind.valueOf(form[1]), KvScaleCollector.kindOf(L0 + form[0]));
        }
        for (String other : new String[]{"self_attn.k_proj.weight_scale", "self_attn.k_proj.input_scale",
                "mlp.down_proj.output_scale", "self_attn.o_proj.output_scale", "self_attn.k_proj.weight"}) {
            assertNull(other, KvScaleCollector.kindOf(L0 + other));
        }
    }

    @Test
    public void scalesAreTakenOutOfTheIndex() throws IOException {
        Result r = collect(L0 + "self_attn.k_proj.k_scale", f(0.5f), L0 + "self_attn.v_proj.v_scale", f(0.25f),
                L1 + "self_attn.q.scale", f(2f), L1 + "self_attn.prob_output_scale", f(3f));
        Map<String, float[]> m = byName(r);
        assertEquals(Arrays.asList("blk.0.attn_k.kv_scale", "blk.0.attn_v.kv_scale", "blk.1.attn_q.q_scale",
                "blk.1.attn_prob_scale"), Arrays.asList(m.keySet().toArray()));
        assertArrayEquals(f(0.5f), m.get("blk.0.attn_k.kv_scale"), 0f);
        assertArrayEquals(f(3f), m.get("blk.1.attn_prob_scale"), 0f);
        assertEquals(Collections.singletonList(L0 + "self_attn.k_proj.weight"), r.index().names());
        assertEquals(1, r.scales().get(2).layer());
        assertEquals(L1 + "self_attn.q.scale", r.scales().get(2).hfName());
    }

    @Test
    public void theDeprecatedKvScaleSetsKeysAndValuesUnlessTheLayerSetsValues() throws IOException {
        Result r = collect(L0 + "self_attn.kv_scale", f(0.5f), L1 + "self_attn.kv_scale", f(0.75f),
                L1 + "self_attn.v_scale", f(0.125f));
        Map<String, float[]> m = byName(r);
        assertArrayEquals(f(0.5f), m.get("blk.0.attn_k.kv_scale"), 0f);
        assertArrayEquals(f(0.5f), m.get("blk.0.attn_v.kv_scale"), 0f);
        assertArrayEquals(f(0.75f), m.get("blk.1.attn_k.kv_scale"), 0f);
        assertArrayEquals("the explicit value scale wins", f(0.125f), m.get("blk.1.attn_v.kv_scale"), 0f);
        assertEquals(4, m.size());
    }

    @Test
    public void kvProjectionOutputScalesAreKvScalesOfTheirModule() throws IOException {
        Result r = collect(L0 + "self_attn.k_proj.output_scale", f(0.5f), L0 + "self_attn.v_proj.output_scale",
                f(0.5f), L0 + "self_attn.q_proj.output_scale", f(1f, 2f, 3f, 4f));
        assertEquals(3, r.scales().size());
        assertTrue(r.outputScaleModules().contains(L0 + "self_attn.k_proj"));
        assertTrue(r.outputScaleModules().contains(L0 + "self_attn.v_proj"));
        assertTrue(r.outputScaleModules().contains(L0 + "self_attn.q_proj"));
        assertArrayEquals("one per query head", f(1, 2, 3, 4), byName(r).get("blk.0.attn_q.q_scale"), 0f);
    }

    @Test
    public void cacheScalesAndZeroPoints() throws IOException {
        Result r = collect(L0 + "self_attn.k_cache.scale", f(0.5f), L0 + "self_attn.v_cache.scale", f(0.5f),
                L0 + "self_attn.k_zero_point", f(0f), L0 + "self_attn.v_zero_point", f(0f, 0f));
        assertEquals(2, r.scales().size());
        assertEquals("zero points are consumed", 1, r.index().names().size());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> collect(L0 + "self_attn.k_zero_point", f(3f)));
        assertTrue(e.getMessage(), e.getMessage().contains("asymmetric"));
    }

    @Test
    public void perHeadCountsFollowTheHeadKind() throws IOException {
        Map<String, float[]> m = byName(collect(L0 + "self_attn.k_scale", f(1f, 2f),
                L0 + "self_attn.q_scale", f(1f, 2f, 3f, 4f)));
        assertArrayEquals(f(1f, 2f), m.get("blk.0.attn_k.kv_scale"), 0f);
        GgufFormatException kWrong = assertThrows(GgufFormatException.class,
                () -> collect(L0 + "self_attn.k_scale", f(1f, 2f, 3f, 4f)));
        assertTrue(kWrong.getMessage(), kWrong.getMessage().contains("expected 1 or 2"));
        assertThrows("queries count query heads", GgufFormatException.class,
                () -> collect(L0 + "self_attn.q_scale", f(1f, 2f)));
        assertThrows("a probability scale is a single value", GgufFormatException.class,
                () -> collect(L0 + "self_attn.prob_scale", f(1f, 2f)));
    }

    @Test
    public void attnHeadScalesStoredAsThreeDimensionsAreAccepted() throws IOException {
        Map<String, RowSource> m = new LinkedHashMap<>();
        m.put(L0 + "self_attn.k_proj.output_scale", RopeTensors.fromFloats(new long[]{2, 1, 1}, f(0.5f, 0.25f)));
        Result r = KvScaleCollector.collect(CFG, HfTensorIndex.of(m));
        assertArrayEquals(f(0.5f, 0.25f), read(r.scales().get(0).source()), 0f);
    }

    @Test
    public void valuesMustBeFiniteAndPositive() {
        for (float bad : new float[]{0f, -1f, Float.NaN, Float.POSITIVE_INFINITY}) {
            GgufFormatException e = assertThrows(String.valueOf(bad), GgufFormatException.class,
                    () -> collect(L0 + "self_attn.k_scale", f(bad)));
            assertTrue(e.getMessage(), e.getMessage().contains("finite and positive"));
        }
    }

    @Test
    public void oneScaleFromTwoTensorsIsRefused() {
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> collect(
                L0 + "self_attn.k_scale", f(1f), L0 + "self_attn.k_proj.output_scale", f(1f)));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.0.attn_k.kv_scale"));
    }

    @Test
    public void aScaleWithoutALayerIsRefused() {
        assertThrows(GgufFormatException.class, () -> collect("lm_head.k_scale", f(1f)));
    }

    @Test
    public void anIndexWithoutScalesIsReturnedAsIs() throws IOException {
        HfTensorIndex in = index();
        Result r = KvScaleCollector.collect(CFG, in);
        assertSame(in, r.index());
        assertTrue(r.scales().isEmpty());
        assertFalse(r.outputScaleModules().iterator().hasNext());
        List<Scale> none = r.scales();
        assertThrows(UnsupportedOperationException.class, () -> none.add(null));
    }
}
