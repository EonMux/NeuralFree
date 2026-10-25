package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link ActQuantPlan#load}: the carrier keys, their validation and the {@code scale_ub} bound. */
public class ActQuantPlanTest {

    private static Map<String, Object> carrier(String[] names, int[] codes) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "llama");
        m.put(GgufKeys.NF_ACT_QUANT_VERSION, 1);
        m.put(GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES, names);
        m.put(GgufKeys.NF_ACT_QUANT_SCHEMES, codes);
        return m;
    }

    private static void expectFormat(Map<String, Object> m, String fragment) {
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> ActQuantPlan.load(GgufKv.ofMap(m)));
        assertTrue(e.getMessage() + " should mention " + fragment, e.getMessage().contains(fragment));
    }

    @Test
    public void noCarrierGivesNone() {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "llama");
        ActQuantPlan p = ActQuantPlan.load(GgufKv.ofMap(m));
        assertSame(ActQuantPlan.NONE, p);
        assertTrue(p.isEmpty());
        assertEquals(Float.POSITIVE_INFINITY, p.scaleUpperBound(), 0f);
    }

    @Test
    public void readsTheSchemesInMetadataOrder() {
        String[] names = {"blk.1.ffn_up.weight", "blk.0.attn_q.weight", "blk.0.ffn_gate_exps.weight"};
        int[] codes = {ActQuantScheme.FP8_TOKEN.code(), ActQuantScheme.INT8_STATIC_ASYM.code(),
                ActQuantScheme.NVFP4.code()};
        ActQuantPlan p = ActQuantPlan.load(GgufKv.ofMap(carrier(names, codes)));
        assertFalse(p.isEmpty());
        assertEquals(Arrays.asList(names), Arrays.asList(p.schemes().keySet().toArray(new String[0])));
        assertEquals(ActQuantScheme.FP8_TOKEN, p.scheme("blk.1.ffn_up.weight"));
        assertEquals(ActQuantScheme.INT8_STATIC_ASYM, p.scheme("blk.0.attn_q.weight"));
        assertEquals(ActQuantScheme.NVFP4, p.scheme("blk.0.ffn_gate_exps.weight"));
        assertNull(p.scheme("output.weight"));
        assertEquals(Float.POSITIVE_INFINITY, p.scaleUpperBound(), 0f);
        assertThrows(UnsupportedOperationException.class,
                () -> p.schemes().put("x.weight", ActQuantScheme.MXFP8));
    }

    @Test
    public void readsTheScaleUpperBound() {
        Map<String, Object> m = carrier(new String[]{"blk.0.attn_q.weight"}, new int[]{4});
        m.put(GgufKeys.NF_ACT_QUANT_SCALE_UB, 1200f);
        assertEquals(1200f, ActQuantPlan.load(GgufKv.ofMap(m)).scaleUpperBound(), 0f);
    }

    @Test
    public void rejectsABoundThatIsNotFiniteAndPositive() {
        for (float bad : new float[]{0f, -1f, Float.NaN, Float.POSITIVE_INFINITY}) {
            Map<String, Object> m = carrier(new String[]{"blk.0.attn_q.weight"}, new int[]{4});
            m.put(GgufKeys.NF_ACT_QUANT_SCALE_UB, bad);
            expectFormat(m, GgufKeys.NF_ACT_QUANT_SCALE_UB);
        }
        Map<String, Object> alone = new HashMap<>();
        alone.put(GgufKeys.NF_ACT_QUANT_SCALE_UB, -2f);
        expectFormat(alone, GgufKeys.NF_ACT_QUANT_SCALE_UB);
    }

    @Test
    public void rejectsAnUnsupportedVersion() {
        Map<String, Object> m = carrier(new String[]{"blk.0.attn_q.weight"}, new int[]{0});
        m.put(GgufKeys.NF_ACT_QUANT_VERSION, 2);
        expectFormat(m, GgufKeys.NF_ACT_QUANT_VERSION);
    }

    @Test
    public void rejectsEntriesWithoutAVersion() {
        Map<String, Object> m = carrier(new String[]{"blk.0.attn_q.weight"}, new int[]{0});
        m.remove(GgufKeys.NF_ACT_QUANT_VERSION);
        expectFormat(m, GgufKeys.NF_ACT_QUANT_VERSION);
    }

    @Test
    public void rejectsMissingOrUnequalArrays() {
        Map<String, Object> noCodes = carrier(new String[]{"blk.0.attn_q.weight"}, new int[]{0});
        noCodes.remove(GgufKeys.NF_ACT_QUANT_SCHEMES);
        expectFormat(noCodes, GgufKeys.NF_ACT_QUANT_SCHEMES);
        Map<String, Object> noNames = carrier(new String[]{"blk.0.attn_q.weight"}, new int[]{0});
        noNames.remove(GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES);
        expectFormat(noNames, GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES);
        expectFormat(carrier(new String[]{"blk.0.attn_q.weight", "blk.0.attn_k.weight"}, new int[]{0}),
                GgufKeys.NF_ACT_QUANT_SCHEMES);
    }

    @Test
    public void rejectsAnUnknownCode() {
        expectFormat(carrier(new String[]{"blk.0.attn_q.weight"}, new int[]{10}), "blk.0.attn_q.weight");
        expectFormat(carrier(new String[]{"blk.0.attn_q.weight"}, new int[]{-1}), "unknown");
    }

    @Test
    public void rejectsDuplicateNames() {
        expectFormat(carrier(new String[]{"blk.0.attn_q.weight", "blk.0.attn_q.weight"}, new int[]{0, 1}),
                "twice");
    }

    @Test
    public void rejectsNamesThatAreNotWeightNames() {
        expectFormat(carrier(new String[]{"blk.0.attn_q.bias"}, new int[]{0}), "<base>.weight");
        expectFormat(carrier(new String[]{".weight"}, new int[]{0}), "<base>.weight");
    }
}
