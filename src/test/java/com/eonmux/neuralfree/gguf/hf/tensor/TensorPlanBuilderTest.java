package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.seq;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.values;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.withSiblings;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link TensorPlanBuilder} and {@link TensorPlan}: validation, typing through the policy and the forced type, the
 * sibling tensors of quantized sources on scaled bases and their folding elsewhere, LoRA factor planning, typed
 * metadata, the consumed and dropped sets, and the {@link HfPlanView} reads.
 */
public class TensorPlanBuilderTest {

    private static TensorPlanBuilder builder(OutType outType) {
        return new TensorPlanBuilder(new DtypePolicy(n -> false, outType), HfForceType.NONE);
    }

    private static RowSource q8(long rows) {
        return ElementTransforms.convert(seq(rows, 32), GGMLType.Q8_0);
    }

    @Test
    public void plansTypedTensorsInOrder() {
        TensorPlanBuilder b = builder(OutType.F16);
        b.add("blk.0.attn_q.weight", seq(4, 32));
        b.add("blk.0.attn_norm.weight", seq(32));
        b.add("blk.0.attn_q.bias", ElementTransforms.convert(seq(4), GGMLType.BF16));
        TensorPlan p = b.build();
        assertEquals(Arrays.asList("blk.0.attn_q.weight", "blk.0.attn_norm.weight", "blk.0.attn_q.bias"), p.names());
        PlannedTensor q = p.tensor("blk.0.attn_q.weight");
        assertEquals(GGMLType.F16, q.type());
        assertArrayEquals(new long[]{32, 4}, q.ne());
        assertEquals(4 * 32 * 2, q.byteSize());
        assertEquals(GGMLType.F32, p.type("blk.0.attn_norm.weight"));
        assertEquals(GGMLType.F32, p.type("blk.0.attn_q.bias"));
        assertArrayEquals(new long[]{4}, p.ne("blk.0.attn_q.bias"));
    }

    @Test
    public void validatesNamesRanksAndForcedTypes() {
        TensorPlanBuilder b = builder(OutType.SOURCE);
        b.add("a.weight", seq(2, 2));
        assertThrows(IllegalArgumentException.class, () -> b.add("a.weight", seq(2, 2)));
        assertThrows(IllegalArgumentException.class, () -> b.add("five.weight", seq(1, 1, 1, 1, 2)));
        TensorPlanBuilder forced = new TensorPlanBuilder(new DtypePolicy(n -> false, OutType.SOURCE),
                (name, nDims, source, outType) -> name.startsWith("q") ? GGMLType.Q8_0 : null);
        forced.add("q.weight", seq(2, 64));
        forced.add("q2.weight", seq(2, 48));
        TensorPlan p = forced.build();
        assertEquals(GGMLType.Q8_0, p.type("q.weight"));
        assertEquals(GGMLType.F16, p.type("q2.weight"));
        assertEquals(1, b.build().tensors().size());
    }

    @Test
    public void outputScaleBecomesASiblingOnScaledBasesAndIsFoldedElsewhere() throws IOException {
        TensorPlanBuilder b = builder(OutType.SOURCE);
        b.add("blk.0.attn_q.weight", withSiblings(q8(2), new float[]{2f}, null, null, null, null));
        b.add("token_embd.weight", withSiblings(q8(2), new float[]{2f}, null, null, null, null));
        TensorPlan p = b.build();
        assertEquals(Arrays.asList("blk.0.attn_q.weight", "blk.0.attn_q.scale", "token_embd.weight"), p.names());
        assertEquals(GGMLType.F32, p.type("blk.0.attn_q.scale"));
        assertArrayEquals(new float[]{2f}, p.readF32("blk.0.attn_q.scale", 1), 0f);
        assertEquals(GGMLType.Q8_0, p.type("token_embd.weight"));
        float[] base = values(q8(2));
        float[] folded = p.readF32("token_embd.weight", 64);
        for (int i = 0; i < base.length; i++) {
            assertEquals(2f * base[i], folded[i], 0f);
        }
        assertArrayEquals(base, p.readF32("blk.0.attn_q.weight", 64), 0f);
    }

    @Test
    public void activationSiblingsAndSchemes() throws IOException {
        TensorPlanBuilder b = builder(OutType.SOURCE);
        int[] perm = new int[32];
        for (int i = 0; i < 32; i++) {
            perm[i] = 31 - i;
        }
        b.add("blk.0.ffn_up.weight", withSiblings(q8(2), null, new float[]{0.5f}, new int[]{3}, perm,
                ActQuantScheme.INT8_STATIC_ASYM));
        b.add("blk.0.ffn_gate_inp.weight", withSiblings(seq(2, 32), null, new float[]{0.5f}, null, null,
                ActQuantScheme.FP8_STATIC));
        b.add("blk.0.attn_kv_a_mqa.weight", withSiblings(seq(2, 32), null, null, null, null,
                ActQuantScheme.FP8_TOKEN));
        TensorPlan p = b.build();
        assertEquals(GGMLType.I32, p.type("blk.0.ffn_up.input_zero_point"));
        assertArrayEquals(new long[]{3}, p.readI64("blk.0.ffn_up.input_zero_point", 1));
        assertEquals(GGMLType.I32, p.type("blk.0.ffn_up.input_perm"));
        assertEquals(31, p.readI64("blk.0.ffn_up.input_perm", 32)[0]);
        assertArrayEquals(new float[]{0.5f}, p.readF32("blk.0.ffn_up.input_scale", 1), 0f);
        assertEquals(ActQuantScheme.INT8_STATIC_ASYM, p.actQuantEntries().get("blk.0.ffn_up.weight"));
        assertFalse(p.has("blk.0.ffn_gate_inp.input_scale"));
        assertNull(p.actQuantEntries().get("blk.0.ffn_gate_inp.weight"));
        assertEquals(ActQuantScheme.FP8_TOKEN, p.actQuantEntries().get("blk.0.attn_kv_a_mqa.weight"));
        assertThrows(IllegalArgumentException.class, () -> builder(OutType.SOURCE).add("blk.0.ffn_up.weight",
                withSiblings(q8(2), null, null, null, null, ActQuantScheme.FP8_STATIC)));
    }

    @Test
    public void activationPreScaleBecomesASiblingOnScaledBases() throws IOException {
        float[] pre = new float[32];
        for (int i = 0; i < pre.length; i++) {
            pre[i] = i % 2 == 0 ? 0.5f : 2f;
        }
        TensorPlanBuilder b = builder(OutType.SOURCE);
        b.add("blk.0.attn_q.weight", withSiblings(q8(2), null, null, null, null, null, pre));
        TensorPlan p = b.build();
        assertEquals(GGMLType.F32, p.type("blk.0.attn_q.input_pre_scale"));
        assertArrayEquals(new long[]{32}, p.ne("blk.0.attn_q.input_pre_scale"));
        assertArrayEquals(pre, p.readF32("blk.0.attn_q.input_pre_scale", 32), 0f);

        float[] perExpert = new float[2 * 32];
        for (int i = 0; i < perExpert.length; i++) {
            perExpert[i] = i % 3 == 0 ? 0.25f : 1f;
        }
        TensorPlanBuilder ex = builder(OutType.SOURCE);
        ex.add("blk.0.ffn_up_exps.weight", withSiblings(seq(2, 2, 32), null, null, null, null, null, perExpert));
        TensorPlan pEx = ex.build();
        assertEquals(GGMLType.F32, pEx.type("blk.0.ffn_up_exps.input_pre_scale"));
        assertArrayEquals(perExpert, pEx.readF32("blk.0.ffn_up_exps.input_pre_scale", 64), 0f);

        // A weight outside the scaled projections has no sibling slot, and a length that is not K or E * K is bad.
        assertThrows(IllegalArgumentException.class, () -> builder(OutType.SOURCE).add("token_embd.weight",
                withSiblings(q8(2), null, null, null, null, null, pre)));
        assertThrows(IllegalArgumentException.class, () -> builder(OutType.SOURCE).add("blk.0.attn_q.weight",
                withSiblings(q8(2), null, null, null, null, null, new float[16])));
    }

    @Test
    public void malformedSiblingValuesAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> builder(OutType.SOURCE).add("blk.0.attn_q.weight",
                withSiblings(q8(2), new float[]{2f, 3f}, null, null, null, null)));
        assertThrows(IllegalArgumentException.class, () -> builder(OutType.SOURCE).add("blk.0.attn_q.weight",
                withSiblings(q8(2), null, new float[]{1f, 1f}, null, null, ActQuantScheme.FP8_STATIC)));
        int[] duplicate = new int[32];
        assertThrows(IllegalArgumentException.class, () -> builder(OutType.SOURCE).add("blk.0.attn_q.weight",
                withSiblings(q8(2), null, null, null, duplicate, null)));
    }

    @Test
    public void permutationIsFoldedOutsideScaledBases() throws IOException {
        TensorPlanBuilder b = builder(OutType.SOURCE);
        b.add("token_embd.weight", withSiblings(seq(1, 3), null, null, null, new int[]{2, 0, 1}, null));
        TensorPlan p = b.build();
        assertEquals(Collections.singletonList("token_embd.weight"), p.names());
        assertArrayEquals(new float[]{1, 2, 0}, p.readF32("token_embd.weight", 3), 0f);
    }

    @Test
    public void loraDeltasArePlannedAsFactors() {
        RowSource a = seq(2, 3);
        RowSource bb = seq(4, 2);
        TensorPlanBuilder b = builder(OutType.F16);
        b.add("blk.0.attn_q.weight", new LoraFactorSource("q", a, bb));
        b.add("token_embd.weight", new LoraFactorSource("e", a, bb));
        TensorPlan p = b.build();
        assertArrayEquals(new long[]{3, 2}, p.ne("blk.0.attn_q.weight.lora_a"));
        assertArrayEquals(new long[]{2, 4}, p.ne("blk.0.attn_q.weight.lora_b"));
        assertEquals(GGMLType.F16, p.type("blk.0.attn_q.weight.lora_a"));
        assertArrayEquals(new long[]{2, 3}, p.ne("token_embd.weight.lora_a"));
    }

    @Test
    public void metadataIsTypedAndNormalized() {
        TensorPlanBuilder b = builder(OutType.SOURCE);
        b.addKv("x.u32", GGUFValueType.UINT32, 4_000_000_000L);
        b.addKv("x.f32", GGUFValueType.FLOAT32, 0.5);
        b.addKv("x.arr", GGUFValueType.ARRAY, GGUFValueType.INT64, new long[]{1, 2});
        b.addKv("x.list", GGUFValueType.ARRAY, GGUFValueType.FLOAT32, Arrays.asList(1, 2.5));
        assertThrows(IllegalArgumentException.class, () -> b.addKv("x.u32", GGUFValueType.UINT32, 1));
        assertThrows(IllegalArgumentException.class, () -> b.addKv("x.u8", GGUFValueType.UINT8, 256));
        assertThrows(IllegalArgumentException.class, () -> b.addKv("x.i", GGUFValueType.INT32, 1.5));
        assertThrows(IllegalArgumentException.class, () -> b.addKv("x.s", GGUFValueType.STRING, 3));
        assertThrows(IllegalArgumentException.class, () -> b.addKv("x.a", GGUFValueType.ARRAY, null, new int[0]));
        List<TensorPlan.KvEntry> kv = b.build().kv();
        assertEquals(4, kv.size());
        assertEquals((int) 4_000_000_000L, kv.get(0).value());
        assertEquals(0.5f, kv.get(1).value());
        assertEquals(Arrays.asList(1L, 2L), kv.get(2).value());
        assertEquals(Arrays.asList(1f, 2.5f), kv.get(3).value());
        assertEquals(GGUFValueType.FLOAT32, kv.get(3).elementType());
    }

    @Test
    public void consumedAndDroppedSets() {
        TensorPlanBuilder b = builder(OutType.SOURCE);
        assertFalse(b.isConsumed("model.layers.0.mlp.experts.1.up_proj.weight"));
        b.consume("model.layers.0.mlp.experts.1.up_proj.weight");
        assertTrue(b.isConsumed("model.layers.0.mlp.experts.1.up_proj.weight"));
        b.drop("rotary_emb.inv_freq", "computed at load");
        b.drop("rotary_emb.inv_freq", "second reason");
        assertEquals("computed at load", b.build().dropped().get("rotary_emb.inv_freq"));
    }

    @Test
    public void planViewReadsAreBounded() throws IOException {
        TensorPlanBuilder b = builder(OutType.SOURCE);
        b.add("d2t", ElementTransforms.keepInteger(RopeTensors.fromLongs(new long[]{3}, new long[]{-1, 0, 5})));
        b.add("rope_freqs.weight", RopeTensors.fromFloats(new long[]{2}, new float[]{1, 8}));
        TensorPlan p = b.build();
        assertEquals(GGMLType.I64, p.type("d2t"));
        assertArrayEquals(new long[]{-1, 0, 5}, p.readI64("d2t", 3));
        assertArrayEquals(new float[]{-1, 0, 5}, p.readF32("d2t", 3), 0f);
        assertThrows(IllegalArgumentException.class, () -> p.readI64("rope_freqs.weight", 2));
        assertThrows(IllegalArgumentException.class, () -> p.readF32("d2t", 2));
        assertThrows(IllegalArgumentException.class, () -> p.readF32("missing", 2));
        assertThrows(IllegalArgumentException.class, () -> p.readF32("d2t", HfPlanView.MAX_READ_ELEMENTS + 1));
        assertThrows(UnsupportedOperationException.class, () -> p.tensors().clear());
    }
}
