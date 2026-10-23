package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Requiredness of {@link TensorRequirement}: model conditions, presence triggers, layer spans, probed alternatives and
 * the {@code nextn_flags} trunk flags.
 */
public class TensorRequirementTest {

    private static ModelHparams llama(int blocks) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "llama");
        m.put("llama.context_length", 4096);
        m.put("llama.embedding_length", 512);
        m.put("llama.block_count", blocks);
        m.put("llama.feed_forward_length", 1024);
        m.put("llama.attention.head_count", 8);
        m.put("llama.attention.layer_norm_rms_epsilon", 1e-6f);
        return HparamsLoader.load(GgufKv.ofMap(m), Arch.LLAMA);
    }

    private static Set<String> names(String... n) {
        return new HashSet<>(Arrays.asList(n));
    }

    @Test
    public void plainModelEntryIsRequiredUnconditionally() {
        ModelHparams hp = llama(2);
        TensorRequirement r = TensorRequirement.model("output_norm.weight");
        assertTrue(r.appliesToModel(hp));
        assertNull(r.trigger());
        assertEquals(Collections.singletonList("output_norm.weight"), r.missing(hp, names()::contains));
        assertTrue(r.missing(hp, names("output_norm.weight")::contains).isEmpty());
    }

    @Test
    public void modelConditionGatesRequiredness() {
        ModelHparams hp = llama(2);
        TensorRequirement off = TensorRequirement.modelWhen(h -> h.nLayer() > 2, "cls.weight");
        assertFalse(off.appliesToModel(hp));
        assertTrue(off.missing(hp, names()::contains).isEmpty());
        TensorRequirement on = TensorRequirement.modelWhen(h -> h.nLayer() == 2, "cls.weight");
        assertTrue(on.appliesToModel(hp));
        assertEquals(1, on.missing(hp, names()::contains).size());
        TensorRequirement both = on.onlyWhen(h -> false);
        assertTrue(both.missing(hp, names()::contains).isEmpty());
    }

    @Test
    public void modelTriggerRequiresTensorOnlyWithItsProbe() {
        ModelHparams hp = llama(2);
        TensorRequirement r = TensorRequirement.model("markov_w2.weight").ifPresent("markov_w1.weight");
        assertEquals("markov_w1.weight", r.trigger());
        assertTrue(r.missing(hp, names()::contains).isEmpty());
        List<String> missing = r.missing(hp, names("markov_w1.weight")::contains);
        assertEquals(Collections.singletonList("markov_w2.weight (required with markov_w1.weight)"), missing);
        assertTrue(r.missing(hp, names("markov_w1.weight", "markov_w2.weight")::contains).isEmpty());
    }

    @Test
    public void layerTriggerIsResolvedAtTheSameLayer() {
        ModelHparams hp = llama(3);
        TensorRequirement r = TensorRequirement.layer("ffn_norm_ada_up.weight")
                .ifPresent("blk.%d.ffn_norm_ada_down.weight");
        List<String> missing = r.missing(hp, names("blk.1.ffn_norm_ada_down.weight")::contains);
        assertEquals(Collections.singletonList(
                "blk.1.ffn_norm_ada_up.weight (required with blk.1.ffn_norm_ada_down.weight)"), missing);
    }

    @Test
    public void layerSpanMayExtendPastTheTrunk() {
        ModelHparams hp = llama(2);
        TensorRequirement plain = TensorRequirement.of(TensorRequirement.Scope.LAYER, true, (h, il) -> true,
                new String[][] {{"dec.blk.%d.attn_norm.weight"}});
        assertEquals(2, plain.layerSlots(hp));
        TensorRequirement wide = plain.overLayers(h -> h.nLayerAll() + 2);
        assertEquals(4, wide.layerSlots(hp));
        assertTrue(wide.appliesTo(hp, 3));
        assertFalse(wide.appliesTo(hp, 4));
        assertEquals(Arrays.asList("dec.blk.0.attn_norm.weight", "dec.blk.1.attn_norm.weight",
                "dec.blk.2.attn_norm.weight", "dec.blk.3.attn_norm.weight"), wide.missing(hp, names()::contains));
    }

    @Test
    public void optionalEntriesAreNeverMissing() {
        ModelHparams hp = llama(2);
        assertTrue(TensorRequirement.modelOptional("output.weight").missing(hp, names()::contains).isEmpty());
        assertTrue(TensorRequirement.layerOptional("attn_q.bias").missing(hp, names()::contains).isEmpty());
    }

    @Test
    public void refinementsRejectTheWrongScope() {
        assertThrows(IllegalStateException.class,
                () -> TensorRequirement.layer("x").onlyWhen(TensorRequirement.ALWAYS));
        assertThrows(IllegalStateException.class,
                () -> TensorRequirement.model("x").overLayers(TensorRequirement.ALL_LAYERS));
        assertThrows(IllegalArgumentException.class, () -> TensorRequirement.model("x").ifPresent("blk.%d.y"));
        assertThrows(IllegalArgumentException.class, () -> TensorRequirement.layer("x").ifPresent("y"));
        assertThrows(IllegalStateException.class,
                () -> TensorRequirement.model("x").ifPresent("y").ifPresent("z"));
    }

    private static TensorRequirement qkvWeights() {
        return TensorRequirement.layerAnyOf(TensorRequirement.TRUNK, new String[] {"attn_qkv.weight"},
                new String[] {"attn_q.weight", "attn_k.weight", "attn_v.weight"});
    }

    @Test
    public void probedEntryCreatesOnlyTheFirstAlternativeWhoseLeadExists() {
        TensorRequirement r = qkvWeights().probed();
        assertTrue(r.isProbed());
        assertEquals(Collections.singletonList("blk.0.attn_qkv.weight"),
                r.created(0, names("blk.0.attn_qkv.weight", "blk.0.attn_q.weight")::contains));
        assertEquals(Arrays.asList("blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"),
                r.created(0, names()::contains));
        assertEquals(4, qkvWeights().created(0, names()::contains).size());
    }

    @Test
    public void probedEntryIsMetOnlyByItsProbedAlternative() {
        ModelHparams hp = llama(1);
        TensorRequirement r = qkvWeights().probed();
        assertTrue(r.missing(hp, names("blk.0.attn_qkv.weight")::contains).isEmpty());
        assertEquals(Collections.singletonList("blk.0.attn_q.weight+blk.0.attn_k.weight+blk.0.attn_v.weight"),
                r.missing(hp, names("blk.0.attn_q.weight")::contains));
        assertEquals(Collections.singletonList(
                "blk.0.attn_qkv.weight or blk.0.attn_q.weight+blk.0.attn_k.weight+blk.0.attn_v.weight"),
                qkvWeights().missing(hp, names()::contains));
        assertThrows(IllegalStateException.class, () -> TensorRequirement.layer("attn_norm.weight").probed());
    }

    @Test
    public void absenceTriggerSelectsTheElseBranch() {
        ModelHparams hp = llama(2);
        TensorRequirement r = TensorRequirement.layer("attn_q.weight").ifAbsent("blk.%d.attn_qkv.weight");
        assertTrue(r.triggerAbsent());
        assertFalse(r.triggerHolds(0, names("blk.0.attn_qkv.weight")::contains));
        assertTrue(r.triggerHolds(1, names("blk.0.attn_qkv.weight")::contains));
        assertEquals(Collections.singletonList(
                "blk.1.attn_q.weight (required without blk.1.attn_qkv.weight)"),
                r.missing(hp, names("blk.0.attn_qkv.weight")::contains));
        assertTrue(r.created(0, names("blk.0.attn_qkv.weight")::contains).isEmpty());
        assertThrows(IllegalStateException.class,
                () -> TensorRequirement.layer("x").ifPresent("blk.%d.y").ifAbsent("blk.%d.z"));
    }

    @Test
    public void skippedBlocksCreateEveryNameOfHelperEntries() {
        TensorRequirement helper = qkvWeights().probed().ifPresent("blk.%d.attn_norm.weight")
                .creatingAllWhenSkipped();
        assertTrue(helper.createsAllWhenSkipped());
        assertEquals(4, helper.createdWhenSkipped(2, names()::contains).size());
        assertTrue(helper.created(2, names()::contains).isEmpty());
        TensorRequirement probing = qkvWeights().probed();
        assertEquals(3, probing.createdWhenSkipped(2, names()::contains).size());
    }

    @Test
    public void trunkFlagsReleaseTrunkTensorsOfAnMtpOnlyFile() {
        ModelHparams hp = llama(3);
        TensorRequirement output = TensorRequirement.model("output.weight").withTrunkFlags();
        assertTrue(output.hasTrunkFlags());
        assertEquals(1, output.missing(hp, names()::contains, false).size());
        assertTrue(output.missing(hp, names()::contains, true).isEmpty());
        assertEquals(1, TensorRequirement.model("token_embd.weight").missing(hp, names()::contains, true).size());

        TensorRequirement norm = TensorRequirement.layer("attn_norm.weight", (h, il) -> true);
        assertEquals(3, norm.missing(hp, names()::contains, false).size());
        assertTrue(norm.missing(hp, names()::contains, true).isEmpty());
        assertThrows(IllegalStateException.class, () -> TensorRequirement.layer("x").withTrunkFlags());
    }
}
