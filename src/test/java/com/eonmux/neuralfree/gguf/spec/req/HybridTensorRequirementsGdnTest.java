package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The qwen4exp block accepts the converter's merged {@code attn_qkv_gate} / {@code ssm_beta_alpha} /
 * {@code hc_*_down_inject} weights or the separate pairs (qwen4exp.cpp:233-281), and names what is missing when
 * neither form is complete.
 */
public class HybridTensorRequirementsGdnTest {

    private static final String[] SITES = {"attn", "ffn"};

    /** Layer 0 is linear attention, layer 1 full attention. */
    private static ModelHparams hparams() {
        return new ModelHparamsBuilder(ArchRegistry.traits(Arch.QWEN4EXP)).layerCounts(2, 0).nEmbd(64)
                .isRecurrent(0, true).build();
    }

    private static List<String> missing(Set<String> present) {
        ModelHparams hp = hparams();
        List<String> out = new ArrayList<>();
        for (TensorRequirement r : HybridTensorRequirementsGdn.qwen4exp()) {
            out.addAll(r.missing(hp, present::contains));
        }
        return out;
    }

    /** Missing entries that mention one of the merged-or-separate projections. */
    private static List<String> projectionGaps(Set<String> present) {
        List<String> gaps = new ArrayList<>();
        for (String m : missing(present)) {
            if (m.contains("attn_qkv") || m.contains("attn_gate") || m.contains("ssm_beta") || m.contains("ssm_alpha")
                    || m.contains(".hc_")) {
                gaps.add(m);
            }
        }
        return gaps;
    }

    private static Set<String> hc(boolean merged) {
        Set<String> s = new HashSet<>();
        for (int il = 0; il < 2; il++) {
            for (String site : SITES) {
                String p = "blk." + il + ".hc_" + site;
                s.add(p + "_norm.weight");
                s.add(p + "_up.weight");
                if (merged) {
                    s.add(p + "_down_inject.weight");
                } else {
                    s.add(p + "_down.weight");
                    s.add(p + "_inject.weight");
                }
            }
        }
        s.add("output_hc_down.weight");
        return s;
    }

    @Test
    public void theMergedWeightsMeetTheBlock() {
        Set<String> present = hc(true);
        present.addAll(Arrays.asList("blk.0.attn_qkv_gate.weight", "blk.0.ssm_beta_alpha.weight"));
        assertEquals(new ArrayList<String>(), projectionGaps(present));
    }

    @Test
    public void theSeparatePairsMeetTheBlock() {
        Set<String> present = hc(false);
        present.addAll(Arrays.asList("blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_beta.weight",
                "blk.0.ssm_alpha.weight"));
        assertEquals(new ArrayList<String>(), projectionGaps(present));
    }

    @Test
    public void anIncompleteSeparatePairIsNamed() {
        Set<String> present = hc(true);
        present.addAll(Arrays.asList("blk.0.attn_qkv.weight", "blk.0.ssm_beta_alpha.weight"));
        present.remove("blk.1.hc_ffn_down_inject.weight");
        present.add("blk.1.hc_ffn_down.weight");
        List<String> gaps = projectionGaps(present);
        assertEquals(gaps.toString(), 2, gaps.size());
        assertTrue(gaps.toString(), gaps.get(0).contains("blk.0.attn_gate.weight")
                || gaps.get(1).contains("blk.0.attn_gate.weight"));
        assertTrue(gaps.toString(), gaps.get(0).contains("blk.1.hc_ffn_inject.weight")
                || gaps.get(1).contains("blk.1.hc_ffn_inject.weight"));
    }
}
