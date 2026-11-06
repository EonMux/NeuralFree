package com.eonmux.neuralfree.gguf.gpu.weights;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The per-tensor placement rules ({@link TensorPlacementOverrides}): the llama.cpp patterns match the llama.cpp
 * names and NeuralFree's {@code .weight}-suffixed names alike, the first matching rule wins, {@code --cpu-moe} and
 * {@code --n-cpu-moe} expand to the reference expert pattern, and every malformed entry fails naming itself.
 */
public class TensorPlacementOverridesTest {

    @Test
    public void noneKeepsEveryTensorOnTheDevice() {
        assertFalse(TensorPlacementOverrides.none().host("blk.0.ffn_up_exps.weight"));
        assertTrue(TensorPlacementOverrides.none().isEmpty());
    }

    @Test
    public void cpuMoeKeepsTheExpertTensorsInTheRam() {
        TensorPlacementOverrides o = TensorPlacementOverrides.cpuMoe();
        assertTrue(o.host("blk.0.ffn_up_exps.weight"));
        assertTrue(o.host("blk.3.ffn_down_exps.weight"));
        assertTrue(o.host("blk.1.ffn_gate_exps.weight"));
        assertTrue(o.host("blk.1.ffn_gate_up_exps.weight"));
        assertTrue("the chunk experts match too", o.host("blk.1.ffn_up_chexps.weight"));
        assertTrue("llama.cpp names without a .weight suffix match", o.host("blk.0.ffn_up_exps"));
        assertFalse("the shared experts are dense", o.host("blk.2.ffn_up_shexp.weight"));
        assertFalse("the router is not an expert tensor", o.host("blk.2.ffn_gate_inp.weight"));
        assertFalse("the FFN biases are not expert tensors", o.host("blk.2.exp_probs_b.bias"));
    }

    @Test
    public void nCpuMoeMatchesTheFirstNBlocksOnly() {
        TensorPlacementOverrides o = TensorPlacementOverrides.nCpuMoe(2);
        assertTrue(o.host("blk.0.ffn_up_exps.weight"));
        assertTrue(o.host("blk.1.ffn_down_exps.weight"));
        assertFalse(o.host("blk.2.ffn_up_exps.weight"));
        assertFalse(o.host("blk.10.ffn_up_exps.weight"));
    }

    @Test
    public void nCpuFfnMatchesTheDenseFfnWeightsOnly() {
        TensorPlacementOverrides o = TensorPlacementOverrides.nCpuFfn(1);
        assertTrue(o.host("blk.0.ffn_up.weight"));
        assertTrue(o.host("blk.0.ffn_gate.weight"));
        assertTrue(o.host("blk.0.ffn_down.weight"));
        assertFalse("the expert tensors are not dense FFN weights", o.host("blk.0.ffn_up_exps.weight"));
        assertFalse(o.host("blk.1.ffn_up.weight"));
    }

    @Test
    public void parsedRulesMatchByNameAndTheFirstMatchWins() {
        TensorPlacementOverrides o = TensorPlacementOverrides.parse("blk\\.1\\..*=GPU , ffn_up_exps=CPU");
        assertTrue(o.host("blk.0.ffn_up_exps.weight"));
        assertFalse("the GPU rule shields layer 1 from the CPU rule", o.host("blk.1.ffn_up_exps.weight"));
        assertTrue(o.host("blk.2.ffn_up_exps.weight"));
        assertFalse(o.host("blk.2.ffn_down_exps.weight"));
    }

    @Test
    public void llamaCppPatternStringsWorkVerbatim() {
        TensorPlacementOverrides o = TensorPlacementOverrides.builder()
                .add(TensorPlacementOverrides.EXPERT_REGEX, TensorPlacementOverrides.Target.HOST).build();
        assertTrue(o.host("blk.7.ffn_gate_up_exps.weight"));
    }

    @Test
    public void withKeepsTheEarlierRuleWinning() {
        TensorPlacementOverrides a = TensorPlacementOverrides.parse("blk\\.0\\..*=GPU");
        TensorPlacementOverrides b = TensorPlacementOverrides.cpuMoe();
        TensorPlacementOverrides both = a.with(b);
        assertFalse("the device rule shields layer 0", both.host("blk.0.ffn_up_exps.weight"));
        assertTrue(both.host("blk.1.ffn_up_exps.weight"));
        assertTrue(a.with(TensorPlacementOverrides.none()) == a);
    }

    @Test
    public void malformedEntriesFailNamingThem() {
        IllegalArgumentException missing = assertThrows(IllegalArgumentException.class,
                () -> TensorPlacementOverrides.parse("ffn_up_exps"));
        assertTrue(missing.getMessage().contains("ffn_up_exps"));
        IllegalArgumentException type = assertThrows(IllegalArgumentException.class,
                () -> TensorPlacementOverrides.parse("exps=CUDA0"));
        assertTrue(type.getMessage().contains("CUDA0"));
        IllegalArgumentException regex = assertThrows(IllegalArgumentException.class,
                () -> TensorPlacementOverrides.parse("([=CPU"));
        assertTrue(regex.getMessage().contains("(["));
        assertThrows(IllegalArgumentException.class, () -> TensorPlacementOverrides.nCpuMoe(0));
        assertThrows(IllegalArgumentException.class, () -> TensorPlacementOverrides.nCpuFfn(-3));
    }

    @Test
    public void rulesExposeTheirParts() {
        TensorPlacementOverrides.Rule r = TensorPlacementOverrides.parse("blk\\.0\\.ffn_up=CPU").rules().get(0);
        assertEquals("blk\\.0\\.ffn_up=CPU", r.toString());
        assertEquals(TensorPlacementOverrides.Target.HOST, r.target());
        assertTrue(r.matches("blk.0.ffn_up.weight"));
        assertFalse(r.matches("blk.1.ffn_up.weight"));
    }
}
