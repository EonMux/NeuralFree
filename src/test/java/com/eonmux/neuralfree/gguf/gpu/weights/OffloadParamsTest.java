package com.eonmux.neuralfree.gguf.gpu.weights;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The parameter name segment matching of {@link OffloadParams} (vLLM's {@code cpu_offload_params}): a segment
 * matches whole dotted parts of the stored tensor name, so {@code experts} matches
 * {@code blk.3.ffn_gate_exps.weight} only when the name contains that exact part, and an empty set matches every
 * tensor.
 */
public class OffloadParamsTest {

    private static final String PROPERTY = "neuralfree.test.params";

    @Test
    public void anEmptySetMatchesEveryTensor() {
        OffloadParams none = OffloadParams.none();

        assertTrue(none.isEmpty());
        assertTrue(none.matches("blk.0.attn_q.weight"));
        assertTrue(none.matches("token_embd.weight"));
        assertTrue(none.unmatched(Arrays.asList("blk.0.attn_q.weight")).isEmpty());
    }

    @Test
    public void aSegmentMatchesWholeDottedPartsOnly() {
        OffloadParams p = OffloadParams.of(PROPERTY, Collections.singletonList("ffn_gate_exps"));

        assertTrue(p.matches("blk.3.ffn_gate_exps.weight"));
        assertFalse(p.matches("blk.3.ffn_gate_exps_scale.weight"));
        assertFalse(p.matches("blk.3.ffn_gate.weight"));
        assertFalse(p.matches("blk.3.ffn_up_exps.weight"));
    }

    @Test
    public void aMultiPartSegmentMatchesTheWholeRun() {
        OffloadParams p = OffloadParams.of(PROPERTY, Collections.singletonList("experts.w2_weight"));

        assertTrue(p.matches("mlp.experts.w2_weight"));
        assertTrue(p.matches("layers.0.mlp.experts.w2_weight.bias"));
        assertFalse(p.matches("mlp.experts.w2_weight_scale"));
        assertFalse(p.matches("mlp.experts.w3_weight"));
    }

    @Test
    public void everySegmentThatMatchesAnyTensorSelectsIt() {
        OffloadParams p = OffloadParams.of(PROPERTY, Arrays.asList("attn_q", "ffn_gate_exps"));

        assertTrue(p.matches("blk.2.attn_q.weight"));
        assertTrue(p.matches("blk.2.ffn_gate_exps.weight"));
        assertFalse(p.matches("blk.2.attn_k.weight"));
        assertEquals(Arrays.asList("attn_q", "ffn_gate_exps"), p.segments());
    }

    @Test
    public void unmatchedReportsOnlyTheSegmentsNamingNoTensor() {
        OffloadParams p = OffloadParams.of(PROPERTY, Arrays.asList("attn_q", "expertz", "ffn_up_exps"));

        assertEquals(Collections.singletonList("expertz"),
                p.unmatched(Arrays.asList("blk.0.attn_q.weight", "blk.0.ffn_up_exps.weight")));
    }

    @Test
    public void aBlankOrEmptyPartIsRefusedByName() {
        IllegalArgumentException blank = assertThrows(IllegalArgumentException.class,
                () -> OffloadParams.of(PROPERTY, Collections.singletonList(" ")));
        assertTrue(blank.getMessage(), blank.getMessage().contains(PROPERTY));

        IllegalArgumentException emptyPart = assertThrows(IllegalArgumentException.class,
                () -> OffloadParams.of(PROPERTY, Collections.singletonList("ffn..exps")));
        assertTrue(emptyPart.getMessage(), emptyPart.getMessage().contains("empty dotted part"));
    }
}
