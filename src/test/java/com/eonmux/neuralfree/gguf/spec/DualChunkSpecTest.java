package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link DualChunkSpec} and the attention-spec combinations dual-chunk attention admits: the chunk/local geometry
 * and its defaults, the sequence scaling factor, and the refusals of the combinations vLLM's dual-chunk backend does
 * not support (non-causal layers, no cache, a sliding window, sinks, softcap, temperature, a position bias, NoPE,
 * the linear replacement and Q/K norms after RoPE).
 */
public class DualChunkSpecTest {

    private static DualChunkSpec spec() {
        return DualChunkSpec.builder().chunkSize(16).localSize(4).originalMaxPositionEmbeddings(1024).build();
    }

    private static AttentionSpec.Builder attention() {
        return AttentionSpec.builder().nHead(4).nHeadKv(2).headDimK(8).headDimV(8).nRot(8).causal(true)
                .usesMemory(true).kvOwnerLayer(0).kqScale(0.25f).dualChunk(spec());
    }

    @Test
    public void defaultsFollowVllm() {
        DualChunkSpec dc = DualChunkSpec.builder().build();
        assertEquals(8192, dc.chunkSize());
        assertEquals(1024, dc.localSize());
        assertEquals(7168, dc.chunkLength());
        assertEquals(0, dc.originalMaxPositionEmbeddings());
        assertFalse(dc.sparseAttentionEnabled());
        assertEquals(32768, dc.sparseAttentionThreshold());
        assertEquals(64, dc.sparseAttentionLastQ());
        assertEquals(1.0f, dc.scalingFactor(1 << 20), 0f);
    }

    @Test
    public void malformedBlocksAreNamed() {
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> DualChunkSpec.builder().chunkSize(8).localSize(8).build()).getMessage()
                .contains("local_size"));
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> DualChunkSpec.builder().chunkSize(8).localSize(0).build()).getMessage()
                .contains("local_size"));
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> DualChunkSpec.builder().originalMaxPositionEmbeddings(-1).build()).getMessage()
                .contains("original_max_position_embeddings"));
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> DualChunkSpec.builder().sparseAttentionThreshold(0).build()).getMessage()
                .contains("sparse_attention_threshold"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> spec().scalingFactor(-1)).getMessage()
                .contains("sequence length"));
    }

    @Test
    public void theAttentionSpecCarriesTheFiveQueryPlanes() {
        AttentionSpec a = attention().build();
        assertTrue(a.hasDualChunk());
        assertEquals(16, a.dualChunk().chunkSize());
        assertEquals(5L * a.qDim(), a.qRopeDim());
        assertEquals(a.qDim(), a.outDim());
    }

    @Test
    public void unsupportedAttentionCombinationsAreRefusedByName() {
        refused(attention().causal(false), "causal");
        refused(attention().usesMemory(false), "KV cache");
        refused(attention().swaLayer(true).swaType(SwaType.STANDARD).nSwa(4), "sliding window");
        refused(attention().sinks(true), "sinks");
        refused(attention().sinkTokens(2), "sinks");
        refused(attention().softcap(30f), "softcap");
        refused(attention().temperature(AttentionSpec.Temperature.of(2f, 1, 0f, false)), "temperature");
        refused(attention().outScale(0.5f), "output logit scale");
        refused(attention().biasMode(AttentionSpec.BiasMode.ALIBI).alibiMaxBias(8f), "position bias");
        refused(attention().noRope(true).nRot(0), "roped");
        refused(attention().linearReplace(true).nHeadKv(0), "linear map");
        refused(attention().qkNorm(QkNormKind.RMS_HEAD).qkNormAfterRope(true), "before RoPE");
        refused(attention().dualRead(1, 64), "second cache");
    }

    private static void refused(AttentionSpec.Builder b, String expected) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, b::build);
        assertTrue(e.getMessage(), e.getMessage().contains(expected));
    }
}
