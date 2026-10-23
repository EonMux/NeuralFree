package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.UnsupportedArchException;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** {@link ModelSpecBuilder} and {@link ModelSpecValidator} over the synthetic GGUF files of {@code SyntheticModels}. */
public class ModelSpecBuilderTest {

    /** Writes {@code w}, parses it and builds its spec with the default memory configuration. */
    private static ModelSpec build(GgufTestWriter w) throws IOException {
        Path path = w.writeTempFile();
        GGUFFileParser parser = null;
        try {
            parser = new GGUFFileParser(path.toString());
            return ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), MemoryConfig.defaults());
        } finally {
            try {
                if (parser != null) {
                    parser.close();
                }
            } finally {
                Files.deleteIfExists(path);
            }
        }
    }

    @Test
    public void llamaResolvesDenseAttentionLayers() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        ModelSpec spec = build(SyntheticModels.llama(c));

        assertSame(Arch.LLAMA, spec.arch());
        assertEquals(MemoryKind.KV, spec.memoryKind());
        assertEquals(c.nLayer(), spec.nLayer());
        assertEquals(c.nLayer(), spec.schedule().slots());
        assertTrue(spec.nextnLayers().isEmpty());
        assertTrue(spec.encoderLayers().isEmpty());
        assertNotNull(spec.embedding().tokenEmbd());
        assertFalse(spec.head().tied());
        for (int il = 0; il < c.nLayer(); il++) {
            LayerSpec l = spec.layer(il);
            assertEquals(il, l.il());
            assertEquals(il, l.weightLayer());
            assertEquals(TensorNames.STACK_BLK, l.stack());
            assertEquals(MixerKind.ATTENTION, l.mixerKind());
            assertEquals(FfnKind.DENSE, l.ffnKind());
            assertTrue("llama layers carry no extras", l.extras().isEmpty());
            AttentionSpec a = l.attention();
            assertEquals(c.nHead(), a.nHead());
            assertEquals(c.nHeadKv(), a.nHeadKv());
            assertEquals(c.headDim(), a.headDimK());
            assertEquals(c.headDim(), a.headDimV());
            assertEquals(AttnGateKind.NONE, a.gate());
            assertTrue(a.causal());
            assertEquals(c.nRot(), l.rope().nDims());
            assertEquals(0, l.rope().nOffs());
            assertNull(l.moe());
            assertEquals(c.nFf(), l.ffn().nFf());
        }
    }

    @Test
    public void qwen3nextResolvesHybridGdnAndGatedAttentionWithSharedExpertMoe() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.qwen3next();
        ModelSpec spec = build(SyntheticModels.qwen3next(c));

        assertSame(Arch.QWEN3NEXT, spec.arch());
        assertEquals(MemoryKind.HYBRID, spec.memoryKind());
        assertEquals(c.nLayer(), spec.nLayer());
        for (int il = 0; il < c.nLayer(); il++) {
            LayerSpec l = spec.layer(il);
            boolean full = (il + 1) % c.fullAttentionInterval() == 0;
            if (full) {
                assertEquals(MixerKind.ATTENTION, l.mixerKind());
                AttentionSpec a = l.attention();
                assertEquals(AttnGateKind.Q_INTERLEAVED, a.gate());
                assertEquals(c.headDim(), a.headDimK());
                assertEquals(QkNormKind.RMS_HEAD, a.qkNorm());
                assertEquals(c.nRot(), a.nRot());
                assertEquals(c.nRot(), l.rope().nDims());
                assertNull(l.recurrent());
            } else {
                assertEquals(MixerKind.GDN, l.mixerKind());
                assertNotNull(l.recurrent());
                assertNull(l.attention());
                assertNull(l.rope());
                assertFalse(l.parallelAttention());
            }
            assertEquals(FfnKind.MOE, l.ffnKind());
            MoeSpec moe = l.moe();
            assertEquals(c.nExpert(), moe.nExpert());
            assertEquals(c.nExpertUsed(), moe.nUsed());
            assertEquals(c.nFfExp(), moe.nFfExp());
            assertNotNull(moe.sharedExpert());
            assertTrue(moe.sharedExpertGated());
            assertNull(l.ffn());
            assertTrue(l.extras().isEmpty());
        }
    }

    @Test
    public void missingRequiredTensorsAreReportedTogether() throws IOException {
        GgufTestWriter w = SyntheticModels.llama();
        w.removeTensor("blk.0.attn_k.weight");
        w.removeTensor("blk.1.ffn_up.weight");
        try {
            build(w);
            fail("a llama file without blk.0.attn_k and blk.1.ffn_up must be rejected");
        } catch (MissingTensorException e) {
            assertEquals("llama", e.arch());
            assertEquals(2, e.missing().size());
            assertTrue(e.getMessage(), e.getMessage().contains("blk.0.attn_k.weight"));
            assertTrue(e.getMessage(), e.getMessage().contains("blk.1.ffn_up.weight"));
        }
    }

    @Test
    public void unsupportedArchitectureIsRejectedWithItsReason() throws IOException {
        GgufTestWriter w = SyntheticModels.llama();
        w.architecture("gptj");
        try {
            build(w);
            fail("gptj has no llama.cpp model implementation");
        } catch (UnsupportedArchException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("gptj"));
        }
    }

    @Test
    public void graphAssertRejectsUnequalHeadSizes() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        GgufTestWriter w = SyntheticModels.llama(c);
        w.u32(w.archKey("attention.key_length"), c.headDim());
        w.u32(w.archKey("attention.value_length"), c.headDim() / 2);
        try {
            build(w);
            fail("the llama graph asserts n_embd_head_k == n_embd_head_v");
        } catch (GgufFormatException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("equal K and V head sizes"));
        }
    }

    @Test
    public void tokenEmbeddingRowsAreCheckedAgainstTheTokenList() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        GgufTestWriter w = SyntheticModels.llama(c);
        w.removeTensor("token_embd.weight");
        w.tensorZeros("token_embd.weight", c.weightType(), c.nEmbd(), (long) c.nVocab() + 1);
        try {
            build(w);
            fail("a token_embd with more rows than tokenizer.ggml.tokens must be rejected");
        } catch (GgufFormatException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("token_embd.weight"));
        }
    }

    @Test
    public void projectionShapesAreCheckedAgainstHparams() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        GgufTestWriter w = SyntheticModels.llama(c);
        w.removeTensor("blk.1.attn_v.weight");
        w.tensorZeros("blk.1.attn_v.weight", c.weightType(), c.nEmbd(), (long) c.nHeadKv() * c.headDim() + 2);
        try {
            build(w);
            fail("an attn_v wider than n_head_kv * head_v must be rejected");
        } catch (GgufFormatException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("blk.1.attn_v.weight"));
        }
    }
}
