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
    public void lfm3ResolvesHybridShortConvAndGatedSwaAttention() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.lfm3();
        ModelSpec spec = build(SyntheticModels.lfm3());

        assertSame(Arch.LFM3, spec.arch());
        assertEquals(MemoryKind.HYBRID, spec.memoryKind());
        assertEquals(c.nLayer(), spec.nLayer());
        assertTrue(spec.head().tied());
        for (int il = 0; il < c.nLayer(); il++) {
            LayerSpec l = spec.layer(il);
            assertEquals(FfnKind.DENSE, l.ffnKind());
            assertTrue(l.extras().isEmpty());
            if (il % 2 == 0) {
                assertEquals(MixerKind.ATTENTION, l.mixerKind());
                AttentionSpec a = l.attention();
                assertEquals(AttnGateKind.PER_HEAD_SIGMOID, a.gate());
                assertEquals(QkNormKind.RMS_HEAD, a.qkNorm());
                assertEquals(c.nHead(), a.nHead());
                assertEquals(c.nHeadKv(), a.nHeadKv());
                assertEquals(il == 0, a.isSwaLayer());
                assertTrue(a.causal());
                assertEquals(c.nRot(), l.rope().nDims());
                // the full-attention layer rotates with the learned factors, the SWA layer with the SWA base
                assertEquals(il == 2, l.rope().freqFactors() != null);
                if (il == 0) {
                    assertEquals(c.ropeFreqBase(), l.rope().freqBase(), 0f);
                }
                assertNull(l.recurrent());
            } else {
                assertEquals(MixerKind.SHORTCONV, l.mixerKind());
                assertNotNull(l.recurrent());
                assertNull(l.attention());
                assertNull(l.rope());
                assertEquals(c.ssmConv(), l.recurrent().lCache());
            }
        }
    }

    @Test
    public void lfm3D1RunsCausallyAndWithoutMemory() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.lfm3();
        ModelSpec spec = build(SyntheticModels.lfm3(c, false));

        assertSame(Arch.LFM3, spec.arch());
        assertEquals(MemoryKind.HYBRID, spec.memoryKind());
        for (int il = 0; il < c.nLayer(); il++) {
            LayerSpec l = spec.layer(il);
            if (il % 2 == 0) {
                AttentionSpec a = l.attention();
                // dense layers read bidirectionally, the sliding-window layer stays causal (lfm3.cpp:22-25)
                assertEquals(il == 0, a.causal());
                assertFalse(a.usesMemory());
                assertEquals(MixerKind.SHORTCONV_CENTERED, spec.layer(il + 1).mixerKind());
            }
        }
    }

    @Test
    public void limiteResolvesValueEmbeddingsXsaMuddAndSigmoidCap() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.limite();
        ModelSpec spec = build(SyntheticModels.limite(c));

        assertSame(Arch.LIMITE, spec.arch());
        assertEquals(MemoryKind.KV, spec.memoryKind());
        assertEquals(c.nLayer(), spec.nLayer());
        assertFalse(spec.head().tied());
        assertTrue(spec.head().sigmoidCapped());
        assertEquals(2.0f, spec.head().sigmoidCapScale(), 0f);
        assertEquals(1.0f, spec.head().sigmoidCapShift(), 0f);
        assertEquals(4.0f, spec.head().sigmoidCapTemperature(), 0f);
        for (int il = 0; il < c.nLayer(); il++) {
            LayerSpec l = spec.layer(il);
            assertEquals(MixerKind.ATTENTION, l.mixerKind());
            assertEquals(FfnKind.DENSE, l.ffnKind());
            AttentionSpec a = l.attention();
            assertTrue(a.valueEmbeddings());
            assertEquals(il == 0, a.valueEmbeddingGate());
            assertEquals(1e-6f, a.xsaEps(), 0f);
            assertEquals(0.125f, a.kqScale(), 0f);
            assertEquals(il == 0, a.isSwaLayer());
            // the full-attention layer keeps no positional encoding (limite.cpp:163-170)
            assertEquals(il == 0, l.rope() != null);
            java.util.List<String> extras = l.extras();
            assertEquals(1, extras.size());
            assertEquals(com.eonmux.neuralfree.gguf.spec.LayerSpec.Extras.MUDD, extras.get(0));
        }
    }

    @Test
    public void mistralLarge4ResolvesMlaDsaKvSourceAndDenseLead() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.mistralLarge4();
        ModelSpec spec = build(SyntheticModels.mistralLarge4(c));

        assertSame(Arch.MISTRAL_LARGE4, spec.arch());
        assertEquals(MemoryKind.DSA, spec.memoryKind());
        assertEquals(c.nLayer(), spec.nLayer());
        assertTrue(spec.head().finalSoftcap() != 0f);
        for (int il = 0; il < c.nLayer(); il++) {
            LayerSpec l = spec.layer(il);
            assertEquals(MixerKind.DSA_MLA, l.mixerKind());
            assertEquals(il == 0 ? FfnKind.DENSE : FfnKind.MOE, l.ffnKind());
            assertNotNull(l.mla());
            assertEquals(0, l.mla().qLoraRank());
            assertEquals(32, l.mla().kvLoraRank());
            assertEquals(c.headDim(), l.mla().headK());
            assertTrue(l.mla().absorbed());
            assertTrue(l.mla().latentCache());
            assertTrue(l.mla().ropeApplied());
            assertNotNull(l.dsa());
            assertTrue(l.dsa().full());
            assertEquals(il, l.dsa().sourceLayer());
            // layer 1 reads layer 0's cache through attention.kv_source_map
            assertEquals(il == 0 ? 0 : 0, l.attention().kvOwnerLayer());
            assertEquals(il == 1, l.attention().sharesKv(il));
            assertEquals(il == 1, spec.hparams().isMoeLayer(il));
        }
        // every layer scores its own indexer here; a shared layer would name its source
        assertEquals(0, spec.layer(0).dsa().sourceLayer());
        assertEquals(1, spec.layer(1).dsa().sourceLayer());
    }

    @Test
    public void mistralLarge4NoPeResolvesTheMixedRopePattern() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.mistralLarge4();
        ModelSpec spec = build(SyntheticModels.mistralLarge4NoPe(c));

        assertTrue(spec.layer(0).mla().ropeApplied());
        assertNotNull(spec.layer(0).rope());
        assertFalse(spec.layer(1).mla().ropeApplied());
        assertNull("a NoPE layer carries no rotary spec", spec.layer(1).rope());
        assertEquals("a NoPE layer keeps the roped cache width", c.headDim(), spec.layer(1).mla().headK());
        for (int il = 0; il < c.nLayer(); il++) {
            assertTrue(spec.layer(il).dsa().ssmax());
            assertTrue(spec.layer(il).dsa().indexerQueryNorm());
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
        w.architecture("clip");
        try {
            build(w);
            fail("clip is a two-tower embedding without a text model");
        } catch (UnsupportedArchException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("clip"));
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
