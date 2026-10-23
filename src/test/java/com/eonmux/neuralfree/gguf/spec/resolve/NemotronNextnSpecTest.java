package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.PostNormRole;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Nemotron-H with an MTP block: trunk layers pick Mamba-2, attention or FFN by {@code n_head_kv} / {@code n_ff}
 * (nemotron-h.cpp:226-234), while the nemotron_h_moe NextN block always runs NoPE attention, the post-attention norm
 * as the FFN input norm and the routed experts with the shared expert (nemotron-h-moe.cpp {@code graph_mtp}), although
 * its {@code feed_forward_length} entry is non-zero. Its MTP-only file (the NextN block, the embedding and the output
 * norm) builds as the drafter of the full model over the full model's trunk, and so does the full file itself, whose
 * trunk the drafter neither reads nor places in memory.
 */
public class NemotronNextnSpecTest {

    private static final String ARCH = NemotronNextnFixture.ARCH;
    private static final int N_EMBD = NemotronNextnFixture.N_EMBD;
    private static final int N_HEAD = NemotronNextnFixture.N_HEAD;
    private static final int N_HEAD_KV = NemotronNextnFixture.N_HEAD_KV;

    private static Map<String, Object> metadata() {
        return NemotronNextnFixture.metadata();
    }

    private static List<GGUFTensorInfo> tensors() {
        return NemotronNextnFixture.tensors();
    }

    @Test
    public void nextnBlockRunsAttentionBeforeItsExperts() {
        ModelSpec spec = ModelSpecBuilder.build(GgufKv.ofMap(metadata()), TensorCatalog.of(ARCH, tensors()),
                MemoryConfig.defaults());

        assertSame(Arch.NEMOTRON_H_MOE, spec.arch());
        assertEquals(3, spec.nLayer());
        assertEquals(MixerKind.MAMBA2, spec.layer(0).mixerKind());
        assertEquals(MixerKind.ATTENTION, spec.layer(1).mixerKind());
        assertEquals(MixerKind.NONE, spec.layer(2).mixerKind());
        assertEquals(FfnKind.MOE, spec.layer(2).ffnKind());

        assertEquals(1, spec.nextnLayers().size());
        LayerSpec mtp = spec.nextnLayers().get(0);
        assertEquals(3, mtp.il());
        assertEquals(MixerKind.ATTENTION, mtp.mixerKind());
        AttentionSpec a = mtp.attention();
        assertNotNull(a);
        assertEquals(N_HEAD, a.nHead());
        assertEquals(N_HEAD_KV, a.nHeadKv());
        assertTrue("nemotron_h attention has no RoPE", a.noRope());
        assertEquals(PostNormRole.PRE_FFN, mtp.norm().postRole());
        assertEquals(FfnKind.MOE, mtp.ffnKind());
        assertNotNull(mtp.moe().sharedExpert());
    }

    /** What {@code convert_hf_to_gguf.py --mtp} keeps: the embedding, the output norm and the NextN block. */
    private static List<GGUFTensorInfo> mtpOnlyTensors() {
        return NemotronNextnFixture.mtpOnly(tensors(), NemotronNextnFixture.TRUNK);
    }

    @Test
    public void mtpOnlyFileDraftsOverTheTargetsTrunk() {
        GgufKv kv = GgufKv.ofMap(metadata());
        ModelSpec target = ModelSpecBuilder.build(kv, TensorCatalog.of(ARCH, tensors()), MemoryConfig.defaults());
        TensorCatalog sidecar = TensorCatalog.of(ARCH, mtpOnlyTensors());
        assertTrue(ModelSpecBuilder.isMtpOnly(kv, sidecar));
        assertFalse(ModelSpecBuilder.isMtpOnly(kv, TensorCatalog.of(ARCH, tensors())));

        ModelSpec draft = ModelSpecBuilder.buildMtpDraft(kv, sidecar, MemoryConfig.defaults(), target);
        assertEquals(target.nLayer(), draft.nLayer());
        for (int i = 0; i < target.layers().size(); i++) {
            assertSame(target.layers().get(i), draft.layers().get(i));
        }
        assertEquals(1, draft.nextnLayers().size());
        LayerSpec mtp = draft.nextnLayers().get(0);
        assertEquals(3, mtp.il());
        assertEquals(MixerKind.ATTENTION, mtp.mixerKind());
        assertEquals(FfnKind.MOE, mtp.ffnKind());
        assertEquals("token_embd.weight", draft.embedding().tokenEmbd().name());
        assertEquals("output_norm.weight", draft.head().outputNorm().name());
    }

    @Test
    public void aFullFileWithNextnBlocksDraftsWithThemOnly() {
        GgufKv kv = GgufKv.ofMap(metadata());
        TensorCatalog full = TensorCatalog.of(ARCH, tensors());
        ModelSpec target = ModelSpecBuilder.build(kv, full, MemoryConfig.defaults());
        assertTrue(ModelSpecBuilder.isNextnDraft(kv, full));

        ModelSpec draft = ModelSpecBuilder.buildMtpDraft(kv, full, MemoryConfig.defaults(), target);
        for (int i = 0; i < target.layers().size(); i++) {
            assertSame(target.layers().get(i), draft.layers().get(i));
        }
        assertEquals(1, draft.nextnLayers().size());
        assertEquals(3, draft.nextnLayers().get(0).il());

        Predicate<String> reads = ModelSpecBuilder.mtpDraftReads(draft);
        assertFalse(reads.test("blk.0.ssm_in.weight"));
        assertFalse(reads.test("blk.1.attn_q.weight"));
        assertFalse(reads.test("blk.2.ffn_up_exps.weight"));
        assertTrue(reads.test("blk.3.nextn.eh_proj.weight"));
        assertTrue(reads.test("blk.3.attn_q.weight"));
        assertTrue(reads.test("token_embd.weight"));
        assertTrue(reads.test("output_norm.weight"));
        Predicate<String> fromMetadata = ModelSpecBuilder.mtpDraftReads(kv, full);
        for (String name : full.names()) {
            assertEquals(name, reads.test(name), fromMetadata.test(name));
        }
    }

    @Test
    public void onlyAFileWithNextnBlocksOfTheTargetsModelDrafts() {
        GgufKv kv = GgufKv.ofMap(metadata());
        TensorCatalog full = TensorCatalog.of(ARCH, tensors());
        TensorCatalog sidecar = TensorCatalog.of(ARCH, mtpOnlyTensors());
        ModelSpec target = ModelSpecBuilder.build(kv, full, MemoryConfig.defaults());
        assertThrows(UnsupportedOperationException.class,
                () -> ModelSpecBuilder.build(kv, sidecar, MemoryConfig.defaults()));
        assertTrue(ModelSpecBuilder.isNextnDraft(kv, sidecar));

        TensorCatalog noNextn = TensorCatalog.of(ARCH,
                NemotronNextnFixture.withoutNextn(tensors(), NemotronNextnFixture.TRUNK));
        assertFalse(ModelSpecBuilder.isNextnDraft(kv, noNextn));
        assertThrows(IllegalArgumentException.class,
                () -> ModelSpecBuilder.buildMtpDraft(kv, noNextn, MemoryConfig.defaults(), target));

        Map<String, Object> wider = metadata();
        wider.put(ARCH + ".embedding_length", 2 * N_EMBD);
        assertThrows(IllegalArgumentException.class, () -> ModelSpecBuilder.buildMtpDraft(GgufKv.ofMap(wider),
                sidecar, MemoryConfig.defaults(), target));
        Map<String, Object> dflash = metadata();
        dflash.put("general.architecture", "dflash");
        assertFalse(ModelSpecBuilder.isNextnDraft(GgufKv.ofMap(dflash), TensorCatalog.of("dflash", tensors())));
    }
}
