package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Byte traces of {@link MemoryPlanner} for representative models (gemma3 ISWA ring, qwen35 hybrid, deepseek32 DSA,
 * mamba2 recurrent) plus rotation (including unrotated latent MLA caches), shared KV, KV dtype validation, the no-memory
 * rules, NextN layers (left empty in the main context, planned alone in the NextN context) and recurrent-state
 * snapshot reservation. Every expected byte count is derived in the test from the llama.cpp sizing rules, and every
 * plan's total must equal the sum of its layer plans, its snapshot shadows and its {@code kv_pos} buffers.
 */
public class MemoryPlannerTest {

    static final int GEMMA3_TRAIN = 8192;
    private static final float KQ_SCALE = 0.0625f;

    private static MemoryConfig f16() {
        return MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).build();
    }

    // gemma3: 5 SWA layers (window 512) then one full layer; 1 KV head of 256; ring = pad256(512 + 512) = 1024 rows.

    @Test
    public void gemma3RingTrace() {
        MemoryPlan plan = MemoryPlanner.planForContext(gemma3(), f16(), 8192);

        assertEquals(MemoryKind.KV, plan.kind());
        assertEquals(2, plan.groups().size());
        MemoryPlan.KvGroup swa = plan.group(0);
        MemoryPlan.KvGroup full = plan.group(1);
        assertEquals(1024, swa.rows());
        assertTrue(swa.ring());
        assertEquals(8192, full.rows());
        assertFalse(full.ring());

        long rowBytes = 256 * 2 + 256 * 2;
        for (int il = 0; il < 5; il++) {
            LayerMemoryPlan l = plan.layer(il);
            assertEquals(LayerMemoryPlan.Kind.KV, l.kind());
            assertEquals(0, l.kvGroup());
            assertEquals(1024 * rowBytes, l.totalBytes());
            assertEquals(1_048_576L, l.totalBytes());
        }
        assertEquals(1, plan.layer(5).kvGroup());
        assertEquals(8_388_608L, plan.layer(5).totalBytes());

        assertEquals(5 * 1_048_576L + 8_388_608L, plan.kvBytes());
        assertEquals(1024 * 4 + 8192 * 4, plan.positionBytes());
        assertEquals(13_668_352L, plan.totalBytes());
        assertEquals(0, plan.stateBytes());
        assertEquals(0, plan.sideCacheBytes());
        assertTotalsMatchLayers(plan);
        assertSame(RotationPlan.NONE, plan.layer(0).rotation());
    }

    @Test
    public void gemma3Q8RotationAndBytes() {
        MemoryConfig q8 = MemoryConfig.builder().typeK(KvDType.Q8_0).typeV(KvDType.Q8_0).build();
        MemoryPlan plan = MemoryPlanner.planForContext(gemma3(), q8, 8192);

        long rowBytes = 2 * (256 / 32 * 34);
        assertEquals(1024 * rowBytes, plan.layer(0).totalBytes());
        assertEquals(8192 * rowBytes, plan.layer(5).totalBytes());
        assertEquals(5 * 557_056L + 4_456_448L + 36_864L, plan.totalBytes());
        assertTotalsMatchLayers(plan);

        for (int il = 0; il < 6; il++) {
            RotationPlan r = plan.layer(il).rotation();
            assertEquals(256, r.nRotK());
            assertEquals(64, r.nRotV());
        }
    }

    @Test
    public void gemma3SwaFullKeepsFullSizeWindowCaches() {
        MemoryConfig cfg = MemoryConfig.builder().swaFull(true).build();
        MemoryPlan plan = MemoryPlanner.planForContext(gemma3(), cfg, 8192);
        assertEquals(8192, plan.layer(0).kv().rows());
        assertFalse(plan.layer(0).kv().ring());
        assertEquals(6 * 8_388_608L + 2 * 8192 * 4L, plan.totalBytes());
        assertTotalsMatchLayers(plan);
    }

    @Test
    public void ubatchIsCappedByContext() {
        MemoryPlan plan = MemoryPlanner.planForContext(gemma3(), f16(), 256);
        assertEquals(256, plan.ubatch());
        assertEquals(256, plan.layer(0).kv().rows());
        assertFalse(plan.layer(0).kv().ring());
    }

    // qwen35: three GDN layers (2 K heads, 4 V heads of 128, d_conv 4) and one full-attention layer (2 KV heads of 256).

    @Test
    public void qwen35HybridTrace() {
        MemoryPlan plan = MemoryPlanner.planForContext(qwen35(), f16(), 4096);

        assertEquals(MemoryKind.HYBRID, plan.kind());
        long convFloats = 3L * (4 * 128 + 2 * 2 * 128);
        long scanFloats = 128L * 128 * 4;
        assertEquals(3072, convFloats);
        assertEquals(65_536, scanFloats);
        for (int il = 0; il < 3; il++) {
            LayerMemoryPlan l = plan.layer(il);
            assertEquals(LayerMemoryPlan.Kind.RECURRENT, l.kind());
            assertNull(l.kv());
            assertEquals(2, l.states().size());
            assertEquals(4 * (convFloats + scanFloats), l.stateBytes());
            assertEquals(274_432L, l.totalBytes());
        }
        LayerMemoryPlan attn = plan.layer(3);
        assertEquals(LayerMemoryPlan.Kind.KV, attn.kind());
        assertEquals(4096L * (2 * 256 * 2) * 2, attn.kvBytes());
        assertEquals(8_388_608L, attn.kvBytes());

        assertEquals(1, plan.groups().size());
        assertEquals(823_296L, plan.stateBytes());
        assertEquals(16_384L, plan.positionBytes());
        assertEquals(9_228_288L, plan.totalBytes());
        assertTrue(plan.hasRecurrentState());
        assertTotalsMatchLayers(plan);
    }

    // deepseek32: two DSA layers, latent MLA rows of 512 + 64, lightning-indexer rows of 128 (F16 by default).

    @Test
    public void deepseek32DsaTrace() {
        MemoryPlan plan = MemoryPlanner.planForContext(deepseek32(), f16(), 4096);

        assertEquals(MemoryKind.DSA, plan.kind());
        for (int il = 0; il < 2; il++) {
            LayerMemoryPlan l = plan.layer(il);
            assertEquals(LayerMemoryPlan.Kind.KV, l.kind());
            KvLayout kv = l.kv();
            assertTrue(kv.vAliasesK());
            assertEquals(576, kv.widthK());
            assertEquals(512, kv.widthV());
            assertEquals(4096, kv.rows());
            assertEquals(4_718_592L, l.kvBytes());

            LayerMemoryPlan.IndexerPlan idx = l.indexer();
            assertEquals(KvDType.F16, idx.dtype());
            assertEquals(128, idx.width());
            assertEquals(4096, idx.rows());
            assertEquals(1_048_576L, idx.bytes());
            assertEquals(128, idx.rotation().nRotK());
            assertEquals(0, idx.rotation().nRotV());
            assertSame(RotationPlan.NONE, l.rotation());
        }
        assertEquals(9_437_184L, plan.kvBytes());
        assertEquals(2_097_152L, plan.sideCacheBytes());
        assertEquals(16_384L, plan.positionBytes());
        assertEquals(11_550_720L, plan.totalBytes());
        assertTotalsMatchLayers(plan);
    }

    // llama.cpp creates the indexer cache with the context's type_k (llama-kv-cache-dsa.cpp:51): the indexer rows
    // follow neuralfree.kv.type like the latent rows; the forced indexer rotation does not depend on the type.

    @Test
    public void deepseek32IndexerFollowsQ8() {
        MemoryConfig q8 = MemoryConfig.builder().typeK(KvDType.Q8_0).typeV(KvDType.Q8_0).build();
        MemoryPlan plan = MemoryPlanner.planForContext(deepseek32(), q8, 4096);
        for (int il = 0; il < 2; il++) {
            LayerMemoryPlan l = plan.layer(il);
            assertEquals(KvDType.Q8_0, l.kv().dtypeK());
            assertEquals(4096L * (576 / 32 * 34), l.kvBytes());
            assertEquals(KvDType.Q8_0, l.indexer().dtype());
            assertEquals(4096L * (128 / 32 * 34), l.indexer().bytes());
            assertEquals(557_056L, l.indexer().bytes());
            assertSame(RotationPlan.NONE, l.rotation());
            assertEquals(128, l.indexer().rotation().nRotK());
            assertEquals(0, l.indexer().rotation().nRotV());
        }
        assertEquals(2 * 557_056L, plan.sideCacheBytes());
        assertTotalsMatchLayers(plan);
    }

    @Test
    public void deepseek32IndexerFollowsF32() {
        MemoryConfig f32 = MemoryConfig.builder().typeK(KvDType.F32).typeV(KvDType.F32).build();
        MemoryPlan plan = MemoryPlanner.planForContext(deepseek32(), f32, 4096);
        for (int il = 0; il < 2; il++) {
            LayerMemoryPlan l = plan.layer(il);
            assertEquals(KvDType.F32, l.kv().dtypeK());
            assertEquals(4096L * 576 * 4, l.kvBytes());
            assertEquals(KvDType.F32, l.indexer().dtype());
            assertEquals(4096L * 128 * 4, l.indexer().bytes());
            assertEquals(2_097_152L, l.indexer().bytes());
            assertEquals(128, l.indexer().rotation().nRotK());
        }
        assertEquals(2 * 2_097_152L, plan.sideCacheBytes());
        assertTotalsMatchLayers(plan);
    }

    @Test
    public void deepseek32IndexerRotationIsForcedOnlyWithRotationDisabled() {
        MemoryConfig q8 = MemoryConfig.builder().typeK(KvDType.Q8_0).typeV(KvDType.Q8_0).kvRotation(false)
                .build();
        LayerMemoryPlan l = MemoryPlanner.planForContext(deepseek32(), q8, 4096).layer(0);
        assertEquals(KvDType.Q8_0, l.indexer().dtype());
        assertEquals(128, l.indexer().rotation().nRotK());
    }

    // deepseek4: one CSA layer (ratio 4), raw head 512, indexer head 128; the indexer (LID) cache holds
    // pad256(ceil(4096 / 4)) = 1024 rows of the K type (llama-kv-cache-dsv4.cpp:1308-1312).

    @Test
    public void deepseek4IndexerFollowsKvType() {
        KvDType[] types = {KvDType.F16, KvDType.F32, KvDType.Q8_0};
        long[] rowBytes = {128 * 2, 128 * 4, 128 / 32 * 34};
        for (int i = 0; i < types.length; i++) {
            MemoryConfig cfg = MemoryConfig.builder().typeK(types[i]).typeV(types[i]).build();
            MemoryPlan plan = MemoryPlanner.planForContext(deepseek4Csa(), cfg, 4096);
            LayerMemoryPlan l = plan.layer(0);
            LayerMemoryPlan.CompressedKvPlan lid = compressed(l, LayerMemoryPlan.CompressedKvPlan.Kind.INDEXER);
            LayerMemoryPlan.CompressedKvPlan csa = compressed(l, LayerMemoryPlan.CompressedKvPlan.Kind.CSA);
            assertEquals(types[i], lid.dtype());
            assertEquals(types[i], csa.dtype());
            assertEquals(128, lid.width());
            assertEquals(1024, lid.rows());
            assertEquals(1024 * rowBytes[i], lid.cacheBytes());
            assertEquals(2L * 8 * 256 * Float.BYTES, lid.stateBytes());
            assertTotalsMatchLayers(plan);
        }
    }

    /**
     * Rollback snapshots ({@code n_rs_seq}) add planes to every DSV4 compressor tensor (llama-kv-cache-dsv4.cpp:968-970),
     * at most one ubatch of them, and count in the plan's totals the context fit sizes from.
     */
    @Test
    public void deepseek4RollbackSnapshotsAddCompressorPlanes() {
        MemoryPlan plain = MemoryPlanner.planForContext(deepseek4Csa(), f16(), 4096);
        MemoryPlan three = MemoryPlanner.planForContext(deepseek4Csa(), f16().forRollback(3), 4096);
        assertEquals(0, plain.rollbackSnapshots());
        assertEquals(3, three.rollbackSnapshots());
        for (LayerMemoryPlan.CompressedKvPlan.Kind kind : new LayerMemoryPlan.CompressedKvPlan.Kind[]{
                LayerMemoryPlan.CompressedKvPlan.Kind.CSA, LayerMemoryPlan.CompressedKvPlan.Kind.INDEXER}) {
            LayerMemoryPlan.CompressedKvPlan p = compressed(plain.layer(0), kind);
            LayerMemoryPlan.CompressedKvPlan t = compressed(three.layer(0), kind);
            assertEquals(1, p.planes());
            assertEquals(4, t.planes());
            assertEquals(4 * p.stateBytes(), t.stateBytes());
            assertEquals(p.cacheBytes(), t.cacheBytes());
        }
        LayerMemoryPlan.CompressedKvPlan csa = compressed(three.layer(0), LayerMemoryPlan.CompressedKvPlan.Kind.CSA);
        assertEquals(2L * 4 * 8 * 1024 * Float.BYTES, csa.stateBytes());
        long extra = 3 * (compressed(plain.layer(0), LayerMemoryPlan.CompressedKvPlan.Kind.CSA).stateBytes()
                + compressed(plain.layer(0), LayerMemoryPlan.CompressedKvPlan.Kind.INDEXER).stateBytes());
        assertEquals(plain.totalBytes() + extra, three.totalBytes());
        assertTotalsMatchLayers(three);

        MemoryPlan clamped = MemoryPlanner.planForContext(deepseek4Csa(), f16().toBuilder().ubatch(8).build()
                .forRollback(64), 4096);
        assertEquals(8, clamped.rollbackSnapshots());
        MemoryPlan noCompressed = MemoryPlanner.planForContext(gemma3(), f16().forRollback(3), 4096);
        assertEquals(0, noCompressed.rollbackSnapshots());
    }

    /**
     * With rollback snapshots the trailing {@code 1 + s} tokens of a call stay in one step, as llama.cpp's
     * {@code split_equal} keeps {@code n_keep_tail} (src/llama-batch.cpp:714-765).
     */
    @Test
    public void stepTokensKeepTheRollbackTailInOneStep() {
        MemoryPlan plain = MemoryPlanner.planForContext(deepseek4Csa(), f16(), 4096);
        assertEquals(8, plain.stepTokens(10, 8));
        assertEquals(2, plain.stepTokens(2, 8));
        MemoryPlan three = MemoryPlanner.planForContext(deepseek4Csa(), f16().forRollback(3), 4096);
        assertEquals("10 = 6 + 4 keeps the 4-token tail whole", 6, three.stepTokens(10, 8));
        assertEquals(4, three.stepTokens(4, 8));
        assertEquals(8, three.stepTokens(12, 8));
        assertEquals(8, three.stepTokens(16, 8));
        assertEquals(8, three.stepTokens(13, 8));
        assertEquals(7, three.stepTokens(11, 8));
        assertEquals(5, three.stepTokens(9, 8));
        assertEquals(8, three.stepTokens(8, 8));
        assertEquals("a step no larger than the tail splits plainly", 4, three.stepTokens(6, 4));
        assertThrows(IllegalArgumentException.class, () -> three.stepTokens(0, 8));
        assertThrows(IllegalArgumentException.class, () -> three.stepTokens(3, 0));
    }

    @Test
    public void contradictingMlaCacheModeIsRejected() {
        MemoryConfig decompressed = MemoryConfig.builder().mlaCache(MemoryConfig.MlaCacheMode.DECOMPRESSED).build();
        assertThrows(IllegalArgumentException.class,
                () -> MemoryPlanner.planForContext(deepseek32(), decompressed, 4096));
    }

    // mamba2: d_inner 1536, d_state 128, one group, d_conv 4.

    @Test
    public void mamba2RecurrentTrace() {
        MemoryPlan plan = MemoryPlanner.planForContext(mamba2(), f16(), 4096);

        assertEquals(MemoryKind.RECURRENT, plan.kind());
        long convFloats = 3L * (1536 + 2 * 128);
        long scanFloats = 128L * 1536;
        for (int il = 0; il < 2; il++) {
            LayerMemoryPlan l = plan.layer(il);
            assertEquals(LayerMemoryPlan.Kind.RECURRENT, l.kind());
            assertEquals(4 * (convFloats + scanFloats), l.totalBytes());
            assertEquals(807_936L, l.totalBytes());
        }
        assertTrue(plan.groups().isEmpty());
        assertEquals(0, plan.kvBytes());
        assertEquals(0, plan.positionBytes());
        assertEquals(1_615_872L, plan.totalBytes());
        assertTotalsMatchLayers(plan);
    }

    @Test
    public void recurrentStateDoesNotGrowWithContext() {
        assertEquals(MemoryPlanner.planForContext(mamba2(), f16(), 256).totalBytes(),
                MemoryPlanner.planForContext(mamba2(), f16(), 65_536).totalBytes());
    }

    @Test
    public void sharedKvLayerReadsItsOwner() {
        MemoryPlan plan = MemoryPlanner.planForContext(gemma3Shared(), f16(), 8192);

        LayerMemoryPlan shared = plan.layer(2);
        assertEquals(LayerMemoryPlan.Kind.SHARED_KV, shared.kind());
        assertEquals(0, shared.sharedFrom().layer());
        assertFalse(shared.sharedFrom().target());
        assertEquals(plan.layer(0).kvGroup(), shared.kvGroup());
        assertSame(plan.layer(0).kv(), shared.kv());
        assertEquals(0, shared.totalBytes());
        assertEquals(1_048_576L + 8_388_608L + 1024 * 4 + 8192 * 4, plan.totalBytes());
        assertTotalsMatchLayers(plan);
    }

    /** llama-context.cpp:4074-4077: MLA (latent) and DeepSeek-V4 caches need equal K and V types. */
    @Test
    public void sharedRowCachesRejectDifferentKAndVTypes() {
        MemoryConfig mixed = MemoryConfig.builder().typeK(KvDType.Q4_0).typeV(KvDType.F16).build();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MemoryPlanner.planForContext(deepseek2(true), mixed, 1024));
        assertTrue(e.getMessage(), e.getMessage().contains("different K (q4_0) and V (f16) cache types"));
        assertThrows(IllegalArgumentException.class, () -> MemoryPlanner.planForContext(deepseek4Csa(), mixed, 4096));
        MemoryPlan plan = MemoryPlanner.planForContext(llamaHead(64, 1), MemoryConfig.builder().typeK(KvDType.Q8_0)
                .typeV(KvDType.F32).build(), 1024);
        assertEquals(KvDType.Q8_0, plan.layer(0).kv().dtypeK());
        assertEquals(KvDType.F32, plan.layer(0).kv().dtypeV());
    }

    /** Every block type keeps whole-block heads and sizes its rows by the ggml block bytes. */
    @Test
    public void blockTypesSizeRowsByTheirBlocks() {
        KvDType[] types = {KvDType.Q4_0, KvDType.Q4_1, KvDType.IQ4_NL, KvDType.Q5_0, KvDType.Q5_1, KvDType.BF16};
        for (KvDType t : types) {
            MemoryConfig cfg = MemoryConfig.builder().typeK(t).typeV(t).build();
            MemoryPlan plan = MemoryPlanner.planForContext(deepseek4Csa(), cfg, 4096);
            LayerMemoryPlan.CompressedKvPlan lid = compressed(plan.layer(0),
                    LayerMemoryPlan.CompressedKvPlan.Kind.INDEXER);
            assertEquals(t, lid.dtype());
            assertEquals(1024 * t.rowBytes(128), lid.cacheBytes());
            assertEquals(t, plan.layer(0).kv().dtypeK());
            assertTotalsMatchLayers(plan);
        }
    }

    /** llama-context.cpp:4090-4110: a quantized K or V type whose block size does not divide the head is refused. */
    @Test
    public void blockTypesThatDoNotDivideTheHeadAreRefused() {
        for (KvDType t : KvDType.all()) {
            if (!t.quantized()) {
                continue;
            }
            MemoryConfig k = MemoryConfig.builder().typeK(t).typeV(KvDType.F16).build();
            IllegalArgumentException ek = assertThrows(IllegalArgumentException.class,
                    () -> MemoryPlanner.planForContext(llamaHead(80, 1), k, 1024));
            assertTrue(ek.getMessage(), ek.getMessage().contains("K cache type "
                    + t.name().toLowerCase(Locale.ROOT) + " with block size 32 does not divide"));
            assertTrue(ek.getMessage(), ek.getMessage().contains(MemoryConfig.PROP_KV_TYPE_K));
            MemoryConfig v = MemoryConfig.builder().typeK(KvDType.F16).typeV(t).build();
            IllegalArgumentException ev = assertThrows(IllegalArgumentException.class,
                    () -> MemoryPlanner.planForContext(llamaHead(80, 2), v, 1024));
            assertTrue(ev.getMessage(), ev.getMessage().contains("V cache type"));
            assertTrue(ev.getMessage(), ev.getMessage().contains(MemoryConfig.PROP_KV_TYPE_V));
        }
        for (KvDType t : new KvDType[]{KvDType.F32, KvDType.F16, KvDType.BF16}) {
            MemoryPlan plan = MemoryPlanner.planForContext(llamaHead(80, 2),
                    MemoryConfig.builder().typeK(t).typeV(t).build(), 1024);
            KvLayout kv = plan.layer(0).kv();
            assertEquals(160, kv.widthK());
            assertEquals(t, kv.dtypeK());
            assertEquals(t, kv.dtypeV());
            assertEquals(1024L * 2 * t.rowBytes(160), plan.layer(0).kvBytes());
        }
    }

    @Test
    public void lfm2DecisionModelHasNoMemory() {
        ModelHparamsBuilder hb = hparams(Arch.LFM2, 2, 4096).causal(false).nLayerDecision(1);
        attentionHeads(hb, 2, 4, 2, 64);
        ModelHparams hp = hb.build();
        List<LayerSpec> layers = new ArrayList<>();
        for (int il = 0; il < 2; il++) {
            layers.add(attentionLayer(il, 64, attention(4, 2, 64).causal(false).kvOwnerLayer(il).build()));
        }
        MemoryPlan plan = MemoryPlanner.planForContext(model(hp, 64, layers, null), f16(), 4096);
        assertEquals(MemoryKind.NONE, plan.kind());
        assertEquals(0, plan.totalBytes());
        for (LayerMemoryPlan l : plan.layers()) {
            assertEquals(LayerMemoryPlan.Kind.NONE, l.kind());
        }
    }

    @Test
    public void memorylessModelKindHasNoMemory() {
        MemoryPlan plan = MemoryPlanner.planForContext(gemma3(MemoryKind.NONE), f16(), 8192);
        assertEquals(MemoryKind.NONE, plan.kind());
        assertTrue(plan.groups().isEmpty());
        assertEquals(0, plan.totalBytes());
    }

    @Test
    public void rejectsNonPositiveContext() {
        assertThrows(IllegalArgumentException.class, () -> MemoryPlanner.planForContext(gemma3(), f16(), 0));
    }

    // MLA: llama.cpp reads latent caches through build_attn_inp_k, which never rotates (llama-graph.cpp:3294-3376);
    // decompressed caches go through build_attn_inp_kv and are rotated like any other cache.

    @Test
    public void latentMlaCacheIsNeverRotated() {
        MemoryConfig q8 = MemoryConfig.builder().typeK(KvDType.Q8_0).typeV(KvDType.Q8_0).build();
        MemoryPlan plan = MemoryPlanner.planForContext(deepseek2(true), q8, 4096);
        for (int il = 0; il < 2; il++) {
            LayerMemoryPlan l = plan.layer(il);
            assertTrue(l.kv().vAliasesK());
            assertEquals(KvDType.Q8_0, l.kv().dtypeK());
            assertEquals(4096L * (576 / 32 * 34), l.kvBytes());
            assertSame(RotationPlan.NONE, l.rotation());
        }
        assertTotalsMatchLayers(plan);
    }

    @Test
    public void decompressedMlaCacheIsRotated() {
        MemoryConfig q8 = MemoryConfig.builder().typeK(KvDType.Q8_0).typeV(KvDType.Q8_0).build();
        MemoryPlan plan = MemoryPlanner.planForContext(deepseek2(false), q8, 4096);
        for (int il = 0; il < 2; il++) {
            LayerMemoryPlan l = plan.layer(il);
            assertFalse(l.kv().vAliasesK());
            assertEquals(64, l.rotation().nRotK());
            assertEquals(64, l.rotation().nRotV());
        }
        assertTotalsMatchLayers(plan);
    }

    // NextN: two trunk layers and one NextN layer, one F16 KV head of 128 each: 4096 * 512 bytes per layer.

    @Test
    public void nextnLayersKeepNoMemoryInTheMainContext() {
        MemoryPlan plan = MemoryPlanner.planForContext(withNextn(), f16(), 4096);
        assertEquals(3, plan.layers().size());
        assertEquals(LayerMemoryPlan.Kind.KV, plan.layer(0).kind());
        assertEquals(LayerMemoryPlan.Kind.KV, plan.layer(1).kind());
        assertEquals(LayerMemoryPlan.Kind.NONE, plan.layer(2).kind());
        assertNull(plan.layer(2).kv());
        assertEquals(1, plan.groups().size());
        assertEquals(2L * 4096 * 512, plan.kvBytes());
        assertEquals(2L * 4096 * 512 + 4096L * 4, plan.totalBytes());
        assertEquals(KvDType.F16, plan.config().typeK());
        assertTotalsMatchLayers(plan);
    }

    @Test
    public void nextnContextPlansOnlyTheNextnLayers() {
        ModelSpec spec = withNextn();
        MemoryPlan target = MemoryPlanner.planForContext(spec, f16(), 4096);
        for (MemoryPlan plan : new MemoryPlan[]{MemoryPlanner.planNextn(spec, target),
                MemoryPlanner.planNextn(spec, target, 2)}) {
            assertEquals(3, plan.layers().size());
            assertEquals(LayerMemoryPlan.Kind.NONE, plan.layer(0).kind());
            assertEquals(LayerMemoryPlan.Kind.NONE, plan.layer(1).kind());
            LayerMemoryPlan nextn = plan.layer(2);
            assertEquals(LayerMemoryPlan.Kind.KV, nextn.kind());
            assertEquals(0, nextn.kvGroup());
            assertEquals(4096, nextn.kv().rows());
            assertEquals(1, plan.groups().size());
            assertEquals(target.nCtx(), plan.nCtx());
            assertEquals(target.ubatch(), plan.ubatch());
            assertEquals(4096L * 512 + 4096L * 4, plan.totalBytes());
            assertTotalsMatchLayers(plan);
        }
        assertEquals(4096L * 512 + 4096L * 4, MemoryPlanner.nextnBytes(spec, target));
        assertEquals(0L, MemoryPlanner.nextnBytes(gemma3(), MemoryPlanner.planForContext(gemma3(), f16(), 4096)));
        assertThrows(IllegalArgumentException.class, () -> MemoryPlanner.planNextn(spec, target, 1));
        assertThrows(IllegalArgumentException.class,
                () -> MemoryPlanner.planNextn(gemma3(), MemoryPlanner.planForContext(gemma3(), f16(), 4096)));
        MemoryPlan handBuilt = MemoryPlan.builder(MemoryKind.KV, 4096, 512)
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.NONE).build())
                .layer(LayerMemoryPlan.builder(1, LayerMemoryPlan.Kind.NONE).build())
                .layer(LayerMemoryPlan.builder(2, LayerMemoryPlan.Kind.NONE).build())
                .build();
        assertThrows(IllegalArgumentException.class, () -> MemoryPlanner.planNextn(spec, handBuilt));
    }

    @Test
    public void nextnContextRunsWithTheDraftCacheTypeWhateverTheTargetUses() {
        ModelSpec spec = withNextn();
        MemoryConfig q8 = MemoryConfig.builder().typeK(KvDType.Q8_0).typeV(KvDType.Q8_0).stateSnapshots(true)
                .nextnMemory(true).build();
        MemoryPlan target = MemoryPlanner.planForContext(spec, q8, 4096);
        assertEquals(KvDType.Q8_0, target.layer(0).kv().dtypeK());
        MemoryPlan plan = MemoryPlanner.planNextn(spec, target, 2);
        LayerMemoryPlan nextn = plan.layer(2);
        assertEquals(MemoryConfig.DRAFT_KV_TYPE, nextn.kv().dtypeK());
        assertEquals(MemoryConfig.DRAFT_KV_TYPE, nextn.kv().dtypeV());
        assertSame(RotationPlan.NONE, nextn.rotation());
        assertEquals(MemoryConfig.DRAFT_KV_TYPE, plan.config().typeK());
        assertFalse(plan.config().stateSnapshots());
        assertFalse(plan.config().nextnMemory());
        assertEquals(4096L * 512 + 4096L * 4, plan.totalBytes());
        assertEquals(plan.totalBytes(), MemoryPlanner.nextnBytes(spec, target));
        assertEquals(plan.totalBytes(), MemoryPlanner.nextnBytes(spec, q8, 4096, target.ubatch()));
        assertEquals(target.totalBytes() + plan.totalBytes(), ContextFitter.requiredBytes(spec, q8, 4096));
        assertEquals(plan.totalBytes(), ContextFitter.reservedBytes(spec, q8, target,
                ContextFitter.Companion.NONE));
    }

    @Test
    public void nextnContextOfAnMtpOnlyFileFollowsATrunkOnlyTarget() {
        ModelSpec sidecar = withNextn();
        MemoryPlan target = MemoryPlanner.planForContext(trunkOnly(), f16(), 4096);
        assertEquals(2, target.layers().size());
        MemoryPlan plan = MemoryPlanner.planNextn(sidecar, target, 2);
        assertEquals(3, plan.layers().size());
        assertEquals(LayerMemoryPlan.Kind.KV, plan.layer(2).kind());
        assertEquals(4096L * 512 + 4096L * 4, plan.totalBytes());
        assertEquals(plan.totalBytes(), MemoryPlanner.nextnBytes(sidecar, f16(), 4096, target.ubatch()));
        assertThrows(IllegalArgumentException.class, () -> MemoryPlanner.nextnBytes(gemma3(), f16(), 4096, 512));
        assertThrows(IllegalArgumentException.class, () -> MemoryPlanner.nextnBytes(sidecar, f16(), 0, 512));
    }

    // Recurrent-state snapshots: one shadow of every state tensor, allocated with the memory.

    @Test
    public void stateSnapshotsReserveASecondCopyOfTheRecurrentState() {
        MemoryConfig snapshots = MemoryConfig.builder().stateSnapshots(true).build();
        MemoryPlan plain = MemoryPlanner.planForContext(mamba2(), f16(), 4096);
        MemoryPlan reserved = MemoryPlanner.planForContext(mamba2(), snapshots, 4096);
        assertFalse(plain.snapshotsReserved());
        assertEquals(0L, plain.snapshotBytes());
        assertTrue(reserved.snapshotsReserved());
        assertEquals(reserved.stateBytes(), reserved.snapshotBytes());
        assertEquals(2 * 1_615_872L, reserved.totalBytes());
        assertTotalsMatchLayers(reserved);

        MemoryPlan noState = MemoryPlanner.planForContext(gemma3(), snapshots, 4096);
        assertFalse(noState.snapshotsReserved());
        assertEquals(0L, noState.snapshotBytes());
        MemoryPlan compressed = MemoryPlanner.planForContext(deepseek4Csa(), snapshots, 4096);
        assertFalse(compressed.snapshotsReserved());
    }

    static void assertTotalsMatchLayers(MemoryPlan plan) {
        long layers = 0;
        for (LayerMemoryPlan l : plan.layers()) {
            layers += l.totalBytes();
        }
        long positions = 0;
        for (MemoryPlan.KvGroup g : plan.groups()) {
            positions += (long) g.rows() * Integer.BYTES;
        }
        assertEquals(layers + plan.snapshotBytes() + positions, plan.totalBytes());
        assertEquals(plan.kvBytes() + plan.stateBytes() + plan.snapshotBytes() + plan.sideCacheBytes()
                + plan.positionBytes(), plan.totalBytes());
    }

    // ---- fixtures ----

    static ModelSpec gemma3() {
        return gemma3(null);
    }

    private static ModelSpec gemma3(MemoryKind override) {
        int nLayer = 6;
        ModelHparamsBuilder hb = hparams(Arch.GEMMA3, nLayer, GEMMA3_TRAIN).swaType(SwaType.STANDARD).nSwa(512)
                .headKSwa(256).headVSwa(256);
        attentionHeads(hb, nLayer, 4, 1, 256);
        List<LayerSpec> layers = new ArrayList<>();
        for (int il = 0; il < nLayer; il++) {
            boolean swa = il < 5;
            hb.isSwa(il, swa);
            layers.add(attentionLayer(il, 1152, swaAttention(swa, il).build()));
        }
        return model(hb.build(), 1152, layers, override);
    }

    private static ModelSpec gemma3Shared() {
        ModelHparamsBuilder hb = hparams(Arch.GEMMA3, 3, GEMMA3_TRAIN).swaType(SwaType.STANDARD).nSwa(512)
                .headKSwa(256).headVSwa(256);
        attentionHeads(hb, 3, 4, 1, 256);
        hb.isSwa(0, true).isSwa(1, false).isSwa(2, true);
        List<LayerSpec> layers = new ArrayList<>();
        layers.add(attentionLayer(0, 1152, swaAttention(true, 0).build()));
        layers.add(attentionLayer(1, 1152, swaAttention(false, 1).build()));
        layers.add(attentionLayer(2, 1152, swaAttention(true, 0).build()));
        return model(hb.build(), 1152, layers, null);
    }

    private static AttentionSpec.Builder swaAttention(boolean swa, int owner) {
        AttentionSpec.Builder a = attention(4, 1, 256).kvOwnerLayer(owner);
        return swa ? a.swaLayer(true).swaType(SwaType.STANDARD).nSwa(512) : a;
    }

    static ModelSpec qwen35() {
        ModelHparamsBuilder hb = hparams(Arch.QWEN35, 4, 32768);
        attentionHeads(hb, 4, 8, 2, 256);
        List<LayerSpec> layers = new ArrayList<>();
        for (int il = 0; il < 3; il++) {
            hb.isRecurrent(il, true);
            RecurrentSpec gdn = RecurrentSpec.builder(MixerKind.GDN).gdnHeads(2, 4, 128, 128).dConv(4)
                    .headMap(GdnHeadMap.TILED).build();
            layers.add(base(il, MixerKind.GDN, 1024).recurrent(gdn).build());
        }
        hb.isRecurrent(3, false);
        layers.add(attentionLayer(3, 1024, attention(8, 2, 256).kvOwnerLayer(3).build()));
        return model(hb.build(), 1024, layers, null);
    }

    static ModelSpec deepseek32() {
        ModelHparamsBuilder hb = hparams(Arch.DEEPSEEK32, 2, 16384).headKFull(576).headVFull(512).indexerHeadSize(128)
                .nLoraKv(512);
        attentionHeads(hb, 2, 16, 1, 576);
        List<LayerSpec> layers = new ArrayList<>();
        for (int il = 0; il < 2; il++) {
            MlaSpec mla = MlaSpec.builder().nHead(16).kvLoraRank(512).headKNope(128).headKRope(64).headV(128)
                    .absorbed(true).latentCache(true).ropeApplied(false).kqScale(KQ_SCALE).build();
            DsaSpec dsa = DsaSpec.builder().indexerHeads(64).indexerHeadDim(128).indexerRopeDim(64).topK(2048)
                    .sourceLayer(il).defaultWeightScale().build();
            layers.add(base(il, MixerKind.DSA_MLA, 2048).mla(mla).dsa(dsa).build());
        }
        return model(hb.build(), 2048, layers, null);
    }

    /** deepseek2: two MLA layers (16 heads, kv_lora 512, rope 64, nope 128, V 128), latent or decompressed. */
    static ModelSpec deepseek2(boolean latent) {
        ModelHparamsBuilder hb = hparams(Arch.DEEPSEEK2, 2, 16384).nLoraKv(512);
        attentionHeads(hb, 2, 16, latent ? 1 : 16, latent ? 576 : 192);
        if (!latent) {
            hb.headVFull(128);
        }
        List<LayerSpec> layers = new ArrayList<>();
        for (int il = 0; il < 2; il++) {
            MlaSpec mla = MlaSpec.builder().nHead(16).kvLoraRank(512).headKNope(128).headKRope(64).headV(128)
                    .absorbed(latent).latentCache(latent).ropeApplied(false).kqScale(KQ_SCALE).build();
            layers.add(base(il, MixerKind.MLA, 2048).mla(mla).build());
        }
        return model(hb.build(), 2048, layers, null);
    }

    /** Two trunk attention layers plus one NextN attention layer (glm4moe), one KV head of 128. */
    /** {@link #withNextn()} without its NextN block, as a target converted without its MTP layers. */
    static ModelSpec trunkOnly() {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.GLM4_MOE)).layerCounts(2, 0)
                .nCtxTrain(8192);
        attentionHeads(hb, 2, 4, 1, 128);
        ModelSpec.Builder b = ModelSpec.builder(hb.build(), TensorCatalog.of(Arch.GLM4_MOE.ggufName(),
                        Collections.emptyList()))
                .schedule(LayerSchedule.linear(2))
                .embedding(EmbeddingSpec.builder(null, 512).build())
                .head(HeadSpec.builder().build());
        for (int il = 0; il < 2; il++) {
            b.layer(attentionLayer(il, 512, attention(4, 1, 128).kvOwnerLayer(il).build()));
        }
        return b.build();
    }

    static ModelSpec withNextn() {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.GLM4_MOE)).layerCounts(3, 1)
                .nCtxTrain(8192);
        attentionHeads(hb, 3, 4, 1, 128);
        List<LayerSpec> trunk = new ArrayList<>();
        for (int il = 0; il < 2; il++) {
            trunk.add(attentionLayer(il, 512, attention(4, 1, 128).kvOwnerLayer(il).build()));
        }
        ModelSpec.Builder b = ModelSpec.builder(hb.build(), TensorCatalog.of(Arch.GLM4_MOE.ggufName(),
                        Collections.emptyList()))
                .schedule(LayerSchedule.linear(trunk.size()))
                .embedding(EmbeddingSpec.builder(null, 512).build())
                .head(HeadSpec.builder().build());
        for (LayerSpec l : trunk) {
            b.layer(l);
        }
        b.nextnLayer(attentionLayer(2, 512, attention(4, 1, 128).kvOwnerLayer(2).build()));
        return b.build();
    }

    static ModelSpec deepseek4Csa() {
        ModelHparamsBuilder hb = hparams(Arch.DEEPSEEK4, 1, 16384).swaType(SwaType.STANDARD).nSwa(128)
                .indexerHeadSize(128);
        attentionHeads(hb, 1, 8, 1, 512);
        DsaSpec idx = DsaSpec.builder().indexerHeads(8).indexerHeadDim(128).indexerRopeDim(64).topK(512).full(true)
                .sourceLayer(0).hadamard(true).defaultWeightScale().build();
        Dsv4Spec v4 = Dsv4Spec.builder().nHead(8).compressRatio(Dsv4Spec.RATIO_CSA).window(128).headDim(512)
                .ropeDim(64).qLoraRank(256).oGroupCount(2).oLoraRank(64).compressRopeBase(160_000f).indexer(idx)
                .kqScale(KQ_SCALE).build();
        List<LayerSpec> layers = Collections.singletonList(base(0, MixerKind.DSV4, 1024).dsv4(v4).build());
        return model(hb.build(), 1024, layers, null);
    }

    private static LayerMemoryPlan.CompressedKvPlan compressed(LayerMemoryPlan l,
                                                               LayerMemoryPlan.CompressedKvPlan.Kind kind) {
        for (LayerMemoryPlan.CompressedKvPlan c : l.compressed()) {
            if (c.kind() == kind) {
                return c;
            }
        }
        throw new AssertionError("layer " + l.layer() + " has no " + kind + " compressed cache");
    }

    static ModelSpec mamba2() {
        ModelHparamsBuilder hb = hparams(Arch.MAMBA2, 2, 1_048_576).ssmDConv(4).ssmDInner(1536).ssmDState(128)
                .ssmNGroup(1);
        List<LayerSpec> layers = new ArrayList<>();
        for (int il = 0; il < 2; il++) {
            hb.isRecurrent(il, true);
            RecurrentSpec m2 = RecurrentSpec.builder(MixerKind.MAMBA2).nEmbd(768).dConv(4).dInner(1536).dState(128)
                    .nGroup(1).nHead(24).headDim(64).build();
            layers.add(base(il, MixerKind.MAMBA2, 768).recurrent(m2).build());
        }
        return model(hb.build(), 768, layers, null);
    }

    private static ModelSpec llamaHead(int headDim, int nHeadKv) {
        ModelHparamsBuilder hb = hparams(Arch.LLAMA, 1, 4096).headKFull(headDim).headVFull(headDim);
        attentionHeads(hb, 1, 8, nHeadKv, headDim);
        List<LayerSpec> layers = Collections.singletonList(
                attentionLayer(0, 8 * headDim, attention(8, nHeadKv, headDim).kvOwnerLayer(0).build()));
        return model(hb.build(), 8 * headDim, layers, null);
    }

    private static ModelHparamsBuilder hparams(Arch arch, int nLayer, int nCtxTrain) {
        return new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(nLayer, 0).nCtxTrain(nCtxTrain);
    }

    private static void attentionHeads(ModelHparamsBuilder hb, int nLayer, int nHead, int nHeadKv, int headDim) {
        hb.headKFull(headDim).headVFull(headDim);
        for (int il = 0; il < nLayer; il++) {
            hb.nHead(il, nHead).nHeadKv(il, nHeadKv);
        }
    }

    private static AttentionSpec.Builder attention(int nHead, int nHeadKv, int headDim) {
        return AttentionSpec.builder().nHead(nHead).nHeadKv(nHeadKv).headDimK(headDim).headDimV(headDim)
                .noRope(true).nRot(0).kqScale(KQ_SCALE).kvOwnerLayer(0);
    }

    private static LayerSpec.Builder base(int il, MixerKind mixer, int nEmbd) {
        return LayerSpec.builder(il, mixer, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, nEmbd).build());
    }

    private static LayerSpec attentionLayer(int il, int nEmbd, AttentionSpec attention) {
        return base(il, MixerKind.ATTENTION, nEmbd).attention(attention).build();
    }

    private static ModelSpec model(ModelHparams hp, int nEmbd, List<LayerSpec> layers, MemoryKind override) {
        ModelSpec.Builder b = ModelSpec.builder(hp, TensorCatalog.of(hp.arch().ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(layers.size()))
                .embedding(EmbeddingSpec.builder(null, nEmbd).build())
                .head(HeadSpec.builder().build());
        for (LayerSpec l : layers) {
            b.layer(l);
        }
        if (override != null) {
            b.memoryKind(override);
        }
        return b.build();
    }
}
