package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.CpuMixerRig;
import com.eonmux.neuralfree.gguf.testutil.PagedMixerCheck;
import com.eonmux.neuralfree.gguf.testutil.RandomGguf;
import org.junit.Test;

import java.io.IOException;
import java.util.function.Function;

/**
 * The paged multi-sequence steps of the CPU QSA k-pool mixer ({@code CpuQsaMixer.batch}) for both variants: the
 * qwen4exp gated GQA layer with mean pools (a full indexer layer and a shared layer reading its selection), and the
 * glm5-next latent MLA layer with gated pools. Two sequences of 13 and 10 positions in 4-position pools with 2
 * selected pools and the tail pool run as paged two-segment steps of uneven segments, their indexer rows and K/V at
 * the step's cells, every segment pooling, scoring and gathering over its own sequence's cells; every row equals
 * the same rows run one sequence at a time on contiguous memory. A non-causal step keeps the two sequences apart.
 */
public class CpuQsaMixerBatchTest {

    private static final int N_EMBD = 32;
    private static final int N_HEAD = 4;
    private static final int N_KV = 2;
    private static final int HD = 16;
    private static final int Q_DIM = N_HEAD * HD;
    private static final int K_DIM = N_KV * HD;
    private static final int IDX = 16;
    private static final int IDX_HEADS = 2;
    private static final int N_ROT = 8;
    private static final int KPOOL = 4;
    private static final int TOP_K = 8;
    private static final int Q_LORA = 16;
    private static final int KV_LORA = 16;
    private static final int NOPE = 16;
    private static final int HEAD_V = 16;
    private static final int N_CTX = 32;
    private static final int UBATCH = 16;
    private static final float TOL = 1e-4f;
    private static final int[] CHUNKS_A = {6, 6, 1};
    private static final int[] CHUNKS_B = {3, 6, 1};

    // ------------------------------------------------------------------ qwen4exp gated GQA, mean pools

    private static RandomGguf gqaWeights() throws IOException {
        RandomGguf.Builder b = RandomGguf.builder(Arch.QWEN4EXP.ggufName(), 7);
        for (int il = 0; il < 2; il++) {
            String blk = "blk." + il + ".";
            b.random(blk + "attn_q.weight", 0.3f, N_EMBD, 2 * Q_DIM)
                    .random(blk + "attn_k.weight", 0.3f, N_EMBD, K_DIM)
                    .random(blk + "attn_v.weight", 0.3f, N_EMBD, K_DIM)
                    .random(blk + "attn_output.weight", 0.3f, Q_DIM, N_EMBD)
                    .constant(blk + "attn_q_norm.weight", 1f, HD)
                    .constant(blk + "attn_k_norm.weight", 1f, HD);
        }
        return b.random("blk.0.indexer.q_proj.weight", 0.5f, N_EMBD, IDX_HEADS * IDX)
                .random("blk.0.indexer.k_proj.weight", 0.5f, N_EMBD, IDX)
                .constant("blk.0.indexer.q_norm.weight", 1f, IDX)
                .constant("blk.0.indexer.k_norm.weight", 1f, IDX)
                .open();
    }

    private static LayerSpec gqaLayer(int il) {
        AttentionSpec a = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_KV).headDimK(HD).headDimV(HD).nRot(N_ROT)
                .causal(true).qkNorm(QkNormKind.RMS_HEAD).gate(AttnGateKind.Q_INTERLEAVED).kqScale(0.25f)
                .kvOwnerLayer(il).build();
        DsaSpec dsa = DsaSpec.builder().indexerHeads(IDX_HEADS).indexerHeadDim(IDX).indexerRopeDim(N_ROT)
                .ropeType(RopeType.NEOX).topK(TOP_K).full(il == 0).sourceLayer(0)
                .kpool(KPOOL, MsaKernels.KPOOL_ROW_MEAN, false, true).weightScale(0.25f).build();
        return LayerSpec.builder(il, MixerKind.MSA, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).qkNorm(a.qkNorm()).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .attention(a)
                .dsa(dsa)
                .rope(RopeSpec.builder(RopeType.NEOX).nDims(N_ROT).build())
                .build();
    }

    // ------------------------------------------------------------------ glm5-next latent MLA, gated pools

    private static RandomGguf mlaWeights() throws IOException {
        return RandomGguf.builder(Arch.GLM5_NEXT.ggufName(), 9)
                .random("blk.0.attn_q_a.weight", 0.3f, N_EMBD, Q_LORA)
                .constant("blk.0.attn_q_a_norm.weight", 1f, Q_LORA)
                .random("blk.0.attn_q_b.weight", 0.3f, Q_LORA, N_HEAD * NOPE)
                .random("blk.0.attn_kv_a_mqa.weight", 0.3f, N_EMBD, KV_LORA)
                .constant("blk.0.attn_kv_a_norm.weight", 1f, KV_LORA)
                .random("blk.0.attn_k_b.weight", 0.3f, NOPE, KV_LORA, N_HEAD)
                .random("blk.0.attn_v_b.weight", 0.3f, KV_LORA, HEAD_V, N_HEAD)
                .random("blk.0.attn_output.weight", 0.3f, N_HEAD * HEAD_V, N_EMBD)
                .random("blk.0.indexer.attn_q_b.weight", 0.5f, Q_LORA, IDX_HEADS * IDX)
                .random("blk.0.indexer.attn_k.weight", 0.5f, N_EMBD, IDX)
                .constant("blk.0.indexer.k_norm.weight", 1f, IDX)
                .constant("blk.0.indexer.k_norm.bias", 0f, IDX)
                .random("blk.0.indexer_compressor_gate.weight", 0.5f, N_EMBD, IDX)
                .random("blk.0.indexer_compressor_ape.weight", 0.5f, IDX * KPOOL)
                .random("blk.0.indexer.proj.weight", 0.5f, N_EMBD, IDX_HEADS)
                .open();
    }

    private static LayerSpec mlaLayer() {
        MlaSpec mla = MlaSpec.builder().nHead(N_HEAD).qLoraRank(Q_LORA).kvLoraRank(KV_LORA).headKNope(NOPE)
                .headKRope(0).headV(HEAD_V).absorbed(true).latentCache(true).ropeApplied(false).kqScale(0.25f)
                .build();
        AttentionSpec a = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD).headDimK(NOPE).headDimV(HEAD_V)
                .nRot(0).noRope(true).qkNorm(QkNormKind.NONE).kqScale(0.25f).kvOwnerLayer(0).build();
        DsaSpec dsa = DsaSpec.builder().indexerHeads(IDX_HEADS).indexerHeadDim(IDX).indexerRopeDim(0)
                .ropeType(RopeType.NONE).topK(TOP_K).full(true).sourceLayer(0)
                .kpool(KPOOL, MsaKernels.KPOOL_ROW_GATED, false, true).weightScale(0.25f).build();
        return LayerSpec.builder(0, MixerKind.MSA, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .mla(mla)
                .attention(a)
                .dsa(dsa)
                .build();
    }

    // ------------------------------------------------------------------ fixtures

    private static ModelSpec spec(Arch arch, TensorCatalog catalog, LayerSpec... layers) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(layers.length, 0)
                .nCtxTrain(4096).nEmbd(N_EMBD).normRmsEps(1e-6f).normEps(1e-6f);
        ModelSpec.Builder b = ModelSpec.builder(hb.build(), catalog)
                .schedule(LayerSchedule.linear(layers.length))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build());
        for (LayerSpec l : layers) {
            b.layer(l);
        }
        return b.build();
    }

    private static Function<Boolean, CpuMixerRig> rigs(RandomGguf w, ModelSpec spec) {
        return paged -> {
            MemoryConfig.Builder cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(KvDType.F32).typeV(KvDType.F32);
            if (paged) {
                cfg = cfg.parallel(2);
            }
            return CpuMixerRig.open(spec, w.weights(), cfg.build(), N_CTX, N_EMBD,
                    layer -> MsaBlocks.cpuMixers().get(0).create(layer));
        };
    }

    @Test
    public void gqaPagedStepsOfTwoSequencesMatchTheContiguousRuns() throws IOException {
        try (RandomGguf w = gqaWeights()) {
            ModelSpec spec = spec(Arch.QWEN4EXP, w.catalog(), gqaLayer(0), gqaLayer(1));
            PagedMixerCheck.assertBatchMatchesSequences(rigs(w, spec), N_EMBD, 2, CHUNKS_A, CHUNKS_B, TOL);
        }
    }

    @Test
    public void gqaNonCausalStepKeepsTheSequencesApart() throws IOException {
        try (RandomGguf w = gqaWeights()) {
            ModelSpec spec = spec(Arch.QWEN4EXP, w.catalog(), gqaLayer(0), gqaLayer(1));
            PagedMixerCheck.assertNonCausalBatchKeepsSequencesApart(rigs(w, spec), N_EMBD, 2, 9, 6, TOL);
        }
    }

    @Test
    public void mlaPagedStepsOfTwoSequencesMatchTheContiguousRuns() throws IOException {
        try (RandomGguf w = mlaWeights()) {
            ModelSpec spec = spec(Arch.GLM5_NEXT, w.catalog(), mlaLayer());
            PagedMixerCheck.assertBatchMatchesSequences(rigs(w, spec), N_EMBD, 1, CHUNKS_A, CHUNKS_B, TOL);
        }
    }

    @Test
    public void mlaNonCausalStepKeepsTheSequencesApart() throws IOException {
        try (RandomGguf w = mlaWeights()) {
            ModelSpec spec = spec(Arch.GLM5_NEXT, w.catalog(), mlaLayer());
            PagedMixerCheck.assertNonCausalBatchKeepsSequencesApart(rigs(w, spec), N_EMBD, 1, 9, 6, TOL);
        }
    }
}
