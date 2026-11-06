package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
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
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MsaSpec;
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
 * The paged multi-sequence steps of the CPU MiniMax-M3 sparse mixer ({@code CpuMsaMixer.batch}): two sequences of
 * 12 and 9 positions in 4-position blocks with 2 selected blocks run as paged two-segment steps of uneven segments,
 * their index keys and K/V at the step's cells and each segment's block selection over its own sequence's cells
 * (the position &rarr; cell map of llama.cpp {@code set_input_pos_slot}); every row equals the same rows run one
 * sequence at a time on contiguous memory. A non-causal step keeps the two sequences apart too.
 */
public class CpuMsaMixerBatchTest {

    private static final int N_EMBD = 32;
    private static final int N_HEAD = 4;
    private static final int N_KV = 2;
    private static final int HD = 16;
    private static final int Q_DIM = N_HEAD * HD;
    private static final int K_DIM = N_KV * HD;
    private static final int IDX = 16;
    private static final int N_ROT = 8;
    private static final int BLK = 4;
    private static final int TOP_K = 2;
    private static final int N_CTX = 32;
    private static final int UBATCH = 16;
    private static final float KQ = 0.25f;
    private static final float TOL = 1e-4f;

    private static RandomGguf weights() throws IOException {
        return RandomGguf.builder(Arch.MINIMAX_M3.ggufName(), 5)
                .random("blk.0.attn_q.weight", 0.3f, N_EMBD, Q_DIM)
                .random("blk.0.attn_k.weight", 0.3f, N_EMBD, K_DIM)
                .random("blk.0.attn_v.weight", 0.3f, N_EMBD, K_DIM)
                .random("blk.0.attn_output.weight", 0.3f, Q_DIM, N_EMBD)
                .constant("blk.0.attn_q_norm.weight", 1f, HD)
                .constant("blk.0.attn_k_norm.weight", 1f, HD)
                .random("blk.0.indexer.q_proj.weight", 0.5f, N_EMBD, N_KV * IDX)
                .random("blk.0.indexer.k_proj.weight", 0.5f, N_EMBD, IDX)
                .constant("blk.0.indexer.q_norm.weight", 1f, IDX)
                .constant("blk.0.indexer.k_norm.weight", 1f, IDX)
                .open();
    }

    private static ModelSpec spec(TensorCatalog catalog) {
        AttentionSpec a = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_KV).headDimK(HD).headDimV(HD).nRot(N_ROT)
                .causal(true).qkNorm(QkNormKind.RMS_HEAD).kqScale(KQ).kvOwnerLayer(0).build();
        MsaSpec msa = MsaSpec.builder().idxHeads(N_KV).idxHeadDim(IDX).idxRopeDim(N_ROT).blockSize(BLK)
                .localBlocks(1).topKBlocks(TOP_K).sparse(true).kqScale(KQ).build();
        LayerSpec layer = LayerSpec.builder(0, MixerKind.MSA, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).qkNorm(a.qkNorm()).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .attention(a)
                .msa(msa)
                .rope(RopeSpec.builder(RopeType.NEOX).nDims(N_ROT).build())
                .build();
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.MINIMAX_M3)).layerCounts(1, 0)
                .nCtxTrain(4096).nEmbd(N_EMBD).normRmsEps(1e-6f);
        return ModelSpec.builder(hb.build(), catalog)
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build())
                .layer(layer)
                .build();
    }

    private static Function<Boolean, CpuMixerRig> rigs(RandomGguf w) {
        ModelSpec spec = spec(w.catalog());
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
    public void pagedStepsOfTwoSequencesMatchTheContiguousRuns() throws IOException {
        try (RandomGguf w = weights()) {
            PagedMixerCheck.assertBatchMatchesSequences(rigs(w), N_EMBD, 1, new int[]{6, 5, 1}, new int[]{3, 5, 1},
                    TOL);
        }
    }

    @Test
    public void aNonCausalStepKeepsTheSequencesApart() throws IOException {
        try (RandomGguf w = weights()) {
            PagedMixerCheck.assertNonCausalBatchKeepsSequencesApart(rigs(w), N_EMBD, 1, 9, 7, TOL);
        }
    }

}
