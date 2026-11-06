package com.eonmux.neuralfree.gguf.ops.mla;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.CpuMixerRig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.PagedMixerCheck;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.SplittableRandom;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;

/**
 * The paged (multi-sequence) steps of the CPU MLA mixer ({@code CpuMixer.batch}): a two-segment paged ubatch over
 * the latent cache — every segment stores its latent rows at the step's cells and attends through its sequence's
 * page table ({@code PagedSequences.prepare} -> {@code beginPagedStep} -> {@code commit}) — computes the same layer
 * outputs as the same tokens run one sequence at a time on the contiguous path, per sequence, over several steps of
 * uneven segments too; a step the context flags non-causal attends without the causal mask and keeps the sequences
 * apart.
 */
public class CpuMlaMixerBatchTest {

    private static final int N_EMBD = 32;
    private static final int N_HEAD = 2;
    private static final int NOPE = 8;
    private static final int ROPE = 8;
    private static final int KV_LORA = 16;
    private static final int HEAD_V = 8;
    private static final int HEAD_K = NOPE + ROPE;
    private static final int WIDTH = KV_LORA + ROPE;
    private static final int N_CTX = 32;
    private static final int UBATCH = 8;
    private static final int SEGMENT = 4;
    private static final float KQ_SCALE = 0.25f;
    private static final float EPS = 1e-6f;

    /** The weight file and its parsed tensors, shared by the two memories. */
    private static final class Fixture implements AutoCloseable {
        final Path path;
        final GGUFFileParser parser;
        final CpuWeights weights;
        final LayerSpec layer;
        final ModelSpec spec;
        final HeapModelMemory contiguous;
        final HeapModelMemory paged;
        final CpuMlaMixer contiguousMixer;
        final CpuMlaMixer pagedMixer;
        final CpuForwardContext pagedContext;

        Fixture() throws IOException {
            SplittableRandom rng = new SplittableRandom(9);
            GgufTestWriter w = new GgufTestWriter().architecture(Arch.DEEPSEEK2.ggufName());
            put(w, rng, TensorNames.ATTN_Q, new long[]{N_EMBD, N_HEAD * HEAD_K});
            put(w, rng, TensorNames.ATTN_KV_A_MQA, new long[]{N_EMBD, WIDTH});
            put(w, rng, TensorNames.ATTN_KV_A_NORM, new long[]{KV_LORA});
            put(w, rng, TensorNames.ATTN_K_B, new long[]{NOPE, KV_LORA, N_HEAD});
            put(w, rng, TensorNames.ATTN_V_B, new long[]{KV_LORA, HEAD_V, N_HEAD});
            put(w, rng, TensorNames.ATTN_OUT, new long[]{N_HEAD * HEAD_V, N_EMBD});
            this.path = w.writeTempFile();
            this.parser = new GGUFFileParser(path.toString());
            this.weights = CpuWeights.open(parser);

            ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.DEEPSEEK2)).layerCounts(1, 0)
                    .nCtxTrain(4096).nEmbd(N_EMBD).normRmsEps(EPS).headKFull(WIDTH).headVFull(KV_LORA);
            hb.nHead(0, N_HEAD).nHeadKv(0, 1);
            MlaSpec mla = MlaSpec.builder().nHead(N_HEAD).qLoraRank(0).kvLoraRank(KV_LORA).headKNope(NOPE)
                    .headKRope(ROPE).headV(HEAD_V).absorbed(true).latentCache(true).ropeApplied(false)
                    .kqScale(KQ_SCALE).build();
            AttentionSpec a = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD).headDimK(HEAD_K)
                    .headDimV(HEAD_V).nRot(0).noRope(true).qkNorm(QkNormKind.NONE).kqScale(KQ_SCALE)
                    .kvOwnerLayer(0).build();
            this.layer = LayerSpec.builder(0, MixerKind.MLA, FfnKind.NONE)
                    .norm(NormSpec.builder(NormKind.RMS, EPS).build())
                    .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                    .mla(mla)
                    .attention(a)
                    .build();
            this.spec = ModelSpec.builder(hb.build(), TensorCatalog.of(parser))
                    .schedule(LayerSchedule.linear(1))
                    .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                    .head(HeadSpec.builder().build())
                    .layer(layer)
                    .build();
            this.contiguous = memory(false);
            this.paged = memory(true);
            this.pagedContext = context(paged);
            this.contiguousMixer = mixer(context(contiguous));
            this.pagedMixer = mixer(pagedContext);
        }

        private HeapModelMemory memory(boolean paged) {
            MemoryConfig.Builder cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(KvDType.F32).typeV(KvDType.F32);
            if (paged) {
                cfg = cfg.parallel(2);
            }
            MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg.build(), N_CTX);
            return HeapModelMemory.allocate(plan, spec, null);
        }

        private CpuForwardContext context(HeapModelMemory memory) {
            return CpuForwardContext.builder(spec, weights, CpuScratch.forModel(spec, UBATCH)).memory(memory).build();
        }

        private CpuMlaMixer mixer(CpuForwardContext ctx) {
            CpuMlaMixer mixer = (CpuMlaMixer) MlaBlocks.cpuMixers().get(0).create(layer);
            mixer.bind(ctx, layer);
            return mixer;
        }

        /** The layer outputs of one contiguous prefill over {@code tokens} positions {@code [0, n)}. */
        float[] runContiguous(float[] in, int n) {
            contiguous.beginStep(0, n);
            float[] out = new float[n * N_EMBD];
            contiguousMixer.prefill(0, n, in, out);
            return out;
        }

        /**
         * The layer outputs of one paged two-segment ubatch: sequence 0's rows {@code [0, 4)} at positions
         * {@code [0, 4)} and sequence 1's rows {@code [4, 8)} at positions {@code [0, 4)}.
         */
        float[] runPagedBatch(float[] in) {
            PagedSequences sequences = paged.pagedSequences();
            assertNotNull(sequences);
            StepLayout layout = StepLayout.of(2 * SEGMENT, new int[]{0, 1}, new int[]{0, SEGMENT},
                    new int[]{SEGMENT, SEGMENT}, new int[]{0, 0}, new int[0], true);
            PagedStep step = sequences.prepare(layout);
            // the ubatch positions: both sequences at [0, SEGMENT)
            int[] positions = new int[2 * SEGMENT];
            for (int r = 0; r < positions.length; r++) {
                positions[r] = r % SEGMENT;
            }
            paged.beginPagedStep(step, positions, null);
            // the embedding stage's positions of the step's rows (section 0 of the POS slot)
            System.arraycopy(positions, 0, pagedContext.scratch().ints(CpuScratch.POS), 0, positions.length);
            try {
                float[] out = new float[2 * SEGMENT * N_EMBD];
                pagedMixer.batch(layout, in, out);
                return out;
            } finally {
                paged.endPagedStep();
                sequences.commit(step);
            }
        }

        /** Rigs over this fixture's spec and weights: paged ({@code true}) or contiguous. */
        Function<Boolean, CpuMixerRig> rigs() {
            return pagedMemory -> {
                MemoryConfig.Builder cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(KvDType.F32)
                        .typeV(KvDType.F32);
                if (pagedMemory) {
                    cfg = cfg.parallel(2);
                }
                return CpuMixerRig.open(spec, weights, cfg.build(), N_CTX, N_EMBD,
                        l -> MlaBlocks.cpuMixers().get(0).create(l));
            };
        }

        private static void put(GgufTestWriter w, SplittableRandom rng, String base, long[] ne) {
            long n = 1;
            for (long dim : ne) {
                n *= dim;
            }
            float[] v = new float[(int) n];
            for (int i = 0; i < v.length; i++) {
                v[i] = (float) (-0.5 + rng.nextDouble());
            }
            w.tensorFromFloats(TensorNames.layerName(0, TensorNames.weight(base)), GGMLType.F32, ne, v);
        }

        @Override
        public void close() throws IOException {
            try {
                contiguousMixer.close();
                pagedMixer.close();
                contiguous.close();
                paged.close();
                weights.close();
            } finally {
                try {
                    parser.close();
                } finally {
                    Files.deleteIfExists(path);
                }
            }
        }
    }

    @Test
    public void aPagedTwoSegmentStepMatchesTheContiguousRunsPerSequence() throws IOException {
        try (Fixture f = new Fixture()) {
            SplittableRandom rng = new SplittableRandom(42);
            float[] inA = new float[SEGMENT * N_EMBD];
            float[] inB = new float[SEGMENT * N_EMBD];
            for (int i = 0; i < inA.length; i++) {
                inA[i] = (float) (-1 + 2 * rng.nextDouble());
                inB[i] = (float) (-1 + 2 * rng.nextDouble());
            }
            float[] refA = f.runContiguous(inA, SEGMENT);
            float[] refB;
            try (Fixture g = new Fixture()) {
                refB = g.runContiguous(inB, SEGMENT);
            }
            float[] batch = new float[2 * SEGMENT * N_EMBD];
            System.arraycopy(inA, 0, batch, 0, inA.length);
            System.arraycopy(inB, 0, batch, inA.length, inB.length);
            float[] out = f.runPagedBatch(batch);
            assertArrayEquals("sequence 0's rows match the contiguous run", refA,
                    java.util.Arrays.copyOfRange(out, 0, SEGMENT * N_EMBD), 1e-4f);
            assertArrayEquals("sequence 1's rows match the contiguous run", refB,
                    java.util.Arrays.copyOfRange(out, SEGMENT * N_EMBD, 2 * SEGMENT * N_EMBD), 1e-4f);
        }
    }

    @Test
    public void pagedStepsOfUnevenSegmentsMatchTheContiguousRuns() throws IOException {
        try (Fixture f = new Fixture()) {
            PagedMixerCheck.assertBatchMatchesSequences(f.rigs(), N_EMBD, 1, new int[]{5, 2, 1}, new int[]{3, 4, 1},
                    1e-4f);
        }
    }

    @Test
    public void aNonCausalStepAttendsBothWaysWithinEachSequence() throws IOException {
        try (Fixture f = new Fixture()) {
            PagedMixerCheck.assertNonCausalBatchKeepsSequencesApart(f.rigs(), N_EMBD, 1, 5, 3, 1e-4f);
        }
    }
}
