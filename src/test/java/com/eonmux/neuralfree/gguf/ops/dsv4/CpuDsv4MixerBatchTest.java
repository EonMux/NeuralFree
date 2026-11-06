package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.CompressedKvPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Blocks.Dsv4Layout;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The paged multi-sequence steps of the CPU DeepSeek-V4 mixer ({@code CpuDsv4Mixer.batch}) over a CSA layer
 * (ratio 4 with the lightning indexer, top-4), an HCA layer (ratio 128) and a window-only layer, all with a 16-row
 * raw window: two sequences of 135 and 137 positions run as paged two-segment steps of uneven segments, each
 * compressing into its own compressed streams and attending its own raw cells (llama.cpp's DSV4 streams,
 * llama-kv-cache-dsv4.cpp:51-60), and every row equals the same rows run one sequence at a time on contiguous memory.
 * The sequence operations keep the streams in step: a whole-sequence copy continues like its source, a tail removal
 * within the rollback snapshots restores the compressor planes, a whole removal starts the sequence afresh, and
 * partial copies, interior removals and non-causal steps are refused by name.
 */
public class CpuDsv4MixerBatchTest {

    private static final int N_EMBD = 64;
    private static final int N_HEAD = 4;
    private static final int HEAD = 64;
    private static final int N_ROT = 16;
    private static final int Q_LORA = 32;
    private static final int N_GROUPS = 2;
    private static final int O_LORA = 16;
    private static final int O_GROUP_DIM = N_HEAD / N_GROUPS * HEAD;
    private static final int IDX_HEADS = 2;
    private static final int IDX_DIM = 64;
    private static final int TOP_K = 4;
    private static final int WINDOW = 16;
    private static final int N_CTX = 512;
    private static final int UBATCH = 32;
    private static final int SNAPSHOTS = 3;
    private static final int[] RATIOS = {Dsv4Spec.RATIO_CSA, Dsv4Spec.RATIO_HCA, 0};
    private static final float TOL = 1e-4f;

    private static RandomGguf weights() throws IOException {
        RandomGguf.Builder b = RandomGguf.builder(Arch.DEEPSEEK4.ggufName(), 9);
        for (int il = 0; il < RATIOS.length; il++) {
            layerTensors(b, il, RATIOS[il]);
        }
        return b.open();
    }

    private static String name(int il, String suffix) {
        return "blk." + il + "." + suffix;
    }

    private static void layerTensors(RandomGguf.Builder b, int il, int ratio) {
        b.random(name(il, Dsv4Layout.WQ_A), 0.3f, N_EMBD, Q_LORA)
                .constant(name(il, Dsv4Layout.Q_A_NORM), 1f, Q_LORA)
                .random(name(il, Dsv4Layout.WQ_B), 0.3f, Q_LORA, (long) N_HEAD * HEAD)
                .random(name(il, Dsv4Layout.WKV), 0.3f, N_EMBD, HEAD)
                .constant(name(il, Dsv4Layout.KV_NORM), 1f, HEAD)
                .random(name(il, Dsv4Layout.WO_A), 0.2f, O_GROUP_DIM, O_LORA, N_GROUPS)
                .random(name(il, Dsv4Layout.WO_B), 0.3f, (long) N_GROUPS * O_LORA, N_EMBD);
        if (ratio != 0) {
            int sw = Dsv4Kernels.stateWidth(ratio, HEAD);
            b.random(name(il, Dsv4Layout.COMP_WKV), 0.3f, N_EMBD, sw)
                    .random(name(il, Dsv4Layout.COMP_WGATE), 0.3f, N_EMBD, sw)
                    .random(name(il, Dsv4Layout.COMP_APE), 0.1f, sw, ratio)
                    .constant(name(il, Dsv4Layout.COMP_NORM), 1f, HEAD);
        }
        if (ratio == Dsv4Spec.RATIO_CSA) {
            int sw = Dsv4Kernels.stateWidth(Dsv4Spec.RATIO_CSA, IDX_DIM);
            b.random(name(il, Dsv4Layout.IDX_Q_B), 0.3f, Q_LORA, (long) IDX_HEADS * IDX_DIM)
                    .random(name(il, Dsv4Layout.IDX_PROJ), 0.3f, N_EMBD, IDX_HEADS)
                    .random(name(il, Dsv4Layout.IDX_COMP_WKV), 0.3f, N_EMBD, sw)
                    .random(name(il, Dsv4Layout.IDX_COMP_WGATE), 0.3f, N_EMBD, sw)
                    .random(name(il, Dsv4Layout.IDX_COMP_APE), 0.1f, sw, Dsv4Spec.RATIO_CSA)
                    .constant(name(il, Dsv4Layout.IDX_COMP_NORM), 1f, IDX_DIM);
        }
    }

    private static LayerSpec layer(int il, int ratio) {
        Dsv4Spec.Builder v4 = Dsv4Spec.builder().nHead(N_HEAD).compressRatio(ratio).window(WINDOW).headDim(HEAD)
                .ropeDim(N_ROT).qLoraRank(Q_LORA).oGroupCount(N_GROUPS).oLoraRank(O_LORA)
                .compressRopeBase(160000f).kqScale(1.0f / (float) Math.sqrt(HEAD));
        if (ratio == Dsv4Spec.RATIO_CSA) {
            v4.indexer(DsaSpec.builder().indexerHeads(IDX_HEADS).indexerHeadDim(IDX_DIM).indexerRopeDim(N_ROT)
                    .ropeType(RopeType.NORM).ropeTail(true).topK(TOP_K).full(true).sourceLayer(il).hadamard(true)
                    .defaultWeightScale().build());
        }
        RopeSpec rope = RopeSpec.builder(RopeType.NORM).nDims(N_ROT).nOffs(HEAD - N_ROT).freqBase(10000f).build();
        return LayerSpec.builder(il, MixerKind.DSV4, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .dsv4(v4.build())
                .rope(rope)
                .build();
    }

    private static ModelSpec spec(TensorCatalog catalog) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.DEEPSEEK4))
                .layerCounts(RATIOS.length, 0).nCtxTrain(4096).nEmbd(N_EMBD).normRmsEps(1e-6f).normEps(1e-6f)
                .swaType(SwaType.STANDARD).nSwa(WINDOW).indexerHeadSize(IDX_DIM);
        hb.headKFull(HEAD).headVFull(HEAD);
        for (int il = 0; il < RATIOS.length; il++) {
            hb.nHead(il, N_HEAD).nHeadKv(il, 1);
        }
        ModelSpec.Builder b = ModelSpec.builder(hb.build(), catalog)
                .schedule(LayerSchedule.linear(RATIOS.length))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build());
        for (int il = 0; il < RATIOS.length; il++) {
            b.layer(layer(il, RATIOS[il]));
        }
        return b.build();
    }

    private static Function<Boolean, CpuMixerRig> rigs(RandomGguf w) {
        ModelSpec spec = spec(w.catalog());
        return paged -> {
            MemoryConfig.Builder cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(KvDType.F32).typeV(KvDType.F32)
                    .rollbackSnapshots(SNAPSHOTS);
            if (paged) {
                cfg = cfg.parallel(2);
            }
            return CpuMixerRig.open(spec, w.weights(), cfg.build(), N_CTX, N_EMBD,
                    layer -> Dsv4Blocks.cpuMixers().get(0).create(layer));
        };
    }

    @Test
    public void thePagedPlanKeepsOneCompressedStreamPerSequence() throws IOException {
        try (RandomGguf w = weights(); CpuMixerRig rig = rigs(w).apply(true)) {
            CompressedKvPlan csa = rig.memory().plan().layer(0).compressed().get(0);
            assertEquals(2, csa.streams());
            assertEquals(2 * (csa.cacheBytes() + csa.stateBytes()), csa.bytes());
            assertNotNull(rig.memory().compressed(0, CompressedKvPlan.Kind.CSA, 1));
            assertTrue("the streams are distinct caches", rig.memory().compressed(0, CompressedKvPlan.Kind.CSA, 0)
                    != rig.memory().compressed(0, CompressedKvPlan.Kind.CSA, 1));
            assertThrows(IndexOutOfBoundsException.class,
                    () -> rig.memory().compressed(0, CompressedKvPlan.Kind.CSA, 2));
        }
    }

    @Test
    public void pagedStepsOfTwoSequencesMatchTheContiguousRuns() throws IOException {
        int[] chunksA = {16, 16, 16, 16, 16, 16, 16, 16, 5, 1, 1};
        int[] chunksB = {7, 16, 16, 16, 16, 16, 16, 16, 16, 1, 1};
        try (RandomGguf w = weights()) {
            PagedMixerCheck.assertBatchMatchesSequences(rigs(w), N_EMBD, RATIOS.length, chunksA, chunksB, TOL);
        }
    }

    /** A whole-sequence copy continues exactly like its source: both decode the same row to the same outputs. */
    @Test
    public void aCopiedSequenceContinuesOverItsCopiedStreams() throws IOException {
        try (RandomGguf w = weights(); CpuMixerRig rig = rigs(w).apply(true)) {
            float[] prompt = PagedMixerCheck.inputs(3, 22, N_EMBD);
            rig.paged(new int[]{0}, new int[]{22}, new int[]{0}, prompt, false);
            SequenceMemory seqs = rig.memory().sequences();
            assertThrows("a partial copy is refused", UnsupportedOperationException.class,
                    () -> seqs.copy(0, 1, 0, 10));
            seqs.copy(0, 1, 0, -1);
            float[] next = PagedMixerCheck.inputs(5, 1, N_EMBD);
            float[] two = new float[2 * N_EMBD];
            System.arraycopy(next, 0, two, 0, N_EMBD);
            System.arraycopy(next, 0, two, N_EMBD, N_EMBD);
            float[] out = rig.paged(new int[]{0, 1}, new int[]{1, 1}, new int[]{22, 22}, two, false);
            for (int l = 0; l < RATIOS.length; l++) {
                assertArrayEquals("layer " + l, rig.rows(out, l, 2, 0, 1), rig.rows(out, l, 2, 1, 1), 0f);
            }
        }
    }

    /**
     * A tail removal within the rollback snapshots restores the compressor planes of that sequence only: its
     * replacement rows equal a sequence that never saw the dropped rows, while the other sequence is untouched.
     */
    @Test
    public void aTailRemovalRestoresTheSequencesSnapshotPlanes() throws IOException {
        float[] head = PagedMixerCheck.inputs(7, 18, N_EMBD);
        float[] dropped = PagedMixerCheck.inputs(8, 2, N_EMBD);
        float[] replacement = PagedMixerCheck.inputs(9, 2, N_EMBD);
        float[] expected;
        try (RandomGguf w = weights()) {
            Function<Boolean, CpuMixerRig> rigs = rigs(w);
            try (CpuMixerRig ref = rigs.apply(true)) {
                ref.paged(new int[]{0}, new int[]{18}, new int[]{0}, head, false);
                expected = ref.paged(new int[]{0}, new int[]{2}, new int[]{18}, replacement, false);
            }
            try (CpuMixerRig rig = rigs.apply(true)) {
                rig.paged(new int[]{0}, new int[]{18}, new int[]{0}, head, false);
                rig.paged(new int[]{0}, new int[]{2}, new int[]{18}, dropped, false);
                SequenceMemory seqs = rig.memory().sequences();
                assertNull(seqs.removeRefusal(0, 18, -1));
                assertTrue("an interior removal is refused by name",
                        seqs.removeRefusal(0, 4, 8).contains("tail only"));
                assertTrue("a removal beyond the last step's rows is refused",
                        seqs.removeRefusal(0, 10, -1).contains("snapshots"));
                seqs.remove(0, 18, -1);
                assertTrue("the rollback is single-use until the next step",
                        seqs.removeRefusal(0, 17, -1).contains("already rolled back"));
                float[] out = rig.paged(new int[]{0}, new int[]{2}, new int[]{18}, replacement, false);
                assertArrayEquals("the replacement rows see the restored compressor state", expected, out, TOL);
            }
        }
    }

    /** Removing a whole sequence zeroes its streams: the slot then runs a new prompt as a fresh sequence. */
    @Test
    public void aWholeRemovalStartsTheSequenceAfresh() throws IOException {
        float[] first = PagedMixerCheck.inputs(13, 20, N_EMBD);
        float[] second = PagedMixerCheck.inputs(14, 12, N_EMBD);
        try (RandomGguf w = weights()) {
            Function<Boolean, CpuMixerRig> rigs = rigs(w);
            float[] fresh;
            try (CpuMixerRig ref = rigs.apply(true)) {
                fresh = ref.paged(new int[]{1}, new int[]{12}, new int[]{0}, second, false);
            }
            try (CpuMixerRig rig = rigs.apply(true)) {
                rig.paged(new int[]{1}, new int[]{20}, new int[]{0}, first, false);
                rig.memory().sequences().remove(1, 0, -1);
                float[] out = rig.paged(new int[]{1}, new int[]{12}, new int[]{0}, second, false);
                assertArrayEquals(fresh, out, TOL);
            }
        }
    }

    @Test
    public void aNonCausalStepAndShiftsAreRefusedByName() throws IOException {
        try (RandomGguf w = weights(); CpuMixerRig rig = rigs(w).apply(true)) {
            float[] in = PagedMixerCheck.inputs(17, 4, N_EMBD);
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> rig.paged(new int[]{0}, new int[]{4}, new int[]{0}, in, true));
            assertTrue(e.getMessage(), e.getMessage().contains("non-causal"));
            assertTrue(rig.memory().sequences().shiftRefusal().contains("cannot shift"));
        }
    }
}
