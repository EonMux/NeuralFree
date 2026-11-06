package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.ops.hc.HcSubBlock;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader.Options;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The catch-up protocol of {@link NextnDraftModel} over a host feed: a small target call (a verification, a decode)
 * defers its pairs, {@code accept} drops the rejected draft tokens' rows, and the next draft's first pass decodes the
 * rest with its own row — one run of the block per draft step. A call too big to defer decodes at once and writes
 * only its caches ({@code kvOnly}), and so does a flush of deferred rows before it. The head's kernels are scripted
 * no-ops, so the tests drive the drafter's wiring.
 */
public class NextnDraftModelCatchupTest {

    private static final int N_EMBD = 64;
    private static final int TARGET_VOCAB = 32;
    private static final int N_CTX = 64;
    private static final int UBATCH = 8;
    private static final String HEAD = "output.weight";
    private static final String NORM = "output_norm.weight";
    private static final String EMBD = "token_embd.weight";

    /**
     * The rounds of a scripted speculation: the prompt call {@code [5, 6, 7]} defers into the first draft's pass
     * {@code [5, 6, 7, 9]}; the verification of {@code [9, 10, 11]} (both drafts confirmed) catches up with
     * {@code [9, 10, 11, 43]}; the next one {@code [43, 12, 13]} (one confirmed, one rolled back) with
     * {@code [43, 12, 99]}; a decode fallback with {@code [99, 101]}; and an eight-row call, too big to defer,
     * writes its caches in one kvOnly run.
     */
    @Test
    public void smallCallsCatchUpWithTheNextDraftsFirstPass() {
        Harness h = new Harness(Options.DEFAULT, picks(10, 11, 12, 13, 14, 15, 16, 17, 18, 19));

        h.verify(new int[]{5, 6, 7}, 0);
        assertArrayEquals(new int[]{10, 11}, h.draft(9, 2));
        h.expectRuns(run(0, 4, new int[]{5, 6, 7, 9}, false), run(4, 1, new int[]{10}, false));

        h.verify(new int[]{9, 10, 11}, 3);
        h.accept(2);
        assertArrayEquals(new int[]{12, 13}, h.draft(43, 2));
        h.expectRuns(run(3, 4, new int[]{9, 10, 11, 43}, false), run(7, 1, new int[]{12}, false));

        h.verify(new int[]{43, 12, 13}, 6);
        h.drop(8);
        h.accept(1);
        assertArrayEquals(new int[]{14, 15}, h.draft(99, 2));
        h.expectRuns(run(6, 3, new int[]{43, 12, 99}, false), run(9, 1, new int[]{14}, false));

        h.verify(new int[]{99}, 8);
        h.accept(0);
        assertArrayEquals(new int[]{16, 17}, h.draft(101, 2));
        h.expectRuns(run(8, 2, new int[]{99, 101}, false), run(10, 1, new int[]{16}, false));

        int[] big = {20, 21, 22, 23, 24, 25, 26, 27};
        h.verify(big, 9);
        assertArrayEquals(new int[]{18, 19}, h.draft(20, 2));
        h.expectRuns(run(9, 8, big.clone(), true), run(17, 1, new int[]{20}, false), run(18, 1, new int[]{18},
                false));
    }

    /** A catch-up the first pass cannot take (the cap of seven rows) decodes before the next call defers. */
    @Test
    public void deferredRowsDecodeBeforeTheNextCallDefers() {
        int[] full = {5, 6, 7, 8, 9, 10, 11};
        Harness h = new Harness(Options.DEFAULT, picks(12, 13));

        h.verify(full, 0);
        h.verify(new int[]{12, 13}, 7);
        h.accept(1);
        assertArrayEquals(new int[]{12}, h.draft(14, 1));
        h.expectRuns(run(0, 7, full.clone(), true), run(7, 3, new int[]{12, 13, 14}, false));
    }

    /** A first pass whose pick falls below p_min leaves the proposal empty; its run still decoded the catch-up. */
    @Test
    public void aStoppedFirstPassLeavesTheDraftEmpty() {
        float[] flat = new float[TARGET_VOCAB];
        Harness h = new Harness(Options.of(0.9f), new float[][]{flat, flat});

        h.verify(new int[]{5, 6, 7}, 0);
        assertEquals(0, h.draft(9, 2).length);
        h.expectRuns(run(0, 4, new int[]{5, 6, 7, 9}, false));
    }

    /** Reset clears the deferred rows with the rest of the drafter's state. */
    @Test
    public void resetClearsTheDeferredRows() {
        Harness h = new Harness(Options.DEFAULT, picks(10, 11, 12));

        h.verify(new int[]{5, 6, 7}, 0);
        assertArrayEquals(new int[]{10, 11}, h.draft(9, 2));
        h.expectRuns(run(0, 4, new int[]{5, 6, 7, 9}, false), run(4, 1, new int[]{10}, false));

        h.verify(new int[]{9, 10, 11}, 3);
        h.accept(0);
        h.drafter.reset();
        h.drop(3);
        assertArrayEquals(new int[]{12}, h.draft(9, 1));
        h.expectRuns(run(3, 1, new int[]{9}, false));
    }

    // ================================================================== expectations

    private static final class Run {
        final int pos0;
        final int n;
        final int[] tokens;
        final boolean kvOnly;

        Run(int pos0, int n, int[] tokens, boolean kvOnly) {
            this.pos0 = pos0;
            this.n = n;
            this.tokens = tokens;
            this.kvOnly = kvOnly;
        }
    }

    private static Run run(int pos0, int n, int[] tokens, boolean kvOnly) {
        return new Run(pos0, n, tokens, kvOnly);
    }

    private static float[][] picks(int... ids) {
        float[][] rows = new float[ids.length][];
        for (int i = 0; i < ids.length; i++) {
            float[] row = new float[TARGET_VOCAB];
            row[ids[i]] = 7f;
            rows[i] = row;
        }
        return rows;
    }

    // ================================================================== harness

    /** The drafter over a scripted head, the tap it installed on its fake target, and the stack runs it made. */
    private static final class Harness {
        final NextnDraftModel drafter;
        final FakeBackend backend;
        final FakeHead head;

        Harness(DraftModelLoader.Options options, float[][] pickLogits) {
            this.head = new FakeHead(targetHeadSpec());
            head.pickLogits = pickLogits;
            this.backend = new FakeBackend(new FixedMemory(targetPlan()));
            this.drafter = new NextnDraftModel(head, head.target(), MtpHead.Source.TARGET, options);
            drafter.bindTarget(target(backend, head.target()));
        }

        /** Feeds one target call: its tokens enter the sequence and its rows reach the tap. */
        void verify(int[] tokens, int pos0) {
            backend.memory.sequence().append(tokens, 0, tokens.length);
            float[] rows = new float[tokens.length * N_EMBD];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = 0.25f * i;
            }
            backend.tap.onHidden(head.target().layers().size(), rows, tokens.length, N_EMBD);
        }

        /** The target rolled back to {@code nPast}, as a rejection does. */
        void drop(int nPast) {
            backend.memory.rollback(backend.memory.sequence().nPast() - nPast);
        }

        int[] draft(int lastToken, int n) {
            return drafter.draft(lastToken, n);
        }

        void accept(int n) {
            drafter.accept(n);
        }

        /** Asserts the runs made since the last check, oldest first. */
        void expectRuns(Run... expected) {
            assertEquals(expected.length, head.runs.size());
            for (int i = 0; i < expected.length; i++) {
                Run e = expected[i];
                Run g = head.runs.get(i);
                assertEquals("run " + i + " pos0", e.pos0, g.pos0);
                assertEquals("run " + i + " n", e.n, g.n);
                assertArrayEquals("run " + i + " tokens", e.tokens, g.tokens);
                assertEquals("run " + i + " kvOnly", e.kvOnly, g.kvOnly);
            }
            head.runs.clear();
        }
    }

    // ================================================================== scripted head

    /**
     * A scripted {@link MtpHead}: shapes and the pick logits answer from fields, every kernel is a no-op, and its
     * stack records its runs ({@code run} for a whole pass, {@code runKV} for a cache-write pass).
     */
    private static final class FakeHead extends MtpHead {
        final List<Run> runs = new ArrayList<>();
        final Set<String> present = new HashSet<>();
        final Map<String, long[]> shapes = new HashMap<>();
        float[][] pickLogits = new float[0][];
        int picks;

        FakeHead(ModelSpec target) {
            super(target);
            present.add(HEAD);
            present.add(EMBD);
            present.add(NORM);
            present.add("blk.1.nextn.eh_proj.weight");
            present.add("blk.1.nextn.enorm.weight");
            present.add("blk.1.nextn.hnorm.weight");
            present.add("blk.1.nextn.shared_head_norm.weight");
            shapes.put(HEAD, new long[]{N_EMBD, TARGET_VOCAB});
        }

        @Override public int ubatch() { return UBATCH; }

        @Override public Rows rows(String name, int width) { return new ScriptedRows(name, width); }

        @Override public boolean has(Source s, String name) { return present.contains(name); }

        @Override public long[] shape(Source s, String name) {
            long[] ne = shapes.get(name);
            if (ne == null) {
                throw new IllegalStateException("no scripted shape for " + name);
            }
            return ne;
        }

        @Override public float[] vector(Source s, String name) { return new float[N_EMBD]; }

        @Override public int[] integers(Source s, String name) { throw new UnsupportedOperationException(name); }

        @Override public void write(Rows dst, float[] src, int nRows) { }

        @Override public void copyRows(Rows src, int srcRow, Rows dst, int dstRow, int n) { }

        @Override public void read(Rows src, int nRows, float[] dst) { }

        @Override public void readRow(Rows src, int row, float[] dst) { }

        @Override public void embed(Source s, String table, int[] tokens, int n, Rows out) { }

        @Override public void norm(NormKind kind, Source s, String weight, float eps, Rows in, Rows out, int n) { }

        @Override public void concat(Rows a, Rows b, Rows out, int n) { }

        @Override public void project(Source s, String weight, Rows in, Rows out, int n) { }

        @Override public void add(Rows a, Rows b, Rows out, int n) { }

        @Override public void scale(Rows x, float s, int n) { }

        @Override public void normStreams(NormKind kind, Source s, String weight, float eps, Rows in, Rows out,
                                           int n, int streams) { }

        @Override public void concatStreams(Rows a, Rows b, Rows out, int n, int streams) { }

        @Override public void projectStreams(Source s, String weight, Rows in, Rows out, int n, int streams) { }

        @Override public Collapse collapse(Source s, HcSubBlock site) {
            throw new UnsupportedOperationException();
        }

        @Override public void logits(Source s, String weight, Rows in, int row, float[] dst) {
            System.arraycopy(pickLogits[Math.min(picks++, pickLogits.length - 1)], 0, dst, 0, dst.length);
        }

        @Override public Memory ownMemory(ModelSpec spec, MemoryPlan plan) { return new ViewMemory(plan); }

        @Override public Memory targetCaches(ModelSpec spec, MemoryPlan plan) {
            throw new UnsupportedOperationException();
        }

        @Override public Stack stack(ModelSpec spec, Source s, Memory memory, List<LayerSpec> layers) {
            return new ScriptedStack();
        }

        @Override public Blocks blocks(ModelSpec spec, Source s, Memory memory, LayerSpec layer) {
            throw new UnsupportedOperationException();
        }

        private final class ScriptedStack implements MtpHead.Stack {
            @Override public void run(int pos0, int n, int[] tokens, MtpHead.Rows x) {
                runs.add(new Run(pos0, n, tokens.clone(), false));
            }

            @Override public void runKV(int pos0, int n, int[] tokens, MtpHead.Rows x) {
                runs.add(new Run(pos0, n, tokens.clone(), true));
            }

            @Override public void close() { }
        }
    }

    private static final class ScriptedRows extends MtpHead.Rows {
        ScriptedRows(String name, int width) {
            super(name, width);
        }
    }

    /** A drafter memory that owns no caches and reads the plan it was built with. */
    private static final class ViewMemory extends MtpHead.Memory {
        private final MemoryPlan plan;

        ViewMemory(MemoryPlan plan) {
            super(false);
            this.plan = plan;
        }

        @Override public ModelMemory memory() { return new FixedMemory(plan); }

        @Override void begin(int pos0, int n, int[] tokens) { }
    }

    // ================================================================== specs

    /** The target spec: one trunk block, its NextN block and a head over the full vocabulary. */
    private static ModelSpec targetHeadSpec() {
        ModelHparams hp = hparams(1);
        return ModelSpec.builder(hp, TensorCatalog.of(Arch.GLM4_MOE.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(TensorRef.of(EMBD, GGMLType.F32, N_EMBD, TARGET_VOCAB), N_EMBD)
                        .build())
                .head(HeadSpec.builder().outputNorm(TensorRef.of(NORM, GGMLType.F32, N_EMBD), null)
                        .output(TensorRef.of(HEAD, GGMLType.F32, N_EMBD, TARGET_VOCAB), false).build())
                .layer(block(0))
                .nextnLayer(block(1))
                .build();
    }

    private static ModelHparams hparams(int nextn) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.GLM4_MOE))
                .layerCounts(1 + nextn, nextn).nCtxTrain(N_CTX).nEmbd(N_EMBD).nVocab(TARGET_VOCAB)
                .headKFull(16).headVFull(16);
        for (int il = 0; il < 1 + nextn; il++) {
            hb.nHead(il, 4).nHeadKv(il, 2);
        }
        return hb.build();
    }

    private static LayerSpec block(int il) {
        return LayerSpec.builder(il, MixerKind.NONE, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-5f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .build();
    }

    // ================================================================== target

    /** The target plan: two layer slots (the trunk block and the NextN block) of one KV group. */
    private static MemoryPlan targetPlan() {
        return MemoryPlan.builder(MemoryKind.KV, N_CTX, UBATCH).config(MemoryConfig.defaults())
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.NONE).build())
                .layer(LayerMemoryPlan.builder(1, LayerMemoryPlan.Kind.NONE).build())
                .build();
    }

    /** The target the drafter binds to: its backend keeps the installed tap beside the scripted memory. */
    private static EngineModel target(FakeBackend backend, ModelSpec spec) {
        return new EngineModel() {
            @Override public ForwardBackend forwardBackend() { return backend; }

            @Override public ModelSpec spec() { return spec; }

            @Override public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP) {
                throw new UnsupportedOperationException();
            }

            @Override public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP,
                                            com.eonmux.neuralfree.gguf.TokenCallback callback) {
                throw new UnsupportedOperationException();
            }

            @Override public int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP) {
                throw new UnsupportedOperationException();
            }

            @Override public int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP,
                                                      com.eonmux.neuralfree.gguf.TokenCallback callback) {
                throw new UnsupportedOperationException();
            }

            @Override public void resetKVCache() { }

            @Override public com.eonmux.neuralfree.gguf.GGUFTokenizer tokenizer() {
                throw new UnsupportedOperationException();
            }

            @Override public void printInfo() { }

            @Override public String architecture() { return spec.arch().ggufName(); }

            @Override public String name() { return architecture(); }

            @Override public int vocabSize() { return TARGET_VOCAB; }

            @Override public int contextLength() { return N_CTX; }

            @Override public int kvCacheContext() { return N_CTX; }

            @Override public int embeddingLength() { return N_EMBD; }

            @Override public int layerCount() { return 1; }

            @Override public int headCount() { return 4; }

            @Override public int headCountKv() { return 2; }

            @Override public int feedForwardLength() { return 0; }

            @Override public float rmsNormEps() { return 1e-5f; }

            @Override public float ropeFreqBase() { return 10000f; }

            @Override public int headDim() { return 16; }

            @Override public int kvDim() { return 32; }

            @Override public void close() { }
        };
    }

    private static final class FakeBackend implements ForwardBackend {
        private final FixedMemory memory;
        HiddenTap tap = HiddenTap.NONE;

        FakeBackend(FixedMemory memory) {
            this.memory = memory;
        }

        @Override public float[] prefill(int[] tokens, int pos0) { throw new UnsupportedOperationException(); }

        @Override public float[] decode(int token, int pos) { throw new UnsupportedOperationException(); }

        @Override public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
            throw new UnsupportedOperationException();
        }

        @Override public float[][] forwardAll(int[] tokens, int[] outputRows) {
            throw new UnsupportedOperationException();
        }

        @Override public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
            throw new UnsupportedOperationException();
        }

        @Override public void reset() { }

        @Override public boolean canRollback(int nTokens) { return memory.canRollback(nTokens); }

        @Override public void rollback(int nTokens) { memory.rollback(nTokens); }

        @Override public void shiftContext(int nKeep, int nDiscard) { throw new UnsupportedOperationException(); }

        @Override public ModelMemory memory() { return memory; }

        @Override public void setHiddenTap(HiddenTap tap) { this.tap = tap; }

        @Override public void close() { }
    }

    /** A memory over a fixed plan with a sequence of its own. */
    private static final class FixedMemory implements ModelMemory {
        private final MemoryPlan plan;
        private final SequenceState sequence = new SequenceState(N_CTX);

        FixedMemory(MemoryPlan plan) {
            this.plan = plan;
        }

        @Override public MemoryPlan plan() { return plan; }

        @Override public SequenceState sequence() { return sequence; }

        @Override public void reset() { sequence.clear(); }

        @Override public boolean canRollback(int nTokens) { return nTokens <= sequence.nPast(); }

        @Override public void rollback(int nTokens) { sequence.truncate(nTokens); }

        @Override public void snapshot() { throw new UnsupportedOperationException(); }

        @Override public void restore() { throw new UnsupportedOperationException(); }

        @Override public void close() { }
    }
}
