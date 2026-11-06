package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
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

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The reduced (FR-Spec) draft head of {@link NextnDraftModel}: an MTP draft file whose LM head is cut to its most
 * frequent tokens, its {@code d2t} row naming the target token of every row (qwen4exp.cpp:287-296). The pick runs
 * over the cut rows and maps through {@code d2t}, which is the reference graph's scatter to the target vocabulary
 * with {@code -inf} elsewhere (qwen4exp.cpp:841-844); a head without {@code d2t} must cover the target vocabulary.
 * The head's kernels are scripted no-ops, so the tests drive the drafter's own wiring.
 */
public class NextnDraftModelD2tTest {

    private static final int N_EMBD = 64;
    private static final int TARGET_VOCAB = 32;
    private static final int DRAFT_VOCAB = 4;
    private static final int[] D2T = {3, 17, 9, 28};
    private static final String HEAD = "output.weight";
    private static final String NORM = "output_norm.weight";
    private static final String EMBD = "token_embd.weight";

    /**
     * The argmax row of the cut head at every draft step, mapped through {@code d2t}: steps 2, 0, 1 draft the
     * target tokens 9, 3, 17.
     */
    @Test
    public void draftMapsTheCutHeadPickThroughD2t() {
        FakeHead head = full();
        NextnDraftModel d = new NextnDraftModel(head, draftSpec(true), MtpHead.Source.DRAFT,
                DraftModelLoader.Options.DEFAULT);
        d.bindTarget(target(head));

        assertEquals("NextnDraftModel{glm4moe, blocks 1..1, d2t 4 of 32}", d.toString());
        assertArrayEquals(new int[]{9, 3, 17}, d.draft(7, 3));
        d.close();
    }

    /** A head spanning the target vocabulary drafts its pick unchanged. */
    @Test
    public void aFullHeadPicksTargetIdsDirectly() {
        FakeHead head = full();
        head.shapes.put(HEAD, new long[]{N_EMBD, TARGET_VOCAB});
        head.pickLogits = new float[][]{{0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
        NextnDraftModel d = new NextnDraftModel(head, draftSpec(false), MtpHead.Source.DRAFT,
                DraftModelLoader.Options.DEFAULT);
        d.bindTarget(target(head));

        assertArrayEquals(new int[]{5}, d.draft(7, 1));
        d.close();
    }

    /** The p_min stop reads the cut distribution, before the mapping (speculative.cpp p_min). */
    @Test
    public void pMinStopsTheDraftBeforeTheMapping() {
        FakeHead head = full();
        head.pickLogits = new float[][]{{5, 1, 9, 0}, {9, 8, 0, 0}};
        NextnDraftModel d = new NextnDraftModel(head, draftSpec(true), MtpHead.Source.DRAFT,
                DraftModelLoader.Options.of(0.9f));
        d.bindTarget(target(head));

        assertArrayEquals(new int[]{9}, d.draft(7, 3));
        d.close();
    }

    @Test
    public void aReducedHeadWithoutD2tIsRefused() {
        FakeHead head = full();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new NextnDraftModel(head, draftSpec(false), MtpHead.Source.DRAFT,
                        DraftModelLoader.Options.DEFAULT));
        assertTrue(e.getMessage(), e.getMessage().contains("no d2t table"));
    }

    @Test
    public void theDeclaredD2tLengthMustMatchTheHeadRows() {
        FakeHead head = full();
        ModelSpec spec = draftSpec(true);
        HeadSpec cut = HeadSpec.builder().outputNorm(TensorRef.of(NORM, GGMLType.F32, N_EMBD), null)
                .output(TensorRef.of(HEAD, GGMLType.F32, N_EMBD, DRAFT_VOCAB), false)
                .d2t(TensorRef.of("d2t", GGMLType.I64, 6)).build();
        ModelSpec wrong = replaceHead(spec, cut);

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new NextnDraftModel(head, wrong, MtpHead.Source.DRAFT, DraftModelLoader.Options.DEFAULT));
        assertTrue(e.getMessage(), e.getMessage().contains("6 entries"));
    }

    @Test
    public void theStoredD2tMustHaveOneEntryPerHeadRow() {
        FakeHead head = full();
        head.d2tRows = new int[]{1, 2, 3};
        NextnDraftModel d = new NextnDraftModel(head, draftSpec(true), MtpHead.Source.DRAFT,
                DraftModelLoader.Options.DEFAULT);

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> d.bindTarget(target(head)));
        assertTrue(e.getMessage(), e.getMessage().contains("has 3 entries for 4 draft tokens"));
        d.close();
    }

    @Test
    public void d2tEntriesMustBeTargetTokenIds() {
        FakeHead head = full();
        head.d2tRows = new int[]{3, TARGET_VOCAB + 8, 9, 28};
        NextnDraftModel d = new NextnDraftModel(head, draftSpec(true), MtpHead.Source.DRAFT,
                DraftModelLoader.Options.DEFAULT);

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> d.bindTarget(target(head)));
        assertTrue(e.getMessage(), e.getMessage().contains("d2t[1] = 40 is outside the target vocabulary of 32"));
        d.close();
    }

    // ================================================================== scripted head

    /** A head with the tensors of one glm4_moe NextN block and the cut {@code output.weight}. */
    private static FakeHead full() {
        FakeHead head = new FakeHead(targetHead(TARGET_VOCAB));
        head.present.add(HEAD);
        head.present.add(EMBD);
        head.present.add(NORM);
        head.present.add("blk.1.nextn.eh_proj.weight");
        head.present.add("blk.1.nextn.enorm.weight");
        head.present.add("blk.1.nextn.hnorm.weight");
        head.present.add("blk.1.nextn.shared_head_norm.weight");
        head.shapes.put(HEAD, new long[]{N_EMBD, DRAFT_VOCAB});
        head.d2tRows = D2T.clone();
        head.pickLogits = new float[][]{{5, 1, 9, 0}, {9, 1, 0, 0}, {0, 1, 0, 0}};
        return head;
    }

    /**
     * A scripted {@link MtpHead}: shapes and the {@code d2t} rows answer from maps, the logits of each draft pick
     * from a script, and every kernel is a no-op, so only the drafter's wiring runs.
     */
    private static final class FakeHead extends MtpHead {
        final Set<String> present = new HashSet<>();
        final Map<String, long[]> shapes = new HashMap<>();
        int[] d2tRows;
        float[][] pickLogits;
        int picks;

        FakeHead(ModelSpec target) {
            super(target);
        }

        @Override public int ubatch() { return 8; }

        @Override public Rows rows(String name, int width) { return new ScriptedRows(width); }

        @Override public boolean has(Source s, String name) { return present.contains(name); }

        @Override public long[] shape(Source s, String name) {
            long[] ne = shapes.get(name);
            if (ne == null) {
                throw new IllegalStateException("no scripted shape for " + name);
            }
            return ne;
        }

        @Override public float[] vector(Source s, String name) { return new float[N_EMBD]; }

        @Override public int[] integers(Source s, String name) {
            if ("d2t".equals(name)) {
                return d2tRows.clone();
            }
            throw new UnsupportedOperationException(name);
        }

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
    }

    private static final class ScriptedRows extends MtpHead.Rows {
        ScriptedRows(int width) {
            super("scripted", width);
        }
    }

    private static final class ScriptedStack implements MtpHead.Stack {
        @Override public void run(int pos0, int n, int[] tokens, MtpHead.Rows x) { }

        @Override public void close() { }
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

    /** The target spec: one trunk block and a head over the full vocabulary. */
    private static ModelSpec targetHead(int vocab) {
        ModelHparams hp = hparams(0);
        return ModelSpec.builder(hp, TensorCatalog.of(Arch.GLM4_MOE.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().outputNorm(TensorRef.of(NORM, GGMLType.F32, N_EMBD), null)
                        .output(TensorRef.of(HEAD, GGMLType.F32, N_EMBD, vocab), false).build())
                .layer(block(0))
                .build();
    }

    /** The MTP draft spec: the target's trunk slot and one NextN block with its own embedding and cut head. */
    private static ModelSpec draftSpec(boolean withD2t) {
        ModelHparams hp = hparams(1);
        HeadSpec.Builder head = HeadSpec.builder()
                .outputNorm(TensorRef.of(NORM, GGMLType.F32, N_EMBD), null)
                .output(TensorRef.of(HEAD, GGMLType.F32, N_EMBD, DRAFT_VOCAB), false);
        if (withD2t) {
            head.d2t(TensorRef.of("d2t", GGMLType.I64, DRAFT_VOCAB));
        }
        return ModelSpec.builder(hp, TensorCatalog.of(Arch.GLM4_MOE.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(TensorRef.of(EMBD, GGMLType.F32, N_EMBD, TARGET_VOCAB), N_EMBD)
                        .build())
                .head(head.build())
                .layer(block(0))
                .nextnLayer(block(1))
                .build();
    }

    private static ModelSpec replaceHead(ModelSpec spec, HeadSpec head) {
        return ModelSpec.builder(spec.hparams(), spec.tensors())
                .schedule(spec.schedule())
                .embedding(spec.embedding())
                .head(head)
                .layer(spec.layers().get(0))
                .nextnLayer(spec.nextnLayers().get(0))
                .build();
    }

    private static ModelHparams hparams(int nextn) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.GLM4_MOE))
                .layerCounts(1 + nextn, nextn).nCtxTrain(4096).nEmbd(N_EMBD).nVocab(TARGET_VOCAB)
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

    /** The target the drafter binds to: its backend carries the scripted memory. */
    private static EngineModel target(FakeHead head) {
        ModelSpec spec = head.target();
        FakeBackend backend = new FakeBackend(new FixedMemory(targetPlan()));
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

            @Override public GGUFTokenizer tokenizer() { throw new UnsupportedOperationException(); }

            @Override public void printInfo() { }

            @Override public String architecture() { return spec.arch().ggufName(); }

            @Override public String name() { return architecture(); }

            @Override public int vocabSize() { return TARGET_VOCAB; }

            @Override public int contextLength() { return 32; }

            @Override public int kvCacheContext() { return 32; }

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

    /** The target plan: two layer slots (the trunk block and the NextN block) of one KV group. */
    private static MemoryPlan targetPlan() {
        return MemoryPlan.builder(MemoryKind.KV, 32, 8).config(MemoryConfig.defaults())
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.NONE).build())
                .layer(LayerMemoryPlan.builder(1, LayerMemoryPlan.Kind.NONE).build())
                .build();
    }

    private static final class FakeBackend implements ForwardBackend {
        private final ModelMemory memory;

        FakeBackend(ModelMemory memory) {
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

        @Override public void setHiddenTap(HiddenTap tap) { }

        @Override public void close() { }
    }

    /** A memory over a fixed plan with a sequence of its own. */
    private static final class FixedMemory implements ModelMemory {
        private final MemoryPlan plan;
        private final SequenceState sequence = new SequenceState(32);

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
