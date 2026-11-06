package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead.DecisionOrder;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;

/**
 * The decision reads over a media prompt ({@link GGUFModel#nextLogits(MediaPrompt)},
 * {@link GGUFModel#classifyMedia}): a prompt whose media rows are the tokens' own embedding rows at the tokens'
 * positions reads exactly as the tokens do, so the label read's logits and a decision head's per-option scores
 * are the same over either form. The memory must be paged.
 */
public class DecisionMediaReadTest {

    private static final float TOL = 1e-4f;
    private static final int[] TOKENS = {9, 21, 33, 45, 57, 69, 81};

    private static EngineConfig paged() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    /** The {@code token_embd} rows of {@code ids} as stored in {@code model}. */
    private static float[][] tokenRows(Path model, int[] ids) throws IOException {
        try (ModelSource source = ModelSources.open(model.toString())) {
            TensorCatalog tc = TensorCatalog.of(source);
            try (CpuWeights w = CpuWeights.open(source, tc,
                    InputTransformLoader.load(source, GgufKv.of(source), tc, ActivationQuant.WEIGHT_ONLY))) {
                CpuTensorRef embd = w.get("token_embd.weight");
                float[][] rows = new float[ids.length][(int) embd.cols()];
                for (int i = 0; i < ids.length; i++) {
                    embd.decodeRow(ids[i], rows[i], 0);
                }
                return rows;
            }
        }
    }

    /** {@link #TOKENS} with tokens {@code [from, to)} given as embedding rows at their positions. */
    private static MediaPrompt withRows(Path model, int from, int to) throws IOException {
        return MediaPrompt.builder()
                .addText(Arrays.copyOfRange(TOKENS, 0, from))
                .addMedia(MediaRopePositions.normal(from, to - from), tokenRows(model, Arrays.copyOfRange(TOKENS,
                        from, to)), false)
                .addText(Arrays.copyOfRange(TOKENS, to, TOKENS.length))
                .build();
    }

    @Test
    public void theLabelReadsLogitsAreTheSameOverTheTokensOwnRows() throws IOException {
        Path model = SyntheticModels.decisionLabels("openjev",
                SyntheticModels.Config.qwen2().toBuilder().nCtxTrain(64).build()).writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), paged())) {
            assertArrayEquals(m.nextLogits(TOKENS), m.nextLogits(withRows(model, 2, 5)), TOL);
        }
    }

    @Test
    public void aDecisionHeadsScoresAreTheSameOverTheTokensOwnRows() throws IOException {
        Path model = SyntheticModels.vllmDecision2(
                SyntheticModels.Config.qwen2().toBuilder().nCtxTrain(64).build()).writeTempFile();
        int[] order = {DecisionOrder.NONE, DecisionOrder.QUESTION_NOUL, DecisionOrder.NONE,
                DecisionOrder.OPTION, DecisionOrder.NONE, DecisionOrder.OPTION, DecisionOrder.NONE};
        try (GGUFModel m = new GGUFModel(model.toString(), paged())) {
            assertArrayEquals(m.classify(TOKENS, order), m.classifyMedia(withRows(model, 2, 5), order), TOL);
        }
    }
}
