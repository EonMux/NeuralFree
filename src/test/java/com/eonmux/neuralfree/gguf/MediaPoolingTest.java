package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Pooling over a media prompt ({@link GGUFModel#encodeMedia}, {@link GGUFModel#classifyOutputMedia}), vLLM's
 * multimodal pooling runners: a text-only prompt pools exactly as {@code encode} pools its tokens, embedding rows
 * equal to the tokens' own embedding rows pool as those tokens do (media rows sit where the prompt places them,
 * at their positions), different rows pool differently, a RANK classifier scores the prompt as it scores the
 * tokens, and a contiguous memory or a model without pooling refuses by name.
 */
public class MediaPoolingTest {

    private static final float TOL = 1e-4f;
    private static final int[] TOKENS = {9, 21, 33, 45, 57, 69, 81};

    private static EngineConfig paged(PoolingKind pooling) {
        EngineConfig.Builder b = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build());
        return (pooling == null ? b : b.pooling(pooling)).build();
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

    /** {@code TOKENS} with tokens {@code [from, to)} given as embedding rows at their positions. */
    private static MediaPrompt withRows(int from, int to, float[][] rows) {
        return MediaPrompt.builder()
                .addText(Arrays.copyOfRange(TOKENS, 0, from))
                .addMedia(MediaRopePositions.normal(from, to - from), rows, false)
                .addText(Arrays.copyOfRange(TOKENS, to, TOKENS.length))
                .build();
    }

    @Test
    public void aTextOnlyPromptPoolsAsItsTokens() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        for (PoolingKind pooling : new PoolingKind[]{PoolingKind.LAST, PoolingKind.MEAN, PoolingKind.CLS}) {
            try (GGUFModel m = new GGUFModel(model.toString(), paged(pooling))) {
                assertArrayEquals(pooling.name(), m.encode(TOKENS), m.encodeMedia(MediaPrompt.ofText(TOKENS)), TOL);
            }
        }
    }

    @Test
    public void embeddingRowsPoolWhereThePromptPlacesThem() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        float[][] rows = tokenRows(model, Arrays.copyOfRange(TOKENS, 2, 5));
        try (GGUFModel m = new GGUFModel(model.toString(), paged(PoolingKind.MEAN))) {
            float[] expected = m.encode(TOKENS);
            assertArrayEquals("the tokens' own rows pool as the tokens", expected,
                    m.encodeMedia(withRows(2, 5, rows)), TOL);
            float[][] other = tokenRows(model, new int[]{100, 110, 120});
            assertFalse("other rows pool differently",
                    Arrays.equals(expected, m.encodeMedia(withRows(2, 5, other))));
            assertArrayEquals("the memory is left empty: a repeat pools the same", expected,
                    m.encodeMedia(withRows(2, 5, rows)), TOL);
        }
    }

    @Test
    public void aRankClassifierScoresTheMediaPromptAsItsTokens() throws IOException {
        GgufTestWriter qwen = SyntheticModels.qwen2vl();
        int width = SyntheticModels.Config.llama().nEmbd();
        qwen.u32(qwen.archKey("pooling_type"), SyntheticModels.POOLING_RANK);
        qwen.tensorRandom("cls.weight", GGMLType.F32, 77, -0.1f, 0.1f, width, width);
        qwen.tensorZeros("cls.bias", GGMLType.F32, width);
        qwen.tensorRandom("cls.output.weight", GGMLType.F32, 78, -0.1f, 0.1f, width, 1);
        qwen.tensorZeros("cls.output.bias", GGMLType.F32, 1);
        Path model = qwen.writeTempFile();
        float[][] rows = tokenRows(model, Arrays.copyOfRange(TOKENS, 1, 4));
        try (GGUFModel m = new GGUFModel(model.toString(), paged(null))) {
            float expected = m.classifyOutput(TOKENS).rankScore();
            assertEquals(expected, m.classifyOutputMedia(MediaPrompt.ofText(TOKENS)).rankScore(), TOL);
            assertEquals("text-position media rows score as their tokens", expected,
                    m.classifyOutputMedia(withRows(1, 4, rows)).rankScore(), TOL);
        }
    }

    @Test
    public void aContiguousMemoryAndAModelWithoutPoolingRefuseByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig contiguous = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .pooling(PoolingKind.LAST).build();
        try (GGUFModel m = new GGUFModel(model.toString(), contiguous)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> m.encodeMedia(MediaPrompt.ofText(TOKENS)));
            assertTrue(e.getMessage(), e.getMessage().contains("paged memory"));
        }
        try (GGUFModel m = new GGUFModel(model.toString(), paged(null))) {
            assertThrows(UnsupportedOperationException.class, () -> m.encodeMedia(MediaPrompt.ofText(TOKENS)));
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> m.classifyOutputMedia(MediaPrompt.ofText(TOKENS)));
            assertTrue(e.getMessage(), e.getMessage().contains("RANK"));
        }
    }
}
