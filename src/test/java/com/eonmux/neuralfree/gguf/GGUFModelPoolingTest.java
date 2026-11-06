package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The pooling heads' service entry points over the synthetic models: {@code nextLogits} answers the
 * distribution a greedy generation samples from and refuses models without token logits; {@code
 * classifyOutput} carries a RANK classifier's scores with their labels and refuses models that pool by
 * something else.
 */
public class GGUFModelPoolingTest {

    private static EngineConfig llamaConfig() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    private static int[] prompt(int seed) {
        int[] t = new int[6];
        for (int i = 0; i < t.length; i++) {
            t[i] = 10 + (seed * 7 + i * 3) % 200;
        }
        return t;
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    @Test
    public void nextLogitsMatchTheGreedyChoice() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = llamaConfig();
        try (GGUFModel m = new GGUFModel(model.toString(), config)) {
            int[] tokens = prompt(2);
            float[] logits = m.nextLogits(tokens);
            assertEquals("one logit per vocabulary entry", m.vocabSize(), logits.length);
            GenerationResult r = GenerationLoop.forModel(m, config).generate(tokens,
                    GenerationRequest.builder().maxNewTokens(1).sampler(SamplerConfig.greedyDecoding()).build(),
                    (TokenCallback) null);
            assertEquals("the greedy sample takes the logit argmax", r.tokens()[0], argmax(logits));
            float[] again = m.nextLogits(prompt(3));
            assertEquals("a different prompt answers through the same entry point", m.vocabSize(), again.length);
        }
    }

    @Test
    public void nextLogitsRunOverAPagedMemory() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), llamaConfig())) {
            int[] tokens = prompt(2);
            float[] contiguous = m.nextLogits(tokens);
            try (GGUFModel paged = new GGUFModel(model.toString(), EngineConfig.builder()
                    .memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                            .typeK(KvDType.F32).typeV(KvDType.F32).build()).build())) {
                float[] logits = paged.nextLogits(tokens);
                assertEquals("the paged memory answers the same distribution", m.vocabSize(), logits.length);
                assertArrayEquals(contiguous, logits, 1e-6f);
                float[] second = paged.nextLogits(prompt(3));
                assertEquals("the scratch sequence releases, so a second forward fits", m.vocabSize(),
                        second.length);
            }
        }
    }

    @Test
    public void nextLogitsRefuseModelsWithoutTokenLogits() throws IOException {
        Path bert = SyntheticModels.bert().writeTempFile();
        try (GGUFModel m = new GGUFModel(bert.toString(), EngineConfig.builder().build())) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> m.nextLogits(prompt(1)));
            assertTrue(e.getMessage(), e.getMessage().contains("generative"));
        }
    }

    @Test
    public void classifyOutputCarriesTheRankClassifierScores() throws IOException {
        Path rank = SyntheticModels.rank("neg", "pos").writeTempFile();
        try (GGUFModel m = new GGUFModel(rank.toString(), EngineConfig.builder().build())) {
            ClassifierOutput out = m.classifyOutput(prompt(4));
            assertEquals("one score per label", 2, out.size());
            assertArrayEquals(new String[]{"neg", "pos"}, out.labels());
            assertFalse("the synthetic head applies no softmax of its own", out.normalized());
            double sum = 0;
            for (float p : out.probabilities()) {
                sum += p;
            }
            assertEquals("the probabilities softmax over the outputs", 1.0, sum, 1e-5);
            assertEquals("the labels answer the scores", out.label(out.best()), out.bestLabel());
            float[] scores = m.classify(prompt(5));
            assertEquals("classify reads the same head", 2, scores.length);
            int[] pair = com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead.rerankInput(m.tokenizer(),
                    prompt(1), prompt(2));
            assertEquals("rerank reads the first classifier output of the pair", m.classify(pair)[0],
                    m.rerank(prompt(1), prompt(2)), 1e-6f);
        }
    }

    @Test
    public void classifyOutputRefusesModelsThatPoolBySomethingElse() throws IOException {
        Path bert = SyntheticModels.bert().writeTempFile();
        try (GGUFModel m = new GGUFModel(bert.toString(), EngineConfig.builder().build())) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> m.classifyOutput(prompt(1)));
            assertTrue(e.getMessage(), e.getMessage().contains("RANK"));
        }
    }
}
