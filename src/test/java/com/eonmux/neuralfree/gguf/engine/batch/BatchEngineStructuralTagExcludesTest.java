package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

/**
 * A dispatching structural tag's free-text exclusions hold on the {@link BatchEngine} as on the single-sequence
 * {@link GenerationLoop}: the {@code excludes} strings of {@code triggered_tags} and the {@code exclude_tokens} of
 * {@code token_dispatch} mask the free text while the constraint waits for its trigger, and the batched tokens equal
 * the single-sequence run of the same request. The excluded string and token are what the unconstrained greedy run
 * writes first, so an ignored exclusion shows in the first token.
 */
public class BatchEngineStructuralTagExcludesTest {

    private static final int N_NEW = 6;
    private static final int[] PROMPT = {11, 24, 37, 50, 63, 76};

    private static EngineConfig config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        if (parallel > 1) {
            mc.parallel(parallel);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static GenerationRequest greedy(ConstraintSpec constraint) {
        GenerationRequest.Builder b = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding());
        if (constraint != null) {
            b.constraint(constraint);
        }
        return b.build();
    }

    private static GenerationResult batched(GGUFModel model, EngineConfig cfg, GenerationRequest request) {
        try (BatchEngine engine = BatchEngine.forModel(model, cfg)) {
            BatchSequence sequence = engine.submit(PROMPT, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(String.valueOf(sequence.error()), BatchSequence.State.DONE, sequence.state());
            return sequence.result();
        }
    }

    private static GenerationResult single(Path model, GenerationRequest request) throws IOException {
        EngineConfig cfg = config(1);
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(PROMPT, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    /** The structural-tag spec of a top-level {@code format}. */
    private static ConstraintSpec structuralTag(String format) {
        return ConstraintSpec.structuralTag("{\"format\": " + format + "}");
    }

    /** A vocabulary id that no token of {@code avoid} equals and whose text is plain ASCII, for a trigger. */
    private static int unusedToken(GGUFTokenizer tokenizer, int[] avoid) {
        for (int id = 1; id < tokenizer.vocabSize(); id++) {
            int candidate = id;
            String text = tokenizer.decode(new int[]{id});
            if (!text.isEmpty() && text.chars().allMatch(c -> c > ' ' && c < 127)
                    && Arrays.stream(avoid).noneMatch(t -> t == candidate)) {
                return id;
            }
        }
        throw new AssertionError("no unused token");
    }

    @Test
    public void triggeredTagsExcludedStringsMaskTheBatchedFreeText() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(2);
        try (GGUFModel two = new GGUFModel(model.toString(), cfg)) {
            GenerationResult free = batched(two, cfg, greedy(null));
            assertTrue(free.tokenCount() > 0);
            String excluded = two.tokenizer().decode(new int[]{free.tokens()[0]});
            assertFalse("the first greedy token writes text", excluded.isEmpty());
            String format = "{\"type\": \"triggered_tags\", \"triggers\": [\"<zq\"], \"excludes\": ["
                    + jsonString(excluded) + "], \"tags\": [{\"begin\": \"<zq>\", \"content\": {\"type\":"
                    + " \"const_string\", \"value\": \"1\"}, \"end\": \"</zq>\"}]}";
            GenerationRequest request = greedy(structuralTag(format));
            GenerationResult out = batched(two, cfg, request);
            assertTrue(out.tokenCount() > 0);
            assertFalse("the batched free text spells the excluded '" + excluded + "': " + out.text(),
                    out.text().contains(excluded));
            assertArrayEquals("the batched run matches the single-sequence run", single(model, request).tokens(),
                    out.tokens());
        }
    }

    @Test
    public void tokenDispatchExcludedTokensMaskTheBatchedFreeText() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(2);
        try (GGUFModel two = new GGUFModel(model.toString(), cfg)) {
            GenerationResult free = batched(two, cfg, greedy(null));
            assertTrue(free.tokenCount() > 0);
            int excluded = free.tokens()[0];
            int trigger = unusedToken(two.tokenizer(), free.tokens());
            String format = "{\"type\": \"token_dispatch\", \"rules\": [[" + trigger + ", {\"type\": \"const_string\","
                    + " \"value\": \"1\"}]], \"exclude_tokens\": [" + excluded + "]}";
            GenerationRequest request = greedy(structuralTag(format));
            GenerationResult out = batched(two, cfg, request);
            assertTrue(out.tokenCount() > 0);
            for (int token : out.tokens()) {
                assertNotEquals("the batched free text sampled the excluded token", excluded, token);
            }
            assertArrayEquals("the batched run matches the single-sequence run", single(model, request).tokens(),
                    out.tokens());
        }
    }

    private static String jsonString(String s) {
        StringBuilder b = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            if (c == '"' || c == '\\') {
                b.append('\\').append(c);
            } else if (c < 0x20) {
                b.append(String.format("\\u%04x", (int) c));
            } else {
                b.append(c);
            }
        }
        return b.append('"').toString();
    }
}
