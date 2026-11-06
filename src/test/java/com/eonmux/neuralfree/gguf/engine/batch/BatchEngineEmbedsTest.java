package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EmbedsPrompt;
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
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Embedding-row submissions on the batch engine (gap prompt-embeds-input): a rows-only submission runs through
 * the same admission, prefill and decode path as a token submission and produces the same tokens and text as the
 * same prompt through {@link GenerationLoop}, an embedding-row and a token sequence batch together, a
 * preempted row submission recomputes its runs, and an oversize row prompt refuses at submit.
 */
public class BatchEngineEmbedsTest {

    private static final int N_TOKENS = 6;
    private static final int N_NEW = 8;

    private static EngineConfig config(boolean paged) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            mc = mc.parallel(2);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    private static float[][] tableRows(Path model, int[] tokens, int nEmbd) throws IOException {
        try (GGUFFileParser parser = new GGUFFileParser(model.toString())) {
            GGUFTensorInfo table = parser.getTensor("token_embd.weight");
            float[] flat = parser.readTensorDataAsFloat(table);
            float[][] rows = new float[tokens.length][];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = Arrays.copyOfRange(flat, tokens[i] * nEmbd, (tokens[i] + 1) * nEmbd);
            }
            return rows;
        }
    }

    /** The reference: the same prompt and request through the single-sequence GenerationLoop. */
    private static GenerationResult reference(Path model, EmbedsPrompt prompt) throws IOException {
        return reference(model, prompt, request());
    }

    private static GenerationResult reference(Path model, EmbedsPrompt prompt, GenerationRequest request)
            throws IOException {
        EngineConfig cfg = config(false);
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    private static GenerationResult reference(Path model, int[] prompt) throws IOException {
        return reference(model, prompt, request());
    }

    private static GenerationResult reference(Path model, int[] prompt, GenerationRequest request)
            throws IOException {
        EngineConfig cfg = config(false);
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    @Test
    public void anEmbedsSubmissionMatchesTheGenerationLoop() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true);
        EmbedsPrompt prompt;
        try (GGUFModel m = new GGUFModel(model.toString(), paged)) {
            prompt = EmbedsPrompt.ofRows(tableRows(model, tokens(1), m.embeddingLength()));
        }
        GenerationResult twin = reference(model, prompt);

        GenerationResult result;
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence a = engine.submit(prompt, request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, a.state());
            result = a.result();
        }
        assertArrayEquals("the batch engine decodes the row prompt like the generation loop",
                twin.tokens(), result.tokens());
        assertEquals(twin.text(), result.text());
        assertEquals(twin.stopReason(), result.stopReason());
    }

    @Test
    public void anEmbedsAndATokenSequenceBatchTogether() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true);
        EmbedsPrompt rows;
        try (GGUFModel m = new GGUFModel(model.toString(), paged)) {
            rows = EmbedsPrompt.ofRows(tableRows(model, tokens(2), m.embeddingLength()));
        }
        GenerationResult twinRows = reference(model, rows);
        GenerationResult twinTokens = reference(model, tokens(3));

        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence a = engine.submit(rows, request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(tokens(3), request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, a.state());
            assertEquals(BatchSequence.State.DONE, b.state());
            assertArrayEquals(twinRows.tokens(), a.result().tokens());
            assertArrayEquals(twinTokens.tokens(), b.result().tokens());
        }
    }

    @Test
    public void aPreemptedEmbedsSubmissionRecomputesAndKeepsItsResult() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true);
        EmbedsPrompt rows;
        try (GGUFModel m = new GGUFModel(model.toString(), paged)) {
            rows = EmbedsPrompt.ofRows(tableRows(model, tokens(4), m.embeddingLength()));
        }
        GenerationRequest longRun = GenerationRequest.builder().maxNewTokens(40).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
        GenerationResult twinRows = reference(model, rows, longRun);
        int[] other = tokens(5);
        GenerationResult twinOther = reference(model, other, longRun);

        GenerationResult soloRows;
        try (GGUFModel m = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(m, paged)) {
            BatchSequence one = engine.submit(rows, longRun, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, one);
            soloRows = one.result();
        }
        // 12 rows + 12 tokens with 40 tokens each overflow the 64 shared cells: the row sequence is preempted
        // mid-decode, recomputes its runs and continues unchanged
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence first = engine.submit(rows, longRun, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence second = engine.submit(other, longRun, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, first, second);
            assertArrayEquals("the row sequence decodes as it does alone", soloRows.tokens(), first.result().tokens());
            assertArrayEquals(twinRows.tokens(), first.result().tokens());
            assertArrayEquals("the token sequence decodes as it does alone", twinOther.tokens(),
                    second.result().tokens());
            assertTrue(engine.idle());
        }
    }

    @Test
    public void aSplicedEmbedsSubmissionMatchesTheGenerationLoop() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true);
        int[] tokens = tokens(5);
        EmbedsPrompt prompt;
        try (GGUFModel m = new GGUFModel(model.toString(), paged)) {
            float[][] rows = tableRows(model, Arrays.copyOfRange(tokens, 3, 6), m.embeddingLength());
            prompt = EmbedsPrompt.of(
                    new int[][] {Arrays.copyOfRange(tokens, 0, 3), new int[0]},
                    new float[][][] {rows});
        }
        GenerationResult twin = reference(model, prompt);

        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence a = engine.submit(prompt, request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertArrayEquals(twin.tokens(), a.result().tokens());
            assertEquals(twin.text(), a.result().text());
        }
    }

    @Test
    public void aSplicedPromptOnAnMRopeModelDecodesLikeItsTokenTwin() throws IOException {
        // the rows run as a NORMAL media chunk: every rope section carries the row position, as the text rows'
        // positions do, so table rows continue exactly like the tokens they came from
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        EngineConfig paged = config(true);
        int[] tokens = tokens(6);
        EmbedsPrompt prompt;
        try (GGUFModel m = new GGUFModel(model.toString(), paged)) {
            float[][] rows = tableRows(model, Arrays.copyOfRange(tokens, 2, 5), m.embeddingLength());
            prompt = EmbedsPrompt.of(
                    new int[][] {Arrays.copyOfRange(tokens, 0, 2), Arrays.copyOfRange(tokens, 5, 6)},
                    new float[][][] {rows});
        }
        GenerationResult loop = reference(model, prompt);
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence rows = engine.submit(prompt, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence twin = engine.submit(tokens, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, rows, twin);
            assertArrayEquals("the spliced rows continue like the tokens they came from", twin.result().tokens(),
                    rows.result().tokens());
            assertArrayEquals("the batch engine decodes the row prompt like the generation loop", loop.tokens(),
                    rows.result().tokens());
        }
    }

    @Test
    public void oversizeRowsRefuseAtSubmit() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true);
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            float[][] rows = new float[65][two.embeddingLength()];
            for (float[] row : rows) {
                Arrays.fill(row, 0.25f);
            }
            ContextFullException e = assertThrows(ContextFullException.class,
                    () -> engine.submit(EmbedsPrompt.ofRows(rows), request(),
                            (com.eonmux.neuralfree.gguf.TokenCallback) null));
            assertEquals(65, e.requiredPositions());
        }
    }

    /** Pumps until every handle finished, failing instead of spinning when the engine stops making progress. */
    private static void pumpUntilFinished(BatchEngine engine, BatchSequence... handles) {
        for (int step = 0; step < 2_000; step++) {
            boolean all = true;
            for (BatchSequence h : handles) {
                all &= h.finished();
            }
            if (all) {
                for (BatchSequence h : handles) {
                    assertEquals("the request completed: " + h.error(), BatchSequence.State.DONE, h.state());
                }
                return;
            }
            engine.pump();
        }
        throw new AssertionError("the engine stopped admitting: a queued request never ran");
    }
}
