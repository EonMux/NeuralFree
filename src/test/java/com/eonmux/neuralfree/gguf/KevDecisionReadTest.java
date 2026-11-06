package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.PagedEvaluations;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.heads.KevScorer;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The kev decision read ({@link GGUFModel#classify(int[], int[])} on a kev-typed checkpoint): every option's score
 * is the dot product of the last prompt token's hidden state with its {@code <|box_end|>} end token's, both rows
 * read as one query and one key half of the output width and scaled by its square root. The reference is computed
 * from the model's own hidden rows.
 */
public class KevDecisionReadTest {

    private static final float TOL = 1e-4f;

    private static EngineConfig paged() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(128).ubatch(8).parallel(2)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    /** A kev prompt of three options over the fixture's template. */
    private static String prompt() {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("type", "choice");
        inp.put("instructions", "Pick one");
        inp.put("state", "the sky is blue");
        List<Map<String, Object>> options = new ArrayList<>();
        for (String[] kv : new String[][] {{"red", "warm"}, {"blue", "cool"}, {"green", "leafy"}}) {
            Map<String, Object> option = new LinkedHashMap<>();
            option.put("key", kv[0]);
            option.put("description", kv[1]);
            options.add(option);
        }
        inp.put("options", options);
        inp.put("images", List.of());
        return MiniJinja.render(SyntheticModels.KEV_TEMPLATE, inp);
    }

    /** The kev read's reference: one dot per {@code <|box_end|} marker, against the last token's row. */
    private static float[] referenceDots(ForwardBackend backend, int[] tokens, int marker, int half) {
        // the same dispatch the model itself uses: paged memory maps its positions through PagedSequences
        float[][] hidden = PagedEvaluations.paged(backend)
                ? PagedEvaluations.hiddenRows(backend, PagedEvaluations.untyped(tokens, null))
                : backend.hiddenRows(tokens, null);
        float[] q = hidden[tokens.length - 1];
        float[] out = new float[3];
        int at = 0;
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] != marker) {
                continue;
            }
            float[] k = hidden[i];
            float dot = 0.0f;
            for (int j = 0; j < half; j++) {
                dot += q[j] * k[half + j];
            }
            out[at++] = dot / (float) Math.sqrt(half);
        }
        assertEquals("the prompt carries one end marker per option", 3, at);
        return out;
    }

    @Test
    public void classifyDotsTheLastTokenWithEveryOptionEndMarker() throws IOException {
        Path model = SyntheticModels.decisionKev().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), paged())) {
            int marker = m.tokenizer().encode(KevScorer.MARKER_TEXT, false, true)[0];
            int[] tokens = m.tokenizer().encode(prompt(), false, true);
            int nMarkers = 0;
            for (int token : tokens) {
                nMarkers += token == marker ? 1 : 0;
            }
            assertEquals("the rendered prompt carries one end marker per option", 3, nMarkers);
            assertArrayEquals(referenceDots(m.forwardBackend(), tokens, marker, 32), m.classify(tokens, null), TOL);
        }
    }

    @Test
    public void aPromptWithoutAnEndMarkerRefusesByName() throws IOException {
        Path model = SyntheticModels.decisionKev().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), paged())) {
            int[] tokens = m.tokenizer().encode("plain text without any option marker", false, true);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> m.classify(tokens, null));
            assertTrue(e.getMessage(), e.getMessage().contains(KevScorer.MARKER_TEXT));
        }
    }

    @Test
    public void aDecisionOrderRefusesOnAKevModel() throws IOException {
        Path model = SyntheticModels.decisionKev().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), paged())) {
            int[] tokens = m.tokenizer().encode(prompt(), false, true);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> m.classify(tokens, new int[] {0, 0}));
            assertTrue(e.getMessage(), e.getMessage().contains("no decision order"));
        }
    }

    @Test
    public void aKevModelRefusesTheEmbeddingRoutes() throws IOException {
        Path model = SyntheticModels.decisionKev().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), paged())) {
            int[] tokens = m.tokenizer().encode(prompt(), false, true);
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> m.encode(tokens));
            assertTrue(e.getMessage(), e.getMessage().contains("is a decision model"));
            assertThrows(UnsupportedOperationException.class, () -> m.encodeTokens(tokens));
            assertThrows(UnsupportedOperationException.class, () -> m.classifyOutput(tokens));
        }
    }

    @Test
    public void aVocabularyWithoutTheEndTokenRefusesByName() throws IOException {
        // the label-family fixture has the kev type but no <|box_end|> token in its vocabulary
        Path model = SyntheticModels.decisionLabels("kev", SyntheticModels.Config.qwen2(),
                SyntheticModels.KEV_TEMPLATE).writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), paged())) {
            IllegalStateException e = assertThrows(IllegalStateException.class,
                    () -> m.classify(m.tokenizer().encode(prompt(), false, true), null));
            assertEquals("decision model has no " + KevScorer.MARKER_TEXT + " token", e.getMessage());
        }
    }
}
