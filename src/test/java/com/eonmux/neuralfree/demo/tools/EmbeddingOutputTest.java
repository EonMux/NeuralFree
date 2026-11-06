package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.LLMModel;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link EmbeddingOutput} against the printf flows of llama.cpp's embedding example: the default preview in each
 * pooling's shape (pooled rows with the cosine similarity matrix, per-token rows, labelled rerank scores) and the
 * json, json+, array and raw formats, byte for byte, under the float and the int16 normalizations.
 */
public class EmbeddingOutputTest {

    private static final String[] NO_LABELS = new String[0];

    private static String render(List<float[]> rows, List<String> prompts, LLMModel.PoolingType pooling,
                                 String[] labels, String format, int embdNorm) {
        StringBuilder out = new StringBuilder();
        EmbeddingOutput.render(out, rows, prompts, pooling, labels, format, embdNorm);
        return out.toString();
    }

    @Test
    public void defaultPreviewPrintsRowsAndTheSimilarityMatrixForSeveralPrompts() {
        List<float[]> rows = Arrays.asList(new float[]{1, 0}, new float[]{0, 1}, new float[]{0.6f, 0.8f});
        String out = render(rows, Arrays.asList("aa", "bb", "cc"), LLMModel.PoolingType.MEAN, NO_LABELS, "", 2);
        assertEquals("\n"
                + "embedding 0:  1.000000  0.000000 \n"
                + "embedding 1:  0.000000  1.000000 \n"
                + "embedding 2:  0.600000  0.800000 \n"
                + "\n"
                + "cosine similarity matrix:\n"
                + "\n"
                + "    aa     bb     cc \n"
                + "  1.00   0.00   0.60 aa\n"
                + "  0.00   1.00   0.80 bb\n"
                + "  0.60   0.80   1.00 cc\n"
                + "\n", out);
    }

    /** The matrix header pads or cuts each prompt to six characters, the row label cuts it to ten. */
    @Test
    public void theMatrixLabelsCutLongPromptsLikePrintf() {
        List<float[]> rows = Arrays.asList(new float[]{1, 0}, new float[]{1, 0});
        String out = render(rows, Arrays.asList("a long first prompt", "b"), LLMModel.PoolingType.CLS, NO_LABELS,
                "", 2);
        assertEquals(true, out.contains("\na long      b \n"));
        assertEquals(true, out.contains("  1.00   1.00 a long fir\n"));
        assertEquals(true, out.contains("  1.00   1.00 b\n"));
    }

    /** Several prompts preview sixteen values each; one prompt prints its whole row. */
    @Test
    public void severalPromptsPreviewSixteenValues() {
        float[] wide = new float[20];
        Arrays.fill(wide, 0.5f);
        String out = render(Arrays.asList(wide, wide), Arrays.asList("x", "y"), LLMModel.PoolingType.MEAN,
                NO_LABELS, "", 2);
        String first = out.split("\n")[1];
        assertEquals(16, first.substring("embedding 0: ".length()).trim().split("\\s+").length);
        String single = render(Collections.singletonList(wide), Collections.singletonList("x"),
                LLMModel.PoolingType.MEAN, NO_LABELS, "", 2);
        assertEquals(20, single.split("\n")[1].substring("embedding 0: ".length()).trim().split("\\s+").length);
    }

    @Test
    public void theSinglePromptPreviewPrintsEveryValueAndNoMatrix() {
        assertEquals("\nembedding 0:  0.250000 -0.500000 \n\n", render(
                Collections.singletonList(new float[]{0.25f, -0.5f}), Collections.singletonList("only"),
                LLMModel.PoolingType.MEAN, NO_LABELS, "", 2));
    }

    @Test
    public void jsonMatchesLlamaCppByteForByte() {
        assertEquals("{\n"
                + "  \"object\": \"list\",\n"
                + "  \"data\": [\n"
                + "    {\n"
                + "      \"object\": \"embedding\",\n"
                + "      \"index\": 0,\n"
                + "      \"embedding\": [1.0000000,2.0000000]\n"
                + "    }\n"
                + "  ]\n"
                + "}\n"
                + "\n", render(Collections.singletonList(new float[]{1, 2}), Collections.singletonList("a"),
                LLMModel.PoolingType.MEAN, NO_LABELS, "json", 2));
    }

    @Test
    public void jsonPlusAppendsTheCosineSimilarityMatrix() {
        assertEquals("{\n"
                + "  \"object\": \"list\",\n"
                + "  \"data\": [\n"
                + "    {\n"
                + "      \"object\": \"embedding\",\n"
                + "      \"index\": 0,\n"
                + "      \"embedding\": [1.0000000,0.0000000]\n"
                + "    },\n"
                + "    {\n"
                + "      \"object\": \"embedding\",\n"
                + "      \"index\": 1,\n"
                + "      \"embedding\": [0.0000000,1.0000000]\n"
                + "    }\n"
                + "  ],\n"
                + "  \"cosineSimilarity\": [\n"
                + "    [  1.00,   0.00 ],\n"
                + "    [  0.00,   1.00 ]\n"
                + "  ]\n"
                + "}\n"
                + "\n", render(Arrays.asList(new float[]{1, 0}, new float[]{0, 1}), Arrays.asList("a", "b"),
                LLMModel.PoolingType.MEAN, NO_LABELS, "json+", 2));
    }

    /** json+ of one prompt has no matrix, as llama.cpp's {@code n_prompts > 1} guard leaves it out. */
    @Test
    public void jsonPlusOfOnePromptHasNoMatrix() {
        String out = render(Collections.singletonList(new float[]{1, 0}), Collections.singletonList("a"),
                LLMModel.PoolingType.MEAN, NO_LABELS, "json+", 2);
        assertEquals(false, out.contains("cosineSimilarity"));
        assertEquals(true, out.endsWith("  ]\n}\n\n"));
    }

    @Test
    public void arrayAndRawMatchLlamaCppByteForByte() {
        assertEquals("[[1.0000000,2.0000000],[3.0000000,4.0000000]]\n\n", render(
                Arrays.asList(new float[]{1, 2}, new float[]{3, 4}), Arrays.asList("a", "b"),
                LLMModel.PoolingType.MEAN, NO_LABELS, "array", 2));
        assertEquals("1.0000000 2.0000000\n3.0000000 4.0000000\n\n", render(
                Arrays.asList(new float[]{1, 2}, new float[]{3, 4}), Arrays.asList("a", "b"),
                LLMModel.PoolingType.MEAN, NO_LABELS, "raw", 2));
    }

    /** The int16 scaling prints llama.cpp's {@code %1.0f} and {@code %6.0f} integer forms. */
    @Test
    public void theInt16NormalizationPrintsIntegerValues() {
        assertEquals(true, render(Collections.singletonList(new float[]{1f, -0.4f}), Collections.singletonList("a"),
                LLMModel.PoolingType.MEAN, NO_LABELS, "json", 0).contains("[1,-0]"));
        assertEquals("\nembedding 0:  32760      0 \n\n", render(Collections.singletonList(new float[]{32760f, 0f}),
                Collections.singletonList("only"), LLMModel.PoolingType.MEAN, NO_LABELS, "", 0));
    }

    @Test
    public void perTokenPreviewPrintsThreeLeadingAndTrailingValues() {
        float[] row = new float[8];
        for (int i = 0; i < row.length; i++) {
            row[i] = i + 1;
        }
        String line = render(Collections.singletonList(row), Collections.singletonList("a"),
                LLMModel.PoolingType.NONE, NO_LABELS, "", 2).split("\n")[1];
        assertEquals("embedding 0:  1.000000  2.000000  3.000000  ...  6.000000  7.000000  8.000000 ", line);
    }

    /** One score prints bare; several print one line per class with the model's label or the index. */
    @Test
    public void rankPoolingPrintsLabelledScores() {
        String out = render(Arrays.asList(new float[]{0.5f}, new float[]{1.25f, -2f, 3f}), Arrays.asList("a", "b"),
                LLMModel.PoolingType.RANK, new String[]{"yes", ""}, "", -1);
        assertEquals("\n"
                + "rerank score 0:    0.500\n"
                + "rerank score 1:    1.250 [yes]\n"
                + "rerank score 1:   -2.000 [1]\n"
                + "rerank score 1:    3.000 [2]\n"
                + "\n", out);
    }

    @Test
    public void renderRefusesAnUnknownFormatAndNoRows() {
        assertThrows(IllegalArgumentException.class, () -> render(Collections.singletonList(new float[]{1}),
                Collections.singletonList("a"), LLMModel.PoolingType.MEAN, NO_LABELS, "csv", 2));
        assertThrows(IllegalArgumentException.class, () -> render(Collections.emptyList(),
                Collections.singletonList("a"), LLMModel.PoolingType.MEAN, NO_LABELS, "", 2));
    }
}
