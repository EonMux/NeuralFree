package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.engine.EmbeddingSimilarity;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * The output block of {@code nftool embedding}, byte for byte the printf flows of llama.cpp's embedding example
 * (examples/embedding/embedding.cpp:292-410): the default preview in the pooling's shape (per-token rows, rerank
 * scores, or one row per prompt with the cosine similarity matrix for several prompts), or the {@code json},
 * {@code json+}, {@code array} and {@code raw} formats of {@code --embd-output-format}. Values print with
 * {@code %1.0f}/{@code %6.0f} under the int16 normalization and {@code %1.7f}/{@code %9.6f} otherwise.
 */
final class EmbeddingOutput {

    /** The {@code --embd-output-format} values; the empty format is the default preview. */
    static final List<String> FORMATS =
            Collections.unmodifiableList(Arrays.asList("", "array", "json", "json+", "raw"));

    private EmbeddingOutput() {
    }

    /**
     * Appends the output block of {@code format} over normalized rows, closed by llama.cpp's final newline.
     *
     * @param prompts  the prompts the rows come from (the matrix labels)
     * @param labels   the classifier labels of a RANK model; a missing or empty one prints as its index
     * @param embdNorm the {@code --embd-normalize} value the rows carry
     * @throws IllegalArgumentException for an unknown format or no rows
     */
    static void render(StringBuilder out, List<float[]> rows, List<String> prompts, LLMModel.PoolingType pooling,
                       String[] labels, String format, int embdNorm) {
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("no embeddings to print");
        }
        switch (format) {
            case "":
                renderDefault(out, rows, prompts, pooling, labels, embdNorm);
                break;
            case "json":
                renderJson(out, rows, prompts.size(), false, embdNorm);
                break;
            case "json+":
                renderJson(out, rows, prompts.size(), true, embdNorm);
                break;
            case "array":
                renderArray(out, rows, embdNorm);
                break;
            case "raw":
                renderRaw(out, rows, embdNorm);
                break;
            default:
                throw new IllegalArgumentException("unknown embedding output format '" + format + "'");
        }
        out.append('\n');
    }

    /** The default preview: llama.cpp's leading blank line, then the rows in the pooling's shape. */
    private static void renderDefault(StringBuilder out, List<float[]> rows, List<String> prompts,
                                      LLMModel.PoolingType pooling, String[] labels, int embdNorm) {
        out.append('\n');
        if (pooling == LLMModel.PoolingType.NONE) {
            renderPerTokenPreview(out, rows, embdNorm);
        } else if (pooling == LLMModel.PoolingType.RANK) {
            renderRankScores(out, rows, labels);
        } else {
            renderPooledPreview(out, rows, prompts, embdNorm);
        }
    }

    /** Per-token rows, first three and last three values (embedding.cpp:295-311). */
    private static void renderPerTokenPreview(StringBuilder out, List<float[]> rows, int embdNorm) {
        for (int j = 0; j < rows.size(); j++) {
            float[] row = rows.get(j);
            out.append("embedding ").append(j).append(": ");
            for (int i = 0; i < Math.min(3, row.length); i++) {
                out.append(preview(row[i], embdNorm));
            }
            out.append(" ... ");
            for (int i = Math.max(0, row.length - 3); i < row.length; i++) {
                out.append(preview(row[i], embdNorm));
            }
            out.append('\n');
        }
    }

    /** Pooled rows, up to sixteen values each for several prompts, and their cosine similarity matrix. */
    private static void renderPooledPreview(StringBuilder out, List<float[]> rows, List<String> prompts,
                                            int embdNorm) {
        int n = prompts.size();
        for (int j = 0; j < n; j++) {
            float[] row = rows.get(j);
            out.append("embedding ").append(j).append(": ");
            int cols = n > 1 ? Math.min(16, row.length) : row.length;
            for (int i = 0; i < cols; i++) {
                out.append(preview(row[i], embdNorm));
            }
            out.append('\n');
        }
        if (n > 1) {
            out.append("\ncosine similarity matrix:\n\n");
            for (String prompt : prompts) {
                out.append(String.format(Locale.ROOT, "%6.6s ", prompt));
            }
            out.append('\n');
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    float sim = EmbeddingSimilarity.cosine(rows.get(i), rows.get(j));
                    out.append(String.format(Locale.ROOT, "%6.2f ", sim));
                }
                out.append(String.format(Locale.ROOT, "%1.10s", prompts.get(i))).append('\n');
            }
        }
    }

    /** RANK pooling's scores (embedding.cpp:312-333): one line per class, labelled when there are several. */
    private static void renderRankScores(StringBuilder out, List<float[]> rows, String[] labels) {
        for (int j = 0; j < rows.size(); j++) {
            float[] row = rows.get(j);
            if (row.length == 1) {
                out.append(String.format(Locale.ROOT, "rerank score %d: %8.3f\n", j, row[0]));
                continue;
            }
            for (int i = 0; i < row.length; i++) {
                out.append(String.format(Locale.ROOT, "rerank score %d: %8.3f [%s]\n", j, row[i], label(labels, i)));
            }
        }
    }

    /** llama.cpp's class label: the model's label, or the index when it has none or an empty one. */
    static String label(String[] labels, int i) {
        String l = labels != null && i < labels.length ? labels[i] : null;
        return l == null || l.isEmpty() ? Integer.toString(i) : l;
    }

    private static String preview(float v, int embdNorm) {
        return String.format(Locale.ROOT, embdNorm == 0 ? "%6.0f " : "%9.6f ", v);
    }

    /** The openai-style list, plus the cosine similarity matrix for {@code json+} (embedding.cpp:366-401). */
    private static void renderJson(StringBuilder out, List<float[]> rows, int nPrompts, boolean plus, int embdNorm) {
        out.append("{\n  \"object\": \"list\",\n  \"data\": [\n");
        for (int j = 0; j < rows.size(); j++) {
            if (j > 0) {
                out.append(",\n");
            }
            out.append("    {\n      \"object\": \"embedding\",\n      \"index\": ").append(j)
                    .append(",\n      \"embedding\": [");
            appendValues(out, rows.get(j), embdNorm);
            out.append("]\n    }");
        }
        out.append("\n  ]");
        if (plus && nPrompts > 1) {
            out.append(",\n  \"cosineSimilarity\": [\n");
            for (int i = 0; i < rows.size(); i++) {
                if (i > 0) {
                    out.append(",\n");
                }
                out.append("    [");
                for (int j = 0; j < rows.size(); j++) {
                    if (j > 0) {
                        out.append(", ");
                    }
                    out.append(String.format(Locale.ROOT, "%6.2f",
                            EmbeddingSimilarity.cosine(rows.get(i), rows.get(j))));
                }
                out.append(" ]");
            }
            out.append("\n  ]");
        }
        out.append("\n}\n");
    }

    /** The {@code [[],[]...]} form (embedding.cpp:366-382 with {@code notArray} false). */
    private static void renderArray(StringBuilder out, List<float[]> rows, int embdNorm) {
        out.append('[');
        for (int j = 0; j < rows.size(); j++) {
            if (j > 0) {
                out.append(',');
            }
            out.append('[');
            appendValues(out, rows.get(j), embdNorm);
            out.append(']');
        }
        out.append("]\n");
    }

    /** One whitespace-delimited embedding per line ({@code print_raw_embeddings}). */
    private static void renderRaw(StringBuilder out, List<float[]> rows, int embdNorm) {
        for (float[] row : rows) {
            for (int i = 0; i < row.length; i++) {
                if (i > 0) {
                    out.append(' ');
                }
                out.append(plainValue(row[i], embdNorm));
            }
            out.append('\n');
        }
    }

    private static void appendValues(StringBuilder out, float[] row, int embdNorm) {
        for (int i = 0; i < row.length; i++) {
            if (i > 0) {
                out.append(',');
            }
            out.append(plainValue(row[i], embdNorm));
        }
    }

    private static String plainValue(float v, int embdNorm) {
        return String.format(Locale.ROOT, embdNorm == 0 ? "%1.0f" : "%1.7f", v);
    }
}
