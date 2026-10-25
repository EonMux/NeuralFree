package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * PLaMo's {@code tokenizer.jsonl} as llama.cpp's {@code _set_vocab_plamo} reads it (LC/conversion/base.py:2720-2800):
 * every line that is not blank is a JSON array {@code [token, score, type, ...]}; the token id is the index among the
 * non-blank lines, and an absent type means {@code "NORMAL"}. Further columns are ignored, as the converter ignores
 * them. Instances are immutable.
 */
public final class PlamoVocabReader {

    /** The type of a row without a third column. */
    public static final String DEFAULT_TYPE = "NORMAL";

    private final String[] tokens;
    private final double[] scores;
    private final String[] types;

    private PlamoVocabReader(String[] tokens, double[] scores, String[] types) {
        this.tokens = tokens;
        this.scores = scores;
        this.types = types;
    }

    /**
     * Reads {@code file} (strict UTF-8, Python text-mode lines, at most {@link HfTokenizerFiles#MAX_FILE_BYTES}).
     *
     * @throws IOException {@code "Invalid Hugging Face checkpoint '<dir>': <file>: line <n>: ..."} for a bad row
     */
    public static PlamoVocabReader read(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        String text = HfTokenizerFiles.readText(file);
        return parse(text, HfTokenizerFiles.filePrefix(file));
    }

    /**
     * Parses the text of a {@code tokenizer.jsonl} file.
     *
     * @throws IOException {@code "Invalid PLaMo vocabulary: line <n>: ..."} for a bad row
     */
    public static PlamoVocabReader parse(String text) throws IOException {
        Objects.requireNonNull(text, "text");
        return parse(text, "Invalid PLaMo vocabulary: ");
    }

    private static PlamoVocabReader parse(String text, String prefix) throws IOException {
        List<String> lines = PythonText.lines(text);
        List<String> tokens = new ArrayList<>();
        List<Double> scores = new ArrayList<>();
        List<String> types = new ArrayList<>();
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i);
            if (PythonText.strip(line).isEmpty()) {
                continue;
            }
            String where = prefix + "line " + (i + 1) + ": ";
            if (tokens.size() == HfTokenizerFiles.MAX_TOKENS) {
                throw new IOException(where + "more than " + HfTokenizerFiles.MAX_TOKENS + " tokens");
            }
            Object row;
            try {
                row = Json.parse(line, Json.Limits.LENIENT);
            } catch (JsonException e) {
                throw new IOException(where + e.getMessage(), e);
            }
            if (!(row instanceof List) || ((List<?>) row).size() < 2) {
                throw new IOException(where + "expected a JSON array [token, score, type, ...]");
            }
            List<?> cells = (List<?>) row;
            if (!(cells.get(0) instanceof String)) {
                throw new IOException(where + "the token (column 1) is not a string");
            }
            if (!(cells.get(1) instanceof Number)) {
                throw new IOException(where + "the score (column 2) is not a number");
            }
            Object type = cells.size() > 2 ? cells.get(2) : DEFAULT_TYPE;
            if (!(type instanceof String)) {
                throw new IOException(where + "the type (column 3) is not a string");
            }
            tokens.add((String) cells.get(0));
            scores.add(((Number) cells.get(1)).doubleValue());
            types.add((String) type);
        }
        double[] scoreArray = new double[scores.size()];
        for (int i = 0; i < scoreArray.length; i++) {
            scoreArray[i] = scores.get(i);
        }
        return new PlamoVocabReader(tokens.toArray(new String[0]), scoreArray, types.toArray(new String[0]));
    }

    /** Number of tokens; ids are {@code 0 .. size() - 1}. */
    public int size() { return tokens.length; }

    public String token(int id) { return tokens[id]; }

    public double score(int id) { return scores[id]; }

    /** The type column ({@code NORMAL}, {@code UNKNOWN}, {@code CONTROL}, {@code BYTE}, ...), as written. */
    public String type(int id) { return types[id]; }

    /** The tokens by id (a copy). */
    public String[] tokens() { return tokens.clone(); }

    /** The scores by id (a copy). */
    public double[] scores() { return scores.clone(); }

    @Override
    public String toString() {
        return "PlamoVocabReader{" + tokens.length + " tokens}";
    }
}
