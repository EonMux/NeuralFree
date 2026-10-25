package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A Mistral tekken tokenizer ({@code tekken.json}) as mistral_common's {@code Tekkenizer.from_file} loads it: the
 * split {@code pattern}, the {@code version}, the special tokens (ids {@code 0 .. numSpecialTokens() - 1}) and the
 * tiktoken ranks of the inner vocabulary (ids {@code numSpecialTokens() + rank}).
 *
 * <p>Rules applied as mistral_common applies them: a file without {@code special_tokens} gets the deprecated default
 * list when its version is at most v7 and is rejected otherwise; special tokens are unique, their rank is their index,
 * and the list is filled with {@code <SPECIAL_<i>>} up to {@code default_num_special_tokens}; the inner vocabulary is
 * the first {@code default_vocab_size - default_num_special_tokens} entries of {@code vocab}, whose rank is their
 * index, whose bytes are unique, and whose first 256 entries are the single bytes. Instances are immutable.</p>
 */
public final class TekkenReader {

    /** The special tokens of tekken files up to v7 that do not list them (mistral_common's deprecated defaults). */
    private static final List<String> DEPRECATED_SPECIAL_TOKENS = Collections.unmodifiableList(Arrays.asList(
            "<unk>", "<s>", "</s>", "[INST]", "[/INST]", "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]", "[TOOL_RESULTS]",
            "[/TOOL_RESULTS]", "[TOOL_CALLS]", "[IMG]", "<pad>", "[IMG_BREAK]", "[IMG_END]", "[PREFIX]", "[MIDDLE]",
            "[SUFFIX]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[TOOL_CONTENT]"));
    private static final int LAST_VERSION_WITH_DEFAULT_SPECIALS = 7;
    private static final Pattern VERSION = Pattern.compile("v(\\d{1,6})");
    private static final String FILLER_FORMAT = "<SPECIAL_%d>";

    private final String version;
    private final int versionNumber;
    private final String pattern;
    private final int vocabSize;
    private final int numVocabTokens;
    private final List<String> specialTokens;
    private final boolean[] specialIsControl;
    private final Map<String, Integer> specialIds;
    private final byte[][] innerVocab;

    private TekkenReader(String version, int versionNumber, String pattern, int vocabSize, int numVocabTokens,
                         List<String> specialTokens, boolean[] specialIsControl, Map<String, Integer> specialIds,
                         byte[][] innerVocab) {
        this.version = version;
        this.versionNumber = versionNumber;
        this.pattern = pattern;
        this.vocabSize = vocabSize;
        this.numVocabTokens = numVocabTokens;
        this.specialTokens = specialTokens;
        this.specialIsControl = specialIsControl;
        this.specialIds = specialIds;
        this.innerVocab = innerVocab;
    }

    /**
     * Reads {@code file} ({@link Json.Limits#LENIENT}, at most {@link HfTokenizerFiles#MAX_FILE_BYTES}).
     *
     * @throws IOException {@code "Invalid Hugging Face checkpoint '<dir>': <file>:<path>: ..."} when it is not a
     *                     valid tekken tokenizer
     */
    public static TekkenReader read(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        try {
            return of(Json.parse(file, HfTokenizerFiles.MAX_FILE_BYTES, Json.Limits.LENIENT), file.toString());
        } catch (JsonException e) {
            throw HfTokenizerFiles.invalid(file, e);
        }
    }

    /**
     * Reads a parsed tekken.json document; {@code source} names it in messages.
     *
     * @throws JsonException when the document is not a valid tekken tokenizer
     */
    public static TekkenReader of(Object parsed, String source) {
        JsonView root = JsonView.of(parsed, Objects.requireNonNull(source, "source"));
        JsonView config = root.obj("config");
        String version = config.str("version");
        Matcher m = VERSION.matcher(version);
        if (!m.matches()) {
            throw new JsonException(config.where("version") + ": '" + version + "' is not a tokenizer version v<N>");
        }
        int versionNumber = Integer.parseInt(m.group(1));
        int vocabSize = config.i32("default_vocab_size");
        int numSpecial = config.i32("default_num_special_tokens");
        if (numSpecial < 0 || vocabSize < numSpecial || vocabSize > HfTokenizerFiles.MAX_TOKENS) {
            throw new JsonException(config.where("default_vocab_size") + ": " + vocabSize + " tokens with "
                    + numSpecial + " special tokens is not a valid vocabulary size (at most "
                    + HfTokenizerFiles.MAX_TOKENS + ")");
        }
        List<String> specials = new ArrayList<>(numSpecial);
        List<Boolean> control = new ArrayList<>(numSpecial);
        readSpecialTokens(root, versionNumber, numSpecial, specials, control);
        Map<String, Integer> ids = new HashMap<>();
        boolean[] isControl = new boolean[numSpecial];
        for (int i = 0; i < numSpecial; i++) {
            ids.put(specials.get(i), i);
            isControl[i] = control.get(i);
        }
        byte[][] inner = readInnerVocab(root, vocabSize - numSpecial, numSpecial);
        return new TekkenReader(version, versionNumber, config.str("pattern"), vocabSize,
                config.optI32("num_vocab_tokens", inner.length), Collections.unmodifiableList(specials), isControl,
                Collections.unmodifiableMap(ids), inner);
    }

    /** {@code config.version}, such as {@code v13}. */
    public String version() { return version; }

    /** The number of {@link #version()}. */
    public int versionNumber() { return versionNumber; }

    /** {@code config.pattern}, the split regex. */
    public String pattern() { return pattern; }

    /** Total vocabulary size ({@code config.default_vocab_size}): special plus inner tokens. */
    public int vocabSize() { return vocabSize; }

    /** {@code config.num_vocab_tokens} as written. */
    public int numVocabTokens() { return numVocabTokens; }

    /** Number of special tokens ({@code config.default_num_special_tokens}). */
    public int numSpecialTokens() { return specialTokens.size(); }

    /** The special tokens by id, filler tokens included (unmodifiable). */
    public List<String> specialTokens() { return specialTokens; }

    public String specialToken(int id) { return specialTokens.get(id); }

    /** The {@code is_control} flag of special token {@code id} (true for the defaults and fillers). */
    public boolean specialIsControl(int id) { return specialIsControl[id]; }

    /** The id of the special token {@code token}, or -1. */
    public int specialId(String token) {
        Integer id = specialIds.get(token);
        return id == null ? -1 : id;
    }

    /** Number of inner (non-special) tokens: {@code vocabSize() - numSpecialTokens()}. */
    public int innerVocabSize() { return innerVocab.length; }

    /** The bytes of the inner token of tiktoken rank {@code rank} (a copy); its id is numSpecialTokens() + rank. */
    public byte[] tokenBytes(int rank) { return innerVocab[rank].clone(); }

    @Override
    public String toString() {
        return "TekkenReader{" + version + ", vocab=" + vocabSize + ", special=" + specialTokens.size() + '}';
    }

    // ------------------------------------------------------------------ reading

    private static void readSpecialTokens(JsonView root, int versionNumber, int numSpecial, List<String> specials,
                                          List<Boolean> control) {
        if (root.raw("special_tokens") == null) {
            if (versionNumber > LAST_VERSION_WITH_DEFAULT_SPECIALS) {
                throw new JsonException(root.where("special_tokens") + ": required for tokenizer version v"
                        + versionNumber + " (later than v" + LAST_VERSION_WITH_DEFAULT_SPECIALS + ")");
            }
            for (String token : DEPRECATED_SPECIAL_TOKENS) {
                specials.add(token);
                control.add(true);
            }
        } else {
            List<Object> raw = root.list("special_tokens");
            for (int i = 0; i < raw.size(); i++) {
                JsonView t = JsonView.of(raw.get(i), root.source(), "special_tokens[" + i + "]");
                int rank = t.i32("rank");
                if (rank != i) {
                    throw new JsonException(t.where("rank") + ": " + rank + " is not the token's index " + i);
                }
                specials.add(t.str("token_str"));
                control.add(t.optBool("is_control", true));
            }
        }
        if (specials.size() > numSpecial) {
            throw new JsonException(root.where("special_tokens") + ": " + specials.size()
                    + " special tokens exceed default_num_special_tokens " + numSpecial);
        }
        for (int i = specials.size(); i < numSpecial; i++) {
            specials.add(String.format(Locale.ROOT, FILLER_FORMAT, i));
            control.add(true);
        }
        Set<String> seen = new HashSet<>();
        for (String token : specials) {
            if (!seen.add(token)) {
                throw new JsonException(root.where("special_tokens") + ": special token '" + token
                        + "' appears more than once");
            }
        }
    }

    private static byte[][] readInnerVocab(JsonView root, int innerSize, int numSpecial) {
        List<Object> raw = root.list("vocab");
        if (raw.size() < innerSize) {
            throw new JsonException(root.where("vocab") + ": " + raw.size() + " entries, but default_vocab_size "
                    + (innerSize + numSpecial) + " with " + numSpecial + " special tokens needs " + innerSize);
        }
        byte[][] inner = new byte[innerSize][];
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < innerSize; i++) {
            JsonView e = JsonView.of(raw.get(i), root.source(), "vocab[" + i + "]");
            int rank = e.i32("rank");
            if (rank != i) {
                throw new JsonException(e.where("rank") + ": " + rank + " is not the entry's index " + i);
            }
            String encoded = e.str("token_bytes");
            byte[] bytes;
            try {
                bytes = Base64.getDecoder().decode(encoded);
            } catch (IllegalArgumentException ex) {
                throw new JsonException(e.where("token_bytes") + ": not valid base64 (" + ex.getMessage() + ")");
            }
            if (i < 256 && (bytes.length != 1 || (bytes[0] & 0xFF) != i)) {
                throw new JsonException(e.where("token_bytes") + ": entry " + i + " must be the single byte " + i);
            }
            if (!seen.add(new String(bytes, StandardCharsets.ISO_8859_1))) {
                throw new JsonException(e.where("token_bytes") + ": the bytes of entry " + i
                        + " repeat an earlier entry");
            }
            inner[i] = bytes;
        }
        return inner;
    }
}
