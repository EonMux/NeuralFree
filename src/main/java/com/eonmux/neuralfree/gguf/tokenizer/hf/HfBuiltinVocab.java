package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Port of the converter's {@code _set_vocab_builtin} (LC/conversion/base.py:2464-2500): the tokenizer of a bundled
 * llama.cpp vocabulary GGUF ({@code ggml-vocab-<name>.gguf}) for checkpoints that ship no tokenizer files (Mamba,
 * mamba.py:44 and :141; OpenELM, openelm.py:45).
 *
 * <p>The model, the pre-tokenizer ({@code mpt} for gpt-neox and {@code default} for llama-spm when the file has
 * none), the tokens, scores (llama-spm only) and types cut to the first {@code vocabSize} entries, the merges (all
 * but llama-spm), the bos/eos/unk/pad ids and the add_bos/add_eos flags are copied. Like the converter, a vocabulary
 * shorter than {@code vocabSize} is not padded. Special ids beyond the cut are dropped.</p>
 */
public final class HfBuiltinVocab {

    private static final Logger LOGGER = LogManager.getLogger(HfBuiltinVocab.class);

    /** The GPT-NeoX vocabulary (gpt2 BPE). */
    public static final String GPT_NEOX = "gpt-neox";
    /** The Llama-2 SentencePiece vocabulary. */
    public static final String LLAMA_SPM = "llama-spm";

    private static final List<String> NAMES = Collections.unmodifiableList(Arrays.asList(GPT_NEOX, LLAMA_SPM));
    private static final String RESOURCE_DIR = "/hf/vocab/";
    private static final List<String> ID_KEYS = Collections.unmodifiableList(Arrays.asList(
            GgufKeys.TOKENIZER_BOS_ID, GgufKeys.TOKENIZER_EOS_ID, GgufKeys.TOKENIZER_UNK_ID,
            GgufKeys.TOKENIZER_PAD_ID));
    private static final List<String> FLAG_KEYS =
            Collections.unmodifiableList(Arrays.asList(GgufKeys.TOKENIZER_ADD_BOS, GgufKeys.TOKENIZER_ADD_EOS));

    private HfBuiltinVocab() {}

    /** The bundled vocabulary names. */
    public static List<String> names() {
        return NAMES;
    }

    /**
     * The tokenizer metadata of the bundled vocabulary {@code name}, cut to {@code vocabSize} tokens.
     *
     * @throws IllegalArgumentException when {@code name} is not a bundled vocabulary or {@code vocabSize} is not
     *                                  positive
     * @throws IOException              when the bundled file cannot be read
     * @throws GgufFormatException      when the bundled file lacks a required key
     */
    public static Map<String, Object> of(String name, int vocabSize) throws IOException {
        Objects.requireNonNull(name, "name");
        if (!NAMES.contains(name)) {
            throw new IllegalArgumentException("unknown built-in vocabulary '" + name + "'; known: " + NAMES);
        }
        if (vocabSize <= 0) throw new IllegalArgumentException("vocabulary size " + vocabSize + " is not positive");
        LOGGER.warn("using the built-in '{}' tokenizer: the checkpoint has no tokenizer files", name);
        Map<String, Object> md = read(name);
        boolean spm = LLAMA_SPM.equals(name);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put(GgufKeys.TOKENIZER_MODEL, required(md, GgufKeys.TOKENIZER_MODEL, name));
        Object pre = md.get(GgufKeys.TOKENIZER_PRE);
        out.put(GgufKeys.TOKENIZER_PRE, pre != null ? pre : spm ? "default" : "mpt");
        List<?> tokens = cut(requiredList(md, GgufKeys.TOKENIZER_LIST, name), vocabSize);
        out.put(GgufKeys.TOKENIZER_LIST, tokens);
        if (spm) out.put(GgufKeys.TOKENIZER_SCORES, cut(requiredList(md, GgufKeys.TOKENIZER_SCORES, name), vocabSize));
        out.put(GgufKeys.TOKENIZER_TOKEN_TYPE, cut(requiredList(md, GgufKeys.TOKENIZER_TOKEN_TYPE, name), vocabSize));
        if (!spm) {
            List<?> merges = requiredList(md, GgufKeys.TOKENIZER_MERGES, name);
            out.put(GgufKeys.TOKENIZER_MERGES, Collections.unmodifiableList(new ArrayList<>(merges)));
        }
        for (String key : ID_KEYS) {
            Object id = md.get(key);
            if (id == null) continue;
            if (id instanceof Number && ((Number) id).longValue() >= 0 && ((Number) id).longValue() < tokens.size()) {
                out.put(key, ((Number) id).intValue());
            } else {
                LOGGER.warn("built-in vocabulary '{}': {} = {} is outside the {} kept tokens; dropped", name, key, id,
                        tokens.size());
            }
        }
        for (String key : FLAG_KEYS) {
            Object flag = md.get(key);
            if (flag != null) out.put(key, flag);
        }
        return Collections.unmodifiableMap(out);
    }

    /** The metadata of the bundled file, read through a temporary copy of the classpath resource. */
    private static Map<String, Object> read(String name) throws IOException {
        String resource = RESOURCE_DIR + "ggml-vocab-" + name + ".gguf";
        Path tmp = Files.createTempFile("neuralfree-vocab-" + name + "-", ".gguf");
        try {
            try (InputStream in = HfBuiltinVocab.class.getResourceAsStream(resource)) {
                if (in == null) throw new IOException("the built-in vocabulary resource " + resource + " is missing");
                Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            }
            try (ModelSource source = ModelSources.open(tmp.toString(), ModelSourceOptions.defaults())) {
                return new LinkedHashMap<>(source.getMetadata());
            }
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    private static Object required(Map<String, Object> md, String key, String name) {
        Object v = md.get(key);
        if (v == null) throw new GgufFormatException("built-in vocabulary '" + name + "' has no " + key);
        return v;
    }

    private static List<?> requiredList(Map<String, Object> md, String key, String name) {
        Object v = required(md, key, name);
        if (!(v instanceof List)) throw new GgufFormatException("built-in vocabulary '" + name + "': " + key
                + " is not an array");
        return (List<?>) v;
    }

    /** {@code values[:size]}, unmodifiable. */
    private static List<?> cut(List<?> values, int size) {
        return Collections.unmodifiableList(new ArrayList<>(values.subList(0, Math.min(size, values.size()))));
    }
}
