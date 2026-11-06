package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.tokenizer.Normalizer;
import com.eonmux.neuralfree.gguf.tokenizer.TokenDecoder;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * The Engram compressed token map of the DeepSeek-V4.1 converter (deepseek41.py:55-78): every token id is decoded on
 * its own, normalized by the converter's fixed chain (NFKC, NFD, strip accents, lowercase, whitespace collapsed, a lone
 * space kept through a private-use sentinel, trim), and tokens that normalize alike collapse to one compressed id.
 * The map is stored as {@code deepseek41.engram.token_map}.
 */
final class HfEngramTokenMap {

    /** A private-use code point that keeps a single-space token distinct from an empty one. */
    private static final char SENTINEL = '';

    private static final char REPLACEMENT = '�';

    private HfEngramTokenMap() {
    }

    /** The compressed ids by token id and the compressed vocabulary size. Immutable. */
    static final class Result {
        private final int[] lookup;
        private final int compressedVocab;

        Result(int[] lookup, int compressedVocab) {
            this.lookup = lookup;
            this.compressedVocab = compressedVocab;
        }

        /** {@code lookup[token id]} is the compressed id. */
        int[] lookup() { return lookup.clone(); }

        /** The tokenizer's id count, {@code len(tokenizer)} in the converter. */
        int tokens() { return lookup.length; }

        int compressedVocab() { return compressedVocab; }

        /** The compressed ids in token order, for the GGUF array key. */
        List<Integer> toList() {
            List<Integer> out = new ArrayList<>(lookup.length);
            for (int id : lookup) {
                out.add(id);
            }
            return Collections.unmodifiableList(out);
        }
    }

    /**
     * Builds the map of the checkpoint's tokenizer.
     *
     * @param files     the checkpoint's tokenizer files
     * @param vocabSize the configuration's {@code vocab_size}
     * @throws IOException          when the tokenizer file cannot be read
     * @throws GgufFormatException  when there is no {@code tokenizer.json}, its decoder or normalizer is not
     *                              supported, or its id count differs from {@code vocabSize}
     */
    static Result build(HfTokenizerFiles files, int vocabSize) throws IOException {
        Objects.requireNonNull(files, "files");
        HfTokenizerJson tj = files.readTokenizerJson();
        if (tj == null) {
            throw new GgufFormatException("DeepSeek-V4.1 Engram: the compressed token map needs the checkpoint's"
                    + " tokenizer.json; an Engram checkpoint without one is not converted");
        }
        HfTokenizerJson.Model model = tj.model();
        String[] tokens = model.vocab();
        Set<String> modelTokens = new HashSet<>();
        for (String token : tokens) {
            if (token != null) {
                modelTokens.add(token);
            }
        }
        Map<String, Integer> addedByContent = new LinkedHashMap<>();
        Map<Integer, HfTokenizerJson.AddedToken> addedById = new LinkedHashMap<>();
        for (HfTokenizerJson.AddedToken added : tj.addedTokens()) {
            if (added.content().isEmpty()) {
                continue;
            }
            addedByContent.put(added.content(), added.id());
            addedById.put(added.id(), added);
        }
        int overlap = 0;
        for (String content : addedByContent.keySet()) {
            if (modelTokens.contains(content)) {
                overlap++;
            }
        }
        int tokenizerSize = model.vocabSize() + addedByContent.size() - overlap;
        if (tokenizerSize != vocabSize) {
            throw new GgufFormatException("DeepSeek-V4.1 Engram: the tokenizer has " + tokenizerSize
                    + " ids but the configuration's vocab_size is " + vocabSize + "; the compressed token map of"
                    + " deepseek41.py:66-68 refuses this mismatch too");
        }
        JsonView root = tj.root();
        Normalizer normalizer = engramNormalizer();
        Normalizer tokenizerNormalizer = Normalizer.fromJson(root.raw("normalizer"),
                tj.source() + ":normalizer");
        TokenDecoder decoder = TokenDecoder.fromJson(root.raw("decoder"), tj.source() + ":decoder");
        Map<String, Integer> keyToNew = new LinkedHashMap<>();
        int[] lookup = new int[tokenizerSize];
        for (int id = 0; id < tokenizerSize; id++) {
            HfTokenizerJson.AddedToken added = addedById.get(id);
            String token = added != null
                    ? addedText(added, tokenizerNormalizer)
                    : (id < tokens.length ? tokens[id] : null);
            String text = token == null ? "" : decoder.decode(Collections.singletonList(token));
            String key;
            if (text.indexOf(REPLACEMENT) >= 0) {
                key = token != null ? token : "";
            } else {
                String normalized = normalizer.normalize(text);
                key = normalized.isEmpty() ? text : normalized;
            }
            Integer compressed = keyToNew.get(key);
            if (compressed == null) {
                compressed = keyToNew.size();
                keyToNew.put(key, compressed);
            }
            lookup[id] = compressed;
        }
        return new Result(lookup, keyToNew.size());
    }

    /** {@code AddedVocabulary::simple_id_to_token}: the content, or its cached normalized form. */
    private static String addedText(HfTokenizerJson.AddedToken added, Normalizer normalizer) {
        if (!added.normalized()) {
            return added.content();
        }
        String normalized = normalizer.normalize(added.content());
        return normalized.equals(added.content()) ? added.content() : normalized;
    }

    /** The converter's fixed Engram normalization chain (deepseek41.py:59-67). */
    static Normalizer engramNormalizer() {
        return Normalizer.sequence(java.util.Arrays.asList(
                Normalizer.nfkc(),
                Normalizer.nfd(),
                Normalizer.stripAccents(),
                Normalizer.lowercase(),
                Normalizer.replaceRegex("[ \\t\\r\\n]+", " "),
                Normalizer.replaceRegex("^ $", String.valueOf(SENTINEL)),
                Normalizer.strip(true, true),
                Normalizer.replace(String.valueOf(SENTINEL), " ")));
    }
}
