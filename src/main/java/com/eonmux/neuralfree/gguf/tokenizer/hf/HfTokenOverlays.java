package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.Normalizer;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;
import com.eonmux.neuralfree.gguf.tokenizer.TokenDecoder;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerAlgorithm;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * The {@code neuralfree.tokenizer.*} overlay keys that carry the Hugging Face behaviour GGUF tokenizer
 * metadata cannot express, and the post-processor ids that {@code tokenizer.encode} adds by default:
 * <ul>
 *   <li>added-token attributes ({@code lstrip}, {@code rstrip}, {@code single_word}, {@code normalized});</li>
 *   <li>the normalizer and decoder chains as canonical JSON, for BPE vocabularies, whose GGUF form implies
 *       neither;</li>
 *   <li>the BPE model options ({@code ignore_merges}, {@code byte_fallback}, {@code unk_token} with
 *       {@code fuse_unk}, {@code continuing_subword_prefix}, {@code end_of_word_suffix});</li>
 *   <li>the SentencePiece prepend scheme;</li>
 *   <li>{@code clean_up_tokenization_spaces} as transformers applies it;</li>
 *   <li>the post-processor's leading and trailing ids when the GGUF add_bos/add_eos flags do not give them.</li>
 * </ul>
 */
final class HfTokenOverlays {

    private static final Logger LOGGER = LogManager.getLogger(HfTokenOverlays.class);

    /** transformers' opt-in for cleaning up BPE output (tokenization_utils_tokenizers.py, {@code _decode}). */
    private static final String CLEAN_UP_BPE_FLAG =
            "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output";
    private static final String CLEAN_UP = "clean_up_tokenization_spaces";

    private HfTokenOverlays() {}

    /** The ids a post-processor puts before and after one encoded sequence. Immutable. */
    static final class PostIds {
        static final PostIds NONE = new PostIds(new int[0], new int[0]);

        private final int[] prefix;
        private final int[] suffix;

        PostIds(int[] prefix, int[] suffix) {
            this.prefix = prefix.clone();
            this.suffix = suffix.clone();
        }

        int[] prefix() { return prefix.clone(); }

        int[] suffix() { return suffix.clone(); }

        /** {@code prefix ++ ids ++ suffix}. */
        int[] wrap(int[] ids) {
            int[] out = new int[prefix.length + ids.length + suffix.length];
            System.arraycopy(prefix, 0, out, 0, prefix.length);
            System.arraycopy(ids, 0, out, prefix.length, ids.length);
            System.arraycopy(suffix, 0, out, prefix.length + ids.length, suffix.length);
            return out;
        }

        /** This post-processor applied around the output of {@code inner}. */
        PostIds around(PostIds inner) {
            return new PostIds(concat(prefix, inner.prefix), concat(inner.suffix, suffix));
        }

        @Override
        public String toString() {
            return "PostIds{" + Arrays.toString(prefix) + ", " + Arrays.toString(suffix) + '}';
        }
    }

    // ------------------------------------------------------------------ post-processor

    /**
     * The ids the tokenizer.json post-processor adds around a single sequence: {@code TemplateProcessing} single
     * (special-token ids before and after {@code $A}), {@code RobertaProcessing} / {@code BertProcessing} (cls and
     * sep), {@code ByteLevel} (none) and {@code Sequence} (each processor wrapping the previous ones' output).
     * Null when {@code tj} is null or has no post-processor: the ids then depend on the transformers tokenizer
     * class, and the GGUF add_bos/add_eos flags written from tokenizer_config stand.
     *
     * @throws GgufFormatException when the post-processor is malformed or of an unknown type
     */
    static PostIds postProcessorIds(HfTokenizerJson tj) {
        if (tj == null || tj.postProcessor() == null) return null;
        return processorIds(tj.postProcessor());
    }

    private static PostIds processorIds(JsonView p) {
        String type = p.str("type");
        switch (type) {
            case "ByteLevel":
                return PostIds.NONE;
            case "TemplateProcessing":
                return templateIds(p);
            case "RobertaProcessing":
                return new PostIds(new int[]{pairId(p, "cls", 0)}, new int[]{pairId(p, "sep", 2)});
            case "BertProcessing":
                return new PostIds(new int[]{pairId(p, "cls", 101)}, new int[]{pairId(p, "sep", 102)});
            case "Sequence": {
                PostIds out = PostIds.NONE;
                List<Object> processors = p.list("processors");
                for (int i = 0; i < processors.size(); i++) {
                    out = processorIds(JsonView.of(processors.get(i), p.source(),
                            p.path() + ".processors[" + i + "]")).around(out);
                }
                return out;
            }
            default:
                throw new GgufFormatException(p.where("type") + ": unsupported post-processor type '" + type + "'");
        }
    }

    /** The id of a {@code [token, id]} member, or {@code defaultId} when the member is absent. */
    private static int pairId(JsonView p, String member, int defaultId) {
        Object raw = p.raw(member);
        if (raw == null) return defaultId;
        if (raw instanceof List && ((List<?>) raw).size() == 2 && ((List<?>) raw).get(1) instanceof Long) {
            long id = (Long) ((List<?>) raw).get(1);
            if (id >= 0 && id < HfTokenizerFiles.MAX_TOKENS) return (int) id;
        }
        throw new GgufFormatException(p.where(member) + ": expected a [token, id] pair, got " + raw);
    }

    private static PostIds templateIds(JsonView p) {
        List<Object> single = p.list("single");
        JsonView specials = p.optObj("special_tokens");
        List<Integer> prefix = new ArrayList<>();
        List<Integer> suffix = new ArrayList<>();
        boolean seenSequence = false;
        for (int i = 0; i < single.size(); i++) {
            JsonView item = JsonView.of(single.get(i), p.source(), p.path() + ".single[" + i + "]");
            if (item.has("Sequence")) {
                seenSequence = true;
            } else if (item.has("SpecialToken")) {
                String name = item.obj("SpecialToken").str("id");
                JsonView token = specials.optObj(name);
                if (token.isEmpty()) {
                    throw new GgufFormatException(item.where("SpecialToken") + ": '" + name
                            + "' is not in the template's special_tokens");
                }
                for (Long id : token.i64List("ids")) {
                    if (id < 0 || id >= HfTokenizerFiles.MAX_TOKENS) {
                        throw new GgufFormatException(token.where("ids") + ": token id " + id + " is out of range");
                    }
                    (seenSequence ? suffix : prefix).add(id.intValue());
                }
            } else {
                throw new GgufFormatException(item.where("") + ": expected a Sequence or SpecialToken piece");
            }
        }
        return new PostIds(toArray(prefix), toArray(suffix));
    }

    // ------------------------------------------------------------------ overlays

    /**
     * Writes {@link GgufKeys#NF_TOKENIZER_PREFIX_IDS} and {@link GgufKeys#NF_TOKENIZER_SUFFIX_IDS} into {@code md}
     * when the vocabulary {@code md} describes (BPE or SentencePiece) would add other ids around a text than the
     * post-processor does; ids outside the vocabulary drop the overlay with a warning. Nothing is written when
     * {@code hf} is null (no post-processor).
     */
    static void writePostProcessorIds(Map<String, Object> md, PostIds hf) {
        if (hf == null) return;
        TokenizerVocab v = TokenizerVocab.fromMetadata(md);
        if (v.type() != TokenizerVocab.Type.BPE && v.type() != TokenizerVocab.Type.SPM) return;
        for (int id : hf.wrap(new int[0])) {
            if (id >= v.size()) {
                LOGGER.warn("post-processor id {} is outside the vocabulary [0, {}); its ids are not applied", id,
                        v.size());
                return;
            }
        }
        SpecialTokens special = SpecialTokens.of(v);
        if (Arrays.equals(leading(special), hf.prefix) && Arrays.equals(trailing(special), hf.suffix)) return;
        md.put(GgufKeys.NF_TOKENIZER_PREFIX_IDS, toList(hf.prefix));
        md.put(GgufKeys.NF_TOKENIZER_SUFFIX_IDS, toList(hf.suffix));
    }

    private static int[] leading(SpecialTokens s) {
        try {
            return s.leadingIds();
        } catch (IllegalStateException e) {
            return null;
        }
    }

    private static int[] trailing(SpecialTokens s) {
        try {
            return s.trailingIds();
        } catch (IllegalStateException e) {
            return null;
        }
    }

    /**
     * {@link GgufKeys#NF_TOKENIZER_TOKEN_ATTRS}: the {@code lstrip}, {@code rstrip}, {@code single_word} and
     * {@code normalized} bits of the added tokens with ids below {@code n}, written when any token has one.
     */
    static void writeTokenAttrs(Map<String, Object> md, List<HfTokenizerJson.AddedToken> added, int n) {
        int[] attrs = new int[n];
        boolean any = false;
        for (HfTokenizerJson.AddedToken t : added) {
            if (t.id() >= n) continue;
            int bits = (t.lstrip() ? TokenizerVocab.ATTR_LSTRIP : 0) | (t.rstrip() ? TokenizerVocab.ATTR_RSTRIP : 0)
                    | (t.singleWord() ? TokenizerVocab.ATTR_SINGLE_WORD : 0)
                    | (t.normalized() ? TokenizerVocab.ATTR_NORMALIZED : 0);
            attrs[t.id()] |= bits;
            any |= bits != 0;
        }
        if (any) md.put(GgufKeys.NF_TOKENIZER_TOKEN_ATTRS, toList(attrs));
    }

    /**
     * The BPE model options of a tokenizer.json BPE model: {@code ignore_merges} and {@code byte_fallback} always,
     * {@code fuse_unk} and the unknown token id (when no special token set it) when the model has an
     * {@code unk_token} in the vocabulary, and the non-empty subword prefix and word suffix.
     */
    static void writeBpeModel(Map<String, Object> md, HfTokenizerJson.Model m, Map<String, Integer> tokenIds) {
        md.put(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES, m.ignoreMerges());
        md.put(GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK, m.byteFallback());
        if (m.unkToken() != null) {
            md.put(GgufKeys.NF_TOKENIZER_BPE_UNK_FUSE, m.fuseUnk());
            Integer unk = tokenIds.get(m.unkToken());
            if (unk != null) {
                md.putIfAbsent(GgufKeys.TOKENIZER_UNK_ID, unk);
            } else {
                LOGGER.warn("BPE unk_token '{}' is not in the vocabulary; unknown characters are dropped",
                        m.unkToken());
            }
        }
        String prefix = m.continuingSubwordPrefix();
        if (prefix != null && !prefix.isEmpty()) md.put(GgufKeys.NF_TOKENIZER_BPE_CONTINUING_SUBWORD_PREFIX, prefix);
        String suffix = m.endOfWordSuffix();
        if (suffix != null && !suffix.isEmpty()) md.put(GgufKeys.NF_TOKENIZER_BPE_END_OF_WORD_SUFFIX, suffix);
    }

    /** The canonical JSON of a tokenizer.json normalizer, or null when it is absent or the identity. */
    static String normalizerJson(JsonView normalizer) {
        if (normalizer == null) return null;
        Normalizer n = Normalizer.fromJson(members(normalizer), normalizer.source() + ":normalizer");
        return n.isIdentity() ? null : n.toCanonicalJson();
    }

    /**
     * The decoder overlay of a BPE vocabulary: the canonical JSON of the tokenizer.json decoder unless it is the
     * GPT-2 byte-level decoding BPE detokenization already performs; for a vocabulary without a decoder whose words
     * are not byte-encoded, the identity (pieces concatenated). Null when no overlay is needed.
     */
    static String bpeDecoderJson(JsonView decoder, boolean byteEncode) {
        if (decoder == null) {
            if (byteEncode) return null;
            LOGGER.warn("tokenizer.json has no decoder; tokens are joined without separators (Hugging Face joins"
                    + " them with spaces)");
            return TokenDecoder.identity().toCanonicalJson();
        }
        TokenDecoder d = decoder(decoder);
        return isByteLevelOnly(d) ? null : d.toCanonicalJson();
    }

    /** The tokenizer.json decoder; the identity when absent. */
    static TokenDecoder decoder(JsonView decoder) {
        return decoder == null ? TokenDecoder.identity()
                : TokenDecoder.fromJson(members(decoder), decoder.source() + ":decoder");
    }

    private static boolean isByteLevelOnly(TokenDecoder d) {
        if (d.kind() == TokenDecoder.Kind.BYTE_LEVEL) return true;
        if (d.kind() != TokenDecoder.Kind.SEQUENCE || d.children().isEmpty()) return false;
        for (TokenDecoder c : d.children()) {
            if (!isByteLevelOnly(c)) return false;
        }
        return true;
    }

    /**
     * {@link GgufKeys#NF_TOKENIZER_CLEAN_SPACES}: tokenizer_config {@code clean_up_tokenization_spaces} (false when
     * absent), which transformers does not apply to a BPE model unless
     * {@code clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output} is set.
     */
    static boolean cleanSpaces(JsonView tokenizerConfig, boolean bpeModel) {
        boolean clean = tokenizerConfig.raw(CLEAN_UP) instanceof Boolean && (Boolean) tokenizerConfig.raw(CLEAN_UP);
        boolean forced = tokenizerConfig.raw(CLEAN_UP_BPE_FLAG) instanceof Boolean
                && (Boolean) tokenizerConfig.raw(CLEAN_UP_BPE_FLAG);
        return clean && (!bpeModel || forced);
    }

    /**
     * {@code clean_up_tokenization_spaces} as transformers 4.x applies it in {@code decode}, the version the
     * llama.cpp converter runs (requirements-convert_legacy_llama.txt): the tokenizer_config value, false when
     * absent.
     */
    static boolean converterCleanSpaces(JsonView tokenizerConfig) {
        return tokenizerConfig.raw(CLEAN_UP) instanceof Boolean && (Boolean) tokenizerConfig.raw(CLEAN_UP);
    }

    /**
     * {@link GgufKeys#NF_TOKENIZER_PREPEND_SCHEME} of a SentencePiece vocabulary: the tokenizer.json
     * {@code Metaspace} pre-tokenizer's {@code prepend_scheme} (or legacy {@code add_prefix_space}) when it is not
     * {@code always}; without tokenizer.json, {@code first} when tokenizer_config sets {@code legacy} false, as the
     * transformers slow-to-fast conversion does. Null when SentencePiece's own prefixing applies.
     */
    static String prependScheme(HfTokenizerJson tj, JsonView tokenizerConfig) {
        if (tj != null) {
            JsonView metaspace = findMetaspace(tj.preTokenizer());
            if (metaspace == null) return null;
            String scheme;
            if (metaspace.has("prepend_scheme")) {
                scheme = metaspace.str("prepend_scheme");
            } else {
                scheme = metaspace.optBool("add_prefix_space", true) ? "always" : "never";
            }
            if (!scheme.equals("always") && !scheme.equals("first") && !scheme.equals("never")) {
                throw new GgufFormatException(metaspace.where("prepend_scheme") + ": '" + scheme
                        + "' is not always, first or never");
            }
            return scheme.equals("always") ? null : scheme;
        }
        return Boolean.FALSE.equals(tokenizerConfig.raw("legacy")) ? "first" : null;
    }

    private static JsonView findMetaspace(JsonView pre) {
        if (pre == null) return null;
        Object type = pre.raw("type");
        if ("Metaspace".equals(type)) return pre;
        if ("Sequence".equals(type) && pre.raw("pretokenizers") instanceof List) {
            List<?> items = (List<?>) pre.raw("pretokenizers");
            for (int i = 0; i < items.size(); i++) {
                if (!(items.get(i) instanceof Map)) continue;
                JsonView found = findMetaspace(JsonView.of(items.get(i), pre.source(),
                        pre.path() + ".pretokenizers[" + i + "]"));
                if (found != null) return found;
            }
        }
        return null;
    }

    /**
     * The converter's {@code tokenizer.decode(tokenizer.encode(token))} for one added token: the token alone through
     * the decoder chain, then transformers' clean-up when enabled.
     */
    static UnaryOperator<String> roundTrip(TokenDecoder decoder, boolean cleanUp) {
        return token -> {
            String text = decoder.decode(Collections.singletonList(token));
            return cleanUp ? TokenizerAlgorithm.cleanUpTokenizationSpaces(text) : text;
        };
    }

    // ------------------------------------------------------------------ candidate encoder

    /**
     * The encoder {@link PreNameResolver} fingerprints with: the tokenizer {@code candidate} describes, encoding with
     * special tokens recognized, wrapped in the post-processor ids (none when {@code post} is null), as
     * {@code tokenizer.encode} does by default.
     */
    static Function<String, int[]> encoder(Map<String, Object> candidate, PostIds post) {
        GGUFTokenizer tokenizer = new GGUFTokenizer(TokenizerVocab.fromMetadata(candidate));
        PostIds ids = post != null ? post : PostIds.NONE;
        return text -> ids.wrap(tokenizer.encode(text, false, true));
    }

    // ------------------------------------------------------------------ helpers

    /** The members of an object view as a map (the parsed JSON object it views). */
    static Map<String, Object> members(JsonView v) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (String key : v.keys()) m.put(key, v.raw(key));
        return m;
    }

    static List<Integer> toList(int[] values) {
        List<Integer> out = new ArrayList<>(values.length);
        for (int v : values) out.add(v);
        return Collections.unmodifiableList(out);
    }

    private static int[] toArray(List<Integer> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < out.length; i++) out[i] = values.get(i);
        return out;
    }

    private static int[] concat(int[] a, int[] b) {
        int[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
