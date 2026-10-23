package com.eonmux.neuralfree.gguf.tokenizer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * The load-time special-token rules of llama.cpp {@code llama_vocab::impl::load}
 * (src/llama-vocab.cpp:2196-3424), evaluated once per vocabulary for {@link SpecialTokens}.
 * Stateless; each {@link #resolve} call works on its own copies.
 */
final class SpecialTokenRules {

    private static final Logger LOGGER = LogManager.getLogger(SpecialTokenRules.class);

    static final int BOS = 0;
    static final int EOS = 1;
    static final int EOT = 2;
    static final int EOM = 3;
    static final int UNK = 4;
    static final int SEP = 5;
    static final int PAD = 6;
    static final int MASK = 7;
    static final int FIM_PRE = 8;
    static final int FIM_SUF = 9;
    static final int FIM_MID = 10;
    static final int FIM_PAD = 11;
    static final int FIM_REP = 12;
    static final int FIM_SEP = 13;
    private static final int COUNT = 14;

    private static final int NULL = SpecialTokens.NULL;

    /** {@code special_token_types} (src/llama-vocab.cpp:2870-2891): GGUF key and slot, in order. */
    private static final String[] OVERRIDE_KEYS = {
            "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.eom_token_id", "tokenizer.ggml.unknown_token_id", "tokenizer.ggml.seperator_token_id",
            "tokenizer.ggml.padding_token_id", "tokenizer.ggml.mask_token_id", "tokenizer.ggml.fim_pre_token_id",
            "tokenizer.ggml.fim_suf_token_id", "tokenizer.ggml.fim_mid_token_id", "tokenizer.ggml.fim_pad_token_id",
            "tokenizer.ggml.fim_rep_token_id", "tokenizer.ggml.fim_sep_token_id",
            "tokenizer.ggml.prefix_token_id", "tokenizer.ggml.suffix_token_id", "tokenizer.ggml.middle_token_id",
    };
    private static final int[] OVERRIDE_SLOTS = {
            BOS, EOS, EOT, EOM, UNK, SEP, PAD, MASK, FIM_PRE, FIM_SUF, FIM_MID, FIM_PAD, FIM_REP, FIM_SEP,
            FIM_PRE, FIM_SUF, FIM_MID,
    };

    private static final String KEY_SUPPRESS_TOKENS = "tokenizer.ggml.suppress_tokens";
    private static final String END_OF_SENTENCE_DS = "<｜end▁of▁sentence｜>";

    /** Slots detected from token text when unset, in llama.cpp's check order (2972-3132). */
    private static final int[] DETECT_SLOTS = {EOT, EOM, FIM_PRE, FIM_SUF, FIM_MID, FIM_PAD, FIM_REP, FIM_SEP};
    private static final List<Set<String>> DETECT_TEXTS = Collections.unmodifiableList(Arrays.asList(
            set("<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>",
                    "<EOT>", "_<EOT>", "[EOT]", END_OF_SENTENCE_DS, "<end_of_utterance>"),
            set("<|eom_id|>"),
            set("<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<｜fim▁begin｜>", "<PRE>",
                    "▁<PRE>", "<|code_prefix|>", "<|prefix|>", "<|plamo:fim_prefix|>"),
            set("<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>", "<｜fim▁hole｜>", "<SUF>",
                    "▁<SUF>", "<|code_suffix|>", "<|suffix|>", "<|plamo:fim_suffix|>"),
            set("<|fim_middle|>", "<fim-middle>", "<fim_middle>", "<｜fim▁end｜>", "<MID>",
                    "▁<MID>", "<|code_middle|>", "<|middle|>", "<|plamo:fim_middle|>"),
            set("<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"),
            set("<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>"),
            set("<|file_sep|>", "<|plamo:file_separator|>")));

    /** Token texts that end generation (3171-3201). */
    private static final Set<String> EOG_TEXTS = set(
            "<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>",
            "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]",
            "<|end_of_text|>", "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>", END_OF_SENTENCE_DS,
            "[e~[", "<|ifm|im_end|>");
    /** Extra stop texts of the FLINT pre-tokenizer (3193). */
    private static final Set<String> FLINT_EOG_TEXTS = set("<|user|>", "<|observation|>");
    /** gpt-oss harmony markers that are always rendered (3219-3228). */
    private static final Set<String> RENDERED_MARKERS = set("<|channel|>", "<|message|>", "<|start|>", "<|constrain|>");

    private SpecialTokenRules() {}

    /** The resolved special-token state of one vocabulary. Arrays are owned by the result. */
    static final class Result {
        final TokenizerVocab vocab;
        final int[] ids;
        final int[] attrs;
        final List<Integer> eog;
        final int[] partitionOrder;
        final int[] suppressTokens;

        Result(TokenizerVocab vocab, int[] ids, int[] attrs, List<Integer> eog, int[] partitionOrder,
               int[] suppressTokens) {
            this.vocab = vocab;
            this.ids = ids;
            this.attrs = attrs;
            this.eog = eog;
            this.partitionOrder = partitionOrder;
            this.suppressTokens = suppressTokens;
        }
    }

    static Result resolve(TokenizerVocab v) {
        int n = v.size();
        int[] attrs = new int[n];
        for (int i = 0; i < n; i++) attrs[i] = v.attr(i);
        int[] ids = new int[COUNT];
        Arrays.fill(ids, NULL);
        if (v.type() == TokenizerVocab.Type.NONE) {
            return new Result(v, ids, attrs, Collections.<Integer>emptyList(), new int[0], new int[0]);
        }
        applyTypeDefaults(v, ids);
        for (int k = 0; k < OVERRIDE_KEYS.length; k++) {
            int slot = OVERRIDE_SLOTS[k];
            ids[slot] = override(v, n, ids[slot], OVERRIDE_KEYS[k]);
        }
        int[] suppress = suppressTokens(v, n);
        detectByText(v, ids, attrs);
        markUnused(v, attrs);
        List<Integer> eog = buildEog(v, ids, attrs);
        int[] order = partitionOrder(v, attrs);
        applyStripRules(v, attrs, order);
        return new Result(v, ids, attrs, eog, order, suppress);
    }

    /** Defaults {bos, eos, unk, sep, pad, mask} per vocabulary type (src/llama-vocab.cpp:2228-2408). */
    private static void applyTypeDefaults(TokenizerVocab v, int[] ids) {
        int[] d;
        switch (v.type()) {
            case SPM: d = new int[]{1, 2, 0, NULL, NULL, NULL}; break;
            case WPM: d = new int[]{101, NULL, 100, 102, 0, 103}; break;
            case BPE:
                d = "gemma4".equals(v.model())
                        ? new int[]{NULL, NULL, NULL, NULL, NULL, NULL}
                        : new int[]{11, 11, NULL, NULL, NULL, NULL};
                break;
            case UGM: d = new int[]{NULL, 1, 2, NULL, 0, NULL}; break;
            case PLAMO2:
            case PLAMO3: d = new int[]{1, 2, 0, NULL, 3, NULL}; break;
            default: d = new int[]{NULL, NULL, NULL, NULL, NULL, NULL}; break;
        }
        ids[BOS] = d[0];
        ids[EOS] = d[1];
        ids[UNK] = d[2];
        ids[SEP] = d[3];
        ids[PAD] = d[4];
        ids[MASK] = d[5];
        PreTokenizerRegistry.PreTokenizer pre = v.preTokenizer();
        if (pre != null && pre.suppressBos()) ids[BOS] = NULL;
    }

    /** One special-token id: the default, disabled when out of range, then an in-range GGUF override. */
    private static int override(TokenizerVocab v, int n, int def, String key) {
        int id = def;
        if (id >= 0 && id >= n) {
            LOGGER.warn("default special token '{}' = {} out of vocab range, disabling", key, id);
            id = NULL;
        }
        Object val = v.metadata(key);
        if (val == null) return id;
        if (!(val instanceof Number)) throw new IllegalArgumentException(key + " is not an integer: " + val);
        long newId = ((Number) val).longValue();
        if (newId < 0 || newId >= n) {
            LOGGER.warn("bad special token: '{}' = {}, using default id {}", key, newId, id);
            return id;
        }
        return (int) newId;
    }

    /** {@code tokenizer.ggml.suppress_tokens}: out-of-range ids are dropped (2948-2965). */
    private static int[] suppressTokens(TokenizerVocab v, int n) {
        Object val = v.metadata(KEY_SUPPRESS_TOKENS);
        if (val == null) return new int[0];
        if (!(val instanceof List)) throw new IllegalArgumentException(KEY_SUPPRESS_TOKENS + " is not an array");
        List<?> list = (List<?>) val;
        int[] out = new int[list.size()];
        int count = 0;
        for (int i = 0; i < list.size(); i++) {
            Object o = list.get(i);
            if (!(o instanceof Number)) {
                throw new IllegalArgumentException(KEY_SUPPRESS_TOKENS + "[" + i + "] is not an integer: " + o);
            }
            long id = ((Number) o).longValue();
            if (id >= 0 && id < n) out[count++] = (int) id;
        }
        return Arrays.copyOf(out, count);
    }

    /** Unset EOT/EOM/FIM ids from token text; detected tokens become CONTROL (2972-3132). */
    private static void detectByText(TokenizerVocab v, int[] ids, int[] attrs) {
        for (int id = 0; id < attrs.length; id++) {
            if (!isMapped(v, id)) continue;
            String text = v.text(id);
            for (int k = 0; k < DETECT_SLOTS.length; k++) {
                int slot = DETECT_SLOTS[k];
                if (ids[slot] == NULL && DETECT_TEXTS.get(k).contains(text)) {
                    ids[slot] = id;
                    forceControl(attrs, id, text);
                }
            }
        }
    }

    /** CONTROL tokens whose text contains "unused" become UNUSED (3136-3157). */
    private static void markUnused(TokenizerVocab v, int[] attrs) {
        for (int id = 0; id < attrs.length; id++) {
            if (isMapped(v, id) && (attrs[id] & TokenizerVocab.ATTR_CONTROL) != 0
                    && (attrs[id] & TokenizerVocab.ATTR_UNUSED) == 0 && v.text(id).contains("unused")) {
                attrs[id] |= TokenizerVocab.ATTR_UNUSED;
            }
        }
    }

    /** The end-of-generation set and its attribute side effects (3162-3313). Sorted ascending. */
    private static List<Integer> buildEog(TokenizerVocab v, int[] ids, int[] attrs) {
        int n = attrs.length;
        PreTokenizerType preType = v.preTokenizer() != null ? v.preTokenizer().type() : PreTokenizerType.DEFAULT;
        TreeSet<Integer> eog = new TreeSet<>();
        for (int slot : new int[]{FIM_PAD, FIM_REP, FIM_SEP}) {
            if (ids[slot] != NULL) eog.add(ids[slot]);
        }
        for (int id = 0; id < n; id++) {
            if (!isMapped(v, id)) continue;
            String text = v.text(id);
            if (EOG_TEXTS.contains(text) || (preType == PreTokenizerType.FLINT && FLINT_EOG_TEXTS.contains(text))) {
                eog.add(id);
                forceControl(attrs, id, text);
            }
        }
        for (int id = 0; id < n; id++) {
            if (isMapped(v, id) && RENDERED_MARKERS.contains(v.text(id))) attrs[id] = TokenizerVocab.ATTR_USER_DEFINED;
        }
        String[] names = {"special_eos_id", "special_eot_id", "special_eom_id"};
        int[] slots = {EOS, EOT, EOM};
        for (int k = 0; k < slots.length; k++) {
            int id = ids[slots[k]];
            if (id != NULL && eog.add(id)) {
                LOGGER.warn("{} is not in special_eog_ids - the tokenizer config may be incorrect", names[k]);
            }
        }
        removeHarmonyEnd(v, attrs, eog);
        removeNormalEndOfSentence(v, attrs, eog);
        return Collections.unmodifiableList(new ArrayList<>(eog));
    }

    /** {@code <|end|>} is a message delimiter for o200k_harmony and solar-open (3248-3284). */
    private static void removeHarmonyEnd(TokenizerVocab v, int[] attrs, TreeSet<Integer> eog) {
        boolean hasReturn = false;
        boolean hasCall = false;
        boolean hasEnd = false;
        boolean hasFlush = false;
        int endId = NULL;
        for (int id : eog) {
            if (id < 0 || id >= attrs.length) continue;
            String text = v.text(id);
            if (text.equals("<|return|>")) {
                hasReturn = true;
            } else if (text.equals("<|call|>") || text.equals("<|calls|>")) {
                hasCall = true;
            } else if (text.equals("<|flush|>")) {
                hasFlush = true;
            } else if (text.equals("<|end|>")) {
                hasEnd = true;
                endId = id;
            }
        }
        if ((hasReturn && hasCall && hasEnd) || (hasCall && hasFlush && hasEnd)) {
            eog.remove(endId);
            attrs[endId] = TokenizerVocab.ATTR_USER_DEFINED;
            LOGGER.warn("removing '<|end|>' from the EOG list (harmony-style message delimiter)");
        }
    }

    /** gemma4, plamo and beam use a normal {@code </s>} token (3287-3313). */
    private static void removeNormalEndOfSentence(TokenizerVocab v, int[] attrs, TreeSet<Integer> eog) {
        boolean hasNormalSMarker = v.tokenId("<rfl:parameter name=\"") >= 0;
        boolean hasS = false;
        int sId = NULL;
        for (int id : eog) {
            if (id < 0 || id >= attrs.length) continue;
            String text = v.text(id);
            if (text.equals("<|tool_response>") || text.equals("<|plamo:eos|>")) {
                hasNormalSMarker = true;
            } else if (text.equals("</s>")) {
                hasS = true;
                sId = id;
            }
        }
        if (hasNormalSMarker && hasS) {
            eog.remove(sId);
            attrs[sId] = TokenizerVocab.ATTR_NORMAL;
            LOGGER.warn("'</s>' is a normal token here, removing it from the EOG list");
        }
    }

    /** CONTROL, USER_DEFINED and UNKNOWN tokens, longest UTF-8 text first, ties by id (3316-3330). */
    private static int[] partitionOrder(TokenizerVocab v, int[] attrs) {
        List<Integer> special = new ArrayList<>();
        for (int id = 0; id < attrs.length; id++) {
            if ((attrs[id] & (TokenizerVocab.ATTR_CONTROL | TokenizerVocab.ATTR_USER_DEFINED
                    | TokenizerVocab.ATTR_UNKNOWN)) != 0) {
                special.add(id);
            }
        }
        int[] lengths = new int[attrs.length];
        for (int id : special) lengths[id] = utf8Length(v.text(id));
        special.sort((a, b) -> lengths[a] != lengths[b]
                ? Integer.compare(lengths[b], lengths[a]) : Integer.compare(a, b));
        int[] order = new int[special.size()];
        for (int i = 0; i < order.length; i++) order[i] = special.get(i);
        return order;
    }

    /** LSTRIP/RSTRIP attributes by pre-tokenizer, architecture and model name (3348-3424). */
    private static void applyStripRules(TokenizerVocab v, int[] attrs, int[] partitionOrder) {
        String pre = stringMeta(v, TokenizerVocab.KEY_PRE);
        String arch = stringMeta(v, TokenizerVocab.KEY_ARCHITECTURE);
        String name = asciiLower(stringMeta(v, TokenizerVocab.KEY_NAME));
        if (containsAny(pre, "jina-v2-de", "jina-v2-es", "jina-v2-code", "mmbert")
                || containsAny(arch, "nomic-bert-moe", "jina-bert-v3")) {
            int id = v.tokenId("<mask>");
            if (id < 0) LOGGER.warn("Mask token is missing in vocab, please reconvert model!");
            else attrs[id] |= TokenizerVocab.ATTR_LSTRIP;
        } else if (containsAny(name, "phi-3", "phi3")) {
            for (int id : partitionOrder) attrs[id] |= TokenizerVocab.ATTR_RSTRIP;
            attrs[required(v, "</s>", name)] |= TokenizerVocab.ATTR_RSTRIP;
            for (String t : new String[]{"<unk>", "<s>", "<|endoftext|>"}) {
                attrs[required(v, t, name)] &= ~TokenizerVocab.ATTR_RSTRIP;
            }
        } else if (containsAny(name, "modern-bert")) {
            int id = v.tokenId("[MASK]");
            if (id < 0) LOGGER.warn("Mask token missing in vocab!");
            else attrs[id] |= TokenizerVocab.ATTR_LSTRIP;
        }
    }

    /** llama.cpp looks these tokens up with {@code token_to_id.at}, which fails the load when absent. */
    private static int required(TokenizerVocab v, String text, String modelName) {
        int id = v.tokenId(text);
        if (id < 0) {
            throw new IllegalArgumentException("Model '" + modelName + "' uses the Phi-3 special-token attributes but"
                    + " its vocabulary has no '" + text + "' token");
        }
        return id;
    }

    /**
     * True when {@code token_to_id} maps the token's text back to this id. llama.cpp scans that
     * map, whose keys keep the hybriddna k-mer marker; a k-mer whose display text was stripped of
     * the marker therefore never matches a special-token text and is skipped here.
     */
    private static boolean isMapped(TokenizerVocab v, int id) {
        return v.tokenId(v.text(id)) == id;
    }

    /** {@code control-looking token ... its type will be overridden}: the token becomes CONTROL. */
    private static void forceControl(int[] attrs, int id, String text) {
        if ((attrs[id] & TokenizerVocab.ATTR_CONTROL) == 0) {
            LOGGER.warn("control-looking token {} '{}' was not control-type; its type will be overridden", id,
                    text);
            attrs[id] |= TokenizerVocab.ATTR_CONTROL;
        }
    }

    /** An optional string key: "" when absent; a value of another type fails the load, as in llama.cpp. */
    private static String stringMeta(TokenizerVocab v, String key) {
        Object val = v.metadata(key);
        if (val == null) return "";
        if (!(val instanceof String)) throw new IllegalArgumentException(key + " is not a string: " + val);
        return (String) val;
    }

    /** {@code std::tolower} per byte in the C locale: ASCII letters only. */
    private static String asciiLower(String s) {
        char[] c = s.toCharArray();
        for (int i = 0; i < c.length; i++) {
            if (c[i] >= 'A' && c[i] <= 'Z') c[i] = (char) (c[i] + 32);
        }
        return new String(c);
    }

    private static boolean containsAny(String s, String... parts) {
        for (String p : parts) if (s.contains(p)) return true;
        return false;
    }

    static int utf8Length(String s) {
        int bytes = 0;
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            bytes += cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
            i += Character.charCount(cp);
        }
        return bytes;
    }

    private static Set<String> set(String... values) {
        return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(values)));
    }
}
