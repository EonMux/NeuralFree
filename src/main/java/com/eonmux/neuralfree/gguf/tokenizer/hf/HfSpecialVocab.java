package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Port of llama.cpp's {@code gguf.SpecialVocab} (LC/gguf-py/gguf/vocab.py:50-386): the merges, the special token ids,
 * the {@code add_*_token} flags and the normalizer flags of a Hugging Face tokenizer, as {@code add_to_gguf} writes
 * them (vocab.py:84-121). The chat template, which SpecialVocab also reads, is resolved by {@link HfChatTemplates}.
 *
 * <p>Every special token id is kept below the vocabulary size given to {@link #load}, and the {@code cls} id, for
 * which the converter's writer has no handler (vocab.py:91-96), is never written. Instances are immutable.</p>
 */
public final class HfSpecialVocab {

    private static final Logger LOGGER = LogManager.getLogger(HfSpecialVocab.class);

    static final String BOS = "bos";
    static final String EOS = "eos";
    static final String UNK = "unk";
    static final String SEP = "sep";
    static final String PAD = "pad";
    static final String CLS = "cls";
    static final String MASK = "mask";
    static final String EOT = "eot";
    static final String EOM = "eom";

    /** The default {@code special_token_types} (vocab.py:73-76). */
    private static final List<String> DEFAULT_TYPES =
            Collections.unmodifiableList(Arrays.asList(BOS, EOS, UNK, SEP, PAD, CLS, MASK));

    /** The {@code add_<typ>_token_id} writer handlers (gguf_writer.py:1293-1381); {@code cls} has none. */
    private static final Map<String, String> ID_KEYS;
    /** The {@code add_add_<typ>_token} writer handlers (gguf_writer.py:1311-1318). */
    private static final Map<String, String> ADD_KEYS;

    static {
        Map<String, String> ids = new LinkedHashMap<>();
        ids.put(BOS, GgufKeys.TOKENIZER_BOS_ID);
        ids.put(EOS, GgufKeys.TOKENIZER_EOS_ID);
        ids.put(UNK, GgufKeys.TOKENIZER_UNK_ID);
        ids.put(SEP, GgufKeys.TOKENIZER_SEP_ID);
        ids.put(PAD, GgufKeys.TOKENIZER_PAD_ID);
        ids.put(MASK, GgufKeys.TOKENIZER_MASK_ID);
        ids.put(EOT, GgufKeys.TOKENIZER_EOT_ID);
        ids.put(EOM, GgufKeys.TOKENIZER_EOM_ID);
        ID_KEYS = Collections.unmodifiableMap(ids);
        Map<String, String> add = new LinkedHashMap<>();
        add.put(BOS, GgufKeys.TOKENIZER_ADD_BOS);
        add.put(EOS, GgufKeys.TOKENIZER_ADD_EOS);
        add.put(SEP, GgufKeys.TOKENIZER_ADD_SEP);
        ADD_KEYS = Collections.unmodifiableMap(add);
    }

    /** The {@code tokenizer_config.json} / {@code special_tokens_map.json} keys naming special tokens. */
    private static final List<String> DECLARED_KEYS = Collections.unmodifiableList(Arrays.asList(
            "bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"));
    private static final List<String> DECLARED_LIST_KEYS =
            Collections.unmodifiableList(Arrays.asList("additional_special_tokens", "extra_special_tokens"));

    /** Qwen-1's first special token id (tokenization_qwen.py {@code SPECIAL_START_ID}) and its extra tokens. */
    private static final int QWEN_SPECIAL_START = 151643;
    private static final int QWEN_EXTRA_TOKENS = 205;

    private final boolean loadMerges;
    private final List<String> merges;
    private final Map<String, Integer> ids;
    private final Map<String, Boolean> add;
    private final Boolean normalizerNfc;
    private final Boolean normalizerLowercase;
    private final Boolean normalizerStripAccents;
    private final int nVocab;

    private HfSpecialVocab(boolean loadMerges, List<String> merges, Map<String, Integer> ids,
                           Map<String, Boolean> add, Boolean nfc, Boolean lowercase, Boolean stripAccents,
                           int nVocab) {
        this.loadMerges = loadMerges;
        this.merges = Collections.unmodifiableList(new ArrayList<>(merges));
        this.ids = Collections.unmodifiableMap(new LinkedHashMap<>(ids));
        this.add = Collections.unmodifiableMap(new LinkedHashMap<>(add));
        this.normalizerNfc = nfc;
        this.normalizerLowercase = lowercase;
        this.normalizerStripAccents = stripAccents;
        this.nVocab = nVocab;
    }

    /**
     * Reads the special vocabulary as {@code SpecialVocab(dir, load_merges, n_vocab=nVocab)} does: tokenizer.json
     * (merges, normalizer, post-processor), tokenizer_config.json (add flags and special token texts, matched
     * against tokenizer.json {@code added_tokens}), then config.json {@code <typ>_token_id} (or its
     * {@code text_config}) for the slots still unset, then {@code merges.txt} when merges were requested and
     * tokenizer.json has none.
     *
     * @param files      the tokenizer files
     * @param tj         the parsed tokenizer.json, or null when there is none
     * @param config     the config.json root object
     * @param loadMerges whether merges are read
     * @param nVocab     the token count; ids at or above it are skipped
     * @throws IOException         when a tokenizer file cannot be read
     * @throws GgufFormatException when a special token id is negative
     */
    public static HfSpecialVocab load(HfTokenizerFiles files, HfTokenizerJson tj, JsonView config,
                                      boolean loadMerges, int nVocab) throws IOException {
        Objects.requireNonNull(files, "files");
        Objects.requireNonNull(config, "config");
        Loader l = new Loader(nVocab);
        if (tj != null && loadMerges) l.merges.addAll(Arrays.asList(tj.model().merges()));
        if (tj != null && tj.normalizer() != null) l.parseNormalizer(tj.normalizer());
        l.loadTokenizerConfig(tj, files.readTokenizerConfig());
        l.loadConfigIds(config);
        if (loadMerges && l.merges.isEmpty()) l.merges.addAll(files.readMergesTxt());
        return new HfSpecialVocab(loadMerges, l.merges, l.ids, l.add, l.nfc, l.lowercase, l.stripAccents, nVocab);
    }

    /** The merges as {@code "a b"} strings (unmodifiable). */
    public List<String> merges() { return merges; }

    /** The special token ids by type ({@code bos}, {@code eos}, ...), in the order they were set (unmodifiable). */
    public Map<String, Integer> ids() { return ids; }

    /** The id of special token type {@code type}, or null when unset. */
    public Integer id(String type) { return ids.get(type); }

    /** The {@code add_<typ>_token} decisions by type (unmodifiable). */
    public Map<String, Boolean> addFlags() { return add; }

    /** This vocabulary with {@code merges} in place of its own ({@code special_vocab.merges = merges}). */
    public HfSpecialVocab withMerges(List<String> newMerges) {
        return new HfSpecialVocab(loadMerges, newMerges, ids, add, normalizerNfc, normalizerLowercase,
                normalizerStripAccents, nVocab);
    }

    /**
     * {@code _set_special_token(type, id)} (vocab.py:155-165): sets {@code type} when it is unset and {@code id} is
     * below the token count; a null id leaves the vocabulary unchanged.
     *
     * @throws GgufFormatException when {@code id} is negative
     */
    public HfSpecialVocab withSpecialToken(String type, Integer id) {
        Objects.requireNonNull(type, "type");
        if (id == null || ids.containsKey(type)) {
            checkNotNegative(type, id);
            return this;
        }
        Map<String, Integer> next = new LinkedHashMap<>(ids);
        setSpecialToken(next, nVocab, type, id);
        return new HfSpecialVocab(loadMerges, merges, next, add, normalizerNfc, normalizerLowercase,
                normalizerStripAccents, nVocab);
    }

    /**
     * {@code special_token_ids[type] = id}: replaces the id of {@code type} whatever it was (the RWKV writer's
     * override, base.py:2460-2462).
     *
     * @throws GgufFormatException when {@code id} is outside the vocabulary
     */
    public HfSpecialVocab withSpecialTokenReplaced(String type, int id) {
        Objects.requireNonNull(type, "type");
        if (id < 0 || id >= nVocab) {
            throw new GgufFormatException("special token " + type + " id " + id + " is outside the vocabulary [0, "
                    + nVocab + ")");
        }
        Map<String, Integer> next = new LinkedHashMap<>(ids);
        next.put(type, id);
        return new HfSpecialVocab(loadMerges, merges, next, add, normalizerNfc, normalizerLowercase,
                normalizerStripAccents, nVocab);
    }

    /**
     * Writes the keys {@code add_to_gguf} writes (vocab.py:84-121): the merges, each special token id with a writer
     * handler, the {@code add_bos/eos/sep_token} flags and the normalizer flags.
     */
    public void writeTo(Map<String, Object> out) {
        if (!merges.isEmpty()) {
            out.put(GgufKeys.TOKENIZER_MERGES, merges);
        } else if (loadMerges) {
            LOGGER.warn("adding merges requested but no merges found, output may be non-functional");
        }
        for (Map.Entry<String, Integer> e : ids.entrySet()) {
            String key = ID_KEYS.get(e.getKey());
            if (key == null) {
                LOGGER.debug("no handler for special token type {} with id {}; skipping", e.getKey(), e.getValue());
                continue;
            }
            out.put(key, e.getValue());
        }
        for (Map.Entry<String, Boolean> e : add.entrySet()) {
            String key = ADD_KEYS.get(e.getKey());
            if (key == null) {
                LOGGER.debug("no handler for add_{}_token with value {}; skipping", e.getKey(), e.getValue());
                continue;
            }
            out.put(key, e.getValue());
        }
        if (normalizerNfc != null) out.put(GgufKeys.TOKENIZER_NORMALIZER_NFC, normalizerNfc);
        if (normalizerLowercase != null) out.put(GgufKeys.TOKENIZER_NORMALIZER_LOWERCASE, normalizerLowercase);
        if (normalizerStripAccents != null) {
            out.put(GgufKeys.TOKENIZER_NORMALIZER_STRIP_ACCENTS, normalizerStripAccents);
        }
    }

    /**
     * The texts transformers treats as special tokens ({@code all_special_tokens}): the {@code <typ>_token} entries
     * and the {@code additional_special_tokens} / {@code extra_special_tokens} of tokenizer_config.json and
     * special_tokens_map.json, as strings or {@code {"content": ...}} objects, in file order.
     */
    public static Set<String> declaredSpecialTokens(JsonView tokenizerConfig, JsonView specialTokensMap) {
        Set<String> out = new LinkedHashSet<>();
        for (JsonView v : Arrays.asList(tokenizerConfig, specialTokensMap)) {
            for (String key : DECLARED_KEYS) {
                String content = tokenContent(v.raw(key));
                if (content != null) out.add(content);
            }
            for (String key : DECLARED_LIST_KEYS) {
                Object raw = v.raw(key);
                Iterable<?> items = raw instanceof List ? (List<?>) raw
                        : raw instanceof Map ? ((Map<?, ?>) raw).values() : Collections.emptyList();
                for (Object item : items) {
                    String content = tokenContent(item);
                    if (content != null) out.add(content);
                }
            }
        }
        return out;
    }

    /** tokenizer_config {@code added_tokens_decoder} as added tokens ({@code normalized} defaults to not special). */
    static List<HfTokenizerJson.AddedToken> configAddedTokens(JsonView tc) {
        Object raw = tc.raw("added_tokens_decoder");
        if (!(raw instanceof Map)) return Collections.emptyList();
        List<HfTokenizerJson.AddedToken> out = new ArrayList<>();
        for (Map.Entry<Integer, Object> e : HfVocabBase.addedTokensDecoder((Map<?, ?>) raw).entrySet()) {
            JsonView t = JsonView.of(e.getValue(), tc.source(), "added_tokens_decoder." + e.getKey());
            boolean special = t.optBool("special", false);
            out.add(new HfTokenizerJson.AddedToken(e.getKey(), t.str("content"), special, t.optBool("lstrip", false),
                    t.optBool("rstrip", false), t.optBool("single_word", false), t.optBool("normalized", !special)));
        }
        return out;
    }

    /**
     * The added tokens of a slow tokenizer converted to a fast one: tokenizer_config {@code added_tokens_decoder},
     * then {@code added_tokens.json}, then the declared special tokens of the base vocabulary (special, not
     * normalized), each text once.
     */
    static List<HfTokenizerJson.AddedToken> slowAddedTokens(HfTokenizerFiles files, JsonView tc, String[] base)
            throws IOException {
        Set<String> declared = declaredSpecialTokens(tc, files.readSpecialTokensMap());
        Map<String, HfTokenizerJson.AddedToken> out = new LinkedHashMap<>();
        for (HfTokenizerJson.AddedToken t : configAddedTokens(tc)) out.put(t.content(), t);
        for (Map.Entry<String, Integer> e : files.readAddedTokens().entrySet()) {
            boolean special = declared.contains(e.getKey());
            out.putIfAbsent(e.getKey(), new HfTokenizerJson.AddedToken(e.getValue(), e.getKey(), special, false, false,
                    false, !special));
        }
        for (int id = 0; id < base.length; id++) {
            if (base[id] != null && declared.contains(base[id])) {
                out.putIfAbsent(base[id], new HfTokenizerJson.AddedToken(id, base[id], true, false, false, false,
                        false));
            }
        }
        return new ArrayList<>(out.values());
    }

    /**
     * Qwen-1's special tokens ({@code tokenizer.special_tokens} of its tokenization_qwen.py): {@code <|endoftext|>},
     * {@code <|im_start|>}, {@code <|im_end|>} and {@code <|extra_0|>} to {@code <|extra_204|>}, from id 151643.
     */
    static Map<String, Integer> qwenSpecials() {
        Map<String, Integer> out = new LinkedHashMap<>();
        int id = QWEN_SPECIAL_START;
        out.put("<|endoftext|>", id++);
        out.put("<|im_start|>", id++);
        out.put("<|im_end|>", id++);
        for (int i = 0; i < QWEN_EXTRA_TOKENS; i++) out.put("<|extra_" + i + "|>", id++);
        return out;
    }

    /** The special tokens of a tiktoken vocabulary listed by tokenizer_config {@code added_tokens_decoder}. */
    static Map<String, Integer> tiktokenSpecials(JsonView tc, Path file) {
        Map<String, Integer> out = new LinkedHashMap<>();
        for (HfTokenizerJson.AddedToken t : configAddedTokens(tc)) out.put(t.content(), t.id());
        if (out.isEmpty()) {
            throw new GgufFormatException(file + ": the special tokens of this tiktoken vocabulary are defined by the"
                    + " checkpoint's Python tokenizer code and tokenizer_config.json lists none");
        }
        return out;
    }

    /** The text of a special token entry: a string, or the string {@code content} of an object; else null. */
    static String tokenContent(Object entry) {
        if (entry instanceof String) return (String) entry;
        if (entry instanceof Map) {
            Object content = ((Map<?, ?>) entry).get("content");
            if (content instanceof String) return (String) content;
        }
        return null;
    }

    @Override
    public String toString() {
        return "HfSpecialVocab{" + merges.size() + " merges, ids=" + ids + ", add=" + add + '}';
    }

    // ------------------------------------------------------------------ loading

    private static void setSpecialToken(Map<String, Integer> ids, int nVocab, String type, Object tid) {
        Long id = integer(tid);
        if (id == null) return;
        checkNotNegative(type, id);
        if (id < nVocab) {
            ids.putIfAbsent(type, id.intValue());
            return;
        }
        LOGGER.warn("special token type {}, id {} out of range, must be under {}; skipping", type, id, nVocab);
    }

    private static void checkNotNegative(String type, Number id) {
        if (id != null && id.longValue() < 0) {
            throw new GgufFormatException("invalid value for special token type " + type + ": " + id);
        }
    }

    /** A JSON integer as a long (saturated when larger), or null for any other value; booleans are not integers. */
    private static Long integer(Object v) {
        if (v instanceof Long || v instanceof Integer) return ((Number) v).longValue();
        if (v instanceof BigInteger) return ((BigInteger) v).signum() < 0 ? Long.MIN_VALUE : Long.MAX_VALUE;
        return null;
    }

    /** Python truthiness of a JSON value. */
    private static boolean truthy(Object v) {
        if (v == null) return false;
        if (v instanceof Boolean) return (Boolean) v;
        if (v instanceof String) return !((String) v).isEmpty();
        if (v instanceof Map) return !((Map<?, ?>) v).isEmpty();
        if (v instanceof List) return !((List<?>) v).isEmpty();
        if (v instanceof Number) return ((Number) v).doubleValue() != 0;
        return true;
    }

    /** The mutable state of one {@link #load}. */
    private static final class Loader {
        final int nVocab;
        final List<String> merges = new ArrayList<>();
        final Map<String, Integer> ids = new LinkedHashMap<>();
        final Map<String, Boolean> add = new LinkedHashMap<>();
        final List<String> types = new ArrayList<>(DEFAULT_TYPES);
        Boolean nfc;
        Boolean lowercase;
        Boolean stripAccents;

        Loader(int nVocab) {
            this.nVocab = nVocab;
        }

        /** {@code _parse_normalizer} (vocab.py:167-188). */
        void parseNormalizer(JsonView n) {
            String type = n.raw("type") instanceof String ? (String) n.raw("type") : "";
            switch (type) {
                case "Lowercase": lowercase = true; break;
                case "NFC": nfc = true; break;
                case "StripAccents": stripAccents = true; break;
                case "BertNormalizer":
                    if (n.has("lowercase")) lowercase = optionalBool(n.raw("lowercase"));
                    if (n.has("strip_accents")) stripAccents = optionalBool(n.raw("strip_accents"));
                    break;
                case "Sequence":
                    Object list = n.raw("normalizers");
                    if (list instanceof List) {
                        List<?> items = (List<?>) list;
                        for (int i = 0; i < items.size(); i++) {
                            if (items.get(i) instanceof Map) {
                                parseNormalizer(JsonView.of(items.get(i), n.source(),
                                        n.path() + ".normalizers[" + i + "]"));
                            }
                        }
                    }
                    break;
                default:
                    break;
            }
        }

        private static Boolean optionalBool(Object v) {
            return v instanceof Boolean ? (Boolean) v : null;
        }

        /** {@code _try_load_from_tokenizer_json} after the merges and normalizer (vocab.py:213-372). */
        void loadTokenizerConfig(HfTokenizerJson tj, JsonView configView) {
            Map<String, Object> tc = new LinkedHashMap<>();
            for (String key : configView.keys()) tc.put(key, configView.raw(key));
            if (tj != null) applyPostProcessor(tj, tc);
            if (tc.isEmpty()) return;
            List<HfTokenizerJson.AddedToken> added = tj != null ? tj.addedTokens()
                    : Collections.<HfTokenizerJson.AddedToken>emptyList();
            for (String typ : types) {
                Object addEntry = tc.get("add_" + typ + "_token");
                if (addEntry instanceof Boolean) {
                    Boolean decided = add.get(typ);
                    if (decided == null) {
                        add.put(typ, (Boolean) addEntry);
                    } else if (!decided.equals(addEntry)) {
                        LOGGER.warn("mismatch between tokenizer_config add_{}_token({}) and tokenizer post_processor"
                                + "<{}>({}); opting for the latter", typ, addEntry, typ, decided);
                    }
                }
                String content = tokenContent(tc.get(typ + "_token"));
                if (content == null) continue;
                for (HfTokenizerJson.AddedToken t : added) {
                    if (t.content().equals(content)) {
                        setSpecialToken(ids, nVocab, typ, (long) t.id());
                        break;
                    }
                }
            }
        }

        /** {@code _try_load_from_config_json} (vocab.py:374-386). */
        void loadConfigIds(JsonView config) {
            Object textConfig = config.raw("text_config");
            for (String typ : types) {
                String key = typ + "_token_id";
                Object id = config.raw(key);
                if (id == null && textConfig instanceof Map) id = ((Map<?, ?>) textConfig).get(key);
                setSpecialToken(ids, nVocab, typ, id);
            }
        }

        /** The post-processor decisions (vocab.py:231-338); {@code tc} is updated as the converter updates it. */
        private void applyPostProcessor(HfTokenizerJson tj, Map<String, Object> tc) {
            Specials s = new Specials(tc);
            JsonView pp = tj.postProcessor();
            if (pp == null) return;
            Object processors = pp.raw("processors");
            List<?> list = processors instanceof List ? (List<?>) processors
                    : Collections.singletonList(HfTokenOverlays.members(pp));
            for (Object item : list) {
                if (!(item instanceof Map)) continue;
                Map<?, ?> processor = (Map<?, ?>) item;
                Object type = processor.get("type");
                if ("ByteLevel".equals(type)) {
                    add.putIfAbsent(BOS, false);
                    add.putIfAbsent(EOS, false);
                } else if ("RobertaProcessing".equals(type)) {
                    roberta(processor, s, tc);
                } else if ("TemplateProcessing".equals(type)) {
                    template(processor, s, tc);
                }
            }
        }

        private void roberta(Map<?, ?> processor, Specials s, Map<String, Object> tc) {
            add.put(BOS, true);
            add.put(EOS, true);
            add.put(SEP, true);
            if (!truthy(s.cls) && !tc.isEmpty()) {
                s.cls = firstOf(processor.get("cls"), s.bos);
                tc.put("cls_token", s.cls);
            }
            if (!truthy(s.sep) && !tc.isEmpty()) {
                s.sep = firstOf(processor.get("sep"), s.eos);
                tc.put("sep_token", s.sep);
            }
        }

        /** {@code processor.get(key, [fallback])[0]}. */
        private static Object firstOf(Object value, Object fallback) {
            if (value == null) return fallback;
            if (value instanceof List && !((List<?>) value).isEmpty()) return ((List<?>) value).get(0);
            throw new GgufFormatException("RobertaProcessing cls/sep is not a [token, id] pair: " + value);
        }

        /** The crude TemplateProcessing parse (vocab.py:256-337). */
        private void template(Map<?, ?> processor, Specials s, Map<String, Object> tc) {
            List<?> single = asList(processor.get("single"));
            List<?> pair = asList(processor.get("pair"));
            Object first = null;
            Object last = null;
            if (single.size() > 1) {
                first = specialId(single.get(0));
                if (truthy(first)) {
                    boolean known = Objects.equals(first, s.bos) || Objects.equals(first, s.cls);
                    if (tc.isEmpty()) {
                        s.bos = first;
                    } else if (!known) {
                        if (!truthy(s.bos)) tc.put("bos_token", s.bos = first);
                        if (!truthy(s.cls)) tc.put("cls_token", s.cls = first);
                    }
                    boolean nowKnown = Objects.equals(first, s.bos) || Objects.equals(first, s.cls);
                    add.put(BOS, nowKnown);
                    if (!nowKnown) {
                        LOGGER.warn("unknown leading special token '{}' in TemplateProcessing<single>", first);
                    }
                }
                last = specialId(single.get(single.size() - 1));
                if (truthy(last)) {
                    if (tc.isEmpty()) {
                        s.eos = last;
                    } else if (!Objects.equals(last, s.eos)) {
                        demoteEos(s, tc, last);
                        tc.put("eos_token", s.eos = last);
                    }
                    add.put(EOS, Objects.equals(last, s.eos));
                }
            }
            if (!pair.isEmpty()) templatePair(pair, first, last, s, tc);
        }

        /** Keeps the replaced EOS as EOT, else EOM (vocab.py:285-296). */
        private void demoteEos(Specials s, Map<String, Object> tc, Object last) {
            if (!types.contains(EOT)) {
                types.add(EOT);
                tc.put("eot_token", s.eos);
            } else if (!types.contains(EOM)) {
                types.add(EOM);
                tc.put("eom_token", s.eos);
            } else {
                LOGGER.warn("overriding EOS token '{}' with '{}' without EOT/EOM fallback", s.eos, last);
            }
        }

        /** The {@code add_sep} decision from TemplateProcessing<pair> (vocab.py:300-337). */
        private void templatePair(List<?> pairItems, Object first, Object last, Specials s, Map<String, Object> tc) {
            int start = truthy(first) && Objects.equals(specialId(pairItems.get(0)), first) ? 1 : 0;
            boolean trimEnd = truthy(last) && Objects.equals(specialId(pairItems.get(pairItems.size() - 1)), last);
            if ((truthy(first) && start == 0) || (truthy(last) && !trimEnd)) {
                LOGGER.warn("TemplateProcessing<single> leading/trailing special tokens do not match"
                        + " TemplateProcessing<pair>");
            }
            int end = trimEnd ? pairItems.size() - 1 : pairItems.size();
            if (end <= start) return;
            List<?> pair = pairItems.subList(start, end);
            Object a = sequenceId(pair.get(0));
            Object b = sequenceId(pair.get(pair.size() - 1));
            if (!"A".equals(a) || !"B".equals(b)) {
                LOGGER.warn("unknown sequence {}...{} in TemplateProcessing<pair>", a, b);
                return;
            }
            if (pair.size() <= 2) return;
            List<?> middle = pair.subList(1, pair.size() - 1);
            boolean addSep = false;
            Object entry = specialId(middle.get(0));
            if (truthy(entry)) {
                boolean isSep = Objects.equals(entry, s.sep) || Objects.equals(entry, s.eos);
                if (isSep && !truthy(last)) addSep = true;
                if (!isSep) LOGGER.warn("unknown separator token '{}' in TemplateProcessing<pair>", entry);
            } else {
                LOGGER.warn("unknown middle sequence {} in TemplateProcessing<pair>", middle.get(0));
            }
            if (middle.size() == 2) {
                Object second = specialId(middle.get(1));
                if (truthy(second)) {
                    boolean isSep = Objects.equals(second, s.sep) || Objects.equals(second, s.eos);
                    if (isSep) addSep = true;
                    if (!isSep) LOGGER.warn("unknown second separator token '{}' in TemplateProcessing<pair>", second);
                } else {
                    LOGGER.warn("unknown second middle sequence {} in TemplateProcessing<pair>", middle.get(1));
                }
            }
            add.put(SEP, addSep);
            if (addSep && !truthy(s.sep) && !tc.isEmpty()) tc.put("sep_token", s.eos);
        }

        private static List<?> asList(Object v) {
            return v instanceof List ? (List<?>) v : Collections.emptyList();
        }

        /** {@code item.get('SpecialToken', {}).get('id')}. */
        private static Object specialId(Object item) {
            return nestedId(item, "SpecialToken");
        }

        /** {@code item.get('Sequence', {}).get('id')}. */
        private static Object sequenceId(Object item) {
            return nestedId(item, "Sequence");
        }

        private static Object nestedId(Object item, String kind) {
            if (!(item instanceof Map)) return null;
            Object inner = ((Map<?, ?>) item).get(kind);
            return inner instanceof Map ? ((Map<?, ?>) inner).get("id") : null;
        }
    }

    /** The special token texts the post-processor parse compares against (vocab.py:224-230). */
    private static final class Specials {
        Object bos;
        Object cls;
        Object eos;
        Object sep;

        Specials(Map<String, Object> tc) {
            bos = tc.get("bos_token");
            cls = tc.get("cls_token");
            eos = tc.get("eos_token");
            sep = tc.get("sep_token");
            if (!truthy(bos) && truthy(cls) && !tc.isEmpty()) tc.put("bos_token", bos = cls);
            if (!truthy(eos) && truthy(sep) && !tc.isEmpty()) tc.put("eos_token", eos = sep);
        }
    }
}
