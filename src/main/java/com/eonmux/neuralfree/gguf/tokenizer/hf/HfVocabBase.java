package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.UnaryOperator;
import java.util.regex.Pattern;

/**
 * The token lists of the converter's vocabulary writers: {@code get_vocab_base} for GPT-2 BPE and WordPiece
 * vocabularies (LC/conversion/base.py:1755-1797), {@code _create_vocab_sentencepiece} (base.py:2308-2396), the T5
 * Unigram list (t5.py:30-117), {@code LlamaHfVocab} (gguf-py/gguf/vocab.py:563-684), the tiktoken vocabulary of
 * {@code _set_vocab_qwen} (base.py:2240-2294) and the tekken vocabulary of {@code MistralVocab}
 * (vocab.py:720-948), with their size rules.
 */
final class HfVocabBase {

    private static final Logger LOGGER = LogManager.getLogger(HfVocabBase.class);

    /** {@code gguf.TokenType} values (gguf-py/gguf/constants.py). */
    static final int NORMAL = 1;
    static final int UNKNOWN = 2;
    static final int CONTROL = 3;
    static final int USER_DEFINED = 4;
    static final int UNUSED = 5;
    static final int BYTE = 6;

    /** SentencePiece's whitespace symbol U+2581. */
    static final String METASPACE = "▁";
    static final float PAD_SCORE = -10000.0f;
    static final float ADDED_SCORE = -1000.0f;

    private static final Pattern BYTE_TOKEN = Pattern.compile("<0x[0-9A-Fa-f]{2}>");
    private static final String[] BYTE_TO_UNICODE = bytesToUnicode();

    private HfVocabBase() {}

    /** An immutable token list: texts, {@code gguf.TokenType} types and optional scores. */
    static final class Tokens {
        private final List<String> texts;
        private final List<Integer> types;
        private final List<Float> scores;

        Tokens(String[] texts, int[] types, float[] scores) {
            if (types.length != texts.length || (scores != null && scores.length != texts.length)) {
                throw new IllegalArgumentException("token, type and score counts differ");
            }
            List<String> t = new ArrayList<>(texts.length);
            List<Integer> ty = new ArrayList<>(texts.length);
            List<Float> sc = scores == null ? null : new ArrayList<>(texts.length);
            for (int i = 0; i < texts.length; i++) {
                t.add(texts[i]);
                ty.add(types[i]);
                if (sc != null) sc.add(scores[i]);
            }
            this.texts = Collections.unmodifiableList(t);
            this.types = Collections.unmodifiableList(ty);
            this.scores = sc == null ? null : Collections.unmodifiableList(sc);
        }

        int size() { return texts.size(); }

        String text(int id) { return texts.get(id); }

        int type(int id) { return types.get(id); }

        /** The tokens with every text replaced by {@code map(text, type)}. */
        Tokens mapTexts(TextMapper map) {
            String[] t = new String[size()];
            int[] ty = new int[size()];
            float[] sc = scores == null ? null : new float[size()];
            for (int i = 0; i < t.length; i++) {
                t[i] = map.apply(texts.get(i), types.get(i));
                ty[i] = types.get(i);
                if (sc != null) sc[i] = scores.get(i);
            }
            return new Tokens(t, ty, sc);
        }

        /** Token text to id; a repeated text keeps its first id. */
        Map<String, Integer> ids() {
            Map<String, Integer> out = new HashMap<>(texts.size() * 2);
            for (int i = 0; i < texts.size(); i++) out.putIfAbsent(texts.get(i), i);
            return out;
        }

        /** Writes {@code tokenizer.ggml.tokens}, {@code token_type} and, when present, {@code scores}. */
        void writeTo(Map<String, Object> out) {
            out.put(GgufKeys.TOKENIZER_LIST, texts);
            if (scores != null) out.put(GgufKeys.TOKENIZER_SCORES, scores);
            out.put(GgufKeys.TOKENIZER_TOKEN_TYPE, types);
        }
    }

    /** A text rewrite that sees the token type. */
    @FunctionalInterface
    interface TextMapper {
        String apply(String text, int type);
    }

    /** A base vocabulary by id (null for ids without a token) and the added tokens, as a fast tokenizer has them. */
    static final class BaseVocab {
        private final String[] base;
        private final List<HfTokenizerJson.AddedToken> added;

        BaseVocab(String[] base, List<HfTokenizerJson.AddedToken> added) {
            this.base = base.clone();
            this.added = Collections.unmodifiableList(new ArrayList<>(added));
        }

        List<HfTokenizerJson.AddedToken> added() { return added; }

        /** {@code tokenizer.get_vocab()}: base tokens, then added tokens replacing base tokens of equal text. */
        Map<String, Integer> vocab() {
            Map<String, Integer> merged = new LinkedHashMap<>();
            for (int id = 0; id < base.length; id++) {
                if (base[id] != null) merged.put(base[id], id);
            }
            for (HfTokenizerJson.AddedToken t : added) merged.put(t.content(), t.id());
            return merged;
        }

        /** The number of base tokens ({@code tokenizer.vocab_size}). */
        int baseCount() {
            int n = 0;
            for (String t : base) if (t != null) n++;
            return n;
        }
    }

    /**
     * {@code does_token_look_special} (base.py:1731-1753): {@code <pad>}, {@code <mask>}, {@code <2mass>},
     * {@code [@BOS@]}, {@code <|...|>}, {@code <｜...｜>} and {@code <unused...>}.
     */
    static boolean looksSpecial(String token) {
        return token.equals("<pad>") || token.equals("<mask>") || token.equals("<2mass>") || token.equals("[@BOS@]")
                || (token.startsWith("<|") && token.endsWith("|>"))
                || (token.startsWith("<｜") && token.endsWith("｜>"))
                || (token.startsWith("<unused") && token.endsWith(">"));
    }

    /**
     * The token count of a vocabulary: {@code configured} when present (else {@code fallback}), at least
     * {@code maxId + 1}, at most {@link HfConfig#MAX_VOCAB}, and cut to the embedding rows when those are fewer and
     * hold every id.
     *
     * @throws GgufFormatException when an id does not fit the size or the size is out of bounds
     */
    static int vocabSize(OptionalInt configured, int fallback, int maxId, OptionalInt embeddingRows, String what) {
        long size = configured.isPresent() ? configured.getAsInt() : fallback;
        if (size <= 0 || size > HfConfig.MAX_VOCAB) {
            throw new GgufFormatException(what + ": vocabulary size " + size + " is outside [1, "
                    + HfConfig.MAX_VOCAB + "]");
        }
        if (maxId >= size) {
            throw new GgufFormatException(what + ": token id " + maxId + " does not fit the vocabulary size " + size);
        }
        if (embeddingRows.isPresent() && embeddingRows.getAsInt() < size) {
            int rows = embeddingRows.getAsInt();
            if (maxId >= rows) {
                throw new GgufFormatException(what + ": token id " + maxId + " has no row in the " + rows
                        + "-row token embedding");
            }
            LOGGER.warn("{}: vocabulary size {} cut to the {} token embedding rows", what, size, rows);
            size = rows;
        }
        return (int) size;
    }

    // ------------------------------------------------------------------ get_vocab_base

    /**
     * {@code get_vocab_base} (base.py:1755-1797): ids without a token become {@code [PAD<i>]} UNUSED; an added token
     * that is not {@code normalized} is first rewritten by {@code roundTrip} (the converter's
     * {@code decode(encode(token))}); added tokens that are special or look special are CONTROL, other added tokens
     * USER_DEFINED (with U+2581 written as a space when {@code metaspaceToSpace}), the rest NORMAL.
     */
    static Tokens vocabBase(BaseVocab v, int size, UnaryOperator<String> roundTrip, boolean metaspaceToSpace) {
        Map<String, Integer> vocab = v.vocab();
        Map<Integer, String> reverse = new HashMap<>();
        for (Map.Entry<String, Integer> e : vocab.entrySet()) reverse.put(e.getValue(), e.getKey());
        Map<String, Integer> addedVocab = new HashMap<>();
        Map<Integer, HfTokenizerJson.AddedToken> addedById = new HashMap<>();
        for (HfTokenizerJson.AddedToken t : v.added()) {
            addedVocab.put(t.content(), t.id());
            addedById.put(t.id(), t);
        }
        String[] texts = new String[size];
        int[] types = new int[size];
        for (int i = 0; i < size; i++) {
            String token = reverse.get(i);
            if (token == null) {
                texts[i] = "[PAD" + i + "]";
                types[i] = UNUSED;
                continue;
            }
            if (addedVocab.containsKey(token)) {
                HfTokenizerJson.AddedToken at = addedById.get(i);
                if (at == null) at = addedById.get(addedVocab.get(token));
                if (!at.normalized()) {
                    String previous = token;
                    token = roundTrip.apply(token);
                    if (!previous.equals(token)) {
                        LOGGER.info("added token '{}' is encoded and decoded back to '{}'", previous, token);
                    }
                }
                if (at.special() || looksSpecial(token)) {
                    types[i] = CONTROL;
                } else {
                    if (metaspaceToSpace) token = token.replace(METASPACE, " ");
                    types[i] = USER_DEFINED;
                }
            } else {
                types[i] = NORMAL;
            }
            texts[i] = token;
        }
        return new Tokens(texts, types, null);
    }

    /** The largest id of a vocabulary map, or -1 when it is empty. */
    static int maxId(Map<String, Integer> vocab) {
        int max = -1;
        for (int id : vocab.values()) max = Math.max(max, id);
        return max;
    }

    // ------------------------------------------------------------------ SentencePiece

    /**
     * {@code _create_vocab_sentencepiece} (base.py:2308-2396): {@code size} slots of {@code [PAD<i>]} (score -10000,
     * UNUSED), then the pieces below {@code size} with their scores and types (USER_DEFINED pieces become NORMAL),
     * then {@code added_tokens.json} (USER_DEFINED, -1000), then tokenizer_config {@code added_tokens_decoder}
     * (CONTROL when special or special-looking, else USER_DEFINED with U+2581 as a space; -1000).
     */
    static Tokens sentencePiece(SentencePieceModel sp, int size, Map<String, Integer> addedJson,
                                JsonView tokenizerConfig) {
        String[] texts = padTexts(size);
        float[] scores = new float[size];
        Arrays.fill(scores, PAD_SCORE);
        int[] types = new int[size];
        Arrays.fill(types, UNUSED);
        for (int id = 0; id < sp.pieceCount(); id++) {
            if (id >= size) {
                LOGGER.warn("ignoring SentencePiece tokens from {}: id is out of range, max={}", id, size - 1);
                break;
            }
            texts[id] = sp.piece(id);
            scores[id] = sp.score(id);
            types[id] = pieceType(sp.type(id), false);
        }
        applyAddedTokensJson(addedJson, texts, scores, types);
        Object decoder = tokenizerConfig.raw("added_tokens_decoder");
        if (decoder instanceof Map) {
            for (Map.Entry<Integer, Object> e : addedTokensDecoder((Map<?, ?>) decoder).entrySet()) {
                applyAddedTokenDecoder(tokenizerConfig, e.getKey(), e.getValue(), texts, scores, types);
            }
        }
        return new Tokens(texts, types, scores);
    }

    /**
     * The T5 Unigram list (t5.py:62-104): {@code size} slots of {@code [PAD<i>]} (score -10000, UNUSED), every piece
     * with its score and type, then {@code added_tokens.json} (USER_DEFINED, -1000).
     *
     * @throws GgufFormatException when the model has more pieces than {@code size}
     */
    static Tokens t5(SentencePieceModel sp, int size, Map<String, Integer> addedJson) {
        if (sp.pieceCount() > size) {
            throw new GgufFormatException("the SentencePiece model has " + sp.pieceCount()
                    + " pieces, more than the vocabulary size " + size);
        }
        String[] texts = padTexts(size);
        float[] scores = new float[size];
        Arrays.fill(scores, PAD_SCORE);
        int[] types = new int[size];
        Arrays.fill(types, UNUSED);
        for (int id = 0; id < sp.pieceCount(); id++) {
            texts[id] = sp.piece(id);
            scores[id] = sp.score(id);
            types[id] = pieceType(sp.type(id), false);
        }
        applyAddedTokensJson(addedJson, texts, scores, types);
        return new Tokens(texts, types, scores);
    }

    /**
     * The pieces of a Mistral SentencePiece model (vocab.py:785-806): every piece with its score and type, USER_DEFINED
     * pieces being NORMAL.
     */
    static Tokens mistralSentencePiece(SentencePieceModel sp) {
        int n = sp.pieceCount();
        String[] texts = new String[n];
        float[] scores = new float[n];
        int[] types = new int[n];
        for (int id = 0; id < n; id++) {
            texts[id] = sp.piece(id);
            scores[id] = sp.score(id);
            types[id] = pieceType(sp.type(id), false);
        }
        return new Tokens(texts, types, scores);
    }

    /** The {@code gguf.TokenType} of a piece; USER_DEFINED is NORMAL unless {@code keepUserDefined}. */
    static int pieceType(SentencePieceModel.PieceType type, boolean keepUserDefined) {
        switch (type) {
            case UNKNOWN: return UNKNOWN;
            case CONTROL: return CONTROL;
            case UNUSED: return UNUSED;
            case BYTE: return BYTE;
            case USER_DEFINED: return keepUserDefined ? USER_DEFINED : NORMAL;
            default: return NORMAL;
        }
    }

    private static String[] padTexts(int size) {
        String[] texts = new String[size];
        for (int i = 0; i < size; i++) texts[i] = "[PAD" + i + "]";
        return texts;
    }

    private static void applyAddedTokensJson(Map<String, Integer> addedJson, String[] texts, float[] scores,
                                             int[] types) {
        for (Map.Entry<String, Integer> e : addedJson.entrySet()) {
            int id = e.getValue();
            if (id >= texts.length) {
                LOGGER.warn("ignoring added token {}: id is out of range, max={}", id, texts.length - 1);
                continue;
            }
            texts[id] = e.getKey();
            scores[id] = ADDED_SCORE;
            types[id] = USER_DEFINED;
        }
    }

    private static void applyAddedTokenDecoder(JsonView config, int id, Object entry, String[] texts, float[] scores,
                                               int[] types) {
        String where = config.where("added_tokens_decoder") + "." + id;
        if (!(entry instanceof Map) || !(((Map<?, ?>) entry).get("content") instanceof String)) {
            throw new GgufFormatException(where + ": expected an object with a string content");
        }
        Map<?, ?> data = (Map<?, ?>) entry;
        String token = (String) data.get("content");
        if (id >= texts.length) {
            LOGGER.warn("ignoring added token {}: id is out of range, max={}", id, texts.length - 1);
            return;
        }
        if (types[id] != UNUSED && !texts[id].equals(token)) {
            LOGGER.warn("replacing token {}: '{}' -> '{}'", id, texts[id], token);
        }
        if (Boolean.TRUE.equals(data.get("special")) || looksSpecial(token)) {
            types[id] = CONTROL;
        } else {
            token = token.replace(METASPACE, " ");
            types[id] = USER_DEFINED;
        }
        scores[id] = ADDED_SCORE;
        texts[id] = token;
    }

    /** An {@code added_tokens_decoder} object keyed by id, in file order. */
    static Map<Integer, Object> addedTokensDecoder(Map<?, ?> decoder) {
        Map<Integer, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : decoder.entrySet()) {
            int id;
            try {
                id = Integer.parseInt(String.valueOf(e.getKey()));
            } catch (NumberFormatException ex) {
                throw new GgufFormatException("added_tokens_decoder key '" + e.getKey() + "' is not a token id", ex);
            }
            if (id < 0 || id >= HfTokenizerFiles.MAX_TOKENS) {
                throw new GgufFormatException("added_tokens_decoder id " + id + " is out of range");
            }
            out.put(id, e.getValue());
        }
        return out;
    }

    // ------------------------------------------------------------------ LlamaHfVocab

    /**
     * {@code LlamaHfVocab.all_tokens} (vocab.py:567-684): the base tokens by id ({@code <0xXX>} BYTE, special ids
     * CONTROL, others NORMAL), then the added tokens with ids past the base vocabulary in id order (CONTROL when a
     * declared special token, else USER_DEFINED); every score is -1000.
     *
     * @throws GgufFormatException when tokenizer.json is not a Llama BPE tokenizer with byte fallback and a Sequence
     *                             decoder, or its ids are not contiguous
     */
    static Tokens llamaHf(HfTokenizerJson tj, Set<String> declaredSpecial) {
        requireLlamaHf(tj);
        BaseVocab base = new BaseVocab(tj.model().vocab(), tj.addedTokens());
        Map<String, Integer> vocab = base.vocab();
        Map<Integer, String> reverse = new HashMap<>();
        for (Map.Entry<String, Integer> e : vocab.entrySet()) reverse.put(e.getValue(), e.getKey());
        Set<Integer> specialIds = new HashSet<>();
        for (String s : declaredSpecial) {
            Integer id = vocab.get(s);
            if (id != null) specialIds.add(id);
        }
        int baseSize = base.baseCount();
        Map<Integer, String> added = new TreeMap<>();
        for (HfTokenizerJson.AddedToken t : tj.addedTokens()) {
            int id = vocab.get(t.content());
            if (id >= baseSize) added.put(id, t.content());
        }
        int n = baseSize + added.size();
        String[] texts = new String[n];
        int[] types = new int[n];
        float[] scores = new float[n];
        Arrays.fill(scores, ADDED_SCORE);
        for (int id = 0; id < baseSize; id++) {
            String token = reverse.get(id);
            if (token == null) {
                throw new GgufFormatException(tj.source() + ": base token id " + id + " has no token; the vocabulary"
                        + " ids are not contiguous");
            }
            texts[id] = token;
            types[id] = BYTE_TOKEN.matcher(token).matches() ? BYTE : specialIds.contains(id) ? CONTROL : NORMAL;
        }
        int next = baseSize;
        for (Map.Entry<Integer, String> e : added.entrySet()) {
            if (e.getKey() != next) {
                throw new GgufFormatException(tj.source() + ": added token '" + e.getValue() + "' has id "
                        + e.getKey() + ", expected the next id " + next);
            }
            texts[next] = e.getValue();
            types[next++] = declaredSpecial.contains(e.getValue()) ? CONTROL : USER_DEFINED;
        }
        return new Tokens(texts, types, scores);
    }

    /**
     * Whether tokenizer.json is the Llama BPE tokenizer {@code LlamaHfVocab} accepts: a BPE model with byte fallback
     * and a Sequence decoder that is not a Llama-3 BPE ({@code ignore_merges} without byte fallback).
     */
    static boolean isLlamaHf(HfTokenizerJson tj) {
        if (tj == null || tj.model().type() != HfTokenizerJson.ModelType.BPE) return false;
        return tj.model().byteFallback() && tj.decoder() != null && "Sequence".equals(tj.decoder().raw("type"));
    }

    private static void requireLlamaHf(HfTokenizerJson tj) {
        if (!isLlamaHf(tj)) {
            throw new GgufFormatException((tj == null ? "the checkpoint" : tj.source())
                    + " has no Llama BPE tokenizer (a tokenizer.json BPE model with byte_fallback and a Sequence"
                    + " decoder)");
        }
    }

    /**
     * The PLaMo token list (base.py:2745-2784): the {@code tokenizer.jsonl} rows, {@code normalTokens} typed NORMAL,
     * the UNKNOWN, CONTROL and BYTE types kept, other {@code <|plamo:...|>} tokens CONTROL and the rest NORMAL, then
     * {@code [PAD1]}, {@code [PAD2]}, ... (UNUSED, -1000) up to {@code size}.
     */
    static Tokens plamo(PlamoVocabReader reader, int size, Set<String> normalTokens) {
        String[] texts = new String[size];
        float[] scores = new float[size];
        int[] types = new int[size];
        for (int i = 0; i < size; i++) {
            if (i >= reader.size()) {
                texts[i] = "[PAD" + (i - reader.size() + 1) + "]";
                scores[i] = ADDED_SCORE;
                types[i] = UNUSED;
                continue;
            }
            texts[i] = reader.token(i);
            scores[i] = (float) reader.score(i);
            types[i] = plamoType(texts[i], reader.type(i), normalTokens);
        }
        return new Tokens(texts, types, scores);
    }

    private static int plamoType(String token, String type, Set<String> normals) {
        if (normals.contains(token)) return NORMAL;
        switch (type) {
            case "UNKNOWN": return UNKNOWN;
            case "CONTROL": return CONTROL;
            case "BYTE": return BYTE;
            default: return token.startsWith("<|plamo:") && token.endsWith("|>") ? CONTROL : NORMAL;
        }
    }

    /**
     * The RWKV world token list (base.py:2424-2443): {@code <s>} (CONTROL), the world tokens as {@code repr} texts
     * (NORMAL), then {@code [PAD<i>]} (UNUSED) up to {@code size}.
     */
    static Tokens rwkvWorld(RwkvVocabReader reader, int size) {
        String[] texts = new String[size];
        int[] types = new int[size];
        texts[0] = "<s>";
        types[0] = CONTROL;
        for (int i = 1; i < size; i++) {
            boolean world = i <= reader.size();
            texts[i] = world ? reader.reprText(i - 1) : "[PAD" + i + "]";
            types[i] = world ? NORMAL : UNUSED;
        }
        return new Tokens(texts, types, null);
    }

    /**
     * The version mistral_common's {@code get_spm_version} reads from a SentencePiece file name: {@code .model} is
     * v1, {@code .model.v<N>} and {@code .model.v<N>m<M>} are v<N>.
     */
    static int mistralSpmVersion(Path file) {
        String name = file.getFileName().toString();
        String last = name.substring(name.lastIndexOf('.') + 1);
        if (last.equals("model")) return 1;
        String v = last.split("m", -1)[0];
        if (!v.matches("v[0-9]{1,6}")) throw new GgufFormatException(file + ": unrecognized tokenizer file name");
        return Integer.parseInt(v.substring(1));
    }

    /** A token → id file as tokens by id; ids must be unique. */
    static String[] byId(Map<String, Integer> vocab, Path file) {
        int max = maxId(vocab);
        String[] out = new String[max + 1];
        for (Map.Entry<String, Integer> e : vocab.entrySet()) {
            if (out[e.getValue()] != null) {
                throw new GgufFormatException(file + ": tokens '" + out[e.getValue()] + "' and '" + e.getKey()
                        + "' share id " + e.getValue());
            }
            out[e.getValue()] = e.getKey();
        }
        return out;
    }

    // ------------------------------------------------------------------ byte-level vocabularies

    /**
     * GPT-2 {@code bytes_to_unicode} (vocab.py:693-716): the printable bytes {@code !..~}, {@code ¡..¬} and
     * {@code ®..ÿ} map to themselves, the others to U+0100 onwards in byte order.
     */
    private static String[] bytesToUnicode() {
        String[] table = new String[256];
        int n = 0;
        for (int b = 0; b < 256; b++) {
            boolean printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            table[b] = new String(Character.toChars(printable ? b : 256 + n++));
        }
        return table;
    }

    /** {@code token_bytes_to_string}: the GPT-2 byte-level text of {@code bytes} (qwen.py:23-26). */
    static String byteLevelText(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length);
        for (byte b : bytes) sb.append(BYTE_TO_UNICODE[b & 0xFF]);
        return sb.toString();
    }

    /**
     * {@code QwenModel.bpe(ranks, token, max_rank=rank)} (qwen.py:29-43): the parts the lowest-rank merges below
     * {@code maxRank} build from the bytes of {@code token}.
     */
    static List<byte[]> bpe(TiktokenReader ranks, byte[] token, int maxRank) {
        List<byte[]> parts = new ArrayList<>(token.length);
        for (byte b : token) parts.add(new byte[]{b});
        while (true) {
            int minIdx = -1;
            int minRank = -1;
            for (int i = 0; i + 1 < parts.size(); i++) {
                int rank = ranks.rankOf(concat(parts.get(i), parts.get(i + 1)));
                if (rank >= 0 && (minRank < 0 || rank < minRank)) {
                    minIdx = i;
                    minRank = rank;
                }
            }
            if (minRank < 0 || minRank >= maxRank) break;
            parts.set(minIdx, concat(parts.get(minIdx), parts.get(minIdx + 1)));
            parts.remove(minIdx + 1);
        }
        return parts;
    }

    /**
     * The merges of a tiktoken vocabulary as {@code _set_vocab_qwen} rebuilds them (base.py:2253-2260): for every
     * token of more than one byte, in file order, the two parts {@link #bpe} leaves below its own rank.
     *
     * @throws GgufFormatException when a token is not the merge of exactly two lower-ranked parts
     */
    static List<String> tiktokenMerges(TiktokenReader ranks) {
        List<String> merges = new ArrayList<>(ranks.size());
        for (int i = 0; i < ranks.size(); i++) {
            byte[] token = ranks.token(i);
            if (token.length == 1) continue;
            List<byte[]> merged = bpe(ranks, token, ranks.rank(i));
            if (merged.size() != 2) {
                throw new GgufFormatException("tiktoken token of rank " + ranks.rank(i) + " ('" + byteLevelText(token)
                        + "') is not the merge of two lower-ranked tokens");
            }
            merges.add(byteLevelText(merged.get(0)) + " " + byteLevelText(merged.get(1)));
        }
        return merges;
    }

    /**
     * The tiktoken token list of {@code _set_vocab_qwen} (base.py:2262-2274): each rank's byte-level text, the
     * special tokens (CONTROL) over it, and {@code [PAD<i>]} UNUSED for ids without a token.
     */
    static Tokens tiktoken(TiktokenReader ranks, Map<String, Integer> specials, int size) {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        for (int i = 0; i < ranks.size(); i++) vocab.put(byteLevelText(ranks.token(i)), ranks.rank(i));
        vocab.putAll(specials);
        Map<Integer, String> reverse = new HashMap<>();
        for (Map.Entry<String, Integer> e : vocab.entrySet()) reverse.put(e.getValue(), e.getKey());
        String[] texts = new String[size];
        int[] types = new int[size];
        for (int i = 0; i < size; i++) {
            String token = reverse.get(i);
            if (token == null) {
                texts[i] = "[PAD" + i + "]";
                types[i] = UNUSED;
            } else {
                texts[i] = token;
                types[i] = specials.containsKey(token) ? CONTROL : NORMAL;
            }
        }
        return new Tokens(texts, types, null);
    }

    /**
     * The tekken token list (vocab.py:808-825): the special tokens (CONTROL), then the inner vocabulary as byte-level
     * texts (NORMAL); every score is 0.
     */
    static Tokens tekken(TekkenReader t) {
        int n = t.numSpecialTokens() + t.innerVocabSize();
        String[] texts = new String[n];
        int[] types = new int[n];
        float[] scores = new float[n];
        for (int i = 0; i < t.numSpecialTokens(); i++) {
            texts[i] = t.specialToken(i);
            types[i] = CONTROL;
        }
        for (int r = 0; r < t.innerVocabSize(); r++) {
            texts[t.numSpecialTokens() + r] = byteLevelText(t.tokenBytes(r));
            types[t.numSpecialTokens() + r] = NORMAL;
        }
        return new Tokens(texts, types, scores);
    }

    /**
     * {@code extract_vocab_merges_from_model} (vocab.py:899-948): for every inner rank from 256, each split of its
     * bytes into two inner tokens, ordered by the parts' ranks, as byte-level {@code "a b"} merges.
     *
     * @throws GgufFormatException when a token has no valid split
     */
    static List<String> tekkenMerges(TekkenReader t) {
        Map<String, Integer> rankOf = new HashMap<>(t.innerVocabSize() * 2);
        for (int r = 0; r < t.innerVocabSize(); r++) rankOf.put(latin1(t.tokenBytes(r)), r);
        List<String> merges = new ArrayList<>();
        for (int r = 256; r < t.innerVocabSize(); r++) {
            byte[] token = t.tokenBytes(r);
            List<int[]> local = new ArrayList<>();
            for (int j = 1; j < token.length; j++) {
                Integer left = rankOf.get(latin1(Arrays.copyOfRange(token, 0, j)));
                Integer right = rankOf.get(latin1(Arrays.copyOfRange(token, j, token.length)));
                if (left != null && right != null) local.add(new int[]{left, right, j});
            }
            if (local.isEmpty()) {
                throw new GgufFormatException("tekken: could not find a valid merge for the token of rank " + r
                        + " ('" + byteLevelText(token) + "')");
            }
            local.sort((a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
            for (int[] m : local) {
                merges.add(byteLevelText(Arrays.copyOfRange(token, 0, m[2])) + " "
                        + byteLevelText(Arrays.copyOfRange(token, m[2], token.length)));
            }
        }
        return merges;
    }

    private static String latin1(byte[] bytes) {
        return new String(bytes, StandardCharsets.ISO_8859_1);
    }

    private static byte[] concat(byte[] a, byte[] b) {
        byte[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
