package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.chat.ChatFormatter;
import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateDetector;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.chat.TemplateCaps;
import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParser;
import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParserAnalysis;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.BpeTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.Gemma4Tokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.IncrementalDetokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.NoVocabTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.Plamo2Tokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.Plamo3Tokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.RwkvTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;
import com.eonmux.neuralfree.gguf.tokenizer.SpmTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.TokenDecoder;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerAlgorithm;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import com.eonmux.neuralfree.gguf.tokenizer.UgmTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.WpmTokenizer;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.IntFunction;

/**
 * Tokenizer facade for a GGUF model: vocabulary, special tokens, the llama.cpp tokenizer
 * algorithm selected by {@code tokenizer.ggml.model}, decoding and chat-template formatting.
 *
 * <p>Routing: llama to SPM, gpt2/hybriddna/whitespace to BPE, gemma4 to Gemma4, t5 to UGM,
 * bert to WPM, rwkv to RWKV, plamo2/plamo3 to PLaMo, none/no_vocab to the no-vocabulary
 * tokenizer. {@link #encode(String)} parses special tokens and applies the vocabulary's
 * BOS/EOS policy, as llama.cpp prompt tokenization does.</p>
 *
 * <p>{@link #decode(int[])} applies the vocabulary's Hugging Face overlays when present: the
 * decoder chain ({@link TokenizerVocab#decoder()}) in place of piece concatenation, then
 * transformers' {@code clean_up_tokenization} ({@link TokenizerVocab#cleanUpTokenizationSpaces()}).
 * Their text depends on neighbouring tokens, so with either overlay {@link #decode(int)} is the
 * one-token sequence's text and {@link #createStreamingDecoder()} streams through
 * {@link IncrementalDetokenizer}; without them both render llama.cpp token pieces.</p>
 */
public final class GGUFTokenizer {

    /**
     * The rerank prompt key: {@code tokenizer.chat_template} with the template name {@code rerank}
     * ({@code llama_model_chat_template(model, "rerank")}, src/llama-model.cpp:3750-3767).
     */
    public static final String KEY_RERANK_TEMPLATE = ChatTemplateSource.KEY_DEFAULT + ".rerank";
    /**
     * The speech chat template key: {@code tokenizer.chat_template.tts} ({@code
     * llama_model_chat_template(model, "tts")}). An LFM3-Audio checkpoint carries it; its rendered prompt ends in
     * the audio start token, which opens the speech pipeline's fresh turn. The chat-template override property
     * does not replace it.
     */
    public static final String KEY_TTS_TEMPLATE = ChatTemplateSource.KEY_DEFAULT + ".tts";
    /**
     * The decision models' systemone template key: {@code tokenizer.chat_template.systemone} ({@code
     * llama_model_chat_template(model, "systemone")}). The vllm-decision2 and clef decision heads carry it; the
     * server renders their joint prompt through it.
     */
    public static final String KEY_SYSTEMONE_TEMPLATE = ChatTemplateSource.KEY_DEFAULT + ".systemone";

    private static final Logger LOGGER = LogManager.getLogger(GGUFTokenizer.class);
    private static final byte[] EMPTY = new byte[0];

    private final TokenizerVocab vocab;
    private final SpecialTokens special;
    private final TokenizerAlgorithm algorithm;
    private final ChatFormatter chat;
    /** The {@code tts} template's formatter, or null when the model carries no speech chat template. */
    private final ChatFormatter ttsChat;
    /** Lazily filled per-token output bytes; idempotent writes make concurrent fills benign. */
    private final byte[][] visibleBytesCache;
    private final String[] decodedStringCache;

    /** The tokenizer of an open model, from its {@code tokenizer.*} metadata. */
    public GGUFTokenizer(ModelSource source) {
        this(TokenizerVocab.fromSource(source));
    }

    public GGUFTokenizer(TokenizerVocab vocab) {
        if (vocab == null) throw new IllegalArgumentException("vocabulary is null");
        this.vocab = vocab;
        String template = templateSource(vocab);
        this.special = SpecialTokens.of(vocab, preservedMarkers(template));
        this.algorithm = algorithmFor(vocab, special);
        this.chat = new ChatFormatter(template, special.bos() >= 0 ? vocab.text(special.bos()) : "",
                special.eos() >= 0 ? vocab.text(special.eos()) : "", vocab.addBos(), vocab.addEos(),
                vocab.architecture(), ChatFormatter.preserveReasoningFromProperties(),
                ChatFormatter.contentFormatFromProperties());
        String tts = templateMetadata(vocab, KEY_TTS_TEMPLATE);
        this.ttsChat = tts == null || tts.isBlank() ? null
                : new ChatFormatter(ChatTemplateSource.resolve(tts, null),
                        special.bos() >= 0 ? vocab.text(special.bos()) : "",
                        special.eos() >= 0 ? vocab.text(special.eos()) : "", vocab.addBos(), vocab.addEos(),
                        vocab.architecture(), ChatFormatter.preserveReasoningFromProperties(),
                        ChatFormatter.contentFormatFromProperties());
        this.visibleBytesCache = new byte[vocab.size()][];
        this.decodedStringCache = new String[vocab.size()];
        LOGGER.info("tokenizer: model={} pre={} vocab={} bos={} eos={} eog={}", vocab.model(), vocab.pre(),
                vocab.size(), special.bos(), special.eos(), special.eogIds().size());
    }

    /**
     * The markers the differential autoparser derives from the template, for the tokenizer to match as tokens
     * ({@link SpecialTokens#of(TokenizerVocab, java.util.List)}); empty when the template does not analyse.
     */
    private static List<String> preservedMarkers(String template) {
        AutoParserAnalysis analysis = AutoParser.analyze(template);
        return analysis == null ? Collections.<String>emptyList() : analysis.preservedTokens();
    }

    /**
     * The template source prompts are rendered with: the override properties when set, else the
     * model's metadata — unless the {@link ChatTemplateSource#JINJA_PROPERTY} switch sends rendering
     * onto the non-Jinja route.
     */
    private static String templateSource(TokenizerVocab vocab) {
        if (!ChatTemplateSource.jinjaFromProperties()) return legacySource(vocab);
        String override = ChatTemplateSource.fromProperties();
        return override != null ? override
                : ChatTemplateSource.resolve(templateMetadata(vocab, ChatTemplateSource.KEY_DEFAULT),
                        templateMetadata(vocab, ChatTemplateSource.KEY_TOOL_USE));
    }

    /**
     * llama.cpp's non-Jinja route: the template's text, kept raw from the override or the model's
     * {@code tokenizer.chat_template}, is read as one of llama.cpp's known templates and replaced
     * by that built-in template's source, as {@code common_chat_templates_init} keeps the text and
     * {@code llm_chat_detect_template} guesses from it (src/llama-chat.cpp:89-240). A template no
     * style fits is refused the way llama.cpp's {@code common_chat_verify_template} refuses it
     * without {@code --jinja} (common/arg.cpp:940-946).
     */
    private static String legacySource(TokenizerVocab vocab) {
        String raw = ChatTemplateSource.rawFromProperties();
        String template = raw != null ? raw : templateMetadata(vocab, ChatTemplateSource.KEY_DEFAULT);
        String toolUse = raw != null ? null : templateMetadata(vocab, ChatTemplateSource.KEY_TOOL_USE);
        String legacy = ChatTemplateSource.legacyTemplate(template, toolUse);
        String detected = ChatTemplateDetector.detect(legacy);
        if (detected == null) {
            throw new IllegalArgumentException("--no-jinja (" + ChatTemplateSource.JINJA_PROPERTY
                    + "=false): " + ChatTemplateDetector.refusal(legacy));
        }
        String source = ChatTemplateSource.detectedSource(detected);
        LOGGER.info("chat template: {} (guessed from the model's template for the non-Jinja route)", detected);
        return source;
    }

    /** A chat template metadata string, or null when the key is absent or not a string. */
    private static String templateMetadata(TokenizerVocab v, String key) {
        Object value = v.metadata(key);
        if (value == null || value instanceof String) return (String) value;
        LOGGER.warn("GGUF metadata '{}' is a {}, not a string; it is ignored", key, value.getClass().getSimpleName());
        return null;
    }

    private static TokenizerAlgorithm algorithmFor(TokenizerVocab v, SpecialTokens s) {
        switch (v.model()) {
            case "llama": return new SpmTokenizer(v, s);
            case "gpt2":
            case "hybriddna":
            case "whitespace": return new BpeTokenizer(v, s);
            case "gemma4": return new Gemma4Tokenizer(v, s);
            case "t5": return new UgmTokenizer(v, s);
            case "bert": return new WpmTokenizer(v, s);
            case "rwkv": return new RwkvTokenizer(v, s);
            case "plamo2": return new Plamo2Tokenizer(v, s);
            case "plamo3": return new Plamo3Tokenizer(v, s);
            case "none":
            case "no_vocab": return new NoVocabTokenizer(v, s);
            default:
                throw new IllegalArgumentException("Unsupported tokenizer model '" + v.model() + "'");
        }
    }

    public TokenizerVocab vocab() { return vocab; }
    public SpecialTokens specialTokens() { return special; }

    public int vocabSize() { return vocab.size(); }
    public String getToken(int id) { return vocab.text(id); }
    public int getTokenId(String token) { return vocab.tokenId(token); }
    public float getScore(int id) { return vocab.score(id); }
    public int bosId() { return special.bos(); }
    public int eosId() { return special.eos(); }
    public int padId() { return special.pad(); }
    public int unkId() { return special.unk(); }
    public String tokenizerModel() { return vocab.model(); }
    public String tokenizerPre() { return vocab.pre(); }
    /**
     * The chat template prompts are rendered with: {@code tokenizer.chat_template}, or when that is absent,
     * empty or {@code chatml}, {@code tokenizer.chat_template.tool_use} or the built-in ChatML template
     * ({@link ChatTemplateSource#resolve}). Never null.
     */
    public String chatTemplate() { return chat.source(); }

    /** The capabilities of {@link #chatTemplate()}, probed as llama.cpp probes them ({@link TemplateCaps}). */
    public TemplateCaps chatTemplateCaps() { return chat.caps(); }

    /**
     * The capabilities {@code request} renders with: its {@code chat_template} override's when it carries one,
     * else those of {@link #chatTemplateCaps()} ({@link ChatFormatter#caps(ChatRequest)}).
     */
    public TemplateCaps chatTemplateCaps(ChatRequest request) { return chat.caps(request); }
    public String architecture() { return vocab.architecture(); }

    /**
     * The rerank prompt ({@value #KEY_RERANK_TEMPLATE}) with {@code {query}}, {@code {document}} and the score
     * request's {@code {instruction}} placeholders, as the converter writes it for Qwen3 rerankers; null when the
     * model has none.
     *
     * @throws IllegalStateException when the metadata value is not a string
     */
    public String rerankTemplate() {
        return stringMetadata(KEY_RERANK_TEMPLATE);
    }

    /**
     * The decision models' systemone chat template ({@value #KEY_SYSTEMONE_TEMPLATE}): the joint prompt of the
     * vllm-decision2 and clef decision heads, rendered with the {@code state}, {@code questions}, {@code images},
     * {@code sep}, {@code mark_question} and {@code mark_option} variables; null when the model carries none.
     *
     * @throws IllegalStateException when the metadata value is not a string
     */
    public String systemoneTemplate() {
        return stringMetadata(KEY_SYSTEMONE_TEMPLATE);
    }

    /**
     * The speech chat template's resolved source ({@code tokenizer.chat_template.tts}, resolved like
     * {@link ChatTemplateSource#resolve(String)} resolves the default), or null when the model carries none: the
     * speech pipeline's prompts render through it, and a model without it has no speech prompt to render
     * ({@code llama_model_chat_template(model, "tts")}). The chat-template override property does not replace it.
     */
    public String ttsTemplate() {
        return ttsChat == null ? null : ttsChat.source();
    }

    /**
     * Renders {@code request} through the model's conversation template ({@link #chatTemplate()}) with the speech
     * variables bound ({@link ChatFormatter#formatSpeech}): the omni chat route's prompt, which the generator
     * continues with the asked mode's switch token.
     *
     * @throws IllegalArgumentException for malformed messages or a template override that does not render
     */
    public String formatSpeechChat(ChatRequest request, String language, String voice) {
        return chat.formatSpeech(request, language, voice);
    }

    /**
     * Renders the plain speech text as one user message through the speech chat template, with the request's
     * {@code language} and {@code voice} bound (common_chat_get_tts_prompt, common/chat.cpp:859-868): the speech
     * pipeline's fresh turn speaks the render, which ends in the audio start token.
     *
     * @throws UnsupportedOperationException when the model carries no {@code tts} template
     */
    public String formatTtsPrompt(String text, String language, String voice) {
        if (ttsChat == null) {
            throw new UnsupportedOperationException("the model has no " + KEY_TTS_TEMPLATE
                    + " chat template to render the speech text with");
        }
        return ttsChat.formatTtsPrompt(text, language, voice);
    }

    /**
     * The token ids appended to every rerank prompt ({@link GgufKeys#NF_RERANK_SUFFIX_IDS}, the jina_vl score token);
     * empty when the model has none.
     *
     * @throws IllegalStateException when the metadata value is not an array of ids inside the vocabulary
     */
    public int[] rerankSuffix() {
        Object value = vocab.metadata(GgufKeys.NF_RERANK_SUFFIX_IDS);
        if (value == null) {
            return new int[0];
        }
        if (!(value instanceof java.util.List)) {
            throw new IllegalStateException("GGUF metadata '" + GgufKeys.NF_RERANK_SUFFIX_IDS + "' is a "
                    + value.getClass().getSimpleName() + ", not an array of token ids");
        }
        java.util.List<?> ids = (java.util.List<?>) value;
        int[] out = new int[ids.size()];
        for (int i = 0; i < out.length; i++) {
            Object id = ids.get(i);
            if (!(id instanceof Number) || ((Number) id).longValue() < 0 || ((Number) id).longValue() >= vocab.size()) {
                throw new IllegalStateException("GGUF metadata '" + GgufKeys.NF_RERANK_SUFFIX_IDS + "[" + i + "]' = "
                        + id + " is not a token id of the " + vocab.size() + "-token vocabulary");
            }
            out[i] = ((Number) id).intValue();
        }
        return out;
    }

    /**
     * The checkpoint's response template ({@value GgufKeys#TOKENIZER_RESPONSE_TEMPLATE}), the JSON text vLLM's
     * {@code hf} and {@code beam} parsers read the assistant message with; null when the model has none.
     *
     * @throws IllegalStateException when the metadata value is not a string
     */
    public String responseTemplate() {
        return stringMetadata(GgufKeys.TOKENIZER_RESPONSE_TEMPLATE);
    }

    private String stringMetadata(String key) {
        Object value = vocab.metadata(key);
        if (value == null || value instanceof String) {
            return (String) value;
        }
        throw new IllegalStateException("GGUF metadata '" + key + "' is a " + value.getClass().getSimpleName()
                + ", not a string");
    }
    /** End-of-generation token ids. */
    public Set<Integer> stopTokenIds() { return special.eogIds(); }
    public boolean isStopToken(int tokenId) { return special.isEog(tokenId); }

    // ---------------------------------------------------------------------------------------------
    // Encoding and decoding
    // ---------------------------------------------------------------------------------------------

    /** Tokenizes a prompt: special tokens are parsed and the vocabulary's BOS/EOS policy applies. */
    public int[] encode(String text) {
        return encode(text, true);
    }

    /** Tokenizes text with special tokens parsed; {@code addSpecial} toggles BOS/EOS insertion. */
    public int[] encode(String text, boolean addSpecial) {
        return encode(text, addSpecial, true);
    }

    /**
     * Tokenizes text ({@code llama_tokenize}): {@code addSpecial} toggles BOS/EOS insertion, and with
     * {@code parseSpecial} false the texts of CONTROL and UNKNOWN tokens are tokenized as plain text.
     */
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        if (text == null) throw new IllegalArgumentException("text is null");
        return algorithm.encode(text, addSpecial, parseSpecial);
    }

    /**
     * How many leading ids {@code encode(a, false, true)} — {@code encodedA}, the caller's encoding of {@code a} —
     * shares with {@code encode(b, false, true)}. Only what the texts do not share is tokenized: both are
     * partitioned around their special tokens, and since every fragment encodes on its own
     * ({@link TokenizerAlgorithm#encodeFragments}), the fragments the two partitions open with alike give the same
     * ids in both; the rest of each is encoded and compared. An algorithm that does not encode fragment by fragment
     * encodes {@code b} whole.
     *
     * @throws IllegalArgumentException when {@code encodedA} does not end in the ids of {@code a}'s unshared
     *         fragments, so it is not {@code a}'s encoding
     */
    public int commonPrefixLength(String a, int[] encodedA, String b) {
        if (a == null || encodedA == null || b == null) throw new IllegalArgumentException("texts and ids are required");
        List<SpecialTokens.Fragment> fa = algorithm.fragments(a, true);
        if (fa == null) return commonPrefix(encodedA, 0, encode(b, false, true), 0);
        List<SpecialTokens.Fragment> fb = algorithm.fragments(b, true);
        int shared = 0;
        while (shared < fa.size() && shared < fb.size() && sameFragment(fa.get(shared), fb.get(shared))) shared++;
        int[] tailA = algorithm.encodeFragments(fa, shared);
        int head = encodedA.length - tailA.length;
        if (head < 0 || commonPrefix(encodedA, head, tailA, 0) != tailA.length) {
            throw new IllegalArgumentException("the ids given are not the encoding of the first text");
        }
        return head + commonPrefix(tailA, 0, algorithm.encodeFragments(fb, shared), 0);
    }

    private static boolean sameFragment(SpecialTokens.Fragment x, SpecialTokens.Fragment y) {
        return x.isToken() ? y.isToken() && x.token() == y.token() : !y.isToken() && x.text().equals(y.text());
    }

    /** The length of the common prefix of {@code x} from {@code xFrom} and {@code y} from {@code yFrom}. */
    private static int commonPrefix(int[] x, int xFrom, int[] y, int yFrom) {
        int n = Math.min(x.length - xFrom, y.length - yFrom);
        int at = 0;
        while (at < n && x[xFrom + at] == y[yFrom + at]) at++;
        return at;
    }

    public String decode(List<Integer> ids) {
        int[] arr = new int[ids.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = ids.get(i);
        return decode(arr);
    }

    /**
     * Detokenizes a sequence; hidden tokens (BOS, PAD, end-of-generation, control) produce no text.
     * With a decoder overlay the texts of the visible tokens go through that chain; with the
     * clean-up overlay set, transformers' clean-up follows. Otherwise this is llama.cpp
     * {@code detokenize}.
     */
    public String decode(int[] ids) {
        if (ids == null) throw new IllegalArgumentException("ids are null");
        TokenDecoder decoder = vocab.decoder();
        String text = decoder != null ? decodeWith(decoder, ids) : algorithm.decode(ids);
        return Boolean.TRUE.equals(vocab.cleanUpTokenizationSpaces())
                ? TokenizerAlgorithm.cleanUpTokenizationSpaces(text) : text;
    }

    /** Hugging Face {@code Tokenizer::decode}: the visible tokens' texts through the decoder chain. */
    private String decodeWith(TokenDecoder decoder, int[] ids) {
        List<String> pieces = new ArrayList<>(ids.length);
        for (int id : ids) {
            if (!special.isHidden(id)) pieces.add(vocab.text(id));
        }
        return decoder.decode(pieces);
    }

    /**
     * Text of a single token; empty for hidden tokens. Without decoder or clean-up overlays this
     * is the token's piece as streamed output renders it; with one, it is
     * {@link #decode(int[])} of the token alone (Hugging Face {@code decode([id])}), and streamed
     * text comes from {@link #createStreamingDecoder()}.
     */
    public String decode(int tokenId) {
        if (tokenId < 0 || tokenId >= decodedStringCache.length) return "";
        String cached = decodedStringCache[tokenId];
        if (cached != null) return cached;
        cached = sequenceDecoding() ? decode(new int[]{tokenId})
                : new String(visibleBytes(tokenId), StandardCharsets.UTF_8);
        decodedStringCache[tokenId] = cached;
        return cached;
    }

    /**
     * The raw bytes of one token as llama.cpp's {@code token_to_piece(id, lstrip = 0, special = true)} renders
     * them: control, unknown and user-defined tokens give their text, whatever role the token plays (BOS, EOS);
     * an out-of-range id gives no bytes. The token dumps of the tools print this piece.
     */
    public byte[] pieceWithSpecial(int tokenId) {
        return algorithm.piece(tokenId).clone();
    }

    /**
     * The raw bytes of one token as llama.cpp's {@code token_to_piece(id, lstrip = 0, special = false)} renders
     * them: control and unknown tokens and out-of-range ids give no bytes, every other token its piece. Unlike
     * {@link #decode(int)}, no decoder chain or clean-up overlay applies and no String round trip happens, so a
     * token that holds part of a UTF-8 sequence keeps its partial bytes. The sampler vocabulary
     * ({@code GgufVocab#piece}) serves it to the grammar, infill and reasoning-budget stages.
     */
    public byte[] piece(int tokenId) {
        if (tokenId < 0 || tokenId >= vocab.size()
                || (vocab.attr(tokenId) & (TokenizerVocab.ATTR_CONTROL | TokenizerVocab.ATTR_UNKNOWN)) != 0) {
            return new byte[0];
        }
        return algorithm.piece(tokenId).clone();
    }

    /** Whether decoding goes through a context-dependent overlay (decoder chain or clean-up). */
    private boolean sequenceDecoding() {
        return vocab.decoder() != null || Boolean.TRUE.equals(vocab.cleanUpTokenizationSpaces());
    }

    private byte[] visibleBytes(int tokenId) {
        if (tokenId < 0 || tokenId >= visibleBytesCache.length || special.isHidden(tokenId)) return EMPTY;
        byte[] cached = visibleBytesCache[tokenId];
        if (cached == null) {
            cached = algorithm.piece(tokenId);
            visibleBytesCache[tokenId] = cached;
        }
        return cached;
    }

    /**
     * Streaming decoder bound to this tokenizer's output rendering: token pieces at UTF-8
     * boundaries, or, with a decoder or clean-up overlay, the incremental text of
     * {@link #decode(int[])}.
     */
    public static final class StreamingDecoder extends com.eonmux.neuralfree.gguf.tokenizer.StreamingDecoder {
        /** The overlay path; null when token pieces are streamed. */
        private final IncrementalDetokenizer sequence;

        private StreamingDecoder(IntFunction<byte[]> visibleBytes, IncrementalDetokenizer sequence) {
            super(visibleBytes);
            this.sequence = sequence;
        }

        @Override
        public String decodeToken(int tokenId) {
            return sequence != null ? sequence.next(tokenId) : super.decodeToken(tokenId);
        }

        @Override
        public String flush() {
            return sequence != null ? sequence.flush() : super.flush();
        }

        @Override
        public void reset() {
            if (sequence != null) sequence.reset();
            else super.reset();
        }
    }

    /** Creates a new streaming decoder for one generation. */
    public StreamingDecoder createStreamingDecoder() {
        return new StreamingDecoder(this::visibleBytes,
                sequenceDecoding() ? new IncrementalDetokenizer(this::decode) : null);
    }

    // ---------------------------------------------------------------------------------------------
    // Chat template formatting
    // ---------------------------------------------------------------------------------------------

    public String formatChat(String systemPrompt, String userMessage) {
        return formatChat(systemPrompt, userMessage, false);
    }

    /**
     * Renders a single-turn prompt: a system message when {@code systemPrompt} is non-empty, then the user
     * message, formatted as {@link #formatChat(List, boolean)} formats a conversation.
     */
    public String formatChat(String systemPrompt, String userMessage, boolean enableThinking) {
        List<Map<String, Object>> messages = new ArrayList<>(2);
        if (systemPrompt != null && !systemPrompt.isEmpty()) messages.add(message("system", systemPrompt));
        messages.add(message("user", userMessage));
        return chat.format(ChatMessages.request(messages, enableThinking));
    }

    /**
     * Multi-turn chat formatting. Each entry is a map with "role" (system/user/assistant) and
     * "content", the structure used by llama.cpp and the OpenAI chat completions API.
     */
    public String formatChat(List<Map<String, String>> messages) {
        return formatChat(messages, false);
    }

    /**
     * Renders a conversation followed by the assistant generation prompt with the model's chat template
     * ({@link #chatTemplate()}), applying llama.cpp's message workarounds; see {@link ChatFormatter}.
     */
    public String formatChat(List<Map<String, String>> messages, boolean enableThinking) {
        return chat.format(ChatMessages.request(messages, enableThinking));
    }

    /**
     * Renders a chat request ({@code messages}, {@code tools}, {@code tool_choice}, {@code
     * chat_template_kwargs}, {@code add_generation_prompt}, {@code continue_final_message},
     * {@code reasoning_effort}, {@code enable_thinking}) with the model's chat template through
     * the full llama.cpp template-input conventions ({@link ChatFormatter#format(ChatRequest)});
     * the overloads above delegate to this path.
     *
     * @throws IllegalArgumentException for malformed messages, tool calls or template kwargs
     * @throws UnsupportedOperationException for a request with image or audio content parts
     */
    public String formatChat(ChatRequest request) {
        return chat.format(request);
    }

    /**
     * {@link #formatChat(ChatRequest)} with the autoparser's analysis: a continued final message is written
     * around the derived reasoning markers ({@link ChatFormatter#format(ChatRequest, boolean,
     * AutoParserAnalysis)}).
     */
    public String formatChat(ChatRequest request, AutoParserAnalysis continuation) {
        return chat.format(request, false, continuation);
    }

    /**
     * {@link #formatChat(ChatRequest)} for the multimodal prompt path: with {@code mediaMarkers} set, image and
     * audio content parts render as {@link ChatMessages#MEDIA_MARKER} instead of being refused
     * ({@link ChatFormatter#format(ChatRequest, boolean)}).
     */
    public String formatChat(ChatRequest request, boolean mediaMarkers) {
        return chat.format(request, mediaMarkers);
    }

    /** {@link #formatChat(ChatRequest, boolean)} over the autoparser's analysis. */
    public String formatChat(ChatRequest request, boolean mediaMarkers, AutoParserAnalysis continuation) {
        return chat.format(request, mediaMarkers, continuation);
    }

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }
}
