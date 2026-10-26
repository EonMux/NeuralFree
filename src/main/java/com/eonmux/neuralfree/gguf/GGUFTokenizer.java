package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.chat.ChatFormatter;
import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
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

    private static final Logger LOGGER = LogManager.getLogger(GGUFTokenizer.class);
    private static final byte[] EMPTY = new byte[0];

    private final TokenizerVocab vocab;
    private final SpecialTokens special;
    private final TokenizerAlgorithm algorithm;
    private final ChatFormatter chat;
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
        this.special = SpecialTokens.of(vocab);
        this.algorithm = algorithmFor(vocab, special);
        String template = ChatTemplateSource.resolve(templateMetadata(vocab, ChatTemplateSource.KEY_DEFAULT),
                templateMetadata(vocab, ChatTemplateSource.KEY_TOOL_USE));
        this.chat = new ChatFormatter(template, special.bos() >= 0 ? vocab.text(special.bos()) : "",
                special.eos() >= 0 ? vocab.text(special.eos()) : "", vocab.addBos(), vocab.addEos(),
                vocab.architecture());
        this.visibleBytesCache = new byte[vocab.size()][];
        this.decodedStringCache = new String[vocab.size()];
        LOGGER.info("tokenizer: model={} pre={} vocab={} bos={} eos={} eog={}", vocab.model(), vocab.pre(),
                vocab.size(), special.bos(), special.eos(), special.eogIds().size());
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
    public String architecture() { return vocab.architecture(); }

    /**
     * The rerank prompt ({@value #KEY_RERANK_TEMPLATE}) with {@code {query}} and {@code {document}} placeholders, as
     * the converter writes it for Qwen3 rerankers; null when the model has none.
     *
     * @throws IllegalStateException when the metadata value is not a string
     */
    public String rerankTemplate() {
        Object value = vocab.metadata(KEY_RERANK_TEMPLATE);
        if (value == null || value instanceof String) {
            return (String) value;
        }
        throw new IllegalStateException("GGUF metadata '" + KEY_RERANK_TEMPLATE + "' is a "
                + value.getClass().getSimpleName() + ", not a string");
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

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }
}
