package com.eonmux.neuralfree.gguf.tokenizer;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.Function;

/**
 * Streams the text of a token sequence whose decoding depends on neighbouring tokens (a Hugging
 * Face decoder chain, transformers' clean-up), by decoding a short window of recent tokens with
 * the full sequence decoder: vLLM's {@code detokenize_incrementally}
 * (vllm/tokenizers/detokenizer_utils.py:176-276). Each step decodes the window before and after
 * the new tokens and emits the difference; text that grows nothing, or ends in U+FFFD (an
 * incomplete character), is held until a later token completes it.
 *
 * <p>The window starts at the first token, as Hugging Face's {@code TextStreamer} decodes from the
 * start of a generation; the streamed text equals the sequence decoder's text of the same tokens
 * whenever a later token only appends to the text of the earlier ones. Not thread-safe.</p>
 */
public final class IncrementalDetokenizer {

    private static final char REPLACEMENT_CHAR = '�';

    private final Function<int[], String> decode;
    private int[] ids = new int[16];
    private int count;
    /** Start of the window: the tokens before the unread ones, decoded as context. */
    private int prefixOffset;
    /** Start of the tokens whose text has not been emitted. */
    private int readOffset;

    /**
     * @param decode the sequence decoder: the text of a whole token sequence
     */
    public IncrementalDetokenizer(Function<int[], String> decode) {
        this.decode = Objects.requireNonNull(decode, "decode");
    }

    /** Adds one token and returns the text it completes; empty while the text is held back. */
    public String next(int tokenId) {
        if (count == ids.length) ids = Arrays.copyOf(ids, count * 2);
        ids[count++] = tokenId;
        return emit(false);
    }

    /** Returns the held-back text, an incomplete character rendered as U+FFFD, and releases it. */
    public String flush() {
        return emit(true);
    }

    /** Drops every token: the next one starts a new sequence. */
    public void reset() {
        count = 0;
        prefixOffset = 0;
        readOffset = 0;
    }

    private String emit(boolean force) {
        if (readOffset == count) return "";
        String prefixText = decodeRange(prefixOffset, readOffset);
        String newText = decodeRange(prefixOffset, count);
        boolean grew = newText.length() > prefixText.length();
        if (!force && (!grew || newText.charAt(newText.length() - 1) == REPLACEMENT_CHAR)) return "";
        String delta = grew ? newText.substring(prefixText.length()) : "";
        advance();
        return delta;
    }

    /** Moves the window past the emitted tokens, keeping the last emitted ones as context. */
    private void advance() {
        int keep = readOffset;
        System.arraycopy(ids, keep, ids, 0, count - keep);
        count -= keep;
        prefixOffset = 0;
        readOffset = count;
    }

    private String decodeRange(int from, int to) {
        return from == to ? "" : decode.apply(Arrays.copyOfRange(ids, from, to));
    }
}
