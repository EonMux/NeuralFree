package com.eonmux.neuralfree.gguf.tokenizer;

import java.nio.charset.StandardCharsets;

/**
 * One llama.cpp tokenizer algorithm bound to a vocabulary.
 *
 * <p>Implementations are immutable after construction and safe for concurrent use; each
 * {@link #encode} call keeps its working state on the stack.</p>
 */
public interface TokenizerAlgorithm {

    /**
     * {@code llama_vocab::impl::tokenize} with {@code parse_special = true}: partitions the text
     * around special tokens and encodes each raw fragment. With {@code addSpecial} the
     * vocabulary's BOS/EOS/SEP policy is applied.
     */
    default int[] encode(String text, boolean addSpecial) {
        return encode(text, addSpecial, true);
    }

    /**
     * {@code llama_vocab::impl::tokenize}: partitions the text around special tokens
     * ({@link SpecialTokens#partition}; with {@code parseSpecial} false, CONTROL and UNKNOWN token
     * texts stay plain text) and encodes each raw fragment. With {@code addSpecial} the
     * vocabulary's BOS/EOS/SEP policy is applied.
     */
    int[] encode(String text, boolean addSpecial, boolean parseSpecial);

    /**
     * {@code llama_vocab::impl::detokenize} with {@code remove_special = true} and
     * {@code unparse_special = false}; tokens hidden by {@link SpecialTokens#isHidden} produce no text.
     */
    String decode(int[] ids);

    /**
     * {@code token_to_piece} with {@code special = true} and no lstrip: the raw bytes a token
     * renders to. Out-of-range ids render to an empty array.
     */
    byte[] piece(int id);

    /** Shared {@code detokenize} for algorithms whose output is the concatenation of pieces. */
    static String detokenize(TokenizerAlgorithm algorithm, TokenizerVocab vocab, SpecialTokens special, int[] ids) {
        int start = 0;
        int end = ids.length;
        boolean removeSpace = vocab.addSpacePrefix();
        if (vocab.addBos() && end > 0 && ids[0] == special.bos()) {
            removeSpace = false;
            start++;
        }
        if (vocab.addEos() && end > start && ids[end - 1] == special.eos()) {
            end--;
        }
        ByteSink out = new ByteSink(Math.max(16, (end - start) * 4));
        for (int i = start; i < end; i++) {
            int id = ids[i];
            if (!special.isHidden(id)) {
                byte[] p = algorithm.piece(id);
                int from = removeSpace && p.length > 0 && p[0] == ' ' ? 1 : 0;
                out.write(p, from, p.length - from);
            }
            removeSpace = false;
        }
        byte[] raw = out.toArray();
        byte[] text = vocab.cleanSpaces() ? cleanSpaces(raw, raw.length) : raw;
        return new String(text, StandardCharsets.UTF_8);
    }

    /** The three {@code clean_spaces} passes of {@code detokenize} (llama-vocab.cpp:4151-4207). */
    static byte[] cleanSpaces(byte[] src, int length) {
        byte[] t = java.util.Arrays.copyOf(src, length);
        int total = length;

        int total1 = total;
        total = total > 0 ? 1 : 0;
        for (int i = 1; i < total1; ++i) {
            byte x = t[i];
            if (t[i - 1] == ' ' && (x == '?' || x == '!' || x == '.' || x == ',')) total--;
            t[total++] = x;
        }

        int total2 = total;
        total = total > 0 ? 1 : 0;
        for (int i = 1; i < total2; ++i) {
            byte x = t[i];
            if (x == '\'' && i + 1 < total2 && t[i - 1] == ' ' && t[i + 1] == ' ') {
                total--;
                t[++i] = 0;
            }
            t[total++] = x;
        }

        int total3 = total;
        total = total > 0 ? 1 : 0;
        for (int i = 1; i < total3; ++i) {
            byte x = t[i];
            if (t[i - 1] == ' ' && x == '\'' && i + 1 < total3) {
                byte x1 = t[i + 1];
                if (x1 == 's' || x1 == 'm') {
                    total--;
                } else if (x1 != 't' && x1 != 'd' && i + 2 < total3) {
                    byte x2 = t[i + 2];
                    if ((x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e')) total--;
                }
            }
            t[total++] = x;
        }
        return java.util.Arrays.copyOf(t, total);
    }

    /**
     * The SPM/UGM/WPM branch of {@code token_to_piece} with {@code special = true}
     * (llama-vocab.cpp:4009-4026): UNKNOWN, CONTROL and USER_DEFINED tokens render their raw
     * text, NORMAL tokens unescape U+2581 to a space, BYTE tokens render their byte, and any
     * other attribute renders nothing.
     */
    static byte[] sentencePiece(TokenizerVocab vocab, int id) {
        if (id < 0 || id >= vocab.size()) return new byte[0];
        String text = vocab.text(id);
        if (vocab.hasAttr(id, TokenizerVocab.ATTR_UNKNOWN | TokenizerVocab.ATTR_CONTROL | TokenizerVocab.ATTR_USER_DEFINED)) {
            return utf8(text);
        }
        if (vocab.isNormal(id)) return utf8(TokenizerVocab.unescapeWhitespace(text));
        if (vocab.isByte(id)) {
            int b = vocab.byteValue(id);
            return b >= 0 ? new byte[]{(byte) b} : utf8(text);
        }
        return new byte[0];
    }

    /** UTF-8 bytes of a string. */
    static byte[] utf8(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }

    /** Growable token-id output buffer used while encoding. */
    final class IdBuffer {
        private int[] data;
        private int length;

        public IdBuffer(int capacity) {
            this.data = new int[Math.max(capacity, 8)];
        }

        public void add(int id) {
            if (length == data.length) data = java.util.Arrays.copyOf(data, data.length * 2);
            data[length++] = id;
        }

        public int length() { return length; }
        public int get(int index) { return data[index]; }

        public int[] toArray() { return java.util.Arrays.copyOf(data, length); }
    }

    /** Growable byte buffer used while detokenizing. */
    final class ByteSink {
        private byte[] bytes;
        private int length;

        public ByteSink(int capacity) {
            this.bytes = new byte[Math.max(capacity, 16)];
        }

        public void write(byte[] src, int offset, int count) {
            if (count <= 0) return;
            if (length + count > bytes.length) {
                bytes = java.util.Arrays.copyOf(bytes, Math.max(bytes.length * 2, length + count));
            }
            System.arraycopy(src, offset, bytes, length, count);
            length += count;
        }

        public void write(int b) {
            if (length == bytes.length) bytes = java.util.Arrays.copyOf(bytes, bytes.length * 2);
            bytes[length++] = (byte) b;
        }

        public int length() { return length; }
        public byte[] toArray() { return java.util.Arrays.copyOf(bytes, length); }
    }
}
