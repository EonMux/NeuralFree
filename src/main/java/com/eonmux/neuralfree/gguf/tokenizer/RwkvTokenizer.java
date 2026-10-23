package com.eonmux.neuralfree.gguf.tokenizer;

import java.util.HashMap;
import java.util.Map;

/**
 * RWKV world tokenizer ({@code tokenizer.ggml.model = "rwkv"}), transcribed from llama.cpp
 * {@code llm_tokenizer_rwkv} / {@code llm_tokenizer_rwkv_session} (src/llama-vocab.cpp:1313-1420) and
 * the RWKV branch of {@code llama_vocab::impl::tokenize} (src/llama-vocab.cpp:3876-3892).
 *
 * <p>Token texts are escaped byte strings ({@code \t}, {@code \n}, {@code \r}, {@code \xNN} and
 * {@code \<c>}); {@link #unescape} recovers their bytes. Encoding walks the UTF-8 bytes of each raw
 * fragment through a trie of those bytes and emits the longest token that matches at the current
 * position. A byte that starts no token becomes the unknown token. The RWKV branch never adds BOS
 * or EOS. Pieces render as the unescaped token bytes.</p>
 */
public final class RwkvTokenizer implements TokenizerAlgorithm {

    private final TokenizerVocab vocab;
    private final SpecialTokens special;
    private final ByteTrie matcher;
    /** Unescaped bytes of every token, indexed by id; the rendering of {@link #piece}. */
    private final byte[][] pieces;

    public RwkvTokenizer(TokenizerVocab vocab, SpecialTokens special) {
        if (vocab == null || special == null) throw new IllegalArgumentException("vocabulary and special tokens are required");
        if (vocab.type() != TokenizerVocab.Type.RWKV) {
            throw new IllegalArgumentException("RwkvTokenizer requires a RWKV vocabulary, got " + vocab.type()
                    + " (model '" + vocab.model() + "')");
        }
        this.vocab = vocab;
        this.special = special;
        this.pieces = new byte[vocab.size()][];
        ByteTrie trie = new ByteTrie();
        for (int id = 0; id < vocab.size(); id++) {
            byte[] bytes = unescape(vocab.text(id));
            pieces[id] = bytes;
            trie.insert(bytes, id);
        }
        this.matcher = trie;
    }

    /** Encodes every raw fragment and passes special tokens through; {@code addSpecial} adds nothing. */
    @Override
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        IdBuffer out = new IdBuffer(Math.max(16, text.length() / 2));
        for (SpecialTokens.Fragment f : special.partition(text, parseSpecial)) {
            if (f.isToken()) out.add(f.token());
            else tokenizeFragment(TokenizerAlgorithm.utf8(f.text()), out);
        }
        return out.toArray();
    }

    @Override
    public String decode(int[] ids) {
        return TokenizerAlgorithm.detokenize(this, vocab, special, ids);
    }

    /** {@code token_to_piece} for RWKV: the unescaped token bytes, whatever the token attribute. */
    @Override
    public byte[] piece(int id) {
        if (id < 0 || id >= pieces.length) return new byte[0];
        return pieces[id].clone();
    }

    /**
     * {@code llm_tokenizer_rwkv_session::tokenize}: from each position, follow the trie as far as the
     * text allows and emit the deepest node that ends a token, then continue after it. A first byte
     * with no trie edge, or a path on which no token ends, emits the unknown token and advances one
     * byte; llama.cpp's vocabularies hold every single byte, so the second case only arises for a
     * vocabulary without them, where llama.cpp would restart at the fragment's first byte.
     */
    private void tokenizeFragment(byte[] text, IdBuffer out) {
        int position = 0;
        while (position < text.length) {
            int node = matcher.child(ByteTrie.ROOT, text[position]);
            int tokenId = -1;
            int tokenEnd = position + 1;
            int cursor = position;
            while (node != ByteTrie.NONE) {
                int value = matcher.value(node);
                if (value >= 0) {
                    tokenId = value;
                    tokenEnd = cursor + 1;
                }
                if (cursor + 1 >= text.length) break;
                node = matcher.child(node, text[++cursor]);
            }
            out.add(tokenId >= 0 ? tokenId : requireUnk(text[position]));
            position = tokenEnd;
        }
    }

    private int requireUnk(byte b) {
        int unk = special.unk();
        if (unk == SpecialTokens.NULL) {
            throw new IllegalStateException(String.format("RWKV vocabulary has no token starting with byte 0x%02X"
                    + " and defines no unknown token (tokenizer.ggml.unknown_token_id)", b & 0xFF));
        }
        return unk;
    }

    /**
     * {@code llama_unescape_rwkv_token}: {@code \t}, {@code \n} and {@code \r} become control bytes,
     * {@code \x} consumes the next two characters as lowercase hex, any other escaped character is
     * kept as is, and unescaped characters are copied. The token text is processed as UTF-8 bytes,
     * and hex digits decode as llama.cpp does ({@code c >= 'a' ? c - 'a' + 10 : c - '0'}, modulo 256).
     */
    static byte[] unescape(String escaped) {
        byte[] in = TokenizerAlgorithm.utf8(escaped);
        ByteSink out = new ByteSink(in.length);
        boolean escaping = false;
        int hexRemaining = 0;
        int hexAcc = 0;
        for (byte b : in) {
            int c = b & 0xFF;
            int signed = b;
            if (hexRemaining != 0) {
                int value = signed >= 'a' ? signed - 'a' + 10 : signed - '0';
                hexAcc = ((hexAcc << 4) + value) & 0xFF;
                hexRemaining--;
                if (hexRemaining == 0) {
                    out.write(hexAcc);
                    hexAcc = 0;
                }
                continue;
            }
            if (escaping) {
                if (c == 't') out.write('\t');
                else if (c == 'n') out.write('\n');
                else if (c == 'r') out.write('\r');
                else if (c == 'x') hexRemaining = 2;
                else out.write(c);
                escaping = false;
                continue;
            }
            if (c == '\\') {
                escaping = true;
                continue;
            }
            out.write(c);
        }
        return out.toArray();
    }

    /**
     * {@code naive_trie} over bytes in a flat layout: node 0 is the root, edges live in one hash map
     * keyed by {@code node * 256 + byte}, and {@code values[node]} is the token ending there or -1.
     * Inserting an existing key replaces its value, as in llama.cpp.
     */
    private static final class ByteTrie {
        static final int ROOT = 0;
        static final int NONE = -1;

        private final Map<Long, Integer> edges = new HashMap<>();
        private int[] values = new int[]{-1};
        private int nodes = 1;

        void insert(byte[] key, int id) {
            int node = ROOT;
            for (byte b : key) {
                long edge = edgeKey(node, b);
                Integer next = edges.get(edge);
                if (next == null) {
                    next = newNode();
                    edges.put(edge, next);
                }
                node = next;
            }
            values[node] = id;
        }

        int child(int node, byte b) {
            Integer next = edges.get(edgeKey(node, b));
            return next != null ? next : NONE;
        }

        int value(int node) {
            return values[node];
        }

        private int newNode() {
            if (nodes == values.length) {
                int grown = values.length * 2;
                values = java.util.Arrays.copyOf(values, grown);
                java.util.Arrays.fill(values, nodes, grown, -1);
            }
            return nodes++;
        }

        private static long edgeKey(int node, byte b) {
            return ((long) node << 8) | (b & 0xFF);
        }
    }
}
