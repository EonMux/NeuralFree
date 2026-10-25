package com.eonmux.neuralfree.gguf.tokenizer.hf;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The mergeable ranks of a tiktoken BPE file ({@code qwen.tiktoken}, Kimi's {@code tiktoken.model}, any
 * {@code *.tiktoken}), as tiktoken's {@code load_tiktoken_bpe} reads them: each non-empty line is
 * {@code <base64 token bytes> <rank>}; lines end with {@code \n}, {@code \r\n} or {@code \r}.
 *
 * <p>Tokens and ranks must both be unique (tiktoken's encoder and decoder must have equal sizes); base64 is decoded
 * strictly and ranks are non-negative decimal integers. Entries keep file order. Instances are immutable.</p>
 */
public final class TiktokenReader {

    private final byte[][] tokens;
    private final int[] ranks;
    private final Map<String, Integer> indexByToken;
    private final Map<Integer, Integer> indexByRank;
    private final int maxRank;

    private TiktokenReader(byte[][] tokens, int[] ranks, Map<String, Integer> indexByToken,
                           Map<Integer, Integer> indexByRank, int maxRank) {
        this.tokens = tokens;
        this.ranks = ranks;
        this.indexByToken = indexByToken;
        this.indexByRank = indexByRank;
        this.maxRank = maxRank;
    }

    /**
     * Reads {@code file} (at most {@link HfTokenizerFiles#MAX_FILE_BYTES}).
     *
     * @throws IOException {@code "Invalid Hugging Face checkpoint '<dir>': <file>: line <n>: ..."} when a line is not
     *                     a valid entry
     */
    public static TiktokenReader read(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        byte[] data = HfTokenizerFiles.readBytes(file);
        return parse(data, HfTokenizerFiles.filePrefix(file));
    }

    /**
     * Parses the bytes of a tiktoken file.
     *
     * @throws IOException {@code "Invalid tiktoken file: line <n>: ..."} when a line is not a valid entry
     */
    public static TiktokenReader parse(byte[] data) throws IOException {
        Objects.requireNonNull(data, "data");
        return parse(data, "Invalid tiktoken file: ");
    }

    private static TiktokenReader parse(byte[] data, String prefix) throws IOException {
        List<byte[]> tokens = new ArrayList<>();
        List<Integer> ranks = new ArrayList<>();
        Map<String, Integer> byToken = new HashMap<>();
        Map<Integer, Integer> byRank = new HashMap<>();
        int maxRank = -1;
        int line = 0;
        int pos = 0;
        while (pos < data.length) {
            int end = pos;
            while (end < data.length && data[end] != '\n' && data[end] != '\r') {
                end++;
            }
            line++;
            if (end > pos) {
                if (tokens.size() == HfTokenizerFiles.MAX_TOKENS) {
                    throw new IOException(prefix + "line " + line + ": more than " + HfTokenizerFiles.MAX_TOKENS
                            + " tokens");
                }
                List<String> fields = fields(new String(data, pos, end - pos, StandardCharsets.ISO_8859_1));
                if (fields.size() != 2) {
                    throw new IOException(prefix + "line " + line + ": expected '<base64 token> <rank>', got "
                            + fields.size() + " field(s)");
                }
                byte[] token = decodeToken(fields.get(0), prefix, line);
                int rank = parseRank(fields.get(1), prefix, line);
                String key = latin1(token);
                int index = tokens.size();
                Integer clash = byToken.putIfAbsent(key, index);
                if (clash != null) {
                    throw new IOException(prefix + "line " + line + ": token " + fields.get(0) + " repeats the token of"
                            + " rank " + ranks.get(clash));
                }
                if (byRank.putIfAbsent(rank, index) != null) {
                    throw new IOException(prefix + "line " + line + ": rank " + rank + " is assigned twice");
                }
                tokens.add(token);
                ranks.add(rank);
                maxRank = Math.max(maxRank, rank);
            }
            pos = end;
            if (pos < data.length) {
                pos += data[pos] == '\r' && pos + 1 < data.length && data[pos + 1] == '\n' ? 2 : 1;
            }
        }
        int[] rankArray = new int[ranks.size()];
        for (int i = 0; i < rankArray.length; i++) {
            rankArray[i] = ranks.get(i);
        }
        return new TiktokenReader(tokens.toArray(new byte[0][]), rankArray, byToken, byRank, maxRank);
    }

    /** Number of entries. */
    public int size() { return tokens.length; }

    /** The token bytes of entry {@code i} in file order (a copy). */
    public byte[] token(int i) { return tokens[i].clone(); }

    /** The rank of entry {@code i}. */
    public int rank(int i) { return ranks[i]; }

    /** The largest rank, or -1 when there are no entries. */
    public int maxRank() { return maxRank; }

    /** The rank of {@code token}, or -1 when it is not a mergeable token. */
    public int rankOf(byte[] token) {
        Integer index = indexByToken.get(latin1(token));
        return index == null ? -1 : ranks[index];
    }

    /** The token with {@code rank} (a copy), or null when no token has it. */
    public byte[] tokenOfRank(int rank) {
        Integer index = indexByRank.get(rank);
        return index == null ? null : tokens[index].clone();
    }

    @Override
    public String toString() {
        return "TiktokenReader{" + tokens.length + " ranks}";
    }

    /** The fields of a line separated by runs of ASCII whitespace, as Python {@code bytes.split()} yields them. */
    private static List<String> fields(String line) {
        List<String> out = new ArrayList<>(2);
        int i = 0;
        int n = line.length();
        while (i < n) {
            while (i < n && isAsciiSpace(line.charAt(i))) {
                i++;
            }
            int start = i;
            while (i < n && !isAsciiSpace(line.charAt(i))) {
                i++;
            }
            if (i > start) {
                out.add(line.substring(start, i));
            }
        }
        return out;
    }

    private static boolean isAsciiSpace(char c) {
        return c == ' ' || c == '\t' || c == '\u000B' || c == '\f';
    }

    /** One char per byte, the key form the converter uses ({@code bytes.decode("latin-1")}). */
    private static String latin1(byte[] b) {
        return new String(b, StandardCharsets.ISO_8859_1);
    }

    private static byte[] decodeToken(String base64, String prefix, int line) throws IOException {
        try {
            return Base64.getDecoder().decode(base64);
        } catch (IllegalArgumentException e) {
            throw new IOException(prefix + "line " + line + ": token '" + base64 + "' is not valid base64 ("
                    + e.getMessage() + ")");
        }
    }

    private static int parseRank(String text, String prefix, int line) throws IOException {
        long rank = 0;
        boolean valid = !text.isEmpty() && text.length() <= 10;
        for (int i = 0; valid && i < text.length(); i++) {
            char c = text.charAt(i);
            valid = c >= '0' && c <= '9';
            rank = rank * 10 + (c - '0');
        }
        if (!valid || rank > Integer.MAX_VALUE) {
            throw new IOException(prefix + "line " + line + ": rank '" + text + "' is not an integer in [0, "
                    + Integer.MAX_VALUE + "]");
        }
        return (int) rank;
    }
}
