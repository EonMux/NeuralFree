package com.eonmux.neuralfree.gguf.tokenizer;

import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A SentencePiece precompiled normalization charsmap: a little-endian uint32 byte size, the
 * XOR-compressed double array (XCDA) of that many bytes mapping UTF-8 keys to offsets, then the
 * NUL-terminated UTF-8 replacement strings.
 *
 * <p>Two readers share the array: {@link #replace} is llama.cpp's {@code xcda_array_view} lookup
 * (longest key at a byte offset, src/llama-vocab.cpp:1188-1304) used by {@link UgmTokenizer},
 * and {@link #normalize} is the Hugging Face {@code Precompiled} normalizer (the
 * {@code spm_precompiled} crate), which looks up whole grapheme clusters and takes the shortest
 * key. Immutable.</p>
 */
public final class PrecompiledCharsmap {

    /** Extended grapheme clusters, the units the Hugging Face normalizer looks up. */
    private static final Pattern GRAPHEME = Pattern.compile("\\X");
    /** Graphemes of at least this many UTF-8 bytes are looked up per character only. */
    private static final int MAX_GRAPHEME_BYTES = 6;

    private final byte[] blob;
    private final int size;
    private final int replacementsOffset;
    private final int replacementsSize;

    /**
     * Reads a charsmap blob (copied).
     *
     * @throws IllegalArgumentException when the blob is shorter than its size header or than the
     *         array size it declares
     */
    public PrecompiledCharsmap(byte[] blob) {
        if (blob == null) throw new IllegalArgumentException("precompiled charsmap is null");
        if (blob.length < 4) {
            throw new IllegalArgumentException("precompiled charsmap of " + blob.length
                    + " bytes is too small for its XCDA size header");
        }
        long bytes = u32(blob, 0);
        if (bytes > blob.length - 4L) {
            throw new IllegalArgumentException("precompiled charsmap declares an XCDA of " + bytes + " bytes but holds "
                    + (blob.length - 4) + " bytes after its header");
        }
        this.blob = blob.clone();
        this.size = (int) (bytes / 4);
        this.replacementsOffset = 4 + (int) bytes;
        this.replacementsSize = blob.length - replacementsOffset;
    }

    /** A copy of the charsmap blob. */
    public byte[] bytes() {
        return blob.clone();
    }

    /** The blob length in bytes. */
    public int length() {
        return blob.length;
    }

    private static long u32(byte[] b, int o) {
        return (b[o] & 0xFFL) | (b[o + 1] & 0xFFL) << 8 | (b[o + 2] & 0xFFL) << 16 | (b[o + 3] & 0xFFL) << 24;
    }

    private long node(int index) {
        if (index < 0 || index >= size) throw new IllegalStateException("Index out of array bounds in XCDA array");
        return u32(blob, 4 + index * 4);
    }

    private int base(int index) {
        long p = node(index);
        return (int) ((p >>> 10) << ((p & (1L << 9)) >>> 6));
    }

    private long lcheck(int index) { return node(index) & ((1L << 31) | 0xFF); }
    private boolean leaf(int index) { return ((node(index) >>> 8) & 1) != 0; }
    private int value(int index) { return (int) (node(index) & ((1L << 31) - 1)); }

    /**
     * llama.cpp {@code normalize_prefix} lookup: the longest key at {@code off}. On a match fills
     * {@code res} with {offset of the replacement in the returned array, its length, consumed
     * input bytes} and returns the array; otherwise returns null.
     *
     * @throws IllegalStateException when the array or a replacement is out of bounds or unterminated
     */
    byte[] replace(byte[] input, int off, int[] res) {
        if (size == 0) return null;
        int longestLength = 0;
        int longestOffset = 0;
        int nodeIndex = base(0);
        for (int p = off; p < input.length; p++) {
            int c = input[p] & 0xFF;
            if (c == 0) break;
            nodeIndex ^= c;
            if (lcheck(nodeIndex) != c) break;
            boolean isLeaf = leaf(nodeIndex);
            nodeIndex ^= base(nodeIndex);
            if (isLeaf) {
                longestLength = p - off + 1;
                longestOffset = value(nodeIndex);
            }
        }
        if (longestLength == 0) return null;
        if (longestOffset >= replacementsSize) {
            throw new IllegalStateException("Index out of array bounds in precompiled charsmap");
        }
        int start = replacementsOffset + longestOffset;
        int maxLen = replacementsSize - longestOffset;
        int len = 0;
        while (len < maxLen && blob[start + len] != 0) len++;
        if (len == maxLen) throw new IllegalStateException("Unterminated string in precompiled charsmap");
        res[0] = start;
        res[1] = len;
        res[2] = longestLength;
        return blob;
    }

    /**
     * Checks that the replacement region is UTF-8, which the Hugging Face {@code Precompiled}
     * normalizer requires of the whole region when it loads a charsmap.
     *
     * @throws IllegalArgumentException when it is not
     */
    public void requireUtf8Replacements() {
        try {
            StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(blob, replacementsOffset, replacementsSize));
        } catch (CharacterCodingException e) {
            throw new IllegalArgumentException("precompiled charsmap replacement strings are not valid UTF-8", e);
        }
    }

    /**
     * {@code spm_precompiled} {@code Precompiled::transform}: the replacement of the shortest
     * prefix of {@code key} that is a charsmap key, or null when none is. A replacement runs to
     * its NUL or to the end of the blob.
     *
     * @throws IllegalStateException when the array is out of bounds or a replacement is not UTF-8
     */
    public String transform(byte[] key) {
        if (key == null) throw new IllegalArgumentException("key is null");
        if (size == 0) return null;
        int nodeIndex = base(0);
        for (byte b : key) {
            int c = b & 0xFF;
            if (c == 0) break;
            nodeIndex ^= c;
            if (lcheck(nodeIndex) != c) return null;
            boolean isLeaf = leaf(nodeIndex);
            nodeIndex ^= base(nodeIndex);
            if (isLeaf) return replacementAt(value(nodeIndex));
        }
        return null;
    }

    private String replacementAt(int offset) {
        if (offset > replacementsSize) {
            throw new IllegalStateException("Index out of array bounds in precompiled charsmap: replacement offset "
                    + offset + " beyond " + replacementsSize + " bytes");
        }
        int start = replacementsOffset + offset;
        int end = start;
        while (end < blob.length && blob[end] != 0) end++;
        try {
            return StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(blob, start, end - start))
                    .toString();
        } catch (CharacterCodingException e) {
            throw new IllegalStateException("precompiled charsmap replacement at offset " + offset + " is not UTF-8", e);
        }
    }

    /**
     * The Hugging Face {@code Precompiled} normalizer: each grapheme cluster shorter than six
     * UTF-8 bytes that has a key as a prefix becomes that key's replacement as a whole; otherwise
     * each of its characters is replaced when it is a key and kept when it is not.
     */
    public String normalize(String text) {
        if (text == null) throw new IllegalArgumentException("text is null");
        Matcher g = GRAPHEME.matcher(text);
        StringBuilder out = new StringBuilder(text.length());
        boolean modified = false;
        while (g.find()) {
            String grapheme = g.group();
            byte[] utf8 = grapheme.getBytes(StandardCharsets.UTF_8);
            if (utf8.length < MAX_GRAPHEME_BYTES) {
                String whole = transform(utf8);
                if (whole != null) {
                    out.append(whole);
                    modified = true;
                    continue;
                }
            }
            for (int i = 0; i < grapheme.length(); ) {
                int cp = grapheme.codePointAt(i);
                int n = Character.charCount(cp);
                String one = grapheme.substring(i, i + n);
                String repl = transform(one.getBytes(StandardCharsets.UTF_8));
                if (repl != null) {
                    out.append(repl);
                    modified = true;
                } else {
                    out.append(one);
                }
                i += n;
            }
        }
        return modified ? out.toString() : text;
    }

    @Override
    public boolean equals(Object o) {
        return this == o || (o instanceof PrecompiledCharsmap && Arrays.equals(blob, ((PrecompiledCharsmap) o).blob));
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(blob);
    }

    @Override
    public String toString() {
        return "PrecompiledCharsmap{" + blob.length + " bytes, " + size + " XCDA units}";
    }
}
