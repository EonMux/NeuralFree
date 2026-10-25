package com.eonmux.neuralfree.gguf.tokenizer.hf;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * llama.cpp's pre-tokenizer fingerprint table: the {@code chkhsh} branches of {@code get_vocab_base_pre}
 * (LC/conversion/base.py:1803-2151) and the {@code chktxt} they hash, read from the generated resources
 * {@value #TABLE_RESOURCE} and {@value #TEXT_RESOURCE}.
 *
 * <p>The converter encodes {@link #checkText()} with the Hugging Face tokenizer (special tokens added), formats the
 * ids as a Python list ({@code "[1, 2, 3]"}) and compares the SHA-256 of that text with each row in order, every
 * matching row overwriting the result. A hash that several rows share therefore resolves to the last of them
 * ({@link #name}); {@link #names} lists all of them in row order.</p>
 *
 * <p>Immutable and thread-safe.</p>
 */
public final class ChkhshTable {

    /** Classpath location of the hash table ({@code sha256, pre_name, repo}). */
    public static final String TABLE_RESOURCE = "hf/chkhsh.tsv";
    /** Classpath location of the fingerprint text (UTF-8, exact bytes). */
    public static final String TEXT_RESOURCE = "hf/chktxt.txt";

    private static final String FILE_NAME = "chkhsh.tsv";
    private static final Pattern HEADER = Pattern.compile(
            "# chkhsh\\.tsv - Generated from llama\\.cpp \\S+ by the updateHfTables Gradle task; do not edit\\."
                    + " Rows: (\\d{1,9})");
    private static final String COLUMNS = "# sha256\tpre_name\trepo";
    private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");
    /** Largest resource read; the generated files are a few kilobytes. */
    private static final int MAX_RESOURCE_BYTES = 1 << 20;

    private static volatile ChkhshTable builtIn;

    /** One table row: a fingerprint, the pre-tokenizer name it selects and the reference repository. */
    public static final class Row {
        private final String sha256;
        private final String preName;
        private final String repo;

        Row(String sha256, String preName, String repo) {
            this.sha256 = sha256;
            this.preName = preName;
            this.repo = repo;
        }

        /** The lowercase hexadecimal SHA-256 of the Python list text of the ids. */
        public String sha256() { return sha256; }

        /** The {@code tokenizer.ggml.pre} value the converter writes for this fingerprint. */
        public String preName() { return preName; }

        /** The Hugging Face repository the fingerprint was computed from. */
        public String repo() { return repo; }

        @Override
        public String toString() {
            return sha256 + " -> " + preName + " (" + repo + ")";
        }
    }

    private final String source;
    private final List<Row> rows;
    private final Map<String, List<String>> namesByHash;
    private final String checkText;

    private ChkhshTable(String source, List<Row> rows, String checkText) {
        this.source = source;
        this.rows = Collections.unmodifiableList(new ArrayList<>(rows));
        Map<String, List<String>> byHash = new LinkedHashMap<>();
        for (Row r : rows) byHash.computeIfAbsent(r.sha256, k -> new ArrayList<>()).add(r.preName);
        Map<String, List<String>> frozen = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> e : byHash.entrySet()) {
            frozen.put(e.getKey(), Collections.unmodifiableList(e.getValue()));
        }
        this.namesByHash = Collections.unmodifiableMap(frozen);
        this.checkText = checkText;
    }

    /**
     * The table bundled with NeuralFree, read once from the classpath.
     *
     * @throws IllegalStateException when a resource is missing or malformed (a defect of the build)
     */
    public static ChkhshTable builtIn() {
        ChkhshTable t = builtIn;
        if (t != null) return t;
        synchronized (ChkhshTable.class) {
            if (builtIn == null) builtIn = loadBuiltIn();
            return builtIn;
        }
    }

    private static ChkhshTable loadBuiltIn() {
        try {
            String table = decodeUtf8(readResource(TABLE_RESOURCE), TABLE_RESOURCE);
            String text = decodeUtf8(readResource(TEXT_RESOURCE), TEXT_RESOURCE);
            return parse(TABLE_RESOURCE, table, text);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the llama.cpp pre-tokenizer hash table: " + e.getMessage(), e);
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException("the bundled llama.cpp pre-tokenizer hash table is malformed: "
                    + e.getMessage(), e);
        }
    }

    /**
     * Reads a table in the generated format: the header line {@code # chkhsh.tsv - Generated from llama.cpp <commit>
     * by the updateHfTables Gradle task; do not edit. Rows: <n>}, the column line, then {@code n} rows of three
     * TAB-separated fields (a lowercase SHA-256, a name without whitespace, a repository).
     *
     * @param source    names the table in messages
     * @param tableText the table file's text
     * @param checkText the exact {@code chktxt}
     * @throws IllegalArgumentException naming the source and line of the first malformed line
     */
    public static ChkhshTable parse(String source, String tableText, String checkText) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(tableText, "tableText");
        Objects.requireNonNull(checkText, "checkText");
        if (checkText.isEmpty()) throw new IllegalArgumentException(source + ": the fingerprint text is empty");
        String[] lines = tableText.split("\n", -1);
        int count = lines.length;
        if (count > 0 && lines[count - 1].isEmpty()) count--;
        if (count < 2) throw new IllegalArgumentException(source + ": missing the header and column lines");
        Matcher header = HEADER.matcher(lines[0]);
        if (!header.matches()) {
            throw new IllegalArgumentException(source + ":1: expected the '# " + FILE_NAME + " - Generated from"
                    + " llama.cpp <commit> by the updateHfTables Gradle task; do not edit. Rows: <n>' header, got '"
                    + lines[0] + "'");
        }
        if (!COLUMNS.equals(lines[1])) {
            throw new IllegalArgumentException(source + ":2: expected the column line '" + COLUMNS.replace("\t", "\\t")
                    + "', got '" + lines[1] + "'");
        }
        int expected = Integer.parseInt(header.group(1));
        List<Row> rows = new ArrayList<>(Math.max(0, count - 2));
        for (int i = 2; i < count; i++) rows.add(parseRow(source, i + 1, lines[i]));
        if (rows.size() != expected) {
            throw new IllegalArgumentException(source + ": the header records " + expected + " rows, the table has "
                    + rows.size());
        }
        return new ChkhshTable(source, rows, checkText);
    }

    private static Row parseRow(String source, int lineNo, String line) {
        String at = source + ":" + lineNo + ": ";
        String[] f = line.split("\t", -1);
        if (f.length != 3) throw new IllegalArgumentException(at + "expected 3 TAB-separated fields, got " + f.length);
        if (!SHA256.matcher(f[0]).matches()) {
            throw new IllegalArgumentException(at + "'" + f[0] + "' is not a lowercase hexadecimal SHA-256");
        }
        if (f[1].isEmpty() || hasWhitespace(f[1])) {
            throw new IllegalArgumentException(at + "pre-tokenizer name '" + f[1] + "' is empty or holds whitespace");
        }
        if (f[2].isEmpty()) throw new IllegalArgumentException(at + "the repository field is empty");
        return new Row(f[0], f[1], f[2]);
    }

    private static boolean hasWhitespace(String s) {
        for (int i = 0; i < s.length(); i++) {
            if (Character.isWhitespace(s.charAt(i)) || Character.isSpaceChar(s.charAt(i))) return true;
        }
        return false;
    }

    /** The table's name in messages. */
    public String source() { return source; }

    /** The text the converter encodes, {@code chktxt}. */
    public String checkText() { return checkText; }

    /** Every row in table order (unmodifiable). */
    public List<Row> rows() { return rows; }

    /** The number of rows. */
    public int size() { return rows.size(); }

    /**
     * The name the converter selects for {@code sha256}: the last row holding it, as the converter's sequence of
     * {@code if} statements does; null when no row holds it.
     */
    public String name(String sha256) {
        List<String> names = namesByHash.get(sha256);
        return names == null ? null : names.get(names.size() - 1);
    }

    /** Every name of the rows holding {@code sha256}, in row order (unmodifiable; empty when none). */
    public List<String> names(String sha256) {
        List<String> names = namesByHash.get(sha256);
        return names == null ? Collections.emptyList() : names;
    }

    /** {@code sha256(str(ids).encode()).hexdigest()}: the fingerprint of an id list. */
    public static String hash(int[] ids) {
        Objects.requireNonNull(ids, "ids");
        byte[] digest = sha256().digest(pythonListText(ids).getBytes(StandardCharsets.UTF_8));
        StringBuilder sb = new StringBuilder(digest.length * 2);
        for (byte b : digest) {
            sb.append(Character.forDigit((b >> 4) & 0xF, 16)).append(Character.forDigit(b & 0xF, 16));
        }
        return sb.toString();
    }

    /** Python {@code str(list)} of integers: {@code "[1, 2, 3]"}, {@code "[]"} when empty. */
    public static String pythonListText(int[] ids) {
        Objects.requireNonNull(ids, "ids");
        StringBuilder sb = new StringBuilder(ids.length * 7 + 2).append('[');
        for (int i = 0; i < ids.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(ids[i]);
        }
        return sb.append(']').toString();
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("the Java runtime provides no SHA-256 implementation", e);
        }
    }

    private static byte[] readResource(String name) throws IOException {
        ClassLoader loader = ChkhshTable.class.getClassLoader();
        try (InputStream in = loader.getResourceAsStream(name)) {
            if (in == null) throw new IOException("resource '" + name + "' is not on the classpath");
            ByteArrayOutputStream out = new ByteArrayOutputStream(16 * 1024);
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) {
                if (out.size() + n > MAX_RESOURCE_BYTES) {
                    throw new IOException("resource '" + name + "' exceeds " + MAX_RESOURCE_BYTES + " bytes");
                }
                out.write(buf, 0, n);
            }
            return out.toByteArray();
        }
    }

    private static String decodeUtf8(byte[] bytes, String name) throws IOException {
        try {
            return StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(bytes))
                    .toString();
        } catch (CharacterCodingException e) {
            throw new IOException("resource '" + name + "' is not valid UTF-8", e);
        }
    }

    @Override
    public String toString() {
        return "ChkhshTable{" + source + ", " + rows.size() + " rows, " + namesByHash.size() + " hashes}";
    }
}
