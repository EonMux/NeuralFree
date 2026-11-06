package com.eonmux.neuralfree.gguf.inspect;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A manifest of GGUF tensor hashes, the file format {@code gguf-hash --check} verifies: one
 * {@code <type> <hash> <target>} line per hash, where {@code type} is {@code xxh64}, {@code sha1},
 * {@code sha256} or {@code uuid}, {@code hash} is the digest text and {@code target} is the file name of a
 * model or {@code <file>:<tensor>} of one tensor. Lines of another type and lines with fewer than three
 * whitespace-separated fields are ignored, as the reference ignores them.
 *
 * <p>{@link #read(Path)} loads a manifest once; {@link #verify(String, String, String)} then looks up one line.
 * The manifest itself is immutable.</p>
 */
public final class HashManifest {

    /** The outcome of looking one hash up in the manifest, in the reference's order of severity. */
    public enum Status {
        /** No line of that type names the target. */
        NOT_FOUND("Not Found"),
        /** A line names the target with a different hash. */
        MISMATCH("Mismatch"),
        /** A line names the target with this hash. */
        OK("Ok");

        private final String text;

        Status(String text) {
            this.text = text;
        }

        /** The text the reference prints for the status. */
        public String text() {
            return text;
        }
    }

    private static final class Entry {
        final String type;
        final String hash;
        final String target;

        Entry(String type, String hash, String target) {
            this.type = type;
            this.hash = hash;
            this.target = target;
        }
    }

    private final List<Entry> entries;
    private final Set<String> knownTypes;

    private HashManifest(List<Entry> entries, Set<String> knownTypes) {
        this.entries = Collections.unmodifiableList(entries);
        this.knownTypes = Collections.unmodifiableSet(knownTypes);
    }

    /**
     * Reads the manifest at {@code file}.
     *
     * @throws IOException when the file cannot be read
     */
    public static HashManifest read(Path file) throws IOException {
        byte[] bytes = Files.readAllBytes(file);
        String text = new String(bytes, StandardCharsets.UTF_8);
        List<Entry> entries = new ArrayList<>();
        Set<String> types = new LinkedHashSet<>();
        int lineStart = 0;
        while (lineStart <= text.length()) {
            int lineEnd = text.indexOf('\n', lineStart);
            if (lineEnd < 0) {
                lineEnd = text.length();
            }
            String line = text.substring(lineStart, lineEnd);
            if (line.endsWith("\r")) {
                line = line.substring(0, line.length() - 1);
            }
            String[] fields = line.trim().split("\\s+");
            if (fields.length >= 1 && GgufHasher.Algorithm.of(fields[0]) != null) {
                // the reference notes the type of any line that starts with one, even a truncated line
                types.add(fields[0]);
                if (fields.length >= 3) {
                    entries.add(new Entry(fields[0], fields[1], fields[2]));
                }
            }
            lineStart = lineEnd + 1;
        }
        return new HashManifest(entries, types);
    }

    /** The hash types at least one usable line names, in file order. */
    public Set<String> knownTypes() {
        return knownTypes;
    }

    /**
     * The status of {@code hash} of {@code target} for the algorithm whose printed name is {@code type}: the
     * first line of that type naming the target decides.
     */
    public Status verify(String type, String hash, String target) {
        for (Entry e : entries) {
            if (e.type.equals(type) && e.target.equals(target)) {
                return e.hash.equals(hash) ? Status.OK : Status.MISMATCH;
            }
        }
        return Status.NOT_FOUND;
    }
}
