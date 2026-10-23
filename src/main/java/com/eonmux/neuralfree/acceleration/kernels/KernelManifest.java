package com.eonmux.neuralfree.acceleration.kernels;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Immutable, ordered list of the source files of one dual-dialect kernel library, read from
 * {@code <root>/manifest.txt} on the classpath.
 *
 * <p>Format: one path per line, relative to the root. {@code #} starts a comment. Two optional, mutually
 * exclusive leading tags exist:</p>
 * <ul>
 *   <li>{@code [opencl]} marks a file that only the OpenCL build uses (CUDA and Vulkan never build it);</li>
 *   <li>{@code [shared]} marks a header that several libraries share: its path is relative to
 *       {@value #SHARED_ROOT} instead of the library root, and every backend builds it. Shared entries
 *       precede every other entry, so the OpenCL concatenation and the CUDA units (which place shared
 *       headers first) see the same order.</li>
 * </ul>
 * <p>Order is significant: a file may use only symbols of files listed above it, and the OpenCL program
 * is the concatenation of every entry in this order. Paths are unique across the manifest, shared or not,
 * because the {@code #line} markers of the assembled program name files by their path alone.</p>
 *
 * <p>The companion resource {@code <root>/inventory.txt}, generated from the sources and checked in, lists
 * every kernel the non-OpenCL-only entries define (one kernel per line, the name in the first tab- or
 * space-separated column); {@link #inventory(ClassLoader, String)} reads it.</p>
 *
 * <p>Manifests and inventories are cached per class loader and root.</p>
 */
public final class KernelManifest {

    /** File name of the manifest inside a library root. */
    public static final String MANIFEST_FILE = "manifest.txt";
    /** File name of the kernel inventory inside a library root. */
    public static final String INVENTORY_FILE = "inventory.txt";
    /** Tag that marks an OpenCL-only entry. */
    public static final String OPENCL_TAG = "[opencl]";
    /** Tag that marks a header resolved against {@link #SHARED_ROOT}. */
    public static final String SHARED_TAG = "[shared]";
    /** Classpath root of the {@code [shared]} entries of every library. */
    public static final String SHARED_ROOT = "kernels";

    private static final Map<ClassLoader, Map<String, KernelManifest>> MANIFESTS = new WeakHashMap<>();
    private static final Map<ClassLoader, Map<String, Optional<Set<String>>>> INVENTORIES = new WeakHashMap<>();

    /** One manifest line: a source path, whether only OpenCL builds it and whether it is shared. */
    public static final class Entry {
        private final String path;
        private final boolean openClOnly;
        private final boolean shared;

        /** A library entry (not shared). */
        public Entry(String path, boolean openClOnly) {
            this(path, openClOnly, false);
        }

        /**
         * @throws IllegalArgumentException when the path is not a clean relative path, or when the entry
         *                                  would be both OpenCL-only and shared
         */
        public Entry(String path, boolean openClOnly, boolean shared) {
            this.path = requireRelativePath(path, "manifest entry");
            if (openClOnly && shared) {
                throw new IllegalArgumentException("manifest entry '" + path + "': " + OPENCL_TAG + " and "
                        + SHARED_TAG + " are mutually exclusive");
            }
            this.openClOnly = openClOnly;
            this.shared = shared;
        }

        /**
         * Path relative to the root the entry resolves against ({@link #SHARED_ROOT} for shared entries,
         * otherwise the library root), using {@code /} separators. It is also the file name of the
         * entry's {@code #line} marker.
         */
        public String path() {
            return path;
        }

        /** True when the entry carries the {@code [opencl]} tag. */
        public boolean openClOnly() {
            return openClOnly;
        }

        /** True when the entry carries the {@code [shared]} tag. */
        public boolean shared() {
            return shared;
        }

        /**
         * Classpath path of the entry's source: {@code SHARED_ROOT + "/" + path} for shared entries,
         * {@code sourceRoot + "/" + path} otherwise.
         *
         * @param sourceRoot root of the library the manifest belongs to
         * @throws IllegalArgumentException when the root is blank or has a leading or trailing {@code /}
         */
        public String resourcePath(String sourceRoot) {
            String libraryRoot = requireRoot(sourceRoot);
            return (shared ? SHARED_ROOT : libraryRoot) + "/" + path;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof Entry)) {
                return false;
            }
            Entry that = (Entry) o;
            return openClOnly == that.openClOnly && shared == that.shared && path.equals(that.path);
        }

        @Override
        public int hashCode() {
            return (path.hashCode() * 31 + (openClOnly ? 1 : 0)) * 31 + (shared ? 1 : 0);
        }

        /** The manifest line form: the tag, if any, then the path. */
        @Override
        public String toString() {
            if (openClOnly) {
                return OPENCL_TAG + " " + path;
            }
            return shared ? SHARED_TAG + " " + path : path;
        }
    }

    private final String root;
    private final List<Entry> entries;

    private KernelManifest(String root, List<Entry> entries) {
        this.root = root;
        this.entries = Collections.unmodifiableList(new ArrayList<>(entries));
    }

    /**
     * Loads (or returns the cached) manifest of a library root.
     *
     * @param cl   class loader whose resources hold the library
     * @param root classpath root without leading or trailing {@code /}, for example {@code kernels/llm}
     * @throws IllegalArgumentException when the manifest is missing or malformed
     * @throws UncheckedIOException     when the manifest cannot be read
     */
    public static KernelManifest load(ClassLoader cl, String root) {
        Objects.requireNonNull(cl, "class loader must not be null");
        String r = requireRoot(root);
        Map<String, KernelManifest> perLoader;
        synchronized (MANIFESTS) {
            perLoader = MANIFESTS.computeIfAbsent(cl, k -> new ConcurrentHashMap<>());
        }
        KernelManifest cached = perLoader.get(r);
        if (cached != null) {
            return cached;
        }
        String path = r + "/" + MANIFEST_FILE;
        String text = readText(cl, path);
        if (text == null) {
            throw new IllegalArgumentException("no kernel manifest at classpath:" + path);
        }
        KernelManifest parsed = parse(r, text, "classpath:" + path);
        KernelManifest raced = perLoader.putIfAbsent(r, parsed);
        return raced != null ? raced : parsed;
    }

    /**
     * Parses manifest text.
     *
     * @param root   library root the paths are relative to
     * @param text   manifest content
     * @param origin where the text came from, used in error messages
     * @throws IllegalArgumentException naming the origin and line of the first malformed line: an unknown,
     *                                  repeated or conflicting tag, a {@code [shared]} entry after a library
     *                                  entry, a malformed or duplicate path, or no entries at all
     */
    public static KernelManifest parse(String root, String text, String origin) {
        String r = requireRoot(root);
        Objects.requireNonNull(text, "manifest text must not be null");
        List<Entry> entries = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        String[] lines = text.split("\r?\n", -1);
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i];
            int hash = line.indexOf('#');
            if (hash >= 0) {
                line = line.substring(0, hash);
            }
            line = line.trim();
            if (line.isEmpty()) {
                continue;
            }
            String where = origin + ":" + (i + 1) + ": ";
            boolean openClOnly = false;
            boolean shared = false;
            while (line.startsWith("[")) {
                int close = line.indexOf(']');
                String tag = close < 0 ? line : line.substring(0, close + 1);
                if (OPENCL_TAG.equals(tag) && !openClOnly) {
                    openClOnly = true;
                } else if (SHARED_TAG.equals(tag) && !shared) {
                    shared = true;
                } else if (OPENCL_TAG.equals(tag) || SHARED_TAG.equals(tag)) {
                    throw new IllegalArgumentException(where + "repeated tag '" + tag + "'");
                } else {
                    throw new IllegalArgumentException(where + "unknown tag '" + tag + "'; the tags are "
                            + OPENCL_TAG + " and " + SHARED_TAG);
                }
                line = line.substring(close + 1).trim();
            }
            if (shared && entries.size() > 0 && !entries.get(entries.size() - 1).shared()) {
                throw new IllegalArgumentException(where + SHARED_TAG + " entry '" + line
                        + "' must precede every library entry");
            }
            Entry entry;
            try {
                entry = new Entry(line, openClOnly, shared);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(where + e.getMessage(), e);
            }
            if (!seen.add(entry.path())) {
                throw new IllegalArgumentException(where + "duplicate entry '" + entry.path() + "'");
            }
            entries.add(entry);
        }
        if (entries.isEmpty()) {
            throw new IllegalArgumentException(origin + ": manifest lists no source files");
        }
        return new KernelManifest(r, entries);
    }

    /**
     * The kernel inventory of a library root: every kernel defined by the entries that are not
     * OpenCL-only. Empty when {@code <root>/inventory.txt} is not on the classpath.
     *
     * @throws IllegalArgumentException when the inventory is malformed (blank name or duplicate)
     * @throws UncheckedIOException     when the inventory cannot be read
     */
    public static Optional<Set<String>> inventory(ClassLoader cl, String root) {
        Objects.requireNonNull(cl, "class loader must not be null");
        String r = requireRoot(root);
        Map<String, Optional<Set<String>>> perLoader;
        synchronized (INVENTORIES) {
            perLoader = INVENTORIES.computeIfAbsent(cl, k -> new ConcurrentHashMap<>());
        }
        Optional<Set<String>> cached = perLoader.get(r);
        if (cached != null) {
            return cached;
        }
        String path = r + "/" + INVENTORY_FILE;
        String text = readText(cl, path);
        Optional<Set<String>> parsed = text == null
                ? Optional.<Set<String>>empty()
                : Optional.of(parseInventory(text, "classpath:" + path));
        Optional<Set<String>> raced = perLoader.putIfAbsent(r, parsed);
        return raced != null ? raced : parsed;
    }

    /**
     * Parses inventory text: one kernel per line, the name in the first column; further columns
     * (scalar and buffer counts) are ignored; {@code #} lines and blank lines are skipped.
     *
     * @throws IllegalArgumentException naming the origin and line of a duplicate or invalid name
     */
    public static Set<String> parseInventory(String text, String origin) {
        Objects.requireNonNull(text, "inventory text must not be null");
        Set<String> names = new LinkedHashSet<>();
        String[] lines = text.split("\r?\n", -1);
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i].trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            String name = line.split("[\\t ]+", 2)[0];
            if (!isIdentifier(name)) {
                throw new IllegalArgumentException(origin + ":" + (i + 1) + ": '" + name
                        + "' is not a kernel name");
            }
            if (!names.add(name)) {
                throw new IllegalArgumentException(origin + ":" + (i + 1) + ": duplicate kernel '" + name + "'");
            }
        }
        return Collections.unmodifiableSet(names);
    }

    /** The library root this manifest belongs to. */
    public String root() {
        return root;
    }

    /** Unmodifiable entries in manifest order. */
    public List<Entry> entries() {
        return entries;
    }

    /**
     * Unmodifiable paths of the entries that every backend builds (the untagged and {@code [shared]}
     * entries), in order.
     */
    public List<String> portablePaths() {
        List<String> out = new ArrayList<>();
        for (Entry e : entries) {
            if (!e.openClOnly()) {
                out.add(e.path());
            }
        }
        return Collections.unmodifiableList(out);
    }

    /** Unmodifiable paths of the {@code [opencl]} entries, in order. */
    public List<String> openClOnlyPaths() {
        List<String> out = new ArrayList<>();
        for (Entry e : entries) {
            if (e.openClOnly()) {
                out.add(e.path());
            }
        }
        return Collections.unmodifiableList(out);
    }

    /** Classpath path of an entry of this manifest: {@code entry.resourcePath(root())}. */
    public String resourcePath(Entry entry) {
        Objects.requireNonNull(entry, "entry must not be null");
        return entry.resourcePath(root);
    }

    @Override
    public String toString() {
        return "KernelManifest{root=" + root + ", entries=" + entries.size() + "}";
    }

    /**
     * Whether a backend build covers the whole inventory of a library: available when every kernel of
     * {@code <sourceRoot>/inventory.txt} is in {@code built}; otherwise unavailable with a reason that
     * names the first missing kernels.
     *
     * @param libraryId library id, used in the reason ("LLM kernel inventory incomplete")
     * @param builtWhat what {@code built} is, for the reason, for example "the CUDA kernel index"
     */
    static Availability inventoryCoverage(ClassLoader cl, String libraryId, String sourceRoot,
                                          Set<String> built, String builtWhat) {
        Optional<Set<String>> inventory;
        try {
            inventory = inventory(cl, sourceRoot);
        } catch (RuntimeException e) {
            return Availability.no("kernel inventory classpath:" + sourceRoot + "/" + INVENTORY_FILE
                    + " is unreadable: " + e.getMessage());
        }
        if (!inventory.isPresent()) {
            return Availability.no("kernel inventory classpath:" + sourceRoot + "/" + INVENTORY_FILE
                    + " is missing; it is a checked-in resource of the kernel sources (regenerate it with the"
                    + " updateKernelInventory Gradle task)");
        }
        List<String> missing = new ArrayList<>();
        for (String name : inventory.get()) {
            if (!built.contains(name)) {
                missing.add(name);
            }
        }
        if (missing.isEmpty()) {
            return Availability.ok();
        }
        Collections.sort(missing);
        StringBuilder sb = new StringBuilder();
        sb.append(libraryId.toUpperCase(java.util.Locale.ROOT)).append(" kernel inventory incomplete: ")
                .append(missing.size()).append(" of ").append(inventory.get().size())
                .append(" kernels are missing from ").append(builtWhat).append(" (");
        int shown = Math.min(missing.size(), 5);
        for (int i = 0; i < shown; i++) {
            sb.append(i == 0 ? "" : ", ").append(missing.get(i));
        }
        sb.append(missing.size() > shown ? ", ...)" : ")");
        return Availability.no(sb.toString());
    }

    /**
     * Reads a classpath text resource as UTF-8.
     *
     * @return the content, or null when the resource does not exist
     * @throws UncheckedIOException when the resource exists but cannot be read
     */
    static String readText(ClassLoader cl, String path) {
        try (InputStream in = cl.getResourceAsStream(path)) {
            if (in == null) {
                return null;
            }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[16384];
            int n;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
            }
            return new String(out.toByteArray(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read classpath:" + path, e);
        }
    }

    /** @throws IllegalArgumentException when the kernel name is null or empty */
    static void requireKernelName(String kernelName) {
        if (kernelName == null || kernelName.isEmpty()) {
            throw new IllegalArgumentException("kernel name must not be null or empty");
        }
    }

    static boolean isIdentifier(String s) {
        if (s == null || s.isEmpty() || (s.charAt(0) >= '0' && s.charAt(0) <= '9')) {
            return false;
        }
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!(c == '_' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
                return false;
            }
        }
        return true;
    }

    static String requireRoot(String root) {
        if (root == null || root.trim().isEmpty()) {
            throw new IllegalArgumentException("library root must not be null or blank");
        }
        if (root.startsWith("/") || root.endsWith("/")) {
            throw new IllegalArgumentException("library root '" + root
                    + "' must be a classpath path without leading or trailing '/'");
        }
        return root;
    }

    private static String requireRelativePath(String path, String what) {
        if (path == null || path.isEmpty()) {
            throw new IllegalArgumentException(what + " path must not be empty");
        }
        for (int i = 0; i < path.length(); i++) {
            if (Character.isWhitespace(path.charAt(i))) {
                throw new IllegalArgumentException(what + " '" + path + "' must not contain whitespace");
            }
        }
        if (path.startsWith("/") || path.indexOf('\\') >= 0) {
            throw new IllegalArgumentException(what + " '" + path
                    + "' must be relative to the library root and use '/' separators");
        }
        for (String part : path.split("/", -1)) {
            if (part.isEmpty() || part.equals(".") || part.equals("..")) {
                throw new IllegalArgumentException(what + " '" + path
                        + "' must not contain empty, '.' or '..' segments");
            }
        }
        return path;
    }
}
