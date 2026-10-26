package com.eonmux.neuralfree.gguf.hub;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * The Hugging Face Hub cache, in the Hub's own layout over its root:
 * {@code <root>/models--user--model/{blobs,refs,snapshots}} (PARITY.md WP TL-7). The root is
 * {@code LLAMA_CACHE} when that environment variable is set, then {@code HF_HUB_CACHE},
 * {@code HUGGINGFACE_HUB_CACHE}, {@code HF_HOME/hub}, {@code XDG_CACHE_HOME/huggingface/hub} and
 * finally {@code $HOME/.cache/huggingface/hub}, llama.cpp's {@code get_cache_directory}
 * ({@code common/hf-cache.cpp:27-53}) at the same precedence. Every path a downloaded file may land
 * on is confined to its repository's own directory ({@link #snapshotPath}): an untrusted repository
 * path (§3.6) escapes the cache by no route of {@code ..}, an absolute path or an empty segment.
 * Immutable; safe for concurrent readers.
 */
public final class HubCache {

    /** A file of a cached snapshot: the repository's own description and its local snapshot path. */
    public static final class CachedFile {

        private final String repoId;
        private final String repoPath;
        private final Path localPath;
        private final GgufFilename filename;

        CachedFile(String repoId, String repoPath, Path localPath, GgufFilename filename) {
            this.repoId = repoId;
            this.repoPath = repoPath;
            this.localPath = localPath;
            this.filename = filename;
        }

        /** The repository id ({@code user/model}). */
        public String repoId() {
            return repoId;
        }

        /** The file's path inside the repository ({@code sub/Model-Q4_K_M-00001-of-00003.gguf}). */
        public String repoPath() {
            return repoPath;
        }

        /** The path inside the snapshot tree the file exists at, already confined. */
        public Path localPath() {
            return localPath;
        }

        /** The parse of {@link #repoPath} into filename parts, or null for other files. */
        public GgufFilename filename() {
            return filename;
        }
    }

    /**
     * The filename grammar llama.cpp's downloader parses ({@code common/download.cpp:524-546}):
     * {@code <prefix>[-<tag>][-<index>-of-<count>].gguf} — the tag is the trailing {@code [-.]}-
     * separated alphanumeric segment, uppercased ({@code Q4_K_M}, {@code F16}); {@code prefix} keeps
     * the name up to the shard suffix, tags included; {@code index} counts from 1, and a {@code count}
     * above 1 marks a shard. Immutable.
     */
    public static final class GgufFilename {

        private final String prefix;
        private final String tag;
        private final int index;
        private final int count;

        GgufFilename(String prefix, String tag, int index, int count) {
            this.prefix = prefix;
            this.tag = tag;
            this.index = index;
            this.count = count;
        }

        /**
         * Parses a repository path into its filename parts, or null when it is not a {@code .gguf}
         * file; the split suffix and the tag rules are llama.cpp's get_gguf_split_info.
         */
        public static GgufFilename parse(String repoPath) {
            if (repoPath == null || repoPath.isEmpty()
                    || !repoPath.toLowerCase(Locale.ROOT).endsWith(".gguf")) {
                return null;
            }
            String name = repoPath.substring(0, repoPath.length() - ".gguf".length()).replace('\\', '/');
            String prefix = name;
            int index = 1;
            int count = 1;
            Matcher split = SPLIT_PATTERN.matcher(name);
            if (split.matches()) {
                prefix = split.group(1);
                index = Integer.parseInt(split.group(2));
                count = Integer.parseInt(split.group(3));
            }
            String tag = "";
            Matcher tagMatcher = TAG_PATTERN.matcher(prefix);
            if (tagMatcher.find()) {
                tag = tagMatcher.group(1).toUpperCase(Locale.ROOT);
            }
            return new GgufFilename(prefix, tag, index, count);
        }

        /** The name up to the shard suffix, tags included; the whole .gguf name when unsplit. */
        public String prefix() {
            return prefix;
        }

        /** The trailing quant tag ({@code Q4_K_M}) uppercased, or empty when the name has none. */
        public String tag() {
            return tag;
        }

        /** The shard index counting from 1; 1 when the name has no split. */
        public int index() {
            return index;
        }

        /** The shard count: 1 when the name has no split, so {@link #isShard()} is false. */
        public int count() {
            return count;
        }

        /** Whether this file is one shard of an {@code -NNNNN-of-NNNNN.gguf} split. */
        public boolean isShard() {
            return count > 1;
        }

        /**
         * The first number of the tag read through its digits (llama.cpp's {@code extract_quant_bits}):
         * {@code Q4_0}→4, {@code F16}→16, {@code NVFP4}→4, {@code Q8_K_M}→8; 0 without a digit.
         */
        public int quantBits() {
            for (int i = 0; i < tag.length(); i++) {
                if (tag.charAt(i) >= '0' && tag.charAt(i) <= '9') {
                    StringBuilder digits = new StringBuilder();
                    while (i < tag.length() && digits.length() < 9
                            && tag.charAt(i) >= '0' && tag.charAt(i) <= '9') {
                        digits.append(tag.charAt(i));
                        i++;
                    }
                    return Integer.parseInt(digits.toString());
                }
            }
            return 0;
        }
    }

    /** The sidecar keywords llama.cpp's downloader excludes from model files ({@code name_is_sidecar}). */
    static final String[] SIDECAR_KEYWORDS = {"mmproj", "mtp-", "dflash-", "eagle3-", "dspark-"};

    private static final Pattern SPLIT_PATTERN =
            Pattern.compile("^(.+)-([0-9]{5})-of-([0-9]{5})$", Pattern.CASE_INSENSITIVE);
    private static final Pattern TAG_PATTERN =
            Pattern.compile("[-.]([A-Za-z0-9_]+)$", Pattern.CASE_INSENSITIVE);

    private final Path root;

    /** A cache rooted at {@code root}, normally {@link #defaultRoot()} or an explicit test root. */
    public HubCache(Path root) {
        this.root = root.toAbsolutePath().normalize();
    }

    /**
     * The cache root from the environment, llama.cpp's precedence: {@code LLAMA_CACHE},
     * {@code HF_HUB_CACHE}, {@code HUGGINGFACE_HUB_CACHE}, {@code HF_HOME/hub},
     * {@code XDG_CACHE_HOME/huggingface/hub} and then {@code $HOME/.cache/huggingface/hub}. Never
     * null; without a home directory there is no root (llama.cpp raises in the same case).
     */
    public static Path defaultRoot() {
        String llamaCache = env("LLAMA_CACHE");
        if (!llamaCache.isEmpty()) {
            return Paths.get(llamaCache);
        }
        String hfHubCache = env("HF_HUB_CACHE");
        if (!hfHubCache.isEmpty()) {
            return Paths.get(hfHubCache);
        }
        String huggingfaceHubCache = env("HUGGINGFACE_HUB_CACHE");
        if (!huggingfaceHubCache.isEmpty()) {
            return Paths.get(huggingfaceHubCache);
        }
        String hfHome = env("HF_HOME");
        if (!hfHome.isEmpty()) {
            return Paths.get(hfHome).resolve("hub");
        }
        String xdgCache = env("XDG_CACHE_HOME");
        if (!xdgCache.isEmpty()) {
            return Paths.get(xdgCache).resolve("huggingface").resolve("hub");
        }
        String home = env("HOME");
        if (home.isEmpty()) {
            throw new IllegalStateException("Failed to determine HF cache directory; neither"
                    + " LLAMA_CACHE, HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE, HF_HOME, XDG_CACHE_HOME nor"
                    + " HOME is set");
        }
        return Paths.get(home).resolve(".cache").resolve("huggingface").resolve("hub");
    }

    /** An environment variable's value, trimmed, or the empty string instead of null. */
    private static String env(String name) {
        String v = System.getenv(name);
        return v == null ? "" : v.trim();
    }

    /** The cache root. */
    public Path root() {
        return root;
    }

    /** The repository's directory {@code <root>/models--user--model}. */
    public Path repoPath(HubRef ref) {
        return root.resolve(ref.folderName());
    }

    /**
     * Where the file {@code repoFilePath} of commit {@code commit} lands in the snapshot tree. The
     * untrusted repository path leaves the snapshot tree by no route: absolute paths, {@code ..} and
     * empty segments are rejected naming the path, the same rule llama.cpp's is_valid_subpath applies.
     */
    public Path snapshotPath(HubRef ref, String commit, String repoFilePath) {
        return confine(repoPath(ref).resolve("snapshots").resolve(validCommit(commit)),
                repoFilePath, "repository path");
    }

    /** The blob an LFS object of the repository lands in ({@code blobs/<oid>}); the oid is 64-hex LFS sha256. */
    public Path blobPath(HubRef ref, String oid) {
        return repoPath(ref).resolve("blobs").resolve(validOid(oid, "LFS oid"));
    }

    /** The ref file of {@code refName}, whose name is a single static path segment. */
    public Path refPath(HubRef ref, String refName) {
        if (!isRefName(refName)) {
            throw new IllegalArgumentException("invalid ref name '" + refName + "': use A-Za-z0-9._- only");
        }
        return repoPath(ref).resolve("refs").resolve(refName);
    }

    /** The commit {@code refName} names, or null when its ref file is absent or carries no 40-hex commit. */
    public String readRef(HubRef ref, String refName) {
        return readRefFile(refPath(ref, refName));
    }

    /** Writes the commit {@code refName} names, creating the refs directory as needed. */
    public void writeRef(HubRef ref, String refName, String commit) throws IOException {
        Path path = refPath(ref, refName);
        Files.createDirectories(path.getParent());
        Files.write(path, (validCommit(commit) + "\n").getBytes(StandardCharsets.UTF_8));
    }

    /**
     * The commit {@code ref} resolves to in the cache: a revision that is already a 40-hex commit
     * hash is returned as itself, an explicit ref name names its own file ("main" first), and no
     * revision at all takes the "main" ref, then the first sorted ref (llama.cpp's get_cached_ref).
     */
    public String cachedCommit(HubRef ref) throws IOException {
        String revision = ref.revision();
        if (revision != null && isValidOid(revision, 40)) {
            return validCommit(revision);
        }
        if (revision != null) {
            return readRef(ref, revision);
        }
        String main = readRef(ref, "main");
        if (main != null) {
            return main;
        }
        return firstRef(ref);
    }

    /** Any ref's commit of the repository, or null (llama.cpp's get_cached_ref fallback). */
    private String firstRef(HubRef ref) throws IOException {
        Path refs = repoPath(ref).resolve("refs");
        if (!Files.isDirectory(refs)) {
            return null;
        }
        try (Stream<Path> entries = Files.list(refs).sorted()) {
            for (Path entry : entries.collect(java.util.stream.Collectors.toList())) {
                String commit = readRefFile(entry);
                if (commit != null) {
                    return commit;
                }
            }
        }
        return null;
    }

    /**
     * The files of the cached snapshots: every repository directory of the layout whose folder name
     * decodes to a valid repository id, through the commit its refs name (main first). Restrictions:
     * an invalid folder name, an invalid commit or a repository path outside the layout yields
     * nothing, never a path outside the cache. {@code repoId} filters the listing when given.
     */
    public List<CachedFile> cachedFiles(String repoId) throws IOException {
        List<CachedFile> out = new ArrayList<>();
        if (repoId != null && !isValidRepoId(repoId)) {
            return out;
        }
        if (!Files.isDirectory(root)) {
            return out;
        }
        List<Path> repos;
        try (Stream<Path> entries = Files.list(root).sorted()) {
            repos = entries.collect(java.util.stream.Collectors.toList());
        }
        for (Path repo : repos) {
            String folder = repo.getFileName().toString();
            if (!folder.startsWith("models--") || !Files.isDirectory(repo.resolve("snapshots")) || !Files.isDirectory(repo.resolve("refs"))) {
                continue;
            }
            String id = folder.substring("models--".length()).replace("--", "/");
            if (!isValidRepoId(id) || (repoId != null && !id.equals(repoId))) {
                continue;
            }
            listRepo(repo, id, out);
        }
        return out;
    }

    /** The snapshot files of one repository directory, through the commit its refs name (main first). */
    private void listRepo(Path repo, String id, List<CachedFile> out) throws IOException {
        Path refs = repo.resolve("refs");
        String commit = readRefFile(refs.resolve("main"));
        if (commit == null) {
            try (Stream<Path> entries = Files.list(refs).sorted()) {
                for (Path entry : entries.collect(java.util.stream.Collectors.toList())) {
                    String candidate = readRefFile(entry);
                    if (candidate != null) {
                        commit = candidate;
                        break;
                    }
                }
            }
        }
        if (commit == null) {
            return;
        }
        Path snapshot = repo.resolve("snapshots").resolve(commit);
        if (!Files.isDirectory(snapshot)) {
            return;
        }
        try (Stream<Path> walk = Files.walk(snapshot)) {
            for (Path p : walk.sorted(Comparator.naturalOrder()).collect(java.util.stream.Collectors.toList())) {
                if (!p.equals(snapshot) && (Files.isRegularFile(p) || Files.isSymbolicLink(p))) {
                    String relative = snapshot.relativize(p).toString().replace('\\', '/');
                    out.add(new CachedFile(id, relative, p, GgufFilename.parse(relative)));
                }
            }
        }
    }

    /** A 40-hex commit read from one ref file ({@code readable}), or null. */
    private String readRefFile(Path file) {
        if (!Files.isRegularFile(file)) {
            return null;
        }
        try {
            String commit = new String(Files.readAllBytes(file), StandardCharsets.UTF_8).trim();
            return isValidOid(commit, 40) ? validCommit(commit) : null;
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * Brings a downloaded blob into its place in the snapshot tree, llama.cpp's finalize_file: a
     * snapshot that already exists stands; a blob is linked into the snapshot tree with a relative
     * symbolic link when the filesystem allows one, and otherwise moved, then copied (the degraded
     * mode llama.cpp switches to and logs). Returns the snapshot path either way.
     */
    public Path finalizeFile(Path localPath, Path snapshotPath) throws IOException {
        if (Files.exists(snapshotPath) || !Files.isRegularFile(localPath)) {
            return snapshotPath;
        }
        Files.createDirectories(snapshotPath.getParent());
        try {
            Path relative = snapshotPath.getParent().relativize(localPath);
            Files.createSymbolicLink(snapshotPath, relative);
            return snapshotPath;
        } catch (IOException | UnsupportedOperationException e) {
            // no link support on this filesystem: llama.cpp moves, then copies the blob itself
        }
        try {
            Files.move(localPath, snapshotPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            Files.copy(localPath, snapshotPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
        return snapshotPath;
    }

    /** Whether {@code repoPath} is a downloadable model file: a non-sidecar, non-imatrix GGUF (llama.cpp's gguf_filename_is_model). */
    public static boolean isModel(String repoPath) {
        if (!repoPath.toLowerCase(Locale.ROOT).endsWith(".gguf")) {
            return false;
        }
        String file = repoPath.replace('\\', '/');
        int slash = file.lastIndexOf('/');
        file = slash >= 0 ? file.substring(slash + 1) : file;
        if (file.contains("imatrix")) {
            return false;
        }
        for (String sidecar : SIDECAR_KEYWORDS) {
            if (file.contains(sidecar)) {
                return false;
            }
        }
        return true;
    }

    /** A snapshot path whose relative repository input is confined: valid segment by valid segment. */
    static Path confine(Path base, String relative, String what) {
        String slash = relative.replace('\\', '/');
        if (slash.isEmpty()) {
            throw new IllegalArgumentException("the " + what + " is empty");
        }
        Path p;
        try {
            p = Paths.get(slash);
        } catch (java.nio.file.InvalidPathException e) {
            throw new IllegalArgumentException("the " + what + " '" + relative
                    + "' is not a path of valid characters", e);
        }
        if (p.isAbsolute()) {
            throw new IllegalArgumentException("the " + what + " '" + relative + "' is absolute");
        }
        for (int i = 0; i < p.getNameCount(); i++) {
            String element = p.getName(i).toString();
            if (element.isEmpty() || element.equals(".") || element.equals("..")) {
                throw new IllegalArgumentException("the " + what + " '" + relative
                        + "' leaves the snapshot tree with segment '" + element + "'");
            }
        }
        Path confined = base.resolve(p).normalize();
        if (!confined.startsWith(base)) {
            throw new IllegalArgumentException("the " + what + " '" + relative + "' leaves the cache root");
        }
        return confined;
    }

    /** The 40-hex commit, llama.cpp's is_valid_commit, lowercased. */
    static String validCommit(String commit) {
        if (isValidOid(commit, 40)) {
            return commit.toLowerCase(Locale.ROOT);
        }
        throw new IllegalArgumentException("invalid commit '" + commit + "': expected 40 hex characters");
    }

    /** Whether {@code oid} is exactly {@code length} hex characters. */
    static boolean isValidOid(String oid, int length) {
        if (oid == null || length != 40 && length != 64 || oid.length() != length) {
            return false;
        }
        for (int i = 0; i < oid.length(); i++) {
            char c = oid.charAt(i);
            boolean hex = (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
            if (!hex) {
                return false;
            }
        }
        return true;
    }

    /** The lowercased validated oid: exactly 40 (git blob) or 64 (LFS sha256) hex characters. */
    public static String validOid(String oid, String what) {
        if (oid == null || (oid.length() != 40 && oid.length() != 64)) {
            throw new IllegalArgumentException("invalid " + what + ": expected 40 or 64 hex characters");
        }
        return oid.toLowerCase(Locale.ROOT);
    }

    /** Whether {@code oid} is a 64-hex LFS sha256 pointer, the digest the cache verifies downloads with. */
    public static boolean isSha256Oid(String oid) {
        return isValidOid(oid, 64);
    }

    /** A single static ref name: A-Za-z0-9._- only, one segment, not the dot itself. */
    private static boolean isRefName(String refName) {
        if (refName == null || refName.isEmpty() || refName.length() > 256) {
            return false;
        }
        for (int i = 0; i < refName.length(); i++) {
            char c = refName.charAt(i);
            boolean ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                    || c == '.' || c == '-' || c == '_';
            if (!ok) {
                return false;
            }
        }
        return !refName.equals(".") && !refName.equals("..");
    }

    /** llama.cpp's is_valid_repo_id, also used on the ids the cache folder names decode back into. */
    static boolean isValidRepoId(String repo) {
        if (repo == null || repo.isEmpty() || repo.length() > 256) {
            return false;
        }
        int slashes = 0;
        boolean lastWasSpecial = true;
        for (int i = 0; i < repo.length(); i++) {
            char c = repo.charAt(i);
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') {
                lastWasSpecial = false;
            } else if (c == '/' || c == '.' || c == '-') {
                if (lastWasSpecial) {
                    return false;
                }
                if (c == '/') {
                    slashes++;
                }
                lastWasSpecial = true;
            } else {
                return false;
            }
        }
        return !lastWasSpecial && slashes == 1;
    }
}
