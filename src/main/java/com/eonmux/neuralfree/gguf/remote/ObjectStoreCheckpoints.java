package com.eonmux.neuralfree.gguf.remote;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.utils.json.Json;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;

/**
 * Object-storage checkpoints, the loading path vLLM's {@code runai_streamer} adds: {@code s3://}, {@code gs://} and
 * {@code az://} locations are mirrored into a per-URI directory under {@link ModelSourceOptions#cacheDir()} and the
 * mirror is then opened by the ordinary local checkpoint readers. A mirror keeps every model file the allow-list
 * names (configurations, tokenizers and weights), downloads the files whose size or entity tag changed in parallel
 * ({@link ModelSourceOptions#remoteConcurrency()}), leaves unchanged files in place, and records its listing in
 * {@value #MANIFEST} so files a later listing no longer carries are removed. Nothing outside the mirror directory is
 * written, and no key that is not a safe relative path is ever materialized.
 */
public final class ObjectStoreCheckpoints {

    private static final Logger LOG = LogManager.getLogger(ObjectStoreCheckpoints.class);

    /** The mirror's listing record, inside the mirror directory. */
    public static final String MANIFEST = ".neuralfree-mirror.json";
    /** The piece of a download in progress; a completed piece is renamed into place. */
    static final String PARTIAL_SUFFIX = ".part";

    private static final String MIRROR_DIR = "remote";
    private static final Set<String> MODEL_SUFFIXES = modelSuffixes();

    private ObjectStoreCheckpoints() {
    }

    /** Whether {@code path} is an object-store URI ({@code s3://}, {@code gs://}, {@code az://}, {@code azure://}). */
    public static boolean isRemote(String path) {
        return ObjectStoreUri.isObjectStorePath(path);
    }

    /**
     * The local path holding the checkpoint of the object-store URI {@code uri}: the URI itself when it is not
     * remote, else the mirror directory (or the single mirrored file, when the URI names one object).
     *
     * @throws IllegalArgumentException when the URI or an object key is malformed or outside the mirror directory
     * @throws IOException              when the store cannot be listed, an object cannot be downloaded, or the
     *                                  location holds no model file
     */
    public static Path localize(String uri, ModelSourceOptions options) throws IOException {
        Objects.requireNonNull(uri, "uri");
        Objects.requireNonNull(options, "options");
        if (!isRemote(uri)) {
            return Paths.get(uri);
        }
        ObjectStoreUri location = ObjectStoreUri.parse(uri);
        Path root = mirrorRoot(options.cacheDir(), location);
        ObjectStore store = ObjectStores.open(location);
        try {
            return mirror(location, store, root, options.remoteConcurrency());
        } finally {
            try {
                store.close();
            } catch (IOException e) {
                LOG.debug("closing {} failed: {}", store.description(), e.getMessage());
            }
        }
    }

    private static Path mirrorRoot(Path cacheDir, ObjectStoreUri uri) throws IOException {
        Path base = cacheDir.resolve(MIRROR_DIR).toAbsolutePath().normalize();
        Files.createDirectories(base);
        return base.resolve(RemoteHttp.sha256Hex(uri.toString()));
    }

    private static Path mirror(ObjectStoreUri uri, ObjectStore store, Path root, int concurrency) throws IOException {
        List<RemoteObject> listing = store.list(uri.prefix());
        List<Entry> entries = new ArrayList<>();
        Map<String, String> relPaths = new HashMap<>();
        for (RemoteObject object : listing) {
            String rel = relativeKey(uri, object.key());
            if (rel == null) {
                continue;
            }
            if (!ModelFiles.accepts(rel)) {
                continue;
            }
            Path local;
            try {
                local = safeResolve(root, rel);
            } catch (IllegalArgumentException e) {
                throw new IOException("the object '" + object.key() + "' cannot be mirrored: " + e.getMessage(), e);
            }
            String previous = relPaths.put(local.toString(), object.key());
            if (previous != null) {
                throw new IOException("the objects '" + object.key() + "' and '" + previous
                        + "' both map to the local path '" + local + "'");
            }
            entries.add(new Entry(object, rel, local));
        }
        if (entries.isEmpty()) {
            throw new IOException("no model files under '" + uri + "': looked for " + ModelFiles.DESCRIPTION
                    + " (" + listing.size() + " object(s) listed)");
        }
        Manifest previous = Manifest.read(root.resolve(MANIFEST));
        List<Entry> downloads = new ArrayList<>();
        for (Entry entry : entries) {
            if (!upToDate(entry, previous)) {
                downloads.add(entry);
            }
        }
        if (!downloads.isEmpty()) {
            downloadAll(store, downloads, Math.min(concurrency, downloads.size()));
        }
        removeStale(root, entries, previous);
        writeManifest(root, uri, entries);
        LOG.info("Mirrored {} object(s) of {} into {} ({} downloaded)", entries.size(), uri, root, downloads.size());
        if (entries.size() == 1) {
            Entry only = entries.get(0);
            if (!only.rel.contains("/") && isSingleCheckpointFile(only.rel)) {
                return only.local;
            }
        }
        return root;
    }

    /** Whether the mirrored file is the object the listing names; the entity tag decides when both carry one. */
    private static boolean upToDate(Entry entry, Manifest previous) throws IOException {
        Path local = entry.local;
        if (!Files.isRegularFile(local) || Files.isSymbolicLink(local)) {
            return false;
        }
        if (entry.object.size() >= 0 && Files.size(local) != entry.object.size()) {
            return false;
        }
        Manifest.Record record = previous.records.get(entry.object.key());
        if (record == null || !record.path.equals(entry.rel)) {
            return false;
        }
        String etag = entry.object.etag();
        if (etag != null) {
            return etag.equals(record.etag);
        }
        return record.size == entry.object.size();
    }

    private static void downloadAll(ObjectStore store, List<Entry> entries, int concurrency) throws IOException {
        ThreadFactory factory = runnable -> {
            Thread thread = new Thread(runnable, "neuralfree-remote-download");
            thread.setDaemon(true);
            return thread;
        };
        ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, concurrency), factory);
        List<Future<Void>> futures = new ArrayList<>(entries.size());
        try {
            for (Entry entry : entries) {
                futures.add(pool.submit(downloadTask(store, entry)));
            }
            IOException first = null;
            for (Future<Void> future : futures) {
                try {
                    future.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    IOException interrupted = new IOException("the object-store download was interrupted", e);
                    if (first == null) {
                        first = interrupted;
                    } else {
                        first.addSuppressed(interrupted);
                    }
                } catch (ExecutionException e) {
                    Throwable cause = e.getCause();
                    IOException failure = cause instanceof IOException ? (IOException) cause
                            : new IOException("the object-store download failed: " + cause, cause);
                    if (first == null) {
                        first = failure;
                    } else {
                        first.addSuppressed(failure);
                    }
                }
            }
            if (first != null) {
                for (Future<Void> future : futures) {
                    future.cancel(true);
                }
                throw first;
            }
        } finally {
            pool.shutdownNow();
        }
    }

    private static Callable<Void> downloadTask(ObjectStore store, Entry entry) {
        return () -> {
            Path parent = entry.local.getParent();
            Files.createDirectories(parent);
            Path partial = parent.resolve(entry.local.getFileName().toString() + PARTIAL_SUFFIX + "-"
                    + Long.toHexString(System.nanoTime()));
            try {
                store.download(entry.object.key(), partial);
                long size = Files.size(partial);
                if (entry.object.size() >= 0 && size != entry.object.size()) {
                    throw new IOException("the object '" + entry.object.key() + "' downloaded " + size
                            + " bytes, but the listing says " + entry.object.size());
                }
                move(partial, entry.local);
            } catch (IOException | RuntimeException e) {
                try {
                    Files.deleteIfExists(partial);
                } catch (IOException cleanup) {
                    e.addSuppressed(cleanup);
                }
                throw e;
            }
            return null;
        };
    }

    private static void move(Path from, Path to) throws IOException {
        try {
            Files.move(from, to, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(from, to, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /** Deletes the files the previous listing had and this one does not, and their empty directories. */
    private static void removeStale(Path root, List<Entry> entries, Manifest previous) throws IOException {
        Set<String> kept = new HashSet<>();
        for (Entry entry : entries) {
            kept.add(entry.rel);
        }
        for (Map.Entry<String, Manifest.Record> old : previous.records.entrySet()) {
            if (kept.contains(old.getValue().path)) {
                continue;
            }
            Path stale;
            try {
                stale = safeResolve(root, old.getValue().path);
            } catch (IllegalArgumentException e) {
                continue;
            }
            if (Files.deleteIfExists(stale)) {
                LOG.debug("removed stale mirror file {}", stale);
                Path parent = stale.getParent();
                while (parent != null && !parent.equals(root) && parent.startsWith(root)) {
                    try (java.util.stream.Stream<Path> children = Files.list(parent)) {
                        if (children.findAny().isPresent()) {
                            break;
                        }
                    }
                    Files.deleteIfExists(parent);
                    parent = parent.getParent();
                }
            }
        }
    }

    private static void writeManifest(Path root, ObjectStoreUri uri, List<Entry> entries) throws IOException {
        Files.createDirectories(root);
        Map<String, Object> objects = new LinkedHashMap<>();
        for (Entry entry : entries) {
            Map<String, Object> record = new LinkedHashMap<>();
            record.put("path", entry.rel);
            record.put("size", entry.object.size());
            record.put("etag", entry.object.etag());
            objects.put(entry.object.key(), record);
        }
        Map<String, Object> document = new LinkedHashMap<>();
        document.put("uri", uri.toString());
        document.put("objects", objects);
        Path manifest = root.resolve(MANIFEST);
        Path temporary = root.resolve(MANIFEST + PARTIAL_SUFFIX);
        Files.write(temporary, Json.toString(document, Json.Style.CANONICAL).getBytes(StandardCharsets.UTF_8));
        move(temporary, manifest);
    }

    /**
     * The path of {@code key} relative to the URI's prefix, or null when the key is not under it. A key equal to the
     * prefix is the single object the URI names, so its last segment is the relative path.
     */
    private static String relativeKey(ObjectStoreUri uri, String key) {
        String prefix = uri.prefix();
        if (prefix.isEmpty()) {
            return key;
        }
        if (key.equals(prefix)) {
            int slash = key.lastIndexOf('/');
            return slash < 0 ? key : key.substring(slash + 1);
        }
        if (key.startsWith(prefix + "/")) {
            return key.substring(prefix.length() + 1);
        }
        return null;
    }

    /**
     * The path of the relative name {@code rel} inside {@code root}. The name must be relative, hold no NUL, no
     * backslash and no {@code ..} segment, and resolve under {@code root}.
     *
     * @throws IllegalArgumentException with the reason when the name breaks a rule
     */
    static Path safeResolve(Path root, String rel) {
        Objects.requireNonNull(root, "root");
        Objects.requireNonNull(rel, "rel");
        if (rel.isEmpty()) {
            throw new IllegalArgumentException("the object key maps to an empty path");
        }
        if (rel.indexOf('\0') >= 0) {
            throw new IllegalArgumentException("the object key holds a NUL character");
        }
        if (rel.indexOf('\\') >= 0) {
            throw new IllegalArgumentException("the object key holds a backslash");
        }
        if (rel.startsWith("/")) {
            throw new IllegalArgumentException("the object key is absolute");
        }
        if (rel.length() >= 2 && rel.charAt(1) == ':' && Character.isLetter(rel.charAt(0))) {
            throw new IllegalArgumentException("the object key is an absolute Windows path");
        }
        for (String segment : rel.split("/", -1)) {
            if (segment.equals("..")) {
                throw new IllegalArgumentException("the object key holds a '..' segment");
            }
        }
        Path relative;
        try {
            relative = Paths.get(rel);
        } catch (InvalidPathException e) {
            throw new IllegalArgumentException("the object key is not a valid path (" + e.getReason() + ")", e);
        }
        if (relative.isAbsolute() || relative.getRoot() != null) {
            throw new IllegalArgumentException("the object key is absolute");
        }
        Path base = root.toAbsolutePath().normalize();
        Path resolved = base.resolve(relative).normalize();
        if (!resolved.startsWith(base) || resolved.equals(base)) {
            throw new IllegalArgumentException("the object key does not resolve to a file inside '" + base + "'");
        }
        return resolved;
    }

    private static boolean isSingleCheckpointFile(String rel) {
        String lower = rel.toLowerCase(Locale.ROOT);
        return lower.endsWith(".safetensors") || lower.endsWith(".tensors") || lower.endsWith(".gguf")
                || lower.endsWith(".bin") || lower.endsWith(".pt") || lower.endsWith(".pth");
    }

    private static Set<String> modelSuffixes() {
        Set<String> out = new HashSet<>();
        Collections.addAll(out, ".json", ".safetensors", ".tensors", ".gguf", ".bin", ".pt", ".pth", ".model",
                ".txt", ".tiktoken", ".jinja", ".spm", ".vocab", ".merges");
        return Collections.unmodifiableSet(out);
    }

    /** One listed object chosen for the mirror. */
    private static final class Entry {
        final RemoteObject object;
        final String rel;
        final Path local;

        Entry(RemoteObject object, String rel, Path local) {
            this.object = object;
            this.rel = rel;
            this.local = local;
        }
    }

    /**
     * The files a checkpoint mirror keeps: the configurations, tokenizers, indexes and weights of a Hugging Face,
     * Mistral native, GGUF or PyTorch checkpoint, and the side files of a sentence-transformers or PEFT checkpoint.
     * Git metadata and the {@code original/} copies are never kept.
     */
    static final class ModelFiles {

        static final String DESCRIPTION = "*.safetensors, *.tensors, *.gguf, *.bin, *.pt, *.pth, *.json, *.model,"
                + " *.txt, *.tiktoken, *.jinja, *.spm, *.vocab and *.merges files (skipping original/ and .git/)";

        private ModelFiles() {
        }

        /** Whether the relative object name is a model file the mirror keeps. */
        static boolean accepts(String rel) {
            String lower = rel.toLowerCase(Locale.ROOT);
            if (lower.startsWith("original/") || lower.startsWith(".git/") || lower.equals("original")) {
                return false;
            }
            int dot = lower.lastIndexOf('.');
            if (dot <= lower.lastIndexOf('/')) {
                return false;
            }
            return MODEL_SUFFIXES.contains(lower.substring(dot));
        }
    }

    /** The previous mirror listing, read from {@value #MANIFEST}; an absent or unreadable file is an empty listing. */
    private static final class Manifest {
        static final Manifest EMPTY = new Manifest(Collections.emptyMap());
        final Map<String, Record> records;

        Manifest(Map<String, Record> records) {
            this.records = records;
        }

        static Manifest read(Path file) {
            if (!Files.isRegularFile(file)) {
                return EMPTY;
            }
            Map<String, Record> records = new LinkedHashMap<>();
            try {
                Object parsed = Json.parse(file, 16L << 20, Json.Limits.LENIENT);
                if (parsed instanceof Map) {
                    Object objects = ((Map<?, ?>) parsed).get("objects");
                    if (objects instanceof Map) {
                        for (Map.Entry<?, ?> e : ((Map<?, ?>) objects).entrySet()) {
                            if (!(e.getKey() instanceof String) || !(e.getValue() instanceof Map)) {
                                continue;
                            }
                            Map<?, ?> value = (Map<?, ?>) e.getValue();
                            Object path = value.get("path");
                            Object size = value.get("size");
                            Object etag = value.get("etag");
                            if (path instanceof String) {
                                records.put((String) e.getKey(), new Record((String) path,
                                        size instanceof Long ? (Long) size : -1L,
                                        etag instanceof String ? (String) etag : null));
                            }
                        }
                    }
                }
            } catch (IOException | RuntimeException e) {
                LOG.warn("ignoring the unreadable mirror manifest {}: {}", file, e.getMessage());
                return EMPTY;
            }
            return new Manifest(records);
        }

        static final class Record {
            final String path;
            final long size;
            final String etag;

            Record(String path, long size, String etag) {
                this.path = path;
                this.size = size;
                this.etag = etag;
            }
        }
    }
}
