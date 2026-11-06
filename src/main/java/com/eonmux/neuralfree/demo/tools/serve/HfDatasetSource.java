package com.eonmux.neuralfree.demo.tools.serve;

import com.eonmux.neuralfree.gguf.hub.HubCache;
import com.eonmux.neuralfree.gguf.hub.HubClient;
import com.eonmux.neuralfree.gguf.hub.HubRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Where an {@code hf} dataset's rows come from: a local file or dataset directory as given, or a
 * Hugging Face Hub dataset repository downloaded through NeuralFree's {@link HubClient} into the
 * Hub cache (the {@code datasets--user--name} layout, the {@code HF_TOKEN} or token-file access, and
 * {@code HF_HUB_OFFLINE} reading the cache). Only data files are fetched ({@code .parquet},
 * {@code .jsonl}, {@code .json}, {@code .csv}, {@code .tsv}), selected by the requested subset and
 * split; when the network is unavailable the cache supplies the snapshot as llama.cpp's downloader
 * falls back to it. The reference's default subset and split ({@link HfRepoCatalog}) apply when the
 * flags are absent.
 */
public final class HfDatasetSource {

    /** The row source and the subset and split selected for it. */
    public static final class Source {

        final Path path;
        final String subset;
        final String split;

        Source(Path path, String subset, String split) {
            this.path = path;
            this.subset = subset;
            this.split = split;
        }

        /** The local file or snapshot directory holding the rows. */
        public Path path() {
            return path;
        }

        /** The subset selected, or null when the rows sit at the source's root. */
        public String subset() {
            return subset;
        }

        /** The split selected, or null when every split is read. */
        public String split() {
            return split;
        }
    }

    private static final List<String> DATA_EXTENSIONS = List.of(".parquet", ".jsonl", ".json", ".csv", ".tsv");

    private HfDatasetSource() {
    }

    /**
     * The rows' source: the existing local path, else the Hub dataset repository named by
     * {@code --dataset-path} or {@code --hf-name}, downloaded into the Hub cache or read from it
     * under {@code HF_HUB_OFFLINE}.
     *
     * @throws IllegalArgumentException naming the missing flag or an invalid repository id
     * @throws IllegalStateException    when the repository cannot be fetched or is not cached
     */
    static Source resolve(ServeBenchOptions opts) {
        Path local = existingLocalPath(opts.datasetPath());
        if (local != null) {
            return new Source(local, opts.hfSubset(), opts.hfSplit());
        }
        String spec = hubSpec(opts);
        HubRef ref;
        try {
            ref = HubRef.parseDataset(spec);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("the hf dataset needs a local --dataset-path or a Hub"
                    + " repository id: " + e.getMessage(), e);
        }
        HfRepoCatalog.Entry defaults = HfRepoCatalog.of(ref.repository());
        String subset = opts.hfSubset() != null ? opts.hfSubset()
                : defaults != null ? defaults.subset : null;
        String split = opts.hfSplit() != null ? opts.hfSplit()
                : defaults != null ? defaults.split : null;
        return fetch(ref, subset, split);
    }

    /**
     * The rows' source of a dataset expression: the existing local path, else a Hugging Face dataset
     * repository downloaded into the Hub cache (or read from it under {@code HF_HUB_OFFLINE}).
     *
     * @param spec   a local file or dataset directory, or a dataset repository id such as
     *               {@code openai/gsm8k@main}
     * @param subset the configuration directory selected, null for every configuration
     * @param split  the split selected, null for every split
     * @throws IllegalArgumentException when {@code spec} names neither a local path nor a repository
     */
    public static Source resolve(String spec, String subset, String split) {
        Path local = existingLocalPath(spec);
        if (local != null) {
            return new Source(local, subset, split);
        }
        HubRef ref;
        try {
            ref = HubRef.parseDataset(spec);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("the dataset source '" + spec + "' is neither a local"
                    + " path nor a Hub dataset repository id: " + e.getMessage(), e);
        }
        return fetch(ref, subset, split);
    }

    /**
     * The rows of a resolved source: the source file itself, or the data files of its subset and
     * split. A source whose rows sit in per-configuration subdirectories (a multi-configuration Hub
     * repository) is read recursively, in path order.
     *
     * @throws java.io.IOException when a row file cannot be read
     */
    public static List<Map<String, Object>> readRows(Source source) throws IOException {
        Path path = source.path;
        if (!Files.isDirectory(path)) {
            return BenchDatasets.readRows(path);
        }
        Path dir = path;
        if (source.subset != null) {
            Path selected = path.resolve(source.subset);
            if (!Files.isDirectory(selected)) {
                throw new IllegalArgumentException("the dataset " + path + " has no '"
                        + source.subset + "' subset directory");
            }
            dir = selected;
        }
        List<Path> files = rowFiles(dir);
        if (source.split != null) {
            List<Path> selected = new ArrayList<>();
            for (Path file : files) {
                if (inSplit(dir, file, source.split)) {
                    selected.add(file);
                }
            }
            // a file without a split marker belongs to the repository's own default split
            if (!selected.isEmpty()) {
                files = selected;
            }
        }
        if (files.isEmpty()) {
            throw new IllegalArgumentException("the dataset directory " + dir + " carries no "
                    + String.join(", ", BenchDatasets.ROW_EXTENSIONS) + " file");
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Path file : files) {
            rows.addAll(BenchDatasets.readFile(file));
        }
        return rows;
    }

    /** A directory's row files, recursively: a Hub snapshot keeps each configuration in its own directory. */
    private static List<Path> rowFiles(Path dir) throws IOException {
        List<Path> out = new ArrayList<>();
        collectRowFiles(dir, out);
        out.sort((a, b) -> a.toString().compareTo(b.toString()));
        return out;
    }

    private static void collectRowFiles(Path dir, List<Path> out) throws IOException {
        try (java.util.stream.Stream<Path> entries = Files.list(dir)) {
            for (Path entry : (Iterable<Path>) entries::iterator) {
                if (Files.isDirectory(entry)) {
                    collectRowFiles(entry, out);
                } else if (Files.isRegularFile(entry) && BenchDatasets.hasRowExtension(entry.getFileName().toString())) {
                    out.add(entry);
                }
            }
        }
    }

    /** Whether a row file belongs to the split by its name or a directory it sits in. */
    private static boolean inSplit(Path root, Path file, String split) {
        Path relative = root.relativize(file);
        for (Path segment : relative) {
            if (segment.toString().equals(split)) {
                return true;
            }
        }
        String name = file.getFileName().toString();
        return name.startsWith(split + ".") || name.startsWith(split + "-") || name.equals(split);
    }

    /** The existing local path, or null when the flag is absent or names no local file. */
    private static Path existingLocalPath(String value) {
        if (value == null) {
            return null;
        }
        Path path;
        try {
            path = Paths.get(value);
        } catch (java.nio.file.InvalidPathException e) {
            return null;
        }
        return Files.exists(path) ? path : null;
    }

    /** The Hub repository expression: the dataset path when it is a repository id, else {@code --hf-name}. */
    private static String hubSpec(ServeBenchOptions opts) {
        String path = opts.datasetPath();
        String name = opts.hfName();
        if (path != null && !path.isEmpty() && isRepoId(path)) {
            return path;
        }
        if (name != null && !name.isEmpty()) {
            return name;
        }
        if (path != null && !path.isEmpty()) {
            return path;
        }
        throw new IllegalArgumentException("--dataset-path or --hf-name must be provided for the hf"
                + " dataset");
    }

    /** Whether {@code value} parses as a dataset repository expression. */
    private static boolean isRepoId(String value) {
        try {
            HubRef.parseDataset(value);
            return true;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    /** Downloads the repository's selected data files, or reads the cached snapshot offline. */
    private static Source fetch(HubRef ref, String subset, String split) {
        HubCache cache = new HubCache(HubCache.defaultRoot());
        HubClient client = new HubClient(cache);
        String commit = null;
        List<HubClient.RepoFile> files = null;
        if (!HubClient.offline()) {
            try {
                commit = client.resolveCommit(ref);
                files = client.listFiles(ref, commit);
            } catch (IOException e) {
                files = null;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("the download of '" + ref + "' was interrupted", e);
            } catch (IllegalArgumentException e) {
                files = null;
            }
        }
        if (files == null || files.isEmpty()) {
            return cachedSnapshot(cache, ref, subset, split);
        }
        String selectedSubset = subset != null ? subset : inferSubset(files);
        List<HubClient.RepoFile> selected = select(files, selectedSubset, split);
        if (selected.isEmpty()) {
            throw new IllegalStateException("the Hub dataset '" + ref + "' carries no data file for"
                    + " subset " + (selectedSubset == null ? "<all>" : "'" + selectedSubset + "'")
                    + " and split " + (split == null ? "<all>" : "'" + split + "'")
                    + " (it has " + dataFiles(files).size() + " data files)");
        }
        Path snapshot = cache.snapshotDir(ref, commit);
        for (HubClient.RepoFile file : selected) {
            try {
                client.download(ref, commit, file);
            } catch (IOException e) {
                throw new IllegalStateException("cannot download '" + file.path() + "' of the Hub"
                        + " dataset '" + ref + "': " + e.getMessage(), e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("the download of '" + file.path() + "' was interrupted", e);
            }
        }
        return new Source(snapshot, selectedSubset, split);
    }

    /** The cached snapshot directory, or a refusal naming the repository and the offline mode. */
    private static Source cachedSnapshot(HubCache cache, HubRef ref, String subset, String split) {
        try {
            String commit = cache.cachedCommit(ref);
            if (commit == null) {
                throw new IllegalStateException("the Hub dataset '" + ref + "' is not in the hub cache"
                        + (HubClient.offline() ? " (HF_HUB_OFFLINE is set)" : "")
                        + " and cannot be downloaded");
            }
            Path snapshot = cache.snapshotDir(ref, commit);
            if (!Files.isDirectory(snapshot)) {
                throw new IllegalStateException("the Hub dataset '" + ref + "' has no cached snapshot at "
                        + snapshot);
            }
            String selectedSubset = subset;
            if (selectedSubset == null) {
                List<String> paths = new ArrayList<>();
                for (HubCache.CachedFile file : cache.cachedFilesOf(ref)) {
                    paths.add(file.repoPath());
                }
                selectedSubset = inferSubsetPaths(paths);
            }
            return new Source(snapshot, selectedSubset, split);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the cached Hub dataset '" + ref + "': "
                    + e.getMessage(), e);
        }
    }

    /** The repository's data files. */
    private static List<HubClient.RepoFile> dataFiles(List<HubClient.RepoFile> files) {
        List<HubClient.RepoFile> out = new ArrayList<>();
        for (HubClient.RepoFile file : files) {
            if (isDataFile(file.path())) {
                out.add(file);
            }
        }
        return out;
    }

    /** A data file's extension: Parquet, JSONL, JSON, CSV or TSV. */
    private static boolean isDataFile(String path) {
        String lower = path.toLowerCase(Locale.ROOT);
        for (String extension : DATA_EXTENSIONS) {
            if (lower.endsWith(extension)) {
                return true;
            }
        }
        return false;
    }

    /** The files of {@code subset} and {@code split}, in repository order. */
    private static List<HubClient.RepoFile> select(List<HubClient.RepoFile> files, String subset,
                                                   String split) {
        List<HubClient.RepoFile> candidates = new ArrayList<>();
        for (HubClient.RepoFile file : files) {
            if (!isDataFile(file.path())) {
                continue;
            }
            if (subset != null && !file.path().startsWith(subset + "/")) {
                continue;
            }
            candidates.add(file);
        }
        if (split == null) {
            return candidates;
        }
        List<HubClient.RepoFile> marked = new ArrayList<>();
        for (HubClient.RepoFile file : candidates) {
            if (inSplit(file.path(), split)) {
                marked.add(file);
            }
        }
        // a file without a split marker belongs to the repository's own default split
        return marked.isEmpty() ? candidates : marked;
    }

    /** Whether a repository path belongs to {@code split} by its file name or a directory it sits in. */
    private static boolean inSplit(String path, String split) {
        String[] segments = path.split("/");
        String name = segments[segments.length - 1];
        for (int i = 0; i < segments.length - 1; i++) {
            if (segments[i].equals(split)) {
                return true;
            }
        }
        return name.startsWith(split + ".") || name.startsWith(split + "-") || name.equals(split);
    }

    /** The single top-level directory holding the data files, or null when they sit at the root. */
    private static String inferSubset(List<HubClient.RepoFile> files) {
        List<String> paths = new ArrayList<>();
        for (HubClient.RepoFile file : files) {
            paths.add(file.path());
        }
        return inferSubsetPaths(paths);
    }

    /** {@link #inferSubset(List)} over repository paths. */
    private static String inferSubsetPaths(List<String> paths) {
        Set<String> subsets = new LinkedHashSet<>();
        boolean root = false;
        for (String path : paths) {
            if (!isDataFile(path)) {
                continue;
            }
            int slash = path.indexOf('/');
            if (slash < 0) {
                root = true;
            } else {
                subsets.add(path.substring(0, slash));
            }
        }
        return !root && subsets.size() == 1 ? subsets.iterator().next() : null;
    }
}
