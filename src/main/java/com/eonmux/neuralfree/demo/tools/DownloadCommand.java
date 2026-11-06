package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.hub.HubCache;
import com.eonmux.neuralfree.gguf.hub.HubClient;
import com.eonmux.neuralfree.gguf.hub.HubRef;
import com.eonmux.neuralfree.gguf.mm.MmprojSources;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * {@code nftool download}: the Hugging Face Hub surface of the tools, the mirror of
 * llama.cpp's {@code -hf} model selection ({@code common/download.cpp:731-834}) around
 * {@link HubClient}: the repository file list is fetched online when {@code HF_HUB_OFFLINE} is not
 * set, and read back from {@link HubCache} otherwise, exactly as llama.cpp's downloader falls back to
 * its cache. From that list the tool picks the model file llama.cpp's find_best_model picks: the
 * {@code -hff} file when given, else the files whose name carries the requested quant tag (or
 * {@code Q4_K_M}, then {@code Q8_0}, then the first model file), skipping sidecars, imatrix files and
 * every shard but the first. The primary model and its split shards ({@link #splitFiles}) download
 * into the hub cache, and the tool prints the snapshot path of the primary model — the path the
 * {@code -hf} option of the demos opens as their model. Every downloaded path stays inside the cache
 * root and is verified (the cache's path confinement and the LFS sha256 pointer), {@code HF_TOKEN} or the
 * token file of {@code HF_TOKEN_PATH} / {@code HF_HOME/token} reaches the client's {@code Authorization}
 * header and is never printed, and the repository listing is
 * untrusted input parsed under the untrusted JSON limits.
 */
public final class DownloadCommand {

    /** An invalid command line or an unresolvable download; the message says which. */
    public static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private static final String USAGE =
            "nftool download [<user>/<model>[:quant][@revision]] [options]\n"
            + "options:\n"
            + "  -hf, -hfr, --hf-repo <user>/<model>[:quant]  the repository, as llama.cpp's -hf\n"
            + "                                               takes it; quant defaults to Q4_K_M\n"
            + "                                               (then Q8_0, then the first model file)\n"
            + "  -hff, --hf-file <file>        the repository file to download, in place of the quant tag\n"
            + "  -cl, --cache-list             show list of models in cache and exit\n"
            + "environment:\n"
            + "  HF_HUB_OFFLINE=1              use the hub cache only; an uncached repository is an error\n"
            + "  HF_TOKEN                      access token, sent in the Authorization header, never printed;\n"
            + "                                when it is unset, the token file HF_TOKEN_PATH names, else\n"
            + "                                HF_HOME/token (what `hf auth login` writes), is sent\n"
            + "  HF_HUB_DISABLE_IMPLICIT_TOKEN send no token from HF_TOKEN or the token file\n"
            + "  LLAMA_CACHE, HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE, HF_HOME,   the cache root\n"
            + "  XDG_CACHE_HOME                 (llama.cpp's precedence; see HubCache.defaultRoot)\n"
            + "  --help, -h                    this text\n";

    private DownloadCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code download}. */
    static void run(String[] args) {
        try {
            String[] spec = specOf(args);
            if (spec == null) {
                return;
            }
            System.out.println("nftool download: " + downloadModel(spec[0], spec[1]));
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Error: download interrupted");
            System.exit(1);
        } catch (IOException | IllegalArgumentException | IllegalStateException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** The repository expression and the {@code -hff} file of the arguments. */
    private static String[] specOf(String[] args) throws UsageException {
        String spec = null;
        String file = null;
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.println(USAGE);
                return null;
            }
            if (a.equals("-cl") || a.equals("--cache-list")) {
                try {
                    com.eonmux.neuralfree.cli.CacheListing.print(System.out,
                            new HubCache(HubCache.defaultRoot()));
                } catch (IOException | IllegalStateException | IllegalArgumentException e) {
                    System.err.println("Error: " + e.getMessage());
                    System.exit(1);
                }
                return null;
            }
            if (a.equals("-hf") || a.equals("-hfr") || a.equals("--hf-repo")) {
                spec = value(a, args, i++);
            } else if (a.equals("-hff") || a.equals("--hf-file")) {
                file = value(a, args, i++);
            } else if (a.startsWith("-")) {
                throw new UsageException("unknown option " + a + " (run with --help for the options)");
            } else if (spec == null) {
                spec = a;
            } else {
                throw new UsageException("too many repository arguments; '" + a + "' follows " + spec);
            }
        }
        if (spec == null) {
            throw new UsageException("download needs a repository, as -hf takes it");
        }
        return new String[]{spec, file};
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /**
     * Resolves {@code <user>/<model>[:quant][@revision]} over the Hub or the cache and returns the
     * snapshot paths of the primary model, printing one hub line per file; the demos' {@code -hf}
     * calls this with the parsed {@link HubRef}. The download runs here, on the caller's thread; the
     * tests only cover {@link HubRef} and the cache paths, so no test starts one.
     */
    public static String downloadModel(String spec, String hfFile)
            throws IOException, InterruptedException, UsageException {
        return downloadModel(spec, hfFile, null);
    }

    /**
     * {@link #downloadModel(String, String)} carrying {@code hfToken} instead of the environment's
     * token (llama-bench's {@code -hft}); a null token keeps the environment's token.
     */
    public static String downloadModel(String spec, String hfFile, String hfToken)
            throws IOException, InterruptedException, UsageException {
        HubRef ref;
        try {
            ref = HubRef.parse(spec);
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
        HubCache cache = new HubCache(HubCache.defaultRoot());
        HubClient client = hfToken == null ? new HubClient(cache) : new HubClient(cache, hfToken);
        List<HubClient.RepoFile> files = new ArrayList<>();
        String commit = null;
        if (!HubClient.offline()) {
            try {
                commit = client.resolveCommit(ref);
                files.addAll(client.listFiles(ref, commit));
            } catch (IOException | IllegalArgumentException e) {
                // llama.cpp: a repository that resolves to nothing reads back from the cache below
            }
        }
        if (files.isEmpty()) {
            String cached = cachedModelPath(cache, ref, hfFile);
            if (cached == null) {
                throw new IOException("no model file of '" + ref + "' is in the repository or the"
                        + " hub cache" + (HubClient.offline() ? " (HF_HUB_OFFLINE)" : ""));
            }
            return cached;
        }
        boolean explicitFile = hfFile != null && !hfFile.isEmpty();
        HubClient.RepoFile primary = explicitFile ? null : bestModel(files, ref.quant());
        if (explicitFile) {
            for (HubClient.RepoFile f : files) {
                if (f.path().equals(hfFile)) {
                    primary = f;
                    break;
                }
            }
            if (primary == null) {
                throw new IOException("file '" + hfFile + "' is not in the repository '" + ref + "'");
            }
            List<HubClient.RepoFile> single = new ArrayList<>();
            single.add(primary);
            return download(client, ref, commit, single);
        }
        if (primary == null) {
            throw new IOException("no GGUF model file is in the repository '" + ref + "'"
                    + (ref.quant() != null ? " for the quant tag " + ref.quant() : ""));
        }
        return download(client, ref, commit, splitFiles(files, primary));
    }

    /**
     * The mmproj sidecar of the repository, llama.cpp's download plan (common/download.cpp): the
     * {@code *mmproj*.gguf} file whose name pairs with the primary model downloads beside it, and the
     * cache answers when the Hub is unreachable, as {@link #downloadModel} reads the model back. Null
     * when no pairing sidecar is in the repository or the cache — the {@code --mmproj-auto} search then
     * finds none and the model runs text-only.
     */
    public static Path downloadMmproj(String spec, String hfFile)
            throws IOException, InterruptedException, UsageException {
        HubRef ref;
        try {
            ref = HubRef.parse(spec);
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
        HubCache cache = new HubCache(HubCache.defaultRoot());
        HubClient client = new HubClient(cache);
        List<HubClient.RepoFile> files = new ArrayList<>();
        String commit = null;
        if (!HubClient.offline()) {
            try {
                commit = client.resolveCommit(ref);
                files.addAll(client.listFiles(ref, commit));
            } catch (IOException | IllegalArgumentException e) {
                // llama.cpp: a repository that resolves to nothing reads back from the cache below
            }
        }
        if (!files.isEmpty()) {
            HubClient.RepoFile primary = primaryOf(files, ref, hfFile);
            if (primary != null) {
                return siblingDownload(client, ref, commit, files, primary.path(), ref.quant());
            }
        }
        return cachedMmproj(cache, ref, hfFile);
    }

    /** The primary model file of the request: the {@code -hff} file, else the best model of the quant tag. */
    private static HubClient.RepoFile primaryOf(List<HubClient.RepoFile> files, HubRef ref, String hfFile) {
        if (hfFile != null && !hfFile.isEmpty()) {
            for (HubClient.RepoFile f : files) {
                if (f.path().equals(hfFile)) {
                    return f;
                }
            }
            return null;
        }
        return bestModel(files, ref.quant());
    }

    /** Downloads the sidecar {@link MmprojSources#bestSibling} names for the primary model, or answers null. */
    private static Path siblingDownload(HubClient client, HubRef ref, String commit,
                                        List<HubClient.RepoFile> files, String primaryPath, String tag)
            throws IOException, InterruptedException {
        List<String> candidates = new ArrayList<>();
        for (HubClient.RepoFile f : files) {
            candidates.add(f.path());
        }
        String sibling = MmprojSources.bestSibling(candidates, primaryPath, tag);
        if (sibling == null) {
            return null;
        }
        for (HubClient.RepoFile f : files) {
            if (f.path().equals(sibling)) {
                Path path = client.download(ref, commit, f);
                System.out.println("hub: " + f.path() + " -> " + path);
                return path;
            }
        }
        return null;
    }

    /**
     * The snapshot path of a cached sidecar of {@code ref}, or null, as {@link #cachedModelPath} picks the
     * model: the primary of the cached files, then the sidecar pairing with it among the cached paths.
     */
    private static Path cachedMmproj(HubCache cache, HubRef ref, String hfFile) throws IOException {
        try {
            String commit = cache.cachedCommit(ref);
            if (commit == null) {
                return null;
            }
            List<HubCache.CachedFile> cached = cache.cachedFiles(ref.repository());
            boolean explicitFile = hfFile != null && !hfFile.isEmpty();
            String primary = null;
            List<String> candidates = new ArrayList<>();
            for (HubCache.CachedFile f : cached) {
                candidates.add(f.repoPath());
                if (primary == null && HubCache.isModel(f.repoPath())
                        && (explicitFile ? f.repoPath().equals(hfFile) : isPrimaryCandidate(f, ref.quant()))) {
                    primary = f.repoPath();
                }
            }
            if (primary == null) {
                return null;
            }
            String sibling = MmprojSources.bestSibling(candidates, primary, ref.quant());
            for (HubCache.CachedFile f : cached) {
                if (f.repoPath().equals(sibling)) {
                    System.out.println("hub: cached " + f.repoPath() + " -> " + f.localPath());
                    return f.localPath();
                }
            }
            return null;
        } catch (IllegalArgumentException e) {
            // an invalid repository id, commit or cached path names nothing the loader may open
            return null;
        }
    }

    /** The snapshot paths of the model files, printed one hub line per file as each resolves. */
    private static String download(HubClient client, HubRef ref, String commit,
                                   List<HubClient.RepoFile> files) throws IOException, InterruptedException {
        String first = null;
        for (HubClient.RepoFile file : files) {
            String path = client.download(ref, commit, file).toString();
            System.out.println("hub: " + file.path() + " -> " + path);
            if (first == null) {
                first = path;
            }
        }
        return first;
    }

    /**
     * The snapshot path of a cached model of {@code ref}, through the cache's own listing, or null:
     * the {@code -hff} file names its path exactly, any other request takes the best model file the
     * quant tag counts for. Only a valid repository id, a valid commit and a confined path can come
     * out of a repository folder of the layout.
     */
    private static String cachedModelPath(HubCache cache, HubRef ref, String hfFile)
            throws IOException {
        try {
            String commit = cache.cachedCommit(ref);
            if (commit == null) {
                return null;
            }
            List<HubCache.CachedFile> cached = cache.cachedFiles(ref.repository());
            boolean explicitFile = hfFile != null && !hfFile.isEmpty();
            for (HubCache.CachedFile f : cached) {
                if (!HubCache.isModel(f.repoPath())) {
                    continue;
                }
                if (explicitFile) {
                    if (f.repoPath().equals(hfFile)) {
                        System.out.println("hub: cached " + f.repoPath() + " -> " + f.localPath());
                        return f.localPath().toString();
                    }
                    continue;
                }
                if (isPrimaryCandidate(f, ref.quant())) {
                    System.out.println("hub: cached " + f.repoPath() + " -> " + f.localPath());
                    return f.localPath().toString();
                }
            }
            return null;
        } catch (IllegalArgumentException e) {
            // an invalid repository id, commit or cached path names nothing the loader may open
            return null;
        }
    }

    /** Whether a cached file is the listed model of its tag, llama.cpp's is_listed_model: the first shard of a tagged, non-sidecar GGUF. */
    private static boolean isPrimaryCandidate(HubCache.CachedFile f, String quant) {
        HubCache.GgufFilename name = f.filename();
        if (name == null || !HubCache.isModel(f.repoPath()) || name.tag().isEmpty()) {
            return false;
        }
        if (name.isShard() && name.index() != 1) {
            return false;
        }
        return quant == null || quant.isEmpty() || name.tag().equalsIgnoreCase(quant);
    }

    /**
     * The best model file llama.cpp's find_best_model selects: with a quant tag, the first model file
     * whose path carries {@code <tag>[.-]}, skipping later shards without the tag; without one, the
     * files with {@code Q4_K_M} and {@code Q8_0} in that order, then the first model file.
     */
    private static HubClient.RepoFile bestModel(List<HubClient.RepoFile> files, String tag) {
        List<String> tags = tag != null && !tag.isEmpty()
                ? List.of(tag) : List.of("Q4_K_M", "Q8_0");
        for (String t : tags) {
            Pattern pattern = Pattern.compile(Pattern.quote(t) + "[.-]",
                    Pattern.CASE_INSENSITIVE);
            for (HubClient.RepoFile f : files) {
                if (!HubCache.isModel(f.path()) || f.filename() == null
                        || (f.filename().isShard() && f.filename().index() != 1)) {
                    continue;
                }
                if (pattern.matcher(f.path()).find()) {
                    return f;
                }
            }
        }
        if (tag == null || tag.isEmpty()) {
            for (HubClient.RepoFile f : files) {
                HubCache.GgufFilename name = f.filename();
                if (!HubCache.isModel(f.path()) || name == null) {
                    continue;
                }
                if (name.isShard() && name.index() != 1) {
                    continue;
                }
                return f;
            }
        }
        return null;
    }

    /**
     * The split's parts of {@code primary}, llama.cpp's get_split_files: every listed file with the
     * same shard count and the same prefix as the primary; a primary of one file is that file.
     */
    private static List<HubClient.RepoFile> splitFiles(List<HubClient.RepoFile> files,
                                                       HubClient.RepoFile primary) {
        HubCache.GgufFilename name = primary.filename();
        if (name == null || !name.isShard()) {
            List<HubClient.RepoFile> single = new ArrayList<>();
            single.add(primary);
            return single;
        }
        List<HubClient.RepoFile> out = new ArrayList<>();
        for (HubClient.RepoFile f : files) {
            HubCache.GgufFilename other = f.filename();
            if (other != null && other.count() == name.count() && other.prefix().equals(name.prefix())) {
                out.add(f);
            }
        }
        out.sort((a, b) -> Integer.compare(a.filename().index(), b.filename().index()));
        return out;
    }
}
