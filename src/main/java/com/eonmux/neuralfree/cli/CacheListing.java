package com.eonmux.neuralfree.cli;

import com.eonmux.neuralfree.gguf.hub.HubCache;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * The hub cache listing behind {@code --cache-list} ({@code -cl}), the mirror of llama.cpp's
 * {@code common_list_cached_models} ({@code common/arg.cpp}): the model files of every cached
 * repository, one entry per {@code repository:quant-tag} pair — the first shard or the unsplit file
 * of each tag, never a sidecar (the {@link HubCache#isModel} split) and never a file without a tag.
 * The printed form is llama.cpp's: a count line, then one {@code <n>. <repo>:<tag>} line per model.
 */
public final class CacheListing {

    /** One cached model: the repository id and its quant tag, llama.cpp's cached_model_info. */
    public static final class CachedModel {

        private final String repo;
        private final String tag;

        CachedModel(String repo, String tag) {
            this.repo = repo;
            this.tag = tag;
        }

        /** The repository id ({@code user/model}). */
        public String repo() {
            return repo;
        }

        /** The quant tag of the model file ({@code Q4_K_M}), uppercased as the cache parses it. */
        public String tag() {
            return tag;
        }

        /** The {@code repo:tag} form the listing prints. */
        @Override
        public String toString() {
            return repo + ":" + tag;
        }
    }

    private CacheListing() {
    }

    /**
     * The distinct cached models of {@code cache}, in the cache's listing order (repository folders
     * sorted, snapshot paths sorted); an unreadable repository contributes nothing, as the listing
     * skips layouts it cannot resolve.
     */
    public static List<CachedModel> models(HubCache cache) throws IOException {
        Set<String> seen = new LinkedHashSet<>();
        List<CachedModel> out = new ArrayList<>();
        for (HubCache.CachedFile f : cache.cachedFiles(null)) {
            if (!isListedModel(f)) {
                continue;
            }
            CachedModel model = new CachedModel(f.repoId(), f.filename().tag());
            if (seen.add(model.toString())) {
                out.add(model);
            }
        }
        return out;
    }

    /** Whether a cached file lists a model under its tag: a sidecar-free tagged GGUF, first shard only. */
    private static boolean isListedModel(HubCache.CachedFile f) {
        HubCache.GgufFilename name = f.filename();
        if (name == null || !HubCache.isModel(f.repoPath()) || name.tag().isEmpty()) {
            return false;
        }
        return !name.isShard() || name.index() == 1;
    }

    /** The listing llama.cpp's {@code --cache-list} prints: the count line, then one line per model. */
    public static String lines(List<CachedModel> models) {
        StringBuilder sb = new StringBuilder();
        sb.append("number of models in cache: ").append(models.size()).append('\n');
        for (int i = 0; i < models.size(); i++) {
            sb.append(String.format("%4d. %s%n", i + 1, models.get(i)));
        }
        return sb.toString();
    }

    /** Prints the listing of {@code cache} to {@code out}. */
    public static void print(Appendable out, HubCache cache) throws IOException {
        out.append(lines(models(cache)));
    }
}
