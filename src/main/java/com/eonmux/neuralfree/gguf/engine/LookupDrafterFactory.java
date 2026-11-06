package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import com.eonmux.neuralfree.gguf.engine.draft.HistoryDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.NgramCache;
import com.eonmux.neuralfree.gguf.engine.draft.NgramCacheFile;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.SuffixDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.SuffixRequestCache;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The lookup drafters an {@link EngineConfig} selects ({@link EngineConfig#lookupTypes()}), as a source of unbound
 * drafter sets: {@link NgramDrafter} for the n-gram types and {@link SuffixDrafter} for {@code suffix}, in
 * llama.cpp's ask order, built from the config's {@link SpeculativeOptions}. The {@code ngram-cache} files are read
 * once, when the factory opens, as llama.cpp reads them (common/speculative.cpp
 * {@code common_speculative_impl_ngram_cache}): the static file must exist; a dynamic file that does not exist yet
 * starts empty. Every drafter set {@link #newDrafters()} builds shares the loaded caches read-only, so the single
 * sequence of {@code LLMModelFactory.speculate} takes one set and the batch engine one set per sequence slot;
 * {@link #saveDynamic} writes the dynamic file with the n-grams every set learned. The batch engine's {@code suffix}
 * drafters share one request tree of past responses ({@link #sharedSuffixRequests()}), as vLLM's suffix proposer
 * keeps one cache for the whole batch. Immutable.
 */
public final class LookupDrafterFactory {

    private static final Logger LOGGER = LogManager.getLogger(LookupDrafterFactory.class);

    private final List<DraftType> types;
    private final SpeculativeOptions options;
    private final NgramCache staticCache;
    private final NgramCache dynamicCache;
    private final Path dynamicFile;

    private LookupDrafterFactory(List<DraftType> types, SpeculativeOptions options, NgramCache staticCache,
                                 NgramCache dynamicCache, Path dynamicFile) {
        this.types = Collections.unmodifiableList(new ArrayList<>(types));
        this.options = options;
        this.staticCache = staticCache;
        this.dynamicCache = dynamicCache;
        this.dynamicFile = dynamicFile;
    }

    /**
     * The factory of {@code cfg}'s lookup types, reading the lookup cache files when {@code ngram-cache} is one of
     * them; empty ({@link #isEmpty()}) when the config selects no lookup type.
     *
     * @throws FileNotFoundException when the static lookup cache file does not exist
     * @throws IOException           when a lookup cache file cannot be read, naming its property
     */
    public static LookupDrafterFactory open(EngineConfig cfg) throws IOException {
        Objects.requireNonNull(cfg, "cfg");
        SpeculativeOptions options = cfg.speculativeOptions();
        List<DraftType> types = cfg.lookupTypes();
        NgramCache staticCache = null;
        NgramCache dynamicCache = null;
        Path dynamicFile = null;
        if (types.contains(DraftType.NGRAM_CACHE)) {
            if (options.lookupCacheStatic() != null) {
                staticCache = loadStatic(options.lookupCacheStatic());
            }
            if (options.lookupCacheDynamic() != null) {
                dynamicFile = Paths.get(options.lookupCacheDynamic());
                dynamicCache = loadDynamic(options.lookupCacheDynamic());
            }
        }
        return new LookupDrafterFactory(types, options, staticCache, dynamicCache, dynamicFile);
    }

    /** The lookup types, in ask order. */
    public List<DraftType> types() {
        return types;
    }

    /** The llama.cpp names of {@link #types()}, in the same order. */
    public List<String> names() {
        List<String> names = new ArrayList<>(types.size());
        for (DraftType t : types) {
            names.add(t.llamaName());
        }
        return names;
    }

    /** Whether no lookup type is configured. */
    public boolean isEmpty() {
        return types.isEmpty();
    }

    /** Whether the {@code ngram-cache} drafters learn into a dynamic cache file ({@link #saveDynamic}). */
    public boolean savesDynamic() {
        return dynamicFile != null;
    }

    /**
     * A fresh, unbound drafter per lookup type, in ask order, over the factory's loaded caches; the {@code suffix}
     * drafter keeps a request tree of its own.
     */
    public List<HistoryDrafter> newDrafters() {
        return newDrafters(null);
    }

    /**
     * A fresh, unbound drafter per lookup type, in ask order, over the factory's loaded caches; the {@code suffix}
     * drafter feeds and drafts from {@code suffixRequests}, a tree of {@link #sharedSuffixRequests()}, or keeps one
     * of its own when it is null.
     */
    public List<HistoryDrafter> newDrafters(SuffixRequestCache suffixRequests) {
        List<HistoryDrafter> drafters = new ArrayList<>(types.size());
        for (DraftType type : types) {
            drafters.add(type == DraftType.SUFFIX ? suffixDrafter(suffixRequests) : ngramDrafter(type));
        }
        return drafters;
    }

    /**
     * A request tree for the {@code suffix} drafters of several sequences to share
     * ({@link #newDrafters(SuffixRequestCache)}), or null when no {@code suffix} type is configured or its
     * {@code suffix_decoding_max_cached_requests} is 0.
     */
    public SuffixRequestCache sharedSuffixRequests() {
        SuffixDrafter.Options o = options.suffixOptions();
        return !types.contains(DraftType.SUFFIX) || o.maxCachedRequests() == 0 ? null
                : SuffixDrafter.sharedRequests(o);
    }

    private SuffixDrafter suffixDrafter(SuffixRequestCache shared) {
        SuffixDrafter.Options o = options.suffixOptions();
        return shared == null ? SuffixDrafter.of(o) : SuffixDrafter.of(o, shared);
    }

    private NgramDrafter ngramDrafter(DraftType type) {
        NgramDrafter.Kind kind = kindOf(type);
        NgramDrafter.Options o = options.ngramOptions(kind);
        if (kind != NgramDrafter.Kind.CACHE) {
            return NgramDrafter.of(o);
        }
        NgramDrafter.Builder b = o.toBuilder();
        if (staticCache != null) {
            b.staticCache(staticCache);
        }
        if (dynamicCache != null) {
            b.dynamicCache(dynamicCache);
        }
        return NgramDrafter.of(b.build());
    }

    /** The {@link NgramDrafter} kind of an n-gram lookup type. */
    static NgramDrafter.Kind kindOf(DraftType type) {
        switch (type) {
            case NGRAM: return NgramDrafter.Kind.SIMPLE;
            case NGRAM_MAP_K: return NgramDrafter.Kind.MAP_K;
            case NGRAM_MAP_K4V: return NgramDrafter.Kind.MAP_K4V;
            case NGRAM_MOD: return NgramDrafter.Kind.MOD;
            case NGRAM_CACHE: return NgramDrafter.Kind.CACHE;
            default: throw new IllegalArgumentException("draft type " + type.llamaName() + " is no n-gram type");
        }
    }

    /**
     * Writes the dynamic lookup cache file: the cache it was loaded with plus the history n-grams of every
     * {@code ngram-cache} drafter among {@code drafters} ({@link NgramDrafter#historyNgrams()}, read after the
     * drafters closed), as llama.cpp saves the dynamic cache when its drafter ends. A no-op without a dynamic file.
     *
     * @throws IOException naming {@code neuralfree.draft.lookup.cache.dynamic} when the file cannot be written
     */
    public void saveDynamic(Collection<? extends HistoryDrafter> drafters) throws IOException {
        if (dynamicFile == null) {
            return;
        }
        NgramCache cache = new NgramCache();
        cache.merge(dynamicCache);
        for (HistoryDrafter d : drafters) {
            if (d instanceof NgramDrafter && ((NgramDrafter) d).options().kind() == NgramDrafter.Kind.CACHE) {
                cache.merge(((NgramDrafter) d).historyNgrams());
            }
        }
        LOGGER.info("saving the dynamic lookup cache ({} n-grams) to {}", cache.ngrams(), dynamicFile);
        try {
            NgramCacheFile.save(cache, dynamicFile);
        } catch (IOException e) {
            throw new IOException(SpeculativeOptions.PROP_LOOKUP_CACHE_DYNAMIC + "=" + dynamicFile + ": "
                    + e.getMessage(), e);
        }
    }

    private static NgramCache loadStatic(String path) throws IOException {
        Path file = Paths.get(path);
        if (!Files.isRegularFile(file)) {
            throw new FileNotFoundException(SpeculativeOptions.PROP_LOOKUP_CACHE_STATIC + "=" + path
                    + ": no such file");
        }
        return load(SpeculativeOptions.PROP_LOOKUP_CACHE_STATIC, file);
    }

    /** The dynamic cache file, or an empty cache when the file does not exist yet (it is written on close). */
    private static NgramCache loadDynamic(String path) throws IOException {
        Path file = Paths.get(path);
        if (Files.notExists(file)) {
            LOGGER.info("the dynamic lookup cache {} does not exist; it starts empty and is written when the"
                    + " session closes", path);
            return new NgramCache();
        }
        return load(SpeculativeOptions.PROP_LOOKUP_CACHE_DYNAMIC, file);
    }

    private static NgramCache load(String property, Path file) throws IOException {
        try {
            return NgramCacheFile.load(file);
        } catch (IOException e) {
            throw new IOException(property + "=" + file + ": " + e.getMessage(), e);
        }
    }

    @Override
    public String toString() {
        return "LookupDrafterFactory{" + String.join(",", names())
                + (dynamicFile != null ? ", dynamic=" + dynamicFile : "") + '}';
    }
}
