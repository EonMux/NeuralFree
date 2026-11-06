package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.draft.HistoryDrafter;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The lookup drafters an {@link EngineConfig} selects, built by its {@link LookupDrafterFactory} and bound to a
 * target: {@link com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter} for the n-gram types and
 * {@link com.eonmux.neuralfree.gguf.engine.draft.SuffixDrafter} for {@code suffix}, in llama.cpp's ask order. The
 * {@code ngram-cache} dynamic file is written when the session closes ({@link #saves()}). Immutable once opened.
 */
final class LookupDrafters {

    private final List<DraftModel> drafters;
    private final List<String> names;
    private final List<AutoCloseable> saves;

    private LookupDrafters(List<DraftModel> drafters, List<String> names, List<AutoCloseable> saves) {
        this.drafters = Collections.unmodifiableList(drafters);
        this.names = Collections.unmodifiableList(names);
        this.saves = Collections.unmodifiableList(saves);
    }

    /**
     * Builds and binds the lookup drafters of {@code cfg} to {@code target}; none when it selects no lookup type.
     * A drafter that fails to bind closes the ones already built.
     *
     * @throws FileNotFoundException    when the static lookup cache file does not exist
     * @throws IOException              when a lookup cache file cannot be read, naming its property
     * @throws IllegalArgumentException when the target cannot run a lookup drafter
     */
    static LookupDrafters open(EngineModel target, EngineConfig cfg) throws IOException {
        Objects.requireNonNull(target, "target");
        LookupDrafterFactory factory = LookupDrafterFactory.open(cfg);
        List<HistoryDrafter> built = factory.newDrafters();
        try {
            for (HistoryDrafter d : built) {
                d.bindTarget(target);
            }
        } catch (RuntimeException | Error e) {
            Throwable first = e;
            for (int i = built.size() - 1; i >= 0; i--) {
                first = LLMModelFactory.closeStep(built.get(i), first);
            }
            throw e;
        }
        List<AutoCloseable> saves = factory.savesDynamic()
                ? Collections.singletonList(() -> factory.saveDynamic(built)) : Collections.emptyList();
        return new LookupDrafters(new ArrayList<>(built), factory.names(), saves);
    }

    /** The bound drafters, in ask order. */
    List<DraftModel> drafters() {
        return drafters;
    }

    /** The llama.cpp type names of {@link #drafters()}, in the same order. */
    List<String> names() {
        return names;
    }

    /**
     * The dynamic lookup cache writes, to run after the drafters are closed: the dynamic file with the
     * n-grams the {@code ngram-cache} drafter learned ({@link LookupDrafterFactory#saveDynamic}).
     */
    List<AutoCloseable> saves() {
        return saves;
    }

    /** Whether no lookup type is configured. */
    boolean isEmpty() {
        return drafters.isEmpty();
    }
}
