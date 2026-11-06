package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Lists of {@link DraftType}: llama.cpp's comma-separated {@code --spec-type a,b,c}, held in the order the drafters
 * are asked ({@link #ordered}), with at most one draft-file type. Stateless.
 */
public final class DraftTypes {

    /** The default list: the draft file's type detected from the file. */
    public static final List<DraftType> DEFAULT = Collections.singletonList(DraftType.AUTO);

    private DraftTypes() {
    }

    /**
     * A comma-separated type list ({@link DraftType#parse} names, empty entries skipped) in ask order.
     *
     * @throws IllegalArgumentException for an empty list, an invalid name, or two draft-file types
     */
    public static List<DraftType> parse(String value) {
        List<DraftType> out = new ArrayList<>();
        for (String part : Objects.requireNonNull(value, "draft types").split(",")) {
            if (!part.trim().isEmpty()) {
                out.add(DraftType.parse(part));
            }
        }
        return ordered(out);
    }

    /**
     * {@code types} without repeats in llama.cpp's ask order (common/speculative.cpp {@code common_speculative_init}):
     * the lookup types first ({@code ngram-simple}, {@code ngram-map-k}, {@code ngram-map-k4v}, {@code ngram-mod},
     * {@code ngram-cache}, then {@code suffix}), the draft-file type ({@code auto}, {@code draft-mtp} or
     * {@code draft-simple}) last. Unmodifiable.
     *
     * @throws IllegalArgumentException for an empty list or two different draft-file types
     */
    public static List<DraftType> ordered(List<DraftType> types) {
        boolean[] seen = new boolean[DraftType.values().length];
        DraftType file = null;
        for (DraftType t : Objects.requireNonNull(types, "draft types")) {
            seen[Objects.requireNonNull(t, "draft type").ordinal()] = true;
            if (!t.lookupWithoutDraftModel()) {
                if (file != null && file != t) {
                    throw new IllegalArgumentException("draft types " + file.llamaName() + " and " + t.llamaName()
                            + " both run the draft file; a target binds one draft-file drafter");
                }
                file = t;
            }
        }
        List<DraftType> out = new ArrayList<>();
        for (DraftType t : DraftType.values()) {
            if (seen[t.ordinal()] && t.lookupWithoutDraftModel()) {
                out.add(t);
            }
        }
        if (file != null) {
            out.add(file);
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException("the draft type list is empty");
        }
        return Collections.unmodifiableList(out);
    }

    /** The draft-file member of an {@link #ordered} list, or {@link DraftType#AUTO} when it has none. */
    public static DraftType fileType(List<DraftType> ordered) {
        DraftType last = ordered.get(ordered.size() - 1);
        return last.lookupWithoutDraftModel() ? DraftType.AUTO : last;
    }

    /** The lookup members of an {@link #ordered} list, in order; unmodifiable. */
    public static List<DraftType> lookups(List<DraftType> ordered) {
        List<DraftType> out = new ArrayList<>(ordered.size());
        for (DraftType t : ordered) {
            if (t.lookupWithoutDraftModel()) {
                out.add(t);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /** The llama.cpp names of {@code types}, comma-separated as {@code --spec-type} takes them. */
    public static String names(List<DraftType> types) {
        StringBuilder sb = new StringBuilder();
        for (DraftType t : types) {
            sb.append(sb.length() == 0 ? "" : ",").append(t.llamaName());
        }
        return sb.toString();
    }
}
