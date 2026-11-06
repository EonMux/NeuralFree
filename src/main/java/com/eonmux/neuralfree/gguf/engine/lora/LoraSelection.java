package com.eonmux.neuralfree.gguf.engine.lora;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The per-request adapters of one generation request, by their {@link LoraRegistry} names: vLLM's
 * {@code lora_request} (one loaded adapter, named as the request's model) or llama.cpp server's per-request
 * {@code lora} list ({@code [{"id", "scale"}]}, server-context.cpp:1952-1966), where every listed adapter runs at its
 * scale. An entry without a scale runs at the adapter's default ({@link LoraHandle#scale()} when the request is
 * submitted); a scale of 0 leaves the adapter out, as llama.cpp's {@code construct_lora_list} disables it.
 *
 * <p>Immutable; build one with {@link #of(String)} or {@link #builder()}. Names are distinct.</p>
 */
public final class LoraSelection {

    /** One selected adapter: its registry name and its scale, NaN for the adapter's default. Immutable. */
    public static final class Entry {
        private final String name;
        private final float scale;

        private Entry(String name, float scale) {
            this.name = name;
            this.scale = scale;
        }

        /** The registry name of the adapter. */
        public String name() {
            return name;
        }

        /** The scale the request runs the adapter at, or NaN for the adapter's default. */
        public float scale() {
            return scale;
        }

        /** True when the request takes the adapter's default scale. */
        public boolean defaultScale() {
            return Float.isNaN(scale);
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Entry)) {
                return false;
            }
            Entry e = (Entry) o;
            return name.equals(e.name) && Float.floatToIntBits(scale) == Float.floatToIntBits(e.scale);
        }

        @Override
        public int hashCode() {
            return 31 * name.hashCode() + Float.floatToIntBits(scale);
        }

        @Override
        public String toString() {
            return defaultScale() ? name : name + ":" + scale;
        }
    }

    private final List<Entry> entries;

    private LoraSelection(List<Entry> entries) {
        this.entries = Collections.unmodifiableList(new ArrayList<>(entries));
    }

    /** The one adapter {@code name} at its default scale (vLLM's per-request {@code lora_request}). */
    public static LoraSelection of(String name) {
        return builder().add(name).build();
    }

    /** An empty builder. */
    public static Builder builder() {
        return new Builder();
    }

    /** The selected adapters in the order they were added. */
    public List<Entry> entries() {
        return entries;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof LoraSelection && entries.equals(((LoraSelection) o).entries);
    }

    @Override
    public int hashCode() {
        return entries.hashCode();
    }

    @Override
    public String toString() {
        return "LoraSelection" + entries;
    }

    /** A builder of an immutable {@link LoraSelection}. */
    public static final class Builder {
        private final List<Entry> entries = new ArrayList<>();

        /** Adds the adapter {@code name} at its default scale. */
        public Builder add(String name) {
            return put(name, Float.NaN);
        }

        /**
         * Adds the adapter {@code name} at {@code scale}.
         *
         * @throws IllegalArgumentException for a blank or repeated name or a non-finite scale
         */
        public Builder add(String name, float scale) {
            if (!Float.isFinite(scale)) {
                throw new IllegalArgumentException("LoRA adapter " + name + " has a non-finite scale " + scale);
            }
            return put(name, scale);
        }

        private Builder put(String name, float scale) {
            if (name == null || name.trim().isEmpty()) {
                throw new IllegalArgumentException("a selected LoRA adapter needs a name, got '" + name + "'");
            }
            for (Entry e : entries) {
                if (e.name.equals(name)) {
                    throw new IllegalArgumentException("LoRA adapter " + name + " is selected twice");
                }
            }
            entries.add(new Entry(name, scale));
            return this;
        }

        /**
         * Builds the selection.
         *
         * @throws IllegalStateException when no adapter was added
         */
        public LoraSelection build() {
            if (entries.isEmpty()) {
                throw new IllegalStateException("a LoRA selection names at least one adapter");
            }
            return new LoraSelection(entries);
        }
    }
}
