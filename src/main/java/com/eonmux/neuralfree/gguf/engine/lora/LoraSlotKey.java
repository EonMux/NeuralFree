package com.eonmux.neuralfree.gguf.engine.lora;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * What one adapter slot of a {@link LoraRegistry} runs: resident adapters, each at a fixed scale, sorted by name —
 * the complete set of the slot's rows, the startup adapters included only when the key lists them. A request's
 * {@link LoraSelection} resolves to one ({@link LoraRegistry#resolve}) when it is submitted, freezing the default
 * scales it takes; requests with equal keys share a slot, so a slot is the adapter-and-scale state keyed per
 * request side. The bare key ({@link #isBare}) holds no adapter: its rows run the base model without the startup
 * adapters.
 *
 * <p>Immutable. Equality is by handle identity and scale bits: two loads of one file are different adapters in
 * process. {@link #identity()} names the content instead (digest and scale), the key under which
 * computed rows are interchangeable, which the prefix cache matches on.</p>
 */
public final class LoraSlotKey {

    /** One adapter of a slot at its scale. Immutable. */
    public static final class Part {
        private final LoraHandle handle;
        private final float scale;

        Part(LoraHandle handle, float scale) {
            this.handle = Objects.requireNonNull(handle, "handle");
            this.scale = scale;
        }

        /** The resident adapter. */
        public LoraHandle handle() {
            return handle;
        }

        /** The scale the slot runs it at. */
        public float scale() {
            return scale;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Part)) {
                return false;
            }
            Part p = (Part) o;
            return handle == p.handle && Float.floatToIntBits(scale) == Float.floatToIntBits(p.scale);
        }

        @Override
        public int hashCode() {
            return 31 * System.identityHashCode(handle) + Float.floatToIntBits(scale);
        }

        @Override
        public String toString() {
            return handle.name() + ":" + scale;
        }
    }

    private static final LoraSlotKey BARE = new LoraSlotKey(Collections.<Part>emptyList());

    private final List<Part> parts;

    LoraSlotKey(List<Part> parts) {
        List<Part> sorted = new ArrayList<>(parts);
        sorted.sort(Comparator.comparing(p -> p.handle.name()));
        this.parts = Collections.unmodifiableList(sorted);
    }

    /** The key of no adapter: the base model without the startup adapters. */
    public static LoraSlotKey bare() {
        return BARE;
    }

    /** The adapters of the slot, sorted by name. */
    public List<Part> parts() {
        return parts;
    }

    /** True when the key holds no adapter. */
    public boolean isBare() {
        return parts.isEmpty();
    }

    /**
     * The activated LoRA (aLoRA) of the key, or null when it holds none: a key holds at most one, alone
     * ({@link LoraRegistry#resolve}).
     */
    public LoraHandle activated() {
        for (Part p : parts) {
            if (p.handle.adapter().isActivated()) {
                return p.handle;
            }
        }
        return null;
    }

    /** True when the slot runs {@code handle}. */
    public boolean contains(LoraHandle handle) {
        for (Part p : parts) {
            if (p.handle == handle) {
                return true;
            }
        }
        return false;
    }

    /**
     * The content identity of the slot: every adapter's content digest with the bits of its scale, in name order;
     * {@code lora} alone for the bare key. Rows computed under equal identities are equal, so the prefix cache keys
     * its entries by it.
     */
    public String identity() {
        StringBuilder sb = new StringBuilder("lora");
        for (Part p : parts) {
            sb.append('|').append(p.handle.digest()).append('*').append(Integer.toHexString(
                    Float.floatToIntBits(p.scale)));
        }
        return sb.toString();
    }

    /**
     * The adapters' registry names, comma-separated in name order: {@code sql} for vLLM's one adapter per request,
     * {@code a,b} for a llama.cpp list. KV cache events report it as vLLM's {@code lora_name}; the scales are part
     * of {@link #identity()}, not of the name.
     */
    public String label() {
        StringBuilder sb = new StringBuilder();
        for (Part p : parts) {
            if (sb.length() > 0) {
                sb.append(',');
            }
            sb.append(p.handle.name());
        }
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof LoraSlotKey && parts.equals(((LoraSlotKey) o).parts);
    }

    @Override
    public int hashCode() {
        return parts.hashCode();
    }

    @Override
    public String toString() {
        return "LoraSlotKey" + parts;
    }
}
