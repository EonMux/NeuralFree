package com.eonmux.neuralfree.gguf.ops.moe;

import java.util.Objects;

/**
 * One expert of an expert ranking: its layer, its expert index and its heat — the routed rows counted for it
 * (Avalon {@code ggml/src/ggml-cuda/moe-cache.cu}, {@code moe_cache_ranked_expert}). Rankings are hottest first;
 * the file fields bound each part: layer and expert to unsigned 16 bits, the heat to unsigned 32 bits.
 *
 * <p>Immutable.</p>
 */
public final class RankedExpert {

    /** Largest layer a ranking can hold ({@code uint16}). */
    public static final int MAX_LAYER = 0xFFFF;
    /** Largest expert index a ranking can hold ({@code uint16}). */
    public static final int MAX_EXPERT = 0xFFFF;
    /** Largest heat a ranking can hold ({@code uint32}). */
    public static final long MAX_HEAT = 0xFFFFFFFFL;

    private final int layer;
    private final int expert;
    private final long heat;

    /**
     * An entry of a ranking.
     *
     * @throws IllegalArgumentException when the layer or the expert is outside {@code [0, 65535]} or the heat is
     *                                 outside {@code [0, 4294967295]}
     */
    public RankedExpert(int layer, int expert, long heat) {
        if (layer < 0 || layer > MAX_LAYER) {
            throw new IllegalArgumentException("ranked expert layer must be in [0, " + MAX_LAYER + "], got " + layer);
        }
        if (expert < 0 || expert > MAX_EXPERT) {
            throw new IllegalArgumentException("ranked expert index must be in [0, " + MAX_EXPERT + "], got "
                    + expert);
        }
        if (heat < 0 || heat > MAX_HEAT) {
            throw new IllegalArgumentException("ranked expert heat must be in [0, " + MAX_HEAT + "], got " + heat);
        }
        this.layer = layer;
        this.expert = expert;
        this.heat = heat;
    }

    /** The trunk layer the expert belongs to. */
    public int layer() {
        return layer;
    }

    /** The expert's index in its layer. */
    public int expert() {
        return expert;
    }

    /** The routed rows counted for the expert ({@code 0} in a version 1 ranking). */
    public long heat() {
        return heat;
    }

    @Override
    public String toString() {
        return "{layer=" + layer + ", expert=" + expert + ", heat=" + heat + "}";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        return o instanceof RankedExpert && layer == ((RankedExpert) o).layer
                && expert == ((RankedExpert) o).expert && heat == ((RankedExpert) o).heat;
    }

    @Override
    public int hashCode() {
        return Objects.hash(layer, expert, heat);
    }
}
