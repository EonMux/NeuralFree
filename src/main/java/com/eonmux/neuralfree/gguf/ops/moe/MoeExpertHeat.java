package com.eonmux.neuralfree.gguf.ops.moe;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The routed rows counted per (layer, expert) of one loaded model (Avalon {@code moe_cache_heat_note},
 * ggml/src/ggml-cuda/moe-cache.cu:4271). The routed MoE blocks note their selections, the close of the session
 * merges the counts with the saved ranking's heats and writes them back through {@link MoeRankingFile}. The
 * recorder follows the reference's bounds: layers above {@link #MAX_NOTE_LAYER} and expert indexes outside
 * {@code [0, 65535]} are not counted, and the counts saturate at {@link RankedExpert#MAX_HEAT} like the file's
 * {@code uint32} heats, while heats merged from a file keep the file's layer range (the format holds every
 * {@code uint16} layer).
 *
 * <p>Not thread-safe: one model's forward path owns it.</p>
 */
public final class MoeExpertHeat {

    /** Largest layer the routed-row recorder counts (the reference's profiler bound). */
    public static final int MAX_NOTE_LAYER = 4095;

    private final Map<Integer, long[]> layers = new HashMap<>();

    /**
     * Counts {@code pairs} routed rows: {@code ids[0 .. pairs)} are expert indexes of {@code layer}. Ids outside
     * {@code [0, 65535]} and layers outside {@code [0, 4095]} are not counted.
     *
     * @throws IllegalArgumentException when {@code ids} is null or does not hold {@code pairs} entries
     */
    public void note(int layer, int[] ids, int pairs) {
        Objects.requireNonNull(ids, "ids");
        if (pairs < 0 || pairs > ids.length) {
            throw new IllegalArgumentException("routed rows must be in [0, ids.length = " + ids.length
                    + "], got " + pairs);
        }
        if (layer < 0 || layer > MAX_NOTE_LAYER) {
            return;
        }
        long[] counts = layers.computeIfAbsent(layer, l -> new long[0]);
        for (int i = 0; i < pairs; i++) {
            int expert = ids[i];
            if (expert < 0 || expert > RankedExpert.MAX_EXPERT) {
                continue;
            }
            if (expert >= counts.length) {
                counts = grow(counts, expert + 1);
                layers.put(layer, counts);
            }
            if (counts[expert] < RankedExpert.MAX_HEAT) {
                counts[expert]++;
            }
        }
    }

    /** Adds file heats to the counts, without the recorder's layer bound. */
    public void merge(List<RankedExpert> heats) {
        Objects.requireNonNull(heats, "heats");
        for (RankedExpert e : heats) {
            long[] counts = layers.computeIfAbsent(e.layer(), l -> new long[0]);
            if (e.expert() >= counts.length) {
                counts = grow(counts, e.expert() + 1);
                layers.put(e.layer(), counts);
            }
            counts[e.expert()] = Math.min(counts[e.expert()] + e.heat(), RankedExpert.MAX_HEAT);
        }
    }

    /** True when nothing was counted and nothing was merged. */
    public boolean isEmpty() {
        for (long[] counts : layers.values()) {
            for (long c : counts) {
                if (c > 0) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * The experts with heat, hottest first ({@link MoeRankingFile#sorted}), the heats clamped to the file's
     * {@code uint32}.
     */
    public List<RankedExpert> snapshot() {
        List<RankedExpert> out = new ArrayList<>();
        for (Map.Entry<Integer, long[]> layer : layers.entrySet()) {
            long[] counts = layer.getValue();
            for (int expert = 0; expert < counts.length; expert++) {
                if (counts[expert] > 0) {
                    out.add(new RankedExpert(layer.getKey(), expert,
                            Math.min(counts[expert], RankedExpert.MAX_HEAT)));
                }
            }
        }
        return MoeRankingFile.sorted(out);
    }

    /** Zeroes every count. */
    public void clear() {
        layers.clear();
    }

    private static long[] grow(long[] counts, int size) {
        long[] grown = new long[Math.max(size, counts.length + counts.length / 2 + 1)];
        System.arraycopy(counts, 0, grown, 0, counts.length);
        return grown;
    }
}
