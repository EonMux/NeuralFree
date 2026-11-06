package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * The assignment of the routed hosted MoE layers to the GPUs whose expert caches serve them (Avalon
 * {@code GGML_MOE_CACHE_LAYER_DEVICE=1}, llama.cpp's {@code llama_moe_cache} over several backends): the layers are
 * taken in layer order, and each GPU takes a run whose expert bytes match its share of the GPUs' free VRAM. The
 * layer's hosted MoE block runs on the GPU its cache serves.
 *
 * <p>The shares are the VRAM each GPU has left after its weights and KV caches; a layer goes to the GPU whose share
 * its middle falls in (Avalon ggml-backend.cpp {@code ggml_backend_sched_prime_moe_cache}). Immutable.</p>
 */
public final class MoeCachePool {

    private final int stages;
    private final int[] ownerByIl;
    private final List<List<LayerSpec>> layersByStage;
    private final long[] bytesByStage;

    private MoeCachePool(int stages, int[] ownerByIl, List<List<LayerSpec>> layersByStage, long[] bytesByStage) {
        this.stages = stages;
        this.ownerByIl = ownerByIl;
        this.layersByStage = layersByStage;
        this.bytesByStage = bytesByStage;
    }

    /**
     * The pool over {@code stages} GPUs of {@code layers}, whose routed expert tensors take {@code layerBytes} bytes
     * each, with each GPU's share of the free VRAM.
     *
     * @param layers     the routed hosted MoE layers in layer order
     * @param layerBytes the experts' device bytes of each layer, one per layer in the same order
     * @param stageBytes the VRAM each GPU has left for its cache
     * @return the plan, or null when the pool cannot divide anything (fewer than two GPUs, no layers, no expert
     *         bytes or no positive share)
     * @throws IllegalArgumentException when the arrays disagree or a layer index is repeated
     */
    public static MoeCachePool plan(int stages, List<LayerSpec> layers, long[] layerBytes, long[] stageBytes) {
        if (layers == null || layerBytes == null || stageBytes == null) {
            throw new IllegalArgumentException("the pool needs its layers and their bytes");
        }
        if (layerBytes.length != layers.size()) {
            throw new IllegalArgumentException("the pool has " + layers.size() + " layer(s) but "
                    + layerBytes.length + " byte count(s)");
        }
        if (stages != stageBytes.length) {
            throw new IllegalArgumentException("the pool has " + stages + " GPU(s) but " + stageBytes.length
                    + " share(s)");
        }
        long total = 0L;
        int maxIl = -1;
        for (int i = 0; i < layers.size(); i++) {
            if (layerBytes[i] < 0) {
                throw new IllegalArgumentException("layer " + layers.get(i).il() + " has a negative expert size "
                        + layerBytes[i]);
            }
            total = Math.addExact(total, layerBytes[i]);
            maxIl = Math.max(maxIl, layers.get(i).il());
        }
        long shareTotal = 0L;
        for (long s : stageBytes) {
            if (s < 0) {
                throw new IllegalArgumentException("a GPU has a negative VRAM share " + s);
            }
            shareTotal = Math.addExact(shareTotal, s);
        }
        if (stages < 2 || layers.isEmpty() || total == 0L || shareTotal == 0L) {
            return null;
        }
        int[] owner = new int[maxIl + 1];
        Arrays.fill(owner, -1);
        List<List<LayerSpec>> byStage = new ArrayList<>(stages);
        long[] bytes = new long[stages];
        for (int d = 0; d < stages; d++) {
            byStage.add(new ArrayList<LayerSpec>());
        }
        int gpu = 0;
        double bound = share(stageBytes, shareTotal, 0, total);
        long done = 0L;
        for (int i = 0; i < layers.size(); i++) {
            LayerSpec layer = layers.get(i);
            if (layer.il() < 0) {
                throw new IllegalArgumentException("layer index " + layer.il() + " is negative");
            }
            long bytesOfLayer = layerBytes[i];
            while (gpu + 1 < stages && (double) done + bytesOfLayer / 2.0 > bound) {
                gpu++;
                bound += share(stageBytes, shareTotal, gpu, total);
            }
            if (owner[layer.il()] >= 0) {
                throw new IllegalArgumentException("layer " + layer.il() + " appears twice in the pool");
            }
            owner[layer.il()] = gpu;
            byStage.get(gpu).add(layer);
            bytes[gpu] = Math.addExact(bytes[gpu], bytesOfLayer);
            done += bytesOfLayer;
        }
        List<List<LayerSpec>> frozen = new ArrayList<>(stages);
        for (List<LayerSpec> l : byStage) {
            frozen.add(Collections.unmodifiableList(l));
        }
        return new MoeCachePool(stages, owner, Collections.unmodifiableList(frozen), bytes);
    }

    private static double share(long[] stageBytes, long total, int stage, long expertBytes) {
        return (double) stageBytes[stage] / (double) total * (double) expertBytes;
    }

    /** The GPUs the pool spreads the layers over. */
    public int stageCount() {
        return stages;
    }

    /** The GPU whose cache serves layer {@code il}, or -1 when the pool does not serve it. */
    public int ownerOf(int il) {
        return il >= 0 && il < ownerByIl.length ? ownerByIl[il] : -1;
    }

    /** True when the cache of GPU {@code stage} serves layer {@code il}. */
    public boolean owns(int stage, int il) {
        return stage >= 0 && stage < stages && ownerOf(il) == stage;
    }

    /** The routed layers GPU {@code stage}'s cache serves, in layer order (unmodifiable). */
    public List<LayerSpec> layersOf(int stage) {
        if (stage < 0 || stage >= stages) {
            throw new IllegalArgumentException("stage " + stage + " is outside the " + stages + " GPU(s) of the"
                    + " pool");
        }
        return layersByStage.get(stage);
    }

    /** The routed experts' bytes GPU {@code stage}'s cache holds. */
    public long bytesOf(int stage) {
        if (stage < 0 || stage >= stages) {
            throw new IllegalArgumentException("stage " + stage + " is outside the " + stages + " GPU(s) of the"
                    + " pool");
        }
        return bytesByStage[stage];
    }

    /**
     * The assignment as runs of consecutive layers, in layer order: each {@link Run} names the GPU index and the
     * layers it serves, for example {@code 0:0-3} then {@code 1:4-7}.
     */
    public List<Run> runs() {
        List<Run> out = new ArrayList<>();
        List<LayerSpec> all = new ArrayList<>();
        for (List<LayerSpec> l : layersByStage) {
            all.addAll(l);
        }
        all.sort((a, b) -> Integer.compare(a.il(), b.il()));
        int i = 0;
        while (i < all.size()) {
            int stage = ownerOf(all.get(i).il());
            int first = all.get(i).il();
            int last = first;
            i++;
            while (i < all.size() && ownerOf(all.get(i).il()) == stage && all.get(i).il() == last + 1) {
                last = all.get(i).il();
                i++;
            }
            out.add(new Run(stage, first, last));
        }
        return out;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("MoeCachePool{");
        List<Run> runs = runs();
        for (int i = 0; i < runs.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(runs.get(i));
        }
        return sb.append('}').toString();
    }

    /** A run of consecutive layers a GPU's cache serves. Immutable. */
    public static final class Run {
        private final int stage;
        private final int firstIl;
        private final int lastIl;

        Run(int stage, int firstIl, int lastIl) {
            this.stage = stage;
            this.firstIl = firstIl;
            this.lastIl = lastIl;
        }

        public int stage() {
            return stage;
        }

        public int firstIl() {
            return firstIl;
        }

        public int lastIl() {
            return lastIl;
        }

        @Override
        public String toString() {
            return stage + ":" + (firstIl == lastIl ? String.valueOf(firstIl) : firstIl + "-" + lastIl);
        }
    }
}
