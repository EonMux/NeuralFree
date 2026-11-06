package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * The device mapping of a rank's routed experts from logical ids to physical replica slots (vLLM's
 * {@code logical_to_physical_map} and {@code logical_replica_count}): per weight layer, for every logical expert, the
 * global physical slots that reproduce it and how many there are.
 *
 * <p>The routed block remaps each selected pair with {@code LLM_MoE_EplbMap}: a pair of token {@code t} selecting
 * logical expert {@code e} becomes physical slot {@code logicalToPhysical[e][hash(t) % replicaCount[e]]}, so exactly
 * one of the expert's replicas computes it. The expert kernels then address physical slots: a rank's bank holds the
 * contiguous range {@code [rank * slots, (rank + 1) * slots)} and the pairs of other ranks' slots are zeroed before
 * the group's sum. The buffers belong to the expert uploader; this object only names them for the forward path.
 * Immutable.</p>
 */
public final class MoeExpertRemap {

    /** One layer's mapping: logical slots per expert and replica counts. */
    public static final class Layer {
        private final GpuBuffer logicalToPhysical;
        private final GpuBuffer replicaCount;
        private final int logical;
        private final int physical;
        private final int maxReplicas;

        /**
         * @param logicalToPhysical device I32 {@code [logical * maxReplicas]} of physical slots, -1 where unused
         * @param replicaCount      device I32 {@code [logical]} of replica counts, at least one
         */
        public Layer(GpuBuffer logicalToPhysical, GpuBuffer replicaCount, int logical, int physical,
                     int maxReplicas) {
            this.logicalToPhysical = Objects.requireNonNull(logicalToPhysical, "logicalToPhysical");
            this.replicaCount = Objects.requireNonNull(replicaCount, "replicaCount");
            if (logical < 1 || physical < logical || maxReplicas < 1) {
                throw new IllegalArgumentException("expert remap needs 1 <= logical <= physical and a replica count,"
                        + " got logical " + logical + ", physical " + physical + ", max replicas " + maxReplicas);
            }
            this.logical = logical;
            this.physical = physical;
            this.maxReplicas = maxReplicas;
        }

        /** Device I32 {@code [logical * maxReplicas]}: the physical slots of each logical expert, -1 unused. */
        public GpuBuffer logicalToPhysical() { return logicalToPhysical; }

        /** Device I32 {@code [logical]}: the physical replica count of each logical expert. */
        public GpuBuffer replicaCount() { return replicaCount; }

        /** The logical expert count (the router's width). */
        public int logical() { return logical; }

        /** The physical slot count of the rank's bank. */
        public int physical() { return physical; }

        /** The largest replica count of a logical expert. */
        public int maxReplicas() { return maxReplicas; }
    }

    private final Map<Integer, Layer> byLayer;

    private MoeExpertRemap(Map<Integer, Layer> byLayer) {
        this.byLayer = Collections.unmodifiableMap(new LinkedHashMap<>(byLayer));
    }

    /** A remap from per-weight-layer mappings. */
    public static MoeExpertRemap of(Map<Integer, Layer> layers) {
        Objects.requireNonNull(layers, "layers");
        for (Map.Entry<Integer, Layer> e : layers.entrySet()) {
            if (e.getKey() == null || e.getValue() == null) {
                throw new IllegalArgumentException("a remapped layer needs a weight layer and a mapping");
            }
        }
        return new MoeExpertRemap(layers);
    }

    /** True when {@code weightLayer}'s routed pairs take the replica mapping. */
    public boolean covers(int weightLayer) {
        return byLayer.containsKey(weightLayer);
    }

    /**
     * The mapping of {@code weightLayer}.
     *
     * @throws IllegalArgumentException when the layer has no mapping
     */
    public Layer layer(int weightLayer) {
        Layer layer = byLayer.get(weightLayer);
        if (layer == null) {
            throw new IllegalArgumentException("weight layer " + weightLayer + " has no expert remap");
        }
        return layer;
    }

    /** The remapped weight layers, ascending. */
    public int[] layers() {
        int[] out = new int[byLayer.size()];
        int i = 0;
        for (Integer layer : byLayer.keySet()) {
            out[i++] = layer;
        }
        java.util.Arrays.sort(out);
        return out;
    }

    @Override
    public String toString() {
        return "MoeExpertRemap{" + byLayer.size() + " layer(s)}";
    }
}
