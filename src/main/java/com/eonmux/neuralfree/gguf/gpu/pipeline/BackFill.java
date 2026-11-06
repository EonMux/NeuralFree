package com.eonmux.neuralfree.gguf.gpu.pipeline;

import java.util.Objects;

/**
 * The back-to-front layer fill of a layer split's automatic fit (llama.cpp common/fit.cpp, step 3): the last device
 * takes the largest run of trailing trunk layers it holds resident, then the one before it, and so on; the layers
 * left in front of every device stay on the host. A device may move the routed experts of the leading layers of its
 * run to the host before it gives up a whole layer, the analog of llama.cpp filling dense-only layers with all MoE
 * tensors in system memory. Every device keeps at least one layer, so the fill never yields an idle GPU. Stateless.
 */
final class BackFill {

    /** The resident capacity of one device's candidate run, with the expert overflow that makes it fit. */
    interface Capacity {

        /**
         * The number of leading layers of {@code [start, end)} whose routed experts must move to the host for the
         * device to hold the run resident, or -1 when even hosting all of them does not make it fit.
         */
        int expertOverflow(int device, int start, int end);
    }

    /** The per-device layer counts of a fill, the experts each device hosts, and the first layer on a GPU. */
    static final class Layout {

        private final int[] layers;
        private final int[] experts;
        private final int firstGpuLayer;

        private Layout(int[] layers, int[] experts, int firstGpuLayer) {
            this.layers = layers;
            this.experts = experts;
            this.firstGpuLayer = firstGpuLayer;
        }

        /** Trunk layers of device {@code d}. */
        int layers(int d) {
            return layers[d];
        }

        /** Leading layers of device {@code d} whose routed experts move to the host. */
        int experts(int d) {
            return experts[d];
        }

        /** The first trunk layer on a GPU; the layers before it run on the host. */
        int firstGpuLayer() {
            return firstGpuLayer;
        }

        int devices() {
            return layers.length;
        }
    }

    private BackFill() {
    }

    /**
     * Fills {@code layers} trunk layers over {@code devices} devices, back to front, at the capacity each reports.
     *
     * @return the layout, or null when a device cannot hold even one layer of its run
     * @throws IllegalArgumentException when {@code devices} is not positive or {@code layers} is negative
     */
    static Layout fill(int devices, int layers, Capacity capacity) {
        Objects.requireNonNull(capacity, "capacity");
        if (devices <= 0) {
            throw new IllegalArgumentException("a back-to-front fill needs at least one device, got " + devices);
        }
        if (layers < 0) {
            throw new IllegalArgumentException("a back-to-front fill needs a non-negative layer count, got " + layers);
        }
        if (layers < devices) {
            return null;
        }
        int[] counts = new int[devices];
        int[] experts = new int[devices];
        int end = layers;
        for (int d = devices - 1; d >= 0; d--) {
            int max = end - d;
            int best = largestFitting(d, end, max, capacity);
            if (best < 0) {
                return null;
            }
            counts[d] = best;
            experts[d] = capacity.expertOverflow(d, end - best, end);
            end -= best;
        }
        return new Layout(counts, experts, end);
    }

    /** The largest count in {@code [1, max]} whose run {@code [end - count, end)} the device holds. */
    private static int largestFitting(int device, int end, int max, Capacity capacity) {
        int best = -1;
        int lo = 1;
        int hi = max;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (capacity.expertOverflow(device, end - mid, end) >= 0) {
                best = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return best;
    }
}
