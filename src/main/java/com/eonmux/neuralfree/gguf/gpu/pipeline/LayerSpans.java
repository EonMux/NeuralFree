package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Arrays;
import java.util.Objects;

/**
 * The layer spans of a layer split: device {@code d} runs the trunk weight layers
 * {@code [starts[d], starts[d + 1])} and holds their KV caches, the counterpart of llama.cpp assigning each layer to
 * the GPU its cumulative {@code tensor_split} fraction falls in (src/llama-model.cpp:1700). Layers that read the KV
 * cache of another layer (gemma3n / gemma4 reuse) are kept with their owner by moving the boundary between them.
 * When the first trunk layers stay on the host (llama.cpp {@code -ngl} below the layer count), the spans cover the
 * device layers {@code [first, nLayer)} only. Immutable.
 */
public final class LayerSpans {

    private final int[] starts;

    private LayerSpans(int[] starts) {
        this.starts = starts;
    }

    /**
     * Spans of {@code devices} stages over the {@code spec} trunk, sized by {@code ratios} (empty for equal shares,
     * otherwise one positive share per stage prefix; llama.cpp {@code --tensor-split}).
     *
     * @throws IllegalArgumentException when {@code devices} is not positive, a ratio is not a finite value above 0,
     *                                  there are more ratios or devices than trunk layers, or no boundary keeps
     *                                  every KV-sharing layer with its owner
     * @throws UnsupportedOperationException when the spec's layer schedule is not linear (looped and HRM stacks
     *                                       re-run their weights on every pass and cannot be split) or the model is
     *                                       an encoder-decoder (its cross-attention reads the encoder's device)
     */
    public static LayerSpans of(ModelSpec spec, int devices, double[] ratios) {
        return of(spec, devices, ratios, 0);
    }

    /**
     * Spans of {@code devices} stages over the device layers {@code [first, nLayer)} of the {@code spec} trunk; the
     * layers before {@code first} run on the host.
     *
     * @throws IllegalArgumentException as {@link #of(ModelSpec, int, double[])}, when {@code first} is outside
     *                                  {@code [0, nLayer)}, there are more devices than device layers, or a
     *                                  KV-sharing layer and its owner fall on the two sides of {@code first}
     */
    public static LayerSpans of(ModelSpec spec, int devices, double[] ratios, int first) {
        Objects.requireNonNull(spec, "spec");
        if (devices <= 0) {
            throw new IllegalArgumentException("a layer split needs at least one device, got " + devices);
        }
        String arch = spec.arch().ggufName();
        if (spec.supportClass() == SupportClass.ENC_DEC) {
            throw new UnsupportedOperationException(arch + " is an encoder-decoder model; "
                    + EngineConfig.PROP_SPLIT_MODE + "=layer cannot split its cross-attention over GPUs");
        }
        LayerSchedule schedule = spec.schedule();
        if (schedule != null && schedule.kind() != LayerSchedule.Kind.LINEAR) {
            throw new UnsupportedOperationException(arch + " runs a " + schedule.kind() + " layer schedule; "
                    + EngineConfig.PROP_SPLIT_MODE + "=layer splits linear stacks only");
        }
        int nLayer = spec.nLayer();
        if (nLayer == 0) {
            throw new UnsupportedOperationException(arch + " has no trunk layers to split");
        }
        if (first < 0 || first >= nLayer) {
            throw new IllegalArgumentException(EngineConfig.PROP_SPLIT_MODE + "=layer needs at least one of the "
                    + nLayer + " layers of " + arch + " on the GPUs, but the host keeps " + first
                    + "; raise " + EngineConfig.PROP_N_GPU_LAYERS);
        }
        if (devices > nLayer - first) {
            throw new IllegalArgumentException(EngineConfig.PROP_SPLIT_MODE + "=layer over " + devices
                    + " devices needs at least one layer per device, but " + arch + " has " + (nLayer - first)
                    + (first > 0 ? " on the GPUs (" + first + " stay on the host; raise "
                    + EngineConfig.PROP_N_GPU_LAYERS + ")" : ""));
        }
        checkHostBoundary(spec, first);
        double[] shares = normalizedShares(ratios, devices);
        int[] starts = boundaries(shares, first, nLayer);
        keepSharedKvWithOwner(spec, starts);
        checkLongcatPairs(spec, starts);
        return new LayerSpans(starts);
    }

    /**
     * Spans of {@code counts.length} stages over the device layers {@code [first, nLayer)} of the {@code spec}
     * trunk, one layer count per stage in pipeline order: the backend of the automatic fit, which fills each device
     * from the back and gives the leading layers to the host. Equal to
     * {@link #of(ModelSpec, int, double[], int)} with the counts as shares.
     *
     * @throws IllegalArgumentException when a count is not positive, the counts do not cover the device layers, or a
     *                                  KV-sharing layer and its owner fall on the two sides of {@code first}
     * @throws UnsupportedOperationException as {@link #of(ModelSpec, int, double[], int)}
     */
    public static LayerSpans fromCounts(ModelSpec spec, int[] counts, int first) {
        Objects.requireNonNull(counts, "counts");
        if (counts.length == 0) {
            throw new IllegalArgumentException("a layer split needs at least one device");
        }
        double[] shares = new double[counts.length];
        int total = 0;
        for (int d = 0; d < counts.length; d++) {
            if (counts[d] < 1) {
                throw new IllegalArgumentException(EngineConfig.PROP_SPLIT_MODE + "=layer over " + counts.length
                        + " devices leaves device " + d + " without a layer; give it at least one");
            }
            shares[d] = counts[d];
            total += counts[d];
        }
        if (first < 0 || total != spec.nLayer() - first) {
            throw new IllegalArgumentException(EngineConfig.PROP_SPLIT_MODE + "=layer gives the " + counts.length
                    + " devices " + total + " of the " + (spec.nLayer() - first) + " device layers");
        }
        return of(spec, counts.length, shares, first);
    }

    /**
     * The two physical blocks of a LongCat-Flash layer share the shortcut MoE output through one scratch temporary of
     * their device context, so a split may not put them on different sides of a boundary.
     *
     * @throws UnsupportedOperationException when a boundary falls between the blocks of one layer
     */
    private static void checkLongcatPairs(ModelSpec spec, int[] starts) {
        if (spec.arch() != com.eonmux.neuralfree.gguf.arch.Arch.LONGCAT_FLASH
                && spec.arch() != com.eonmux.neuralfree.gguf.arch.Arch.LONGCAT_FLASH_NGRAM) {
            return;
        }
        for (int d = 0; d < starts.length - 1; d++) {
            if (starts[d] % 2 != 0) {
                throw new UnsupportedOperationException(spec.arch().ggufName() + ": " + EngineConfig.PROP_SPLIT_MODE
                        + "=layer puts a boundary at layer " + starts[d] + ", between the two blocks of one"
                        + " LongCat-Flash layer; the second block reads the first block's shortcut MoE output from"
                        + " its own device's scratch, so the pair must stay together");
            }
        }
    }

    /** A layer reading the KV cache of another layer must sit on the same side of the host boundary as its owner. */
    private static void checkHostBoundary(ModelSpec spec, int first) {
        for (LayerSpec l : spec.layers()) {
            if (l.attention() == null || !l.attention().sharesKv(l.il())) {
                continue;
            }
            int owner = l.attention().kvOwnerLayer();
            if ((l.weightLayer() < first) != (owner < first)) {
                throw new IllegalArgumentException(spec.arch().ggufName() + ": layer " + l.il() + " reads the KV"
                        + " cache of layer " + owner + ", and " + EngineConfig.PROP_N_GPU_LAYERS + " puts the two on"
                        + " different sides of the host boundary at layer " + first + "; keep them together");
            }
        }
    }

    /** Shares normalized to sum 1; an empty list is equal shares, otherwise one share per device is required. */
    private static double[] normalizedShares(double[] ratios, int devices) {
        if (ratios.length == 0) {
            double[] equal = new double[devices];
            Arrays.fill(equal, 1.0 / devices);
            return equal;
        }
        if (ratios.length != devices) {
            throw new IllegalArgumentException(EngineConfig.PROP_TENSOR_SPLIT + " names " + ratios.length
                    + " shares for " + devices + " device(s); give exactly one share per device, or drop the idle"
                    + " device from " + EngineConfig.PROP_GPU_DEVICES);
        }
        double total = 0;
        for (double r : ratios) {
            if (!(r > 0.0) || !Double.isFinite(r)) {
                throw new IllegalArgumentException(EngineConfig.PROP_TENSOR_SPLIT + ": every share must be a finite"
                        + " value above 0, got " + r);
            }
            total += r;
        }
        double[] out = new double[devices];
        for (int i = 0; i < devices; i++) {
            out[i] = ratios[i] / total;
        }
        return out;
    }

    /** Largest-remainder boundaries over {@code [first, nLayer)}: each device takes floor(share * layers) layers,
     * the leftover units go to the largest fractional remainders (ties to the lower device), so the counts sum to
     * the device layers. */
    private static int[] boundaries(double[] shares, int first, int nLayer) {
        int devices = shares.length;
        int layers = nLayer - first;
        int[] counts = new int[devices];
        int assigned = 0;
        Integer[] order = new Integer[devices];
        for (int d = 0; d < devices; d++) {
            double ideal = shares[d] * layers;
            counts[d] = (int) Math.floor(ideal);
            assigned += counts[d];
            order[d] = d;
        }
        final double[] remainder = new double[devices];
        for (int d = 0; d < devices; d++) {
            remainder[d] = shares[d] * layers - counts[d];
        }
        java.util.Arrays.sort(order, (a, b) -> remainder[b] > remainder[a] + 1e-12 ? 1
                : remainder[b] < remainder[a] - 1e-12 ? -1 : a - b);
        for (int i = 0; i < layers - assigned; i++) {
            counts[order[i]]++;
        }
        int[] starts = new int[devices + 1];
        starts[0] = first;
        for (int d = 0; d < devices; d++) {
            starts[d + 1] = starts[d] + counts[d];
        }
        if (starts[devices] != nLayer) {
            throw new IllegalStateException("layer spans cover " + starts[devices] + " of " + nLayer + " layers");
        }
        return starts;
    }

    /**
     * Moves boundaries so every layer that reads the KV cache of another layer
     * ({@code AttentionSpec.kvOwnerLayer()}) stays in its owner's span: a reader left of its owner's device takes
     * the layers back to itself one boundary per pass, until the two meet. A pass that leaves a span empty takes a
     * layer from the neighbor that can spare one; a layout where the two rules keep fighting is refused.
     */
    private static void keepSharedKvWithOwner(ModelSpec spec, int[] starts) {
        int nLayer = spec.nLayer();
        for (int pass = 0; pass < starts.length * 2 + 8; pass++) {
            boolean moved = false;
            for (LayerSpec l : spec.layers()) {
                if (l.attention() == null || !l.attention().sharesKv(l.il()) || l.weightLayer() < starts[0]) {
                    continue;
                }
                int owner = l.attention().kvOwnerLayer();
                int readerDev = deviceOf(starts, l.weightLayer());
                int ownerDev = deviceOf(starts, owner);
                if (readerDev > ownerDev) {
                    starts[readerDev] = l.weightLayer() + 1;
                    moved = true;
                } else if (ownerDev > readerDev) {
                    starts[ownerDev] = owner + 1;
                    moved = true;
                }
            }
            if (!moved && !hasEmptySpan(starts)) {
                return;
            }
            fillEmptySpans(starts, nLayer);
        }
        throw new IllegalArgumentException(spec.arch().ggufName() + ": no layer split keeps every KV-sharing layer"
                + " with its owner; give " + EngineConfig.PROP_TENSOR_SPLIT
                + " shares that put each reader on its owner's device");
    }

    private static boolean hasEmptySpan(int[] starts) {
        for (int d = 0; d < starts.length - 1; d++) {
            if (span(starts, d) == 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * Gives an empty span a layer from the neighbor that can spare one, without crossing another boundary; the first
     * device layer and the end stay where they are.
     */
    private static void fillEmptySpans(int[] starts, int nLayer) {
        int first = starts[0];
        for (int d = 0; d < starts.length - 1; d++) {
            if (span(starts, d) > 0) {
                continue;
            }
            if (d > 0 && span(starts, d - 1) > 1) {
                starts[d]--;
            } else if (d + 1 < starts.length - 1 && span(starts, d + 1) > 1) {
                starts[d + 1]++;
            }
        }
        starts[0] = first;
        starts[starts.length - 1] = nLayer;
        for (int d = 1; d < starts.length; d++) {
            starts[d] = Math.max(starts[d], starts[d - 1]);
        }
    }

    private static int span(int[] starts, int d) {
        return starts[d + 1] - starts[d];
    }

    /** The device whose span holds weight layer {@code w} (the last one past the end, so NextN tensors follow). */
    public int deviceOfWeightLayer(int w) {
        int last = starts.length - 2;
        for (int d = 0; d <= last; d++) {
            if (w < starts[d + 1]) {
                return d;
            }
        }
        return Math.max(0, last);
    }

    /** The device whose span holds absolute stack layer {@code il}. */
    private static int deviceOf(int[] starts, int w) {
        int last = starts.length - 2;
        for (int d = 0; d <= last; d++) {
            if (w < starts[d + 1]) {
                return d;
            }
        }
        return Math.max(0, last);
    }

    /** The device that runs {@code l} and holds its weights and caches. */
    public int deviceOfLayer(LayerSpec l) {
        return deviceOfWeightLayer(l.weightLayer());
    }

    /** First weight layer of device {@code d}. */
    public int start(int d) {
        return starts[d];
    }

    /** Weight layers of device {@code d}. */
    public int spanLayers(int d) {
        return starts[d + 1] - starts[d];
    }

    /** Device count. */
    public int devices() {
        return starts.length - 1;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("LayerSpans{");
        for (int d = 0; d < starts.length - 1; d++) {
            sb.append(d == 0 ? "" : ", ").append(starts[d]).append("..").append(starts[d + 1] - 1);
        }
        return sb.append('}').toString();
    }
}
