package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.IntPredicate;
import java.util.function.Predicate;

/**
 * Chooses the weight placement of a GPU load across the device and the host from free device memory: the
 * single-device form of llama.cpp's {@code --fit} (common/fit.cpp, {@code llama_params_fit_impl}). Where
 * {@link ContextFitter} sizes only the context, this fit also decides how many trunk weight layers stay resident
 * ({@code n_gpu_layers}), and which expert stacks ({@code *_exps.weight} / {@code *_chexps.weight},
 * {@link WeightShards#isExpert}) stream from the host while the dense tensors of their layers stay resident — the
 * analog of llama.cpp's {@code tensor_buft_overrides} that place the overflowing MoE tensors of a layer in the CPU
 * buffer. The host tier is the streamed one: its weights stay in the model file (mmap) behind a
 * {@link StreamingLayerWeightSource}.
 *
 * <p><b>The ladder.</b> Candidates run from the most device-resident down, and the first whose context plan fits the
 * usable bytes wins:</p>
 * <ol>
 *   <li>{@link Mode#RESIDENT}: every weight on the device, the context fitted beside them (as llama.cpp's step 1:
 *       no changes needed). An automatic context must reach the configured floor, or the load streams instead;</li>
 *   <li>{@link Mode#MOE_OVERFLOW} / {@link Mode#MOE_UPGRADED} for a model with expert tensors: the dense tensors of
 *       every trunk layer stay on the device, every expert stack streams (llama.cpp's all-MoE-tensors-to-CPU
 *       measurement, common/fit.cpp {@code pattern_moe_all}); then the expert stacks of the leading layers come back
 *       onto the device while they fit (its step 4, front to back);</li>
 *   <li>{@link Mode#PARTIAL}: the first trunk weight layers stream entirely and the rest keep their dense tensors
 *       resident (llama.cpp's back-to-front layer fill, its step 3; the smallest such prefix wins, so as many layers
 *       as possible stay);</li>
 *   <li>{@link Mode#STREAMED}: every trunk layer streams; the automatic context is the largest that fits beside the
 *       streaming slots (capped at the floor), a user context is kept or refused by name.</li>
 * </ol>
 *
 * <p>Streaming candidates run at the streaming context: an automatic context takes the floor, a user context and a
 * model without memory keep theirs. Feasibility compares the planned device bytes (resident weights, the
 * {@linkplain #MIN_SLOTS reserved streaming slots} and the context plan) against the usable bytes, so the fit is
 * pure planning: nothing is allocated. One stage of a multi-GPU layer split fits the same way over its own layers
 * ({@link Builder#layers}): its placed tensors are its span's, and its context plan holds only its layers' caches.
 * Stateless and thread-safe.</p>
 */
public final class PlacementFitter {

    private static final Logger LOGGER = LogManager.getLogger(PlacementFitter.class);

    /**
     * Streaming slots a fit reserves before the context is fitted, so uploads overlap compute; the opened source
     * grows into the VRAM that is left ({@code GGUFModelGPU} reserved the same minimum before this fit existed).
     */
    public static final int MIN_SLOTS = 2;

    /**
     * Smallest automatic context that keeps every weight resident before the trunk is streamed; a model trained on
     * a shorter context uses its training context ({@link #contextFloor(ModelSpec)}).
     */
    public static final int AUTO_CONTEXT_FLOOR = 4096;

    private static final double MIB = 1024.0 * 1024.0;

    /** Expert stacks are the tensors {@link WeightShards#isExpert} matches. */
    private static final Predicate<TensorRef> EXPERT = ref -> WeightShards.isExpert(ref.name(), ref.rank());

    private PlacementFitter() {
    }

    /** The automatic context floor of {@code spec}: {@link #AUTO_CONTEXT_FLOOR}, or its shorter training context. */
    public static int contextFloor(ModelSpec spec) {
        int train = spec.hparams().nCtxTrain();
        return train > 0 ? Math.min(train, AUTO_CONTEXT_FLOOR) : AUTO_CONTEXT_FLOOR;
    }

    /** How a fit places the weights. */
    public enum Mode {
        /** Every weight is resident; nothing streams. */
        RESIDENT,
        /** The dense tensors of every trunk layer are resident; every expert stack streams from the host. */
        MOE_OVERFLOW,
        /** As {@link #MOE_OVERFLOW}, with the expert stacks of the first layers back on the device. */
        MOE_UPGRADED,
        /** The first trunk weight layers stream entirely; the rest keep their dense tensors resident. */
        PARTIAL,
        /** Every trunk layer streams; only model-level tensors stay resident. */
        STREAMED
    }

    /** The placement a fit chose. Immutable. */
    public static final class Fit {

        private final String arch;
        private final Mode mode;
        private final int nCtx;
        private final int streamedLayers;
        private final int fullLayers;
        private final int weightLayers;
        private final List<LayerWeightSet> streamedSets;
        private final long deviceWeightBytes;
        private final long hostWeightBytes;
        private final long slotBytes;

        private Fit(String arch, Mode mode, int nCtx, int streamedLayers, int fullLayers, int weightLayers,
                    List<LayerWeightSet> streamedSets, long deviceWeightBytes, long hostWeightBytes,
                    long slotBytes) {
            this.arch = arch;
            this.mode = mode;
            this.nCtx = nCtx;
            this.streamedLayers = streamedLayers;
            this.fullLayers = fullLayers;
            this.weightLayers = weightLayers;
            this.streamedSets = Collections.unmodifiableList(new ArrayList<>(streamedSets));
            this.deviceWeightBytes = deviceWeightBytes;
            this.hostWeightBytes = hostWeightBytes;
            this.slotBytes = slotBytes;
        }

        public Mode mode() { return mode; }
        /** The context the placement runs with. */
        public int nCtx() { return nCtx; }
        /** Trunk weight layers that stream entirely: the first ones. */
        public int streamedLayers() { return streamedLayers; }
        /** Leading trunk weight layers whose expert tensors stay on the device (0 without expert overflow). */
        public int fullLayers() { return fullLayers; }
        /** Trunk weight layers of the model. */
        public int weightLayers() { return weightLayers; }
        /**
         * Trunk weight layers with a device-resident presence: the {@code n_gpu_layers} of llama.cpp's fit.
         */
        public int gpuLayers() { return weightLayers - streamedLayers; }
        /** The sets to stream, one per streamed weight layer with tensors, in weight-layer order (unmodifiable). */
        public List<LayerWeightSet> streamedSets() { return streamedSets; }
        /** Placed weight bytes kept on the device: everything but the streamed sets. */
        public long deviceWeightBytes() { return deviceWeightBytes; }
        /** Placed weight bytes read from the host: the streamed sets. */
        public long hostWeightBytes() { return hostWeightBytes; }
        /** Device bytes of the streaming slots the fit reserved. */
        public long slotBytes() { return slotBytes; }

        /** The placement as a log line. */
        public String describe() {
            switch (mode) {
                case RESIDENT:
                    return arch + ": all " + weightLayers + " trunk weight layers stay on the device ("
                            + mib(deviceWeightBytes) + " MiB), context " + nCtx;
                case MOE_OVERFLOW:
                case MOE_UPGRADED:
                    return arch + ": the dense tensors of all " + weightLayers + " trunk weight layers stay on the"
                            + " device (" + mib(deviceWeightBytes) + " MiB), the expert tensors of "
                            + (weightLayers - fullLayers) + " layer(s) stream from the host (" + mib(hostWeightBytes)
                            + " MiB through " + mib(slotBytes) + " MiB of slots)"
                            + (fullLayers > 0 ? ", the experts of the first " + fullLayers + " layer(s) stay on the"
                            + " device" : "") + ", context " + nCtx;
                case PARTIAL:
                    return arch + ": " + gpuLayers() + " of " + weightLayers + " trunk weight layers stay on the"
                            + " device, the first " + streamedLayers + " stream from the host (" + mib(hostWeightBytes)
                            + " MiB through " + mib(slotBytes) + " MiB of slots), context " + nCtx;
                default:
                    return arch + ": all " + weightLayers + " trunk weight layers stream from the host ("
                            + mib(hostWeightBytes) + " MiB through " + mib(slotBytes) + " MiB of slots), context "
                            + nCtx;
            }
        }

        @Override
        public String toString() {
            return "Fit{" + mode + ", nCtx=" + nCtx + ", streamed=" + streamedLayers + "/" + weightLayers
                    + ", full=" + fullLayers + ", device=" + deviceWeightBytes + " B, host=" + hostWeightBytes
                    + " B, slots=" + slotBytes + " B}";
        }
    }

    /** Inputs of a fit. */
    public static final class Builder {

        private final ModelSpec spec;
        private final ModelSpec setSpec;
        private final TensorCatalog catalog;
        private final LayerWeightSet.Options options;
        private final Predicate<TensorRef> placed;
        private long usableBytes = -1L;
        private long weightBytes = -1L;
        private MemoryConfig memory;
        private ContextFitter.Companion companion = ContextFitter.Companion.NONE;
        private int contextFloor = -1;
        private String backendType;
        private IntPredicate layers;

        /**
         * @param spec     the model the context plans with (the load's initial spec)
         * @param setSpec  the model whose trunk the streamed sets come from (the forward spec of the same load)
         * @param catalog  the model's tensors
         * @param options  upload planning of the streamed sets (device types, shard limits, input transforms)
         * @param placed   which tensors a placement may count (the {@code WeightPlacement} of the load)
         */
        public Builder(ModelSpec spec, ModelSpec setSpec, TensorCatalog catalog, LayerWeightSet.Options options,
                       Predicate<TensorRef> placed) {
            this.spec = Objects.requireNonNull(spec, "spec");
            this.setSpec = Objects.requireNonNull(setSpec, "setSpec");
            this.catalog = Objects.requireNonNull(catalog, "catalog");
            this.options = Objects.requireNonNull(options, "options");
            this.placed = Objects.requireNonNull(placed, "placed");
        }

        /** Device bytes the placement may use: weights, streaming slots and the context plan together. */
        public Builder usableBytes(long v) {
            if (v < 0) {
                throw new IllegalArgumentException("usable device bytes must be >= 0, got " + v);
            }
            this.usableBytes = v;
            return this;
        }

        /** Planned device bytes of every placed tensor of the model. */
        public Builder weightBytes(long v) {
            if (v < 0) {
                throw new IllegalArgumentException("weight bytes must be >= 0, got " + v);
            }
            this.weightBytes = v;
            return this;
        }

        /** The memory settings the context plans with. */
        public Builder memory(MemoryConfig v) {
            this.memory = Objects.requireNonNull(v, "memory");
            return this;
        }

        /** A companion model whose memory the context fit holds beside the context (a drafter); none by default. */
        public Builder companion(ContextFitter.Companion v) {
            this.companion = Objects.requireNonNull(v, "companion");
            return this;
        }

        /** Smallest automatic context a streaming placement runs with; positive. */
        public Builder contextFloor(int v) {
            if (v <= 0) {
                throw new IllegalArgumentException("context floor must be positive, got " + v);
            }
            this.contextFloor = v;
            return this;
        }

        /**
         * Plans the context over the layers {@code v} accepts (their {@code il}) only, the caches one stage of a
         * layer split holds, and places only the trunk weight layers with placed tensors, the stage's span; every
         * layer by default. A companion of a stage fit is held on that stage's device.
         */
        public Builder layers(IntPredicate v) {
            this.layers = Objects.requireNonNull(v, "layers");
            return this;
        }

        /** The backend type failures name ({@link GpuOutOfMemoryException}); for example {@code OPENCL}. */
        public Builder backendType(String v) {
            if (v == null || v.trim().isEmpty()) {
                throw new IllegalArgumentException("backend type is blank");
            }
            this.backendType = v.trim();
            return this;
        }

        /**
         * Runs the fit.
         *
         * @throws IllegalArgumentException for missing or inconsistent inputs (a negative byte figure, a blank
         *                                  backend type, a model without trunk weight layers, or weight bytes below
         *                                  the trunk tensors' planned bytes)
         * @throws GpuOutOfMemoryException  when not even the streamed fallback fits the usable bytes
         * @throws ContextFitter.ContextBudgetException when the context of the streamed fallback does not fit either
         *                                  (a user context names {@code neuralfree.ctx})
         */
        public Fit build() {
            checkDefaults();
            Weights w = Weights.of(this);
            return fit(this, w);
        }

        private void checkDefaults() {
            if (memory == null) {
                throw new IllegalArgumentException("memory settings are required");
            }
            if (contextFloor <= 0) {
                throw new IllegalArgumentException("contextFloor must be set to a positive context, got "
                        + contextFloor);
            }
            if (backendType == null) {
                throw new IllegalArgumentException("backendType must be set to the backend type failures name");
            }
            if (usableBytes < 0 || weightBytes < 0) {
                throw new IllegalArgumentException("usableBytes and weightBytes must be set to non-negative sizes");
            }
        }
    }

    private static Fit fit(Builder b, Weights w) {
        boolean memoryless = b.spec.memoryKind() == MemoryKind.NONE;
        if (memoryless) {
            if (b.weightBytes <= b.usableBytes) {
                return resident(b, w, b.memory.memorylessContext(b.spec));
            }
        } else {
            int residentCtx = fitOrZero(b, b.usableBytes - b.weightBytes);
            if (residentCtx > 0 && (!b.memory.autoContext() || residentCtx >= b.contextFloor)) {
                return resident(b, w, residentCtx);
            }
        }
        int streamCtx = memoryless ? b.memory.memorylessContext(b.spec)
                : (b.memory.autoContext() ? b.contextFloor : b.memory.nCtx());
        long ctxBytes = b.layers != null
                ? ContextFitter.requiredLayerBytes(b.spec, b.memory, streamCtx, b.layers, b.companion)
                : ContextFitter.requiredBytes(b.spec, b.memory, streamCtx, b.companion);

        if (w.expertTotal > 0) {
            long denseResident = w.base + w.denseTotal;
            List<LayerWeightSet> experts = w.expertRange(0, w.n);
            if (denseResident + slots(b, experts) + ctxBytes <= b.usableBytes) {
                return moeFit(b, w, upgrade(b, w, denseResident, ctxBytes), streamCtx, denseResident);
            }
        }
        int s = partialScan(b, w, ctxBytes);
        if (s >= 0) {
            return partialFit(b, w, s, streamCtx);
        }
        return streamedFit(b, w);
    }

    /** The resident fit; the caller checked the context or the weights fit. */
    private static Fit resident(Builder b, Weights w, int nCtx) {
        return new Fit(b.spec.arch().ggufName(), Mode.RESIDENT, nCtx, 0, 0, w.n,
                Collections.<LayerWeightSet>emptyList(), b.weightBytes, 0L, 0L);
    }

    /**
     * The largest count of leading layers whose expert stacks fit back on the device over the dense-only placement
     * (llama.cpp's step 4, front to back); 0 keeps every expert on the host.
     */
    private static int upgrade(Builder b, Weights w, long denseResident, long ctxBytes) {
        int f = 0;
        long expertBytes = 0L;
        while (f < w.n) {
            long next = expertBytes + w.expertBytes[f];
            if (denseResident + next + slots(b, w.expertRange(f + 1, w.n)) + ctxBytes > b.usableBytes) {
                break;
            }
            f++;
            expertBytes = next;
        }
        return f;
    }

    private static Fit moeFit(Builder b, Weights w, int f, int nCtx, long denseResident) {
        List<LayerWeightSet> sets = w.expertRange(f, w.n);
        long host = streamedBytes(sets);
        long slotBytes = slots(b, sets);
        Fit fit = new Fit(b.spec.arch().ggufName(), f > 0 ? Mode.MOE_UPGRADED : Mode.MOE_OVERFLOW, nCtx, 0, f, w.n,
                sets, denseResident + prefixBytes(w.expertBytes, f), host, slotBytes);
        LOGGER.info("{}", (Object) fit.describe());
        return fit;
    }

    /** The smallest prefix of trunk weight layers that streams entirely while the rest fit (llama.cpp step 3). */
    private static int partialScan(Builder b, Weights w, long ctxBytes) {
        long denseTail = w.denseTotal;
        for (int s = 1; s <= w.n; s++) {
            denseTail -= w.denseBytes[s - 1];
            if (w.base + denseTail + slots(b, w.zoneSets(s)) + ctxBytes <= b.usableBytes) {
                return s;
            }
        }
        return -1;
    }

    private static Fit partialFit(Builder b, Weights w, int s, int nCtx) {
        List<LayerWeightSet> sets = w.zoneSets(s);
        long host = streamedBytes(sets);
        long slotBytes = slots(b, sets);
        Fit fit = new Fit(b.spec.arch().ggufName(), Mode.PARTIAL, nCtx, s, 0, w.n, sets,
                b.weightBytes - host, host, slotBytes);
        LOGGER.info("{}", (Object) fit.describe());
        return fit;
    }

    /**
     * The fallback: every trunk layer streams and the context takes what is left (the placement
     * {@code GGUFModelGPU} chose before this fit existed). A model without memory keeps its context.
     *
     * @throws GpuOutOfMemoryException when the streamed fallback exceeds the usable bytes
     * @throws ContextFitter.ContextBudgetException when its context does not fit either
     */
    private static Fit streamedFit(Builder b, Weights w) {
        long slotBytes = slots(b, w.allSets);
        long base = w.base;
        if (base + slotBytes > b.usableBytes) {
            throw new GpuOutOfMemoryException(b.spec.arch().ggufName() + ": " + mib(b.weightBytes)
                    + " MiB of weights do not fit the " + mib(b.usableBytes) + " MiB of usable VRAM, even with the"
                    + " trunk layers streamed", b.weightBytes, b.usableBytes, b.backendType);
        }
        int nCtx = b.memory.memorylessContext(b.spec);
        if (b.spec.memoryKind() != MemoryKind.NONE) {
            nCtx = fitContext(b, b.usableBytes - base - slotBytes);
            if (b.memory.autoContext()) {
                nCtx = Math.min(nCtx, b.contextFloor);
            }
        }
        Fit fit = new Fit(b.spec.arch().ggufName(), Mode.STREAMED, nCtx, w.n, 0, w.n, w.allSets, base,
                streamedBytes(w.allSets), slotBytes);
        LOGGER.info("{}", (Object) fit.describe());
        return fit;
    }

    private static int fitOrZero(Builder b, long budget) {
        if (budget < 0) {
            return 0;
        }
        try {
            return fitContext(b, budget);
        } catch (ContextFitter.ContextBudgetException e) {
            return 0;
        }
    }

    /** The context that fits {@code budget}, over the builder's layers when it names a subset. */
    private static int fitContext(Builder b, long budget) {
        return b.layers != null ? ContextFitter.fitLayers(b.spec, b.memory, budget, b.layers, b.companion)
                : ContextFitter.fit(b.spec, b.memory, budget, b.companion);
    }

    /** Device bytes of the streaming slots for {@code sets} at the reserved minimum depth. */
    private static long slots(Builder b, List<LayerWeightSet> sets) {
        return sets.isEmpty() ? 0L
                : StreamingLayerWeightSource.plannedBytes(b.setSpec, sets, Math.min(MIN_SLOTS, sets.size()));
    }

    private static long streamedBytes(List<LayerWeightSet> sets) {
        long sum = 0L;
        for (LayerWeightSet s : sets) {
            sum += s.uploadBytes();
        }
        return sum;
    }

    private static long prefixBytes(long[] bytes, int end) {
        long sum = 0L;
        for (int i = 0; i < end; i++) {
            sum += bytes[i];
        }
        return sum;
    }

    private static String mib(long bytes) {
        return String.format(Locale.ROOT, "%.1f", bytes / MIB);
    }

    /**
     * Per trunk weight layer: the placed tensors of the layer as a whole and as expert stacks only, with their
     * planned device bytes. Immutable once built.
     */
    private static final class Weights {

        final int n;
        final LayerWeightSet[] all;
        final LayerWeightSet[] experts;
        final long[] allBytes;
        final long[] expertBytes;
        final long[] denseBytes;
        final long base;
        final long denseTotal;
        final long expertTotal;
        final List<LayerWeightSet> allSets;

        private Weights(int n, LayerWeightSet[] all, LayerWeightSet[] experts, long[] allBytes, long[] expertBytes,
                        long base) {
            this.n = n;
            this.all = all;
            this.experts = experts;
            this.allBytes = allBytes;
            this.expertBytes = expertBytes;
            this.denseBytes = new long[n];
            long denseTotal = 0L;
            long expertTotal = 0L;
            for (int i = 0; i < n; i++) {
                this.denseBytes[i] = allBytes[i] - expertBytes[i];
                denseTotal += this.denseBytes[i];
                expertTotal += expertBytes[i];
            }
            this.denseTotal = denseTotal;
            this.expertTotal = expertTotal;
            this.base = base;
            List<LayerWeightSet> sets = new ArrayList<>(n);
            for (LayerWeightSet set : all) {
                if (set != null) {
                    sets.add(set);
                }
            }
            this.allSets = Collections.unmodifiableList(sets);
        }

        /**
         * The trunk weight layers of the model; for a fit over a layer subset ({@link Builder#layers}) only those
         * with placed tensors, the layers of the stage, so its counts and its streamed prefix are the span's.
         */
        static Weights of(Builder b) {
            Map<Integer, String> stacks = LayerWeightSet.stacksByWeightLayer(b.setSpec);
            List<LayerWeightSet> wholes = new ArrayList<>(stacks.size());
            List<LayerWeightSet> expertSets = new ArrayList<>(stacks.size());
            for (Map.Entry<Integer, String> e : stacks.entrySet()) {
                LayerWeightSet whole = LayerWeightSet.of(b.catalog, e.getValue(), e.getKey(),
                        filtered(b.options, b.placed));
                if (b.layers != null && whole.entries().isEmpty()) {
                    continue;
                }
                wholes.add(whole);
                expertSets.add(LayerWeightSet.of(b.catalog, e.getValue(), e.getKey(),
                        filtered(b.options, b.placed.and(EXPERT))));
            }
            if (wholes.isEmpty()) {
                throw new IllegalArgumentException(b.spec.arch().ggufName() + ": the "
                        + (b.layers != null ? "layer span has" : "model has") + " no trunk weight layers to place");
            }
            int n = wholes.size();
            LayerWeightSet[] all = new LayerWeightSet[n];
            LayerWeightSet[] experts = new LayerWeightSet[n];
            long[] allBytes = new long[n];
            long[] expertBytes = new long[n];
            long trunkTotal = 0L;
            for (int i = 0; i < n; i++) {
                LayerWeightSet whole = wholes.get(i);
                LayerWeightSet expert = expertSets.get(i);
                all[i] = whole.entries().isEmpty() ? null : whole;
                experts[i] = expert.entries().isEmpty() ? null : expert;
                allBytes[i] = whole.uploadBytes();
                expertBytes[i] = expert.uploadBytes();
                trunkTotal = Math.addExact(trunkTotal, allBytes[i]);
            }
            if (trunkTotal > b.weightBytes) {
                throw new IllegalArgumentException(b.spec.arch().ggufName() + ": the trunk weight layers plan "
                        + trunkTotal + " bytes, more than the " + b.weightBytes + " placed weight bytes given");
            }
            return new Weights(n, all, experts, allBytes, expertBytes, b.weightBytes - trunkTotal);
        }

        /** The input options with a filter for the placed tensors of one zone. */
        private static LayerWeightSet.Options filtered(LayerWeightSet.Options o, Predicate<TensorRef> zone) {
            return LayerWeightSet.Options.builder(o.policy())
                    .filter(o.filter().and(zone))
                    .maxShardBytes(o.maxShardBytes())
                    .transforms(o.transforms())
                    .build();
        }

        /** The expert-stack sets of weight layers {@code [from, to)}. */
        List<LayerWeightSet> expertRange(int from, int to) {
            List<LayerWeightSet> out = new ArrayList<>(Math.max(0, to - from));
            for (int i = from; i < to; i++) {
                if (experts[i] != null) {
                    out.add(experts[i]);
                }
            }
            return out;
        }

        /** The streamed sets of {@link Mode#PARTIAL} at {@code s}: whole layers {@code [0, s)}, experts {@code [s, n)}. */
        List<LayerWeightSet> zoneSets(int s) {
            List<LayerWeightSet> out = new ArrayList<>(n);
            for (int i = 0; i < s; i++) {
                if (all[i] != null) {
                    out.add(all[i]);
                }
            }
            out.addAll(expertRange(s, n));
            return out;
        }
    }
}
