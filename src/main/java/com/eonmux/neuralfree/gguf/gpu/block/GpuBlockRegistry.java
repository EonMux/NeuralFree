package com.eonmux.neuralfree.gguf.gpu.block;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Resolves the GPU mixer, FFN block and layer extras of each layer from registered factories: one factory per
 * {@link MixerKind}, per {@link FfnKind} and per extra feature id.
 *
 * <p>Immutable. A layer whose kind or extra has no factory fails with {@link UnsupportedOperationException} naming the
 * kind, the layer and the architecture, so a model the GPU path cannot run is rejected at load time by
 * {@link #validate(ModelSpec)}.</p>
 */
public final class GpuBlockRegistry {

    private final Map<MixerKind, GpuMixerFactory> mixers;
    private final Map<FfnKind, GpuFfnFactory> ffns;
    private final Map<String, GpuLayerExtraFactory> extras;

    /**
     * @throws IllegalArgumentException when two factories serve the same kind or feature id, or a factory serves
     *                                  {@code NONE}
     */
    public GpuBlockRegistry(List<GpuMixerFactory> mixerFactories, List<GpuFfnFactory> ffnFactories,
                            List<GpuLayerExtraFactory> extraFactories) {
        Map<MixerKind, GpuMixerFactory> m = new EnumMap<>(MixerKind.class);
        for (GpuMixerFactory f : Objects.requireNonNull(mixerFactories, "mixerFactories")) {
            MixerKind k = Objects.requireNonNull(Objects.requireNonNull(f, "mixer factory").kind(), "mixer kind");
            if (k == MixerKind.NONE) {
                throw new IllegalArgumentException("mixer factory " + f + " serves MixerKind.NONE");
            }
            if (m.put(k, f) != null) {
                throw new IllegalArgumentException("two GPU mixer factories for " + k);
            }
        }
        Map<FfnKind, GpuFfnFactory> fm = new EnumMap<>(FfnKind.class);
        for (GpuFfnFactory f : Objects.requireNonNull(ffnFactories, "ffnFactories")) {
            FfnKind k = Objects.requireNonNull(Objects.requireNonNull(f, "FFN factory").kind(), "FFN kind");
            if (k == FfnKind.NONE) {
                throw new IllegalArgumentException("FFN factory " + f + " serves FfnKind.NONE");
            }
            if (fm.put(k, f) != null) {
                throw new IllegalArgumentException("two GPU FFN factories for " + k);
            }
        }
        Map<String, GpuLayerExtraFactory> em = new LinkedHashMap<>();
        for (GpuLayerExtraFactory f : Objects.requireNonNull(extraFactories, "extraFactories")) {
            String id = Objects.requireNonNull(f, "extra factory").featureId();
            if (id == null || id.isEmpty()) {
                throw new IllegalArgumentException("layer-extra factory " + f + " has no feature id");
            }
            if (em.put(id, f) != null) {
                throw new IllegalArgumentException("two GPU layer-extra factories for '" + id + "'");
            }
        }
        this.mixers = Collections.unmodifiableMap(m);
        this.ffns = Collections.unmodifiableMap(fm);
        this.extras = Collections.unmodifiableMap(em);
    }

    /** True when a factory serves this mixer kind. */
    public boolean hasMixer(MixerKind k) {
        return mixers.containsKey(k);
    }

    /** True when a factory serves this FFN kind. */
    public boolean hasFfn(FfnKind k) {
        return ffns.containsKey(k);
    }

    /** True when a factory serves this layer-extra feature id. */
    public boolean hasExtra(String featureId) {
        return extras.containsKey(featureId);
    }

    /**
     * An unbound mixer for the layer.
     *
     * @throws IllegalArgumentException      when the layer has no mixer ({@link MixerKind#NONE})
     * @throws UnsupportedOperationException when no factory serves the layer's mixer kind
     */
    public GpuMixer mixer(LayerSpec layer) {
        MixerKind k = layer.mixerKind();
        if (k == MixerKind.NONE) {
            throw new IllegalArgumentException("layer " + layer.il() + " has no mixer");
        }
        GpuMixerFactory f = mixers.get(k);
        if (f == null) {
            throw new UnsupportedOperationException("no GPU mixer for " + k + " (layer " + describe(layer) + ")");
        }
        return Objects.requireNonNull(f.create(layer), "mixer factory for " + k + " returned null");
    }

    /**
     * An unbound FFN block for the layer.
     *
     * @throws IllegalArgumentException      when the layer has no FFN ({@link FfnKind#NONE})
     * @throws UnsupportedOperationException when no factory serves the layer's FFN kind
     */
    public GpuFfnBlock ffn(LayerSpec layer) {
        FfnKind k = layer.ffnKind();
        if (k == FfnKind.NONE) {
            throw new IllegalArgumentException("layer " + layer.il() + " has no FFN");
        }
        GpuFfnFactory f = ffns.get(k);
        if (f == null) {
            throw new UnsupportedOperationException("no GPU FFN block for " + k + " (layer " + describe(layer) + ")");
        }
        return Objects.requireNonNull(f.create(layer), "FFN factory for " + k + " returned null");
    }

    /**
     * Unbound extras of the layer, in {@code LayerSpec.extras()} order.
     *
     * @throws UnsupportedOperationException when no factory serves one of the layer's extras
     */
    public List<GpuLayerExtra> extras(LayerSpec layer) {
        List<GpuLayerExtra> out = new ArrayList<>(layer.extras().size());
        try {
            for (String id : layer.extras()) {
                GpuLayerExtraFactory f = extras.get(id);
                if (f == null) {
                    throw new UnsupportedOperationException("no GPU layer extra for '" + id + "' (layer "
                            + describe(layer) + ")");
                }
                out.add(Objects.requireNonNull(f.create(layer), "layer-extra factory for '" + id + "' returned null"));
            }
        } catch (RuntimeException e) {
            for (GpuLayerExtra x : out) {
                try {
                    x.close();
                } catch (Exception closeFailure) {
                    e.addSuppressed(closeFailure);
                }
            }
            throw e;
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * Checks that every layer of the model (main, next-n and encoder stacks) has a factory for its mixer, FFN and
     * extras, and an {@link MixerKind#ATTENTION} factory for the separate softmax-attention branch the executors bind
     * next to the mixer (T5 decoder self-attention beside {@link MixerKind#CROSS_ATTENTION_T5}, falcon-h1
     * {@link LayerSpec#parallelAttention() parallel attention}).
     *
     * @throws UnsupportedOperationException naming the architecture and the first missing kind or feature
     */
    public void validate(ModelSpec spec) {
        List<LayerSpec> all = new ArrayList<>(spec.layers());
        all.addAll(spec.nextnLayers());
        all.addAll(spec.encoderLayers());
        String arch = spec.arch().ggufName();
        for (LayerSpec l : all) {
            if (l.mixerKind() != MixerKind.NONE && !mixers.containsKey(l.mixerKind())) {
                throw new UnsupportedOperationException(arch + ": no GPU mixer for " + l.mixerKind() + " (layer "
                        + describe(l) + ")");
            }
            boolean attentionBranch = l.mixerKind() == MixerKind.CROSS_ATTENTION_T5
                    || (l.mixerKind() != MixerKind.NONE && l.parallelAttention());
            if (attentionBranch && !mixers.containsKey(MixerKind.ATTENTION)) {
                throw new UnsupportedOperationException(arch + ": no GPU mixer for the ATTENTION branch beside "
                        + l.mixerKind() + " (layer " + describe(l) + ")");
            }
            if (l.ffnKind() != FfnKind.NONE && !ffns.containsKey(l.ffnKind())) {
                throw new UnsupportedOperationException(arch + ": no GPU FFN block for " + l.ffnKind() + " (layer "
                        + describe(l) + ")");
            }
            for (String id : l.extras()) {
                if (!extras.containsKey(id)) {
                    throw new UnsupportedOperationException(arch + ": no GPU layer extra for '" + id + "' (layer "
                            + describe(l) + ")");
                }
            }
        }
    }

    private static String describe(LayerSpec l) {
        return l.stack() + "." + l.il();
    }

    @Override
    public String toString() {
        return "GpuBlockRegistry{mixers=" + mixers.keySet() + ", ffns=" + ffns.keySet() + ", extras="
                + extras.keySet() + '}';
    }
}
