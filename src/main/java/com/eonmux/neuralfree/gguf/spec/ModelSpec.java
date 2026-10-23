package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The resolved model: hparams, tensor catalog, the trunk layers in execution order, the schedule mapping them
 * onto weights, and the embedding and head stages. NextN / MTP layers are kept apart for speculative decoding;
 * T5-style encoder layers are kept apart for the encoder pass. Immutable.
 */
public final class ModelSpec {

    private final ModelHparams hparams;
    private final TensorCatalog tensors;
    private final List<LayerSpec> layers;
    private final LayerSchedule schedule;
    private final EmbeddingSpec embedding;
    private final HeadSpec head;
    private final List<LayerSpec> nextnLayers;
    private final List<LayerSpec> encoderLayers;
    private final MemoryKind memoryKind;

    private ModelSpec(Builder b) {
        this.hparams = b.hparams;
        this.tensors = b.tensors;
        this.layers = Collections.unmodifiableList(new ArrayList<>(b.layers));
        this.schedule = b.schedule;
        this.embedding = b.embedding;
        this.head = b.head;
        this.nextnLayers = Collections.unmodifiableList(new ArrayList<>(b.nextnLayers));
        this.encoderLayers = Collections.unmodifiableList(new ArrayList<>(b.encoderLayers));
        this.memoryKind = b.memoryKind;
    }

    public static Builder builder(ModelHparams hparams, TensorCatalog tensors) {
        return new Builder(hparams, tensors);
    }

    public ModelHparams hparams() { return hparams; }
    public ArchTraits traits() { return hparams.traits(); }
    public Arch arch() { return hparams.arch(); }
    public TensorCatalog tensors() { return tensors; }
    /** Trunk layers in execution order; index {@code i} has {@code il == i} (unmodifiable). */
    public List<LayerSpec> layers() { return layers; }
    public LayerSpec layer(int il) {
        if (il < 0 || il >= layers.size()) {
            throw new IndexOutOfBoundsException("layer " + il + " outside the trunk of " + layers.size() + " layers");
        }
        return layers.get(il);
    }
    public int nLayer() { return layers.size(); }
    public LayerSchedule schedule() { return schedule; }
    public EmbeddingSpec embedding() { return embedding; }
    public HeadSpec head() { return head; }
    /** NextN / MTP layers ({@code il} continues after the trunk); used only by speculative decoding (unmodifiable). */
    public List<LayerSpec> nextnLayers() { return nextnLayers; }
    /** Encoder layers of encoder-decoder models, run without memory (unmodifiable). */
    public List<LayerSpec> encoderLayers() { return encoderLayers; }
    /** Memory organisation; NONE for encoders and diffusion models that run without a cache. */
    public MemoryKind memoryKind() { return memoryKind; }
    public SupportClass supportClass() { return hparams.traits().support(); }

    /** Widest residual stream over the trunk ({@code n_embd}, or {@code hc * n_embd}). */
    public int maxStreamWidth() {
        int w = 0;
        for (LayerSpec l : layers) {
            w = Math.max(w, l.residual().streamWidth());
        }
        return w;
    }

    /**
     * Why the context of this model cannot be shifted, or null when it can: llama.cpp {@code get_can_shift}
     * (src/llama-kv-cache.cpp:1213-1222) refuses step35 (per-layer RoPE dims) and models with more than one position
     * per token ({@code n_pos_per_embd > 1}: MROPE / IMROPE rope types, sensenova_u1 whose model rope type is MROPE,
     * src/llama-model.cpp:3648-3651, and any layer whose RoPE reads several positions). Forward backends check it
     * before the memory moves any row.
     */
    public String contextShiftRefusal() {
        String refusal = archShiftRefusal();
        if (refusal != null) {
            return refusal;
        }
        for (LayerSpec l : layers) {
            if (l.rope() != null && l.rope().positionsPerToken() > 1) {
                return severalPositions() + " (layer " + l.il() + ")";
            }
        }
        return null;
    }

    /**
     * {@link #contextShiftRefusal()} restricted to one layer: the model-wide refusals, or the layer's RoPE reading
     * several positions per token; null when the layer's keys can be re-rotated.
     */
    public String contextShiftRefusal(LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        String refusal = archShiftRefusal();
        if (refusal == null && layer.rope() != null && layer.rope().positionsPerToken() > 1) {
            refusal = severalPositions();
        }
        return refusal;
    }

    private String archShiftRefusal() {
        Arch arch = arch();
        if (arch == Arch.STEP35) {
            return arch.ggufName() + " uses per-layer RoPE dims; the K shift assumes one n_rot";
        }
        if (arch == Arch.SENSENOVA_U1 || traits().ropeType().usesSections() || hparams.nPosPerEmbd() > 1) {
            return severalPositions();
        }
        return null;
    }

    private String severalPositions() {
        return arch().ggufName() + " has several positions per token (n_pos_per_embd > 1)";
    }

    @Override
    public String toString() {
        return "ModelSpec{" + arch() + ", layers=" + layers.size() + ", nextn=" + nextnLayers.size() + ", encoder="
                + encoderLayers.size() + ", " + schedule + ", memory=" + memoryKind + '}';
    }

    public static final class Builder {
        private final ModelHparams hparams;
        private final TensorCatalog tensors;
        private final List<LayerSpec> layers = new ArrayList<>();
        private LayerSchedule schedule;
        private EmbeddingSpec embedding;
        private HeadSpec head;
        private final List<LayerSpec> nextnLayers = new ArrayList<>();
        private final List<LayerSpec> encoderLayers = new ArrayList<>();
        private MemoryKind memoryKind;

        private Builder(ModelHparams hparams, TensorCatalog tensors) {
            this.hparams = Objects.requireNonNull(hparams, "hparams");
            this.tensors = Objects.requireNonNull(tensors, "tensors");
            this.memoryKind = hparams.traits().memoryKind();
        }

        public Builder layer(LayerSpec v) { this.layers.add(Objects.requireNonNull(v, "layer")); return this; }
        public Builder schedule(LayerSchedule v) { this.schedule = v; return this; }
        public Builder embedding(EmbeddingSpec v) { this.embedding = v; return this; }
        public Builder head(HeadSpec v) { this.head = v; return this; }
        public Builder nextnLayer(LayerSpec v) {
            this.nextnLayers.add(Objects.requireNonNull(v, "nextn layer"));
            return this;
        }
        public Builder encoderLayer(LayerSpec v) {
            this.encoderLayers.add(Objects.requireNonNull(v, "encoder layer"));
            return this;
        }
        /** Overrides the trait memory kind (encoders and diffusion models run with {@link MemoryKind#NONE}). */
        public Builder memoryKind(MemoryKind v) {
            this.memoryKind = Objects.requireNonNull(v, "memoryKind");
            return this;
        }

        /**
         * @throws IllegalArgumentException when a stage is missing, layer indices are not contiguous, or the layers
         *                                  disagree with the schedule
         */
        public ModelSpec build() {
            String arch = hparams.arch().ggufName();
            if (schedule == null || embedding == null || head == null) {
                throw new IllegalArgumentException(arch + ": ModelSpec needs a schedule, an embedding stage and a head");
            }
            if (layers.size() != schedule.slots()) {
                throw new IllegalArgumentException(arch + ": " + layers.size() + " trunk layers but the schedule has "
                        + schedule.slots() + " slots");
            }
            for (int i = 0; i < layers.size(); i++) {
                LayerSpec l = layers.get(i);
                if (l.il() != i) {
                    throw new IllegalArgumentException(arch + ": trunk layer at position " + i + " has il " + l.il());
                }
                if (l.weightLayer() != schedule.weightLayer(i)) {
                    throw new IllegalArgumentException(arch + ": layer " + i + " uses weights " + l.weightLayer()
                            + " but the schedule maps it to " + schedule.weightLayer(i));
                }
            }
            for (int i = 0; i < nextnLayers.size(); i++) {
                if (nextnLayers.get(i).il() != layers.size() + i) {
                    throw new IllegalArgumentException(arch + ": NextN layer " + i + " has il " + nextnLayers.get(i).il()
                            + ", expected " + (layers.size() + i));
                }
            }
            for (int i = 0; i < encoderLayers.size(); i++) {
                LayerSpec l = encoderLayers.get(i);
                if (l.il() != i || !TensorNames.STACK_ENC.equals(l.stack())) {
                    throw new IllegalArgumentException(arch + ": encoder layer " + i + " must have il " + i + " and stack "
                            + TensorNames.STACK_ENC + ", got " + l);
                }
            }
            return new ModelSpec(this);
        }
    }
}
