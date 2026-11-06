package com.eonmux.neuralfree.gguf.mm;

import java.util.Locale;
import java.util.Objects;

/**
 * How the vision encoder spreads over the rank devices of an engine (vLLM's {@code mm_encoder_tp_mode},
 * {@code vllm/config/multimodal.py}):
 *
 * <ul>
 *   <li>{@link #WEIGHTS}: the encoder's weights are split across the ranks, the default tensor-parallel layout of
 *       a language model. NeuralFree runs the encoder graphs whole on one device, so this mode is the ordinary
 *       single-device encoder and a rank group of more than one device is refused
 *       ({@link MmprojOptions#PROP_DEVICES}).</li>
 *   <li>{@link #DATA}: every rank hosts the full weights and the images of one encode batch are spread over the
 *       ranks, as vLLM's {@code mm_encoder_tp_mode=data} disables the encoder's tensor parallelism
 *       ({@code disable_tp}) and shards the input ({@code run_dp_sharded_mrope_vision_model}).</li>
 * </ul>
 *
 * A single rank makes the two modes the same encoder, as vLLM's {@code tensor_parallel_size=1} does.
 */
public enum MmEncoderTpMode {

    /** Weights split across the ranks (the default); the encoder runs whole on one device here. */
    WEIGHTS("weights"),

    /** Full weights on every rank, the batch of images spread over the ranks. */
    DATA("data");

    private final String label;

    MmEncoderTpMode(String label) {
        this.label = label;
    }

    /** vLLM's name of the mode ({@code weights} or {@code data}). */
    public String label() {
        return label;
    }

    /**
     * The mode of a vLLM name, ignoring case.
     *
     * @throws IllegalArgumentException for an unknown name, naming the accepted ones
     */
    public static MmEncoderTpMode parse(String value) {
        String s = Objects.requireNonNull(value, "tp mode").trim().toLowerCase(Locale.ROOT);
        for (MmEncoderTpMode mode : values()) {
            if (mode.label.equals(s)) {
                return mode;
            }
        }
        throw new IllegalArgumentException("'" + value + "' is not an encoder tp mode; use weights or data");
    }

    @Override
    public String toString() {
        return label;
    }
}
