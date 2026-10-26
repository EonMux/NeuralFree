package com.eonmux.neuralfree.gguf.engine;

import java.util.EnumSet;

/**
 * Per-row products a batched forward step can produce (llama.cpp {@code n_outputs} split by head kind). A backend
 * that does not produce a kind for a row refuses the batch through {@code ForwardBackend.batchRefusal()}.
 *
 * <p>{@link #LOGITS} is the ordinary product of every backend. {@link #HIDDEN} (final hidden states) and
 * {@link #ROUTING} (mixture-of-experts router probabilities) are requested explicitly; batches of the stages that
 * need them set them on the rows that carry them.</p>
 */
public enum OutputKind {

    /** Logits over the output vocabulary: the product every backend produces on an output row by default. */
    LOGITS,

    /** Final hidden states after the output norm: the LM or classifier head input (SA-4, PL-1). */
    HIDDEN,

    /** Mixture-of-experts router probabilities of the row (SA-4). */
    ROUTING;

    /** Single-kind bitmask (one set bit per kind, by {@link #ordinal()}). */
    public int bit() {
        return 1 << ordinal();
    }

    /** Bitmask of a set of kinds. */
    public static int bits(EnumSet<OutputKind> kinds) {
        int mask = 0;
        for (OutputKind kind : kinds) {
            mask |= 1 << kind.ordinal();
        }
        return mask;
    }

    /** The kinds of a bitmask (a fresh set). */
    public static EnumSet<OutputKind> kinds(int bitMask) {
        EnumSet<OutputKind> out = EnumSet.noneOf(OutputKind.class);
        for (OutputKind kind : values()) {
            if ((bitMask & (1 << kind.ordinal())) != 0) {
                out.add(kind);
            }
        }
        return out;
    }
}
