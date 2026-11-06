package com.eonmux.neuralfree.gguf.hf.arch;

import java.util.Collections;
import java.util.List;

/**
 * Hugging Face text architectures llama.cpp's converter registers whose checkpoints NeuralFree cannot convert at load
 * time: an entry is {@link HfArchEntry#rejected rejected} with a reason that names the architecture and the way to run
 * it (a GGUF written by llama.cpp's {@code convert_hf_to_gguf.py}, where NeuralFree loads that GGUF architecture).
 * {@link HfArchRegistry} merges this table with the other tables; a Hugging Face name appears in exactly one table per
 * {@link HfArchEntry.Kind}.
 */
final class HfArchTableGgufOnly {

    private HfArchTableGgufOnly() {
    }

    /** The entries of this table; currently none, every registered converter class has a conversion. */
    static List<HfArchEntry> entries() {
        return Collections.emptyList();
    }
}
