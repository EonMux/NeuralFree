package com.eonmux.neuralfree.gguf.hf.arch;

import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries for the standalone draft checkpoints the converter does not register (EAGLE-1,
 * standalone MTP and further DFlash and DSpark variants). {@link HfArchRegistry} merges this table with the other
 * tables; a Hugging Face name appears in exactly one table per {@link HfArchEntry.Kind}.
 */
final class HfArchTableDrafts {

    private HfArchTableDrafts() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.emptyList();
    }
}
