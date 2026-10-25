package com.eonmux.neuralfree.gguf.hf.arch;

import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries for the vision and audio projector checkpoints; every entry has kind {@link
 * HfArchEntry.Kind#MMPROJ}. {@link HfArchRegistry} merges this table with the other tables; a Hugging Face name
 * appears in exactly one table per {@link HfArchEntry.Kind}.
 */
final class HfArchTableMmproj {

    private HfArchTableMmproj() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.emptyList();
    }
}
