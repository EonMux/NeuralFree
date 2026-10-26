package com.eonmux.neuralfree.gguf.plugin;

import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;

import java.util.List;

/**
 * A plugin's rows for the Hugging Face architecture registry ({@code com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry}):
 * the {@code HfArchEntry} rows the plugin contributes when the registry builds its table, keyed by the
 * {@code architectures[0]} name of a checkpoint. A plugin can register a Hugging Face name no built-in table
 * knows, which is the usual way an out-of-tree model family lands natively.
 *
 * <p>Each row is the registry's existing {@link HfArchEntry} row type. A row whose Hugging Face name (for its
 * {@link HfArchEntry.Kind}) is registered by a built-in table or returned by another provider is rejected when
 * the registry builds its table, with both sources named: providers never replace built-ins.
 */
public interface HfArchProvider {

    /**
     * The rows contributed, in any order (the registry merges them in provider-name order). Never
     * {@code null}; a plugin that registers no rows returns an empty list.
     */
    List<HfArchEntry> hfArchEntries();
}
