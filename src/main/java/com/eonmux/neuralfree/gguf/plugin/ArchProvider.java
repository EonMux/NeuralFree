package com.eonmux.neuralfree.gguf.plugin;

import com.eonmux.neuralfree.gguf.arch.ArchEntry;

import java.util.List;

/**
 * A plugin's rows for the GGUF architecture registry ({@code com.eonmux.neuralfree.gguf.arch.ArchRegistry}):
 * the rows the plugin contributes to it when the registry builds its table.
 *
 * <p>Each row is the registry's existing {@link ArchEntry} row type. A row whose architecture is registered by
 * a built-in table or returned by another provider is rejected when the registry builds its table, with both
 * sources named: providers never replace built-ins. Rows for an architecture no table registers are accepted.
 */
public interface ArchProvider {

    /**
     * The rows contributed, in any order (the registry merges them in provider-name order). Never
     * {@code null}; a plugin that registers no rows returns an empty list.
     */
    List<ArchEntry> archEntries();
}
