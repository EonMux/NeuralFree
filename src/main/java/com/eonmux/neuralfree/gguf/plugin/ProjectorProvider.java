package com.eonmux.neuralfree.gguf.plugin;

import java.util.List;

/**
 * A plugin's rows for the projector registry ({@code G/mm/ProjectorRegistry}, built the {@code ArchRegistry}
 * way by MM-1): the {@code ProjectorEntry} rows the plugin contributes when that registry builds its table,
 * keyed by the projector's {@code architectures[0]} name.
 *
 * <p>The row type {@code R} is the registry's own row type ({@code ProjectorEntry} at MM-1), which is a
 * compile-package later than this SPI, so the interface carries it as its type parameter; the registry loads
 * the rows with {@link Plugins#load(Class)} and casts each provider to {@code ProjectorProvider<ProjectorEntry>}.
 * A row whose name collides with a built-in row or with a row of another provider is rejected when
 * the registry builds its table, with both sources named: providers never replace built-ins.
 *
 * @param <R> the projector registry's row type
 */
public interface ProjectorProvider<R> {

    /**
     * The rows contributed, in any order (the registry merges them in provider-name order). Never
     * {@code null}; a plugin that registers no rows returns an empty list.
     */
    List<R> projectorEntries();
}
