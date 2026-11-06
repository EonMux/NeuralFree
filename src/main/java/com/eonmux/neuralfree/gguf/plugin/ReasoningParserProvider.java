package com.eonmux.neuralfree.gguf.plugin;

import java.util.List;

/**
 * A plugin's rows for the reasoning parser registry
 * ({@link com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers}, which serves the
 * {@code neuralfree.reasoning.parser} names): the parser factories the plugin contributes when that registry
 * builds its table, keyed by the parser name.
 *
 * <p>The factory type {@code R} is the registry's factory type,
 * {@link com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers.Factory}, which the interface carries as its
 * type parameter; the registry loads the rows with {@link Plugins#load(Class)} and casts each provider to the
 * parameterized type. A factory whose parser name collides with a built-in factory or with a factory of another
 * provider is rejected when the registry builds its table, with both sources named: providers never replace
 * built-ins.
 *
 * @param <R> the reasoning-parser registry's parser-factory type
 */
public interface ReasoningParserProvider<R> {

    /**
     * The factories contributed, in any order (the registry merges them in provider-name order). Never
     * {@code null}; a plugin that registers no parsers returns an empty list.
     */
    List<R> reasoningParserFactories();
}
