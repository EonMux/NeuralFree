package com.eonmux.neuralfree.gguf.plugin;

import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessorFactory;

import java.util.List;

/**
 * A plugin's custom logits processors (vLLM's entry-point group {@code vllm.logits_processors}): the
 * {@link LogitsProcessorFactory} rows the plugin contributes when the engine builds its logits-processor table,
 * always loaded like every other provider of this SPI.
 *
 * <p>The rows merge in provider-class order with the factory classes {@code --logits-processors} names. A name or
 * alias registered twice is rejected when the table builds, with both sources named: providers never replace one
 * another. A factory is consulted for every generation and decides from the request's {@code vllm_xargs} arguments
 * whether it applies; its {@link LogitsProcessorFactory#create} returns null for the requests it leaves alone.
 */
public interface LogitsProcessorProvider {

    /**
     * The factories contributed, in any order (the engine merges them in provider-name order). Never {@code null};
     * a plugin that contributes no processors returns an empty list.
     */
    List<LogitsProcessorFactory> logitsProcessorFactories();
}
