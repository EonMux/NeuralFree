package com.eonmux.neuralfree.gguf.plugin;

import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessorFactory;

import java.util.Collections;
import java.util.List;

/**
 * The logits-processor plugin of the tests: it contributes no factory, so registering it through
 * {@code META-INF/services} proves the SPI is wired without changing another test's engine.
 */
public final class EmptyLogitsProcessorProvider implements LogitsProcessorProvider {

    public EmptyLogitsProcessorProvider() {
    }

    @Override
    public List<LogitsProcessorFactory> logitsProcessorFactories() {
        return Collections.emptyList();
    }
}
