package com.eonmux.neuralfree.gguf.engine.sampling;

import java.util.Collections;
import java.util.List;

/**
 * One custom logits processor (vLLM's {@code LogitsProcessor}): a named factory of per-generation
 * {@link SamplerStage stages} that a request's {@code vllm_xargs} arguments configure.
 *
 * <p>Server-wide installation has two forms, as in vLLM: a plugin jar declares a
 * {@link com.eonmux.neuralfree.gguf.plugin.LogitsProcessorProvider} in its {@code META-INF/services} entry (vLLM's
 * entry-point group) and its factories always load, or the operator names the factory's fully-qualified class name
 * with {@code --logits-processors} (vLLM's FQCN form), which needs a public no-argument constructor. A name or
 * alias registered twice fails the table build with both sources named.
 *
 * <p>Every loaded factory is consulted for every generation ({@link #create}); a factory whose
 * {@link #create} returns null leaves that request alone, which is how a processor activates per request from its
 * arguments. {@link #validate} runs at request admission, so a malformed argument fails the request with a message
 * naming it instead of reaching the sampler. A factory must be stateless and thread-safe: the engine shares one
 * instance across generations, and each {@link #create} call must return fresh stage state for its generation. A
 * stage that keeps state across accepted tokens overrides {@link SamplerStage#saveState} and
 * {@link SamplerStage#restoreState} so speculative verification can roll its history back.
 */
public interface LogitsProcessorFactory {

    /** The processor's primary name, which the registry reports and rejects as a duplicate. */
    String name();

    /** Further names of the same processor; empty when it has none. */
    default List<String> aliases() {
        return Collections.emptyList();
    }

    /**
     * Checks the arguments {@code args} for a request this engine is admitting.
     *
     * @throws IllegalArgumentException naming the argument ({@code vllm_xargs.<name>}) when it is malformed
     */
    default void validate(LogitsProcessorArgs args) {
    }

    /**
     * The processor stage of one generation, configured from the request's arguments, or null when the processor
     * does not apply to this request. Called once per generation; the stage may keep per-generation state.
     *
     * @throws IllegalArgumentException naming the argument ({@code vllm_xargs.<name>}) when it is malformed
     */
    SamplerStage create(LogitsProcessorArgs args);
}
