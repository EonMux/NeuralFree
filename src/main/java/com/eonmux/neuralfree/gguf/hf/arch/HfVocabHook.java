package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Map;

/**
 * A converter class's {@code set_vocab}: the {@code tokenizer.*} metadata of the checkpoint, keyed by concrete GGUF key
 * names, with values as {@link String}, {@link Integer} or {@link Long}, {@link Float}, {@link Boolean} or a
 * {@link java.util.List} of those. The values are typed through
 * {@link com.eonmux.neuralfree.gguf.hf.meta.GgufKeyTable} when they are written.
 */
@FunctionalInterface
public interface HfVocabHook {

    /**
     * The tokenizer metadata.
     *
     * @param pre resolves the {@code tokenizer.ggml.pre} name of a BPE tokenizer
     * @throws IOException when a tokenizer file cannot be read
     */
    Map<String, Object> build(HfLoadContext ctx, PreNameResolver pre) throws IOException;
}
