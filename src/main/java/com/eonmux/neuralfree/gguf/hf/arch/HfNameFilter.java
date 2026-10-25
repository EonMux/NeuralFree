package com.eonmux.neuralfree.gguf.hf.arch;

import java.util.function.UnaryOperator;

/**
 * A converter class's own {@code filter_tensors} classmethod: the per-name renames and drops it applies around the
 * inherited filters ({@code super().filter_tensors}). Every checkpoint name goes through it, both the names of the
 * tensor index ({@link com.eonmux.neuralfree.gguf.hf.names.HfTensorFilters#apply}) and the names side files refer to
 * (the {@code hadamard_packing.json} records), so both land in the same namespace. Whole-index work (side-file
 * tensors, merges that need several tensors) stays in {@link HfIndexHook}.
 *
 * <p>A filter registered with {@link HfArchEntry#withNameFilter} is a pure function of the name; for example LlamaModel
 * (LC/conversion/llama.py:200-206) is {@code (n, base) -> base.apply(n.replace("text_model.", ""))}. A class whose
 * {@code filter_tensors} reads instance state (the original block count and {@code no_mtp} of
 * {@code _QwenMtpMixin}, LC/conversion/qwen.py:318-358) registers a factory with
 * {@link HfArchEntry#withLoadNameFilter}; the filter it builds is a pure function of the name for that load.</p>
 */
@FunctionalInterface
public interface HfNameFilter {

    /** No class filter: only the inherited filters apply. */
    HfNameFilter INHERITED = (name, base) -> base.apply(name);

    /**
     * The converter name of {@code name}, or null when the tensor is dropped.
     *
     * @param base the inherited filters ({@code super().filter_tensors}): null for a dropped name, else the renamed
     *             name; a class calls it at the point its {@code filter_tensors} calls {@code super()}
     */
    String filter(String name, UnaryOperator<String> base);
}
