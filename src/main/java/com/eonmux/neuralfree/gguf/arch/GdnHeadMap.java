package com.eonmux.neuralfree.gguf.arch;

/**
 * Mapping from value heads to key heads in gated delta-net layers. Qwen3-Next GGUFs keep the grouped
 * order; Qwen3.5 GGUFs are reordered to tiled by the converter.
 */
public enum GdnHeadMap {
    /** Value head {@code h} uses key head {@code h % n_k_heads}. */
    TILED(0),
    /** Value head {@code h} uses key head {@code h / (n_v_heads / n_k_heads)}. */
    GROUPED(1);

    private final int code;

    GdnHeadMap(int code) {
        this.code = code;
    }

    /** Kernel code. */
    public int code() {
        return code;
    }
}
