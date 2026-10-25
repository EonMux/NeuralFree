package com.eonmux.neuralfree.gguf;

/** Container format a {@link ModelSource} reads its checkpoint from. */
public enum ModelFormat {
    /** A GGUF file, or the splits of one ({@code <prefix>-0000N-of-0000M.gguf}). */
    GGUF,
    /** A Hugging Face checkpoint: {@code *.safetensors} files with their configuration. */
    SAFETENSORS
}
