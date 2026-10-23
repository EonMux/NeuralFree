package com.eonmux.neuralfree.gguf;

import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * How GGUF tensor bytes reach host memory: memory-mapped in place, or read into allocated buffers (llama.cpp
 * {@code use_mmap} / {@code --no-mmap}, src/llama-model-loader.cpp). The setting is process-wide and comes from one
 * property, {@value #PROP_DISABLE_MMAP}; every reader of tensor data ({@code CpuWeights}, {@link MappedWeightStore})
 * consults {@link #current()} when it opens a model.
 */
public enum TensorReadMode {

    /** Tensors are memory-mapped read-only; the OS page cache backs them (the default). */
    MMAP,
    /** Tensors are read into buffers owned by the process; nothing is mapped. */
    READ;

    /** {@code true} selects {@link #READ}; absent or {@code false} selects {@link #MMAP}. */
    public static final String PROP_DISABLE_MMAP = "neuralfree.disable.mmap";

    /**
     * The mode a property set selects.
     *
     * @throws IllegalArgumentException when {@value #PROP_DISABLE_MMAP} is neither {@code true} nor {@code false}
     */
    public static TensorReadMode fromProperties(Properties p) {
        Objects.requireNonNull(p, "properties");
        String v = p.getProperty(PROP_DISABLE_MMAP);
        if (v == null || v.trim().isEmpty()) {
            return MMAP;
        }
        String s = v.trim().toLowerCase(Locale.ROOT);
        if (s.equals("true")) {
            return READ;
        }
        if (s.equals("false")) {
            return MMAP;
        }
        throw new IllegalArgumentException(PROP_DISABLE_MMAP + "=" + v + " is not a boolean; use true or false");
    }

    /**
     * The mode of this process ({@link System#getProperties()}).
     *
     * @throws IllegalArgumentException when {@value #PROP_DISABLE_MMAP} is neither {@code true} nor {@code false}
     */
    public static TensorReadMode current() {
        return fromProperties(System.getProperties());
    }
}
