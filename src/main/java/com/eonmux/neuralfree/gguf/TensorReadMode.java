package com.eonmux.neuralfree.gguf;

import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * How GGUF tensor bytes reach host memory (llama.cpp's load modes, {@code include/llama.h:207-214} plus its
 * lazy mode and direct I/O): memory-mapped in place, copied into allocated buffers, copied through sector-aligned
 * reads, or read on demand. The setting is process-wide and comes from the properties below; every reader of
 * tensor data ({@code CpuWeights}, {@link MappedWeightStore}) consults {@link #current()} when it opens a model.
 * <ul>
 *   <li>{@value #PROP_DISABLE_MMAP}: the boolean shorthand, {@code --no-mmap}; {@code true} selects READ, absent
 *       or {@code false} selects MMAP.</li>
 *   <li>{@value #PROP_READ_MODE}: mmap | read | direct | lazy, the full form. When it is set it decides and
 *       {@value #PROP_DISABLE_MMAP} is ignored (the two name the same choice: read stands for
 *       {@code --no-mmap}).</li>
 * </ul>
 */
public enum TensorReadMode {

    /** Tensors are memory-mapped read-only; the OS page cache backs them (the default). */
    MMAP,
    /** Tensors are read into buffers owned by the process; nothing is mapped ({@code --no-mmap}). */
    READ,
    /** Tensors are read like READ but through sector-aligned channel reads, with nothing relied on from the page
     *  cache beyond the reads themselves (direct I/O; see {@code FileTensorRegions}). */
    DIRECT,
    /**
     * Tensor bytes are read on demand, per region as the gather asks ({@code --lazy-mode}, common/arg.cpp:2714-2722):
     * per-layer embeddings and other row-gathered tensors are not materialized; their rows are read when gathered.
     */
    LAZY;

    /** {@code true} selects {@link #READ}; absent or {@code false} selects {@link #MMAP}. */
    public static final String PROP_DISABLE_MMAP = "neuralfree.disable.mmap";
    /** The full form of the same choice: mmap | read | direct | lazy (see the class header). */
    public static final String PROP_READ_MODE = "neuralfree.tensor.read";

    /**
     * The mode of a property set.
     *
     * @throws IllegalArgumentException when {@value #PROP_READ_MODE} names no mode or
     *                                  {@value #PROP_DISABLE_MMAP} is neither {@code true} nor {@code false}
     */
    public static TensorReadMode fromProperties(Properties p) {
        Objects.requireNonNull(p, "properties");
        String v = p.getProperty(PROP_READ_MODE);
        if (v != null && !v.trim().isEmpty()) {
            String s = v.trim().toUpperCase(Locale.ROOT).replace('-', '_');
            StringBuilder valid = new StringBuilder();
            for (TensorReadMode m : TensorReadMode.values()) {
                if (m.name().equals(s)) {
                    return m;
                }
                valid.append(valid.length() == 0 ? "" : ", ").append(m.name().toLowerCase(Locale.ROOT));
            }
            throw new IllegalArgumentException(PROP_READ_MODE + "=" + v + " is invalid; use " + valid);
        }
        v = p.getProperty(PROP_DISABLE_MMAP);
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
     * @throws IllegalArgumentException when the properties name no mode (see {@link #fromProperties})
     */
    public static TensorReadMode current() {
        return fromProperties(System.getProperties());
    }
}
