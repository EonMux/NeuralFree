package com.eonmux.neuralfree.gguf;

import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * How GGUF tensor bytes reach host memory (llama.cpp's load modes, {@code include/llama.h:207-214} plus its
 * lazy mode and direct I/O): memory-mapped in place, copied into allocated buffers, copied through sector-aligned
 * reads, or read on demand, each optionally pinned in RAM with {@code mlock}. The setting is process-wide and comes
 * from the properties below; every reader of tensor data ({@code CpuWeights}, {@link MappedWeightStore}) consults
 * {@link #current()} when it opens a model.
 * <ul>
 *   <li>{@value #PROP_DISABLE_MMAP}: the boolean shorthand, {@code --no-mmap}; {@code true} selects READ, absent
 *       or {@code false} selects MMAP.</li>
 *   <li>{@value #PROP_READ_MODE}: mmap | read | direct | lazy | mlock | mmap+mlock, the full form. When it is set it
 *       decides and {@value #PROP_DISABLE_MMAP} is ignored (the two name the same choice: read stands for
 *       {@code --no-mmap}).</li>
 * </ul>
 *
 * <p>The modes llama.cpp's {@code --load-mode} names map one to one: {@code none} is {@link #READ}, {@code mmap}
 * is {@link #MMAP}, {@code mlock} is {@link #MLOCK}, {@code mmap+mlock} is {@link #MMAP_MLOCK} and {@code dio} is
 * {@link #DIRECT}. {@link #MLOCK} reads like {@link #READ} and pins what it reads; {@link #MMAP_MLOCK} maps and pins
 * the mapping. Direct I/O bypasses the page cache, so it has no locked variant, and a lazily read tensor has no
 * resident bytes to pin.
 */
public enum TensorReadMode {

    /** Tensors are memory-mapped read-only; the OS page cache backs them (the default). */
    MMAP("mmap"),
    /** Tensors are read into buffers owned by the process and those buffers are pinned in RAM. */
    MLOCK("mlock"),
    /** Tensors are memory-mapped read-only and the mapping is pinned in RAM. */
    MMAP_MLOCK("mmap+mlock"),
    /** Tensors are read into buffers owned by the process; nothing is mapped ({@code --no-mmap}). */
    READ("read"),
    /** Tensors are read like READ but through sector-aligned channel reads, with nothing relied on from the page
     *  cache beyond the reads themselves (direct I/O; see {@code FileTensorRegions}). */
    DIRECT("direct"),
    /**
     * Tensor bytes are read on demand, per region as the gather asks ({@code --lazy-mode}, common/arg.cpp:2714-2722):
     * per-layer embeddings and other row-gathered tensors are not materialized; their rows are read when gathered.
     */
    LAZY("lazy");

    /** {@code true} selects {@link #READ}; absent or {@code false} selects {@link #MMAP}. */
    public static final String PROP_DISABLE_MMAP = "neuralfree.disable.mmap";
    /** The full form of the same choice: mmap | read | direct | lazy | mlock | mmap+mlock (see the class header). */
    public static final String PROP_READ_MODE = "neuralfree.tensor.read";

    private final String flagName;

    TensorReadMode(String flagName) {
        this.flagName = flagName;
    }

    /** The name {@code --load-mode} and {@value #PROP_READ_MODE} spell this mode with. */
    public String flagName() {
        return flagName;
    }

    /** Whether tensors are served from a memory mapping (one whose pages may be pinned under {@link #MMAP_MLOCK}). */
    public boolean usesMmap() {
        return this == MMAP || this == MMAP_MLOCK;
    }

    /** Whether the tensor bytes are pinned in RAM with {@code mlock}. */
    public boolean locksMemory() {
        return this == MLOCK || this == MMAP_MLOCK;
    }

    /**
     * The mode with memory mapping turned on or off ({@code --no-mmap} and its inverse), keeping a pin:
     * {@link #MMAP_MLOCK} becomes {@link #MLOCK} when mapping is turned off, {@link #MLOCK} becomes
     * {@link #MMAP_MLOCK} when it is turned on, and {@link #MMAP} and {@link #READ} become each other. The modes with
     * another reader ({@link #DIRECT}, {@link #LAZY}) are returned unchanged.
     */
    public TensorReadMode withMmap(boolean mmap) {
        switch (this) {
            case MMAP:
                return mmap ? MMAP : READ;
            case READ:
                return mmap ? MMAP : READ;
            case MLOCK:
                return mmap ? MMAP_MLOCK : MLOCK;
            case MMAP_MLOCK:
                return mmap ? MMAP_MLOCK : MLOCK;
            default:
                return this;
        }
    }

    /**
     * The mode with RAM pinning turned on or off ({@code --mlock} and its inverse).
     *
     * @throws IllegalArgumentException when pinning is asked of a mode without resident tensor bytes ({@link #DIRECT}
     *                                  bypasses the page cache; {@link #LAZY} reads rows on demand)
     */
    public TensorReadMode withMlock(boolean mlock) {
        if (!mlock) {
            switch (this) {
                case MLOCK:
                    return READ;
                case MMAP_MLOCK:
                    return MMAP;
                default:
                    return this;
            }
        }
        switch (this) {
            case MMAP:
            case MMAP_MLOCK:
                return MMAP_MLOCK;
            case READ:
            case MLOCK:
                return MLOCK;
            default:
                throw new IllegalArgumentException("mlock cannot pin " + flagName + " tensors: "
                        + (this == DIRECT
                        ? "direct I/O bypasses the page cache, so nothing stays resident to lock"
                        : "lazily read tensors have no resident bytes to lock"));
        }
    }

    /**
     * Parses one mode name: a {@linkplain #flagName() flag name} or the enum name in lower case.
     *
     * @throws IllegalArgumentException naming the mode when it is unknown
     */
    public static TensorReadMode parse(String value) {
        String s = Objects.requireNonNull(value, "read mode").trim().toLowerCase(Locale.ROOT);
        if (s.equals("none")) {
            // llama.cpp's LLAMA_LOAD_MODE_NONE: no mmap, no direct I/O
            return READ;
        }
        String alias = s.replace('-', '_');
        for (TensorReadMode m : values()) {
            if (m.flagName.equals(s) || m.name().toLowerCase(Locale.ROOT).equals(alias)) {
                return m;
            }
        }
        StringBuilder valid = new StringBuilder();
        for (TensorReadMode m : values()) {
            valid.append(valid.length() == 0 ? "" : ", ").append(m.flagName);
        }
        throw new IllegalArgumentException(PROP_READ_MODE + "=" + value + " is invalid; use " + valid);
    }

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
            return parse(v);
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
