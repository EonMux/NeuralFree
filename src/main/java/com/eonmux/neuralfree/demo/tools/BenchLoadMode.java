package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.TensorReadMode;

/**
 * One llama-bench {@code -lm}/{@code --load-mode} value of {@code nftool bench}: llama.cpp's
 * {@code llama_load_mode} names ({@code auto}, {@code none}, {@code mmap}, {@code mlock}, {@code mmap+mlock},
 * {@code dio}) mapped onto the engine's process-wide tensor read mode ({@link TensorReadMode}): {@code none} reads
 * tensors into process buffers, {@code mmap} maps them, {@code mlock} reads them and pins the pages in RAM,
 * {@code mmap+mlock} maps and pins them, and {@code dio} reads them directly. {@code auto} leaves the configured
 * read mode. Immutable.
 */
enum BenchLoadMode {

    AUTO("auto", null),
    NONE("none", TensorReadMode.READ),
    MMAP("mmap", TensorReadMode.MMAP),
    MLOCK("mlock", TensorReadMode.MLOCK),
    MMAP_MLOCK("mmap+mlock", TensorReadMode.MMAP_MLOCK),
    DIRECT_IO("dio", TensorReadMode.DIRECT);

    private final String llamaName;
    private final TensorReadMode readMode;

    BenchLoadMode(String llamaName, TensorReadMode readMode) {
        this.llamaName = llamaName;
        this.readMode = readMode;
    }

    /** The name llama-bench accepts and prints. */
    String llamaName() {
        return llamaName;
    }

    /** The read mode this value selects, or null for {@code auto} (the configured mode). */
    TensorReadMode readMode() {
        return readMode;
    }

    /**
     * Parses one {@code -lm} value, case-sensitively like llama-bench.
     *
     * @param flag name of the flag being parsed, for the error messages
     * @throws BenchCommand.UsageException for an unknown name
     */
    static BenchLoadMode parse(String value, String flag) throws BenchCommand.UsageException {
        String v = value.trim();
        for (BenchLoadMode m : values()) {
            if (m.llamaName.equals(v)) {
                return m;
            }
        }
        throw new BenchCommand.UsageException(flag + " needs one of auto, none, mmap, mlock, mmap+mlock or dio,"
                + " not '" + v + "'");
    }
}
