package com.eonmux.neuralfree.demo.tools;

/**
 * The machine-readable output formats of {@code nftool bench}, llama.cpp llama-bench's {@code -o} and
 * {@code -oe} set ({@code tools/llama-bench/llama-bench.cpp}): {@code none}, {@code csv}, {@code json},
 * {@code jsonl}, {@code md} (Markdown) and {@code sql}. Immutable.
 */
enum BenchOutput {

    NONE,
    CSV,
    JSON,
    JSONL,
    MARKDOWN,
    SQL;

    /** The name llama.cpp's flag accepts and its usage prints. */
    String flagName() {
        return this == MARKDOWN ? "md" : name().toLowerCase(java.util.Locale.ROOT);
    }

    /**
     * Parses a format name, case-sensitive like the reference.
     *
     * @param flag name of the flag being parsed, for the error message
     * @throws BenchCommand.UsageException for any other value
     */
    static BenchOutput parse(String value, String flag) throws BenchCommand.UsageException {
        for (BenchOutput f : values()) {
            if (f.flagName().equals(value)) {
                return f;
            }
        }
        throw new BenchCommand.UsageException(flag + " needs one of md, none, csv, json, jsonl or sql, not '"
                + value + "'");
    }
}
