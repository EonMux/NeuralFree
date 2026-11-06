package com.eonmux.neuralfree.gguf.hub.slice;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * What one slice run found and did: the repository and the commit it read, how many tensors the sources hold and
 * the slice keeps, the slice's byte size, the directory it wrote (null for {@code --inspect} and {@code --dry-run}),
 * the inspect report when one was asked for, the config changes, the auxiliary files the size rule skipped and the
 * warnings about heuristics that found nothing. Immutable.
 */
public final class SliceReport {

    private final String repository;
    private final String commit;
    private final int keptTensors;
    private final int totalTensors;
    private final long bytes;
    private final Path directory;
    private final String inspect;
    private final boolean dryRun;
    private final List<String> configPatches;
    private final List<String> skippedAux;
    private final List<String> warnings;

    SliceReport(String repository, String commit, int keptTensors, int totalTensors, long bytes, Path directory,
                String inspect, boolean dryRun, List<String> configPatches, List<String> skippedAux,
                List<String> warnings) {
        this.repository = repository;
        this.commit = commit;
        this.keptTensors = keptTensors;
        this.totalTensors = totalTensors;
        this.bytes = bytes;
        this.directory = directory;
        this.inspect = inspect;
        this.dryRun = dryRun;
        this.configPatches = Collections.unmodifiableList(new ArrayList<>(configPatches));
        this.skippedAux = Collections.unmodifiableList(new ArrayList<>(skippedAux));
        this.warnings = Collections.unmodifiableList(new ArrayList<>(warnings));
    }

    /** The repository id that was read. */
    public String repository() {
        return repository;
    }

    /** The commit the repository resolved to. */
    public String commit() {
        return commit;
    }

    /** How many tensors the slice keeps. */
    public int keptTensors() {
        return keptTensors;
    }

    /** How many tensors the source shards hold. */
    public int totalTensors() {
        return totalTensors;
    }

    /** The byte length of the kept tensors. */
    public long bytes() {
        return bytes;
    }

    /** The directory the slice was written to; null when the run wrote nothing. */
    public Path directory() {
        return directory;
    }

    /** The inspect report, or null when none was asked for. */
    public String inspect() {
        return inspect;
    }

    /** Whether the run planned without writing. */
    public boolean dryRun() {
        return dryRun;
    }

    /** One line per config entry the slice changed. */
    public List<String> configPatches() {
        return configPatches;
    }

    /** The auxiliary files that were too large to copy, with their sizes. */
    public List<String> skippedAux() {
        return skippedAux;
    }

    /** Warnings about the heuristics, such as a model whose layers were not recognized. */
    public List<String> warnings() {
        return warnings;
    }

    /** The byte count in the reference skill's units. */
    public String humanBytes() {
        return SlicePlan.human(bytes);
    }
}
