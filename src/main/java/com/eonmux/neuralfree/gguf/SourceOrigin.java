package com.eonmux.neuralfree.gguf;

import java.nio.file.Path;
import java.util.Objects;

/**
 * The checkpoint a loaded component came from: its path and container format. Components built over another model
 * (LoRA adapters, drafters) receive it to locate that model's files. Immutable.
 */
public final class SourceOrigin {

    private final Path path;
    private final ModelFormat format;

    public SourceOrigin(Path path, ModelFormat format) {
        this.path = Objects.requireNonNull(path, "path");
        this.format = Objects.requireNonNull(format, "format");
    }

    /** The file or directory the model was opened from. */
    public Path path() { return path; }

    /** The model's container format. */
    public ModelFormat format() { return format; }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof SourceOrigin)) {
            return false;
        }
        SourceOrigin other = (SourceOrigin) o;
        return path.equals(other.path) && format == other.format;
    }

    @Override
    public int hashCode() {
        return 31 * path.hashCode() + format.hashCode();
    }

    @Override
    public String toString() {
        return format + " " + path;
    }
}
