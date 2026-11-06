package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.Arch;

/** The projector type of an mmproj file has no entry in the projector registry. */
public class UnsupportedProjectorException extends UnsupportedOperationException {

    private static final long serialVersionUID = 1L;

    private final String projectorType;

    public UnsupportedProjectorException(String projectorType, String message) {
        super(message);
        this.projectorType = projectorType;
    }

    /** The {@code clip.projector_type} value that has no entry, or the raw file value when unparsable. */
    public String projectorType() {
        return projectorType;
    }
}
