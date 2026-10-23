package com.eonmux.neuralfree.gguf.arch;

/** The architecture cannot be run: it is unknown, has no registry entry, or is {@link SupportClass#NA}. */
public class UnsupportedArchException extends UnsupportedOperationException {

    private static final long serialVersionUID = 1L;

    private final Arch arch;

    public UnsupportedArchException(Arch arch, String message) {
        super(message);
        this.arch = arch;
    }

    /** The rejected architecture. */
    public Arch arch() {
        return arch;
    }
}
