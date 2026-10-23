package com.eonmux.neuralfree.acceleration.backend;

/**
 * Physical class of a compute device, used to rank devices across APIs.
 *
 * <p>The declaration order is the preference order for backend selection:
 * a discrete GPU is preferred over an integrated one, which is preferred over a
 * virtual or CPU device.</p>
 */
public enum DeviceClass {
    /** A GPU with its own memory, attached over PCIe or a similar bus. */
    DISCRETE,
    /** A GPU that shares system memory with the host CPU. */
    INTEGRATED,
    /** A virtualised or paravirtualised GPU. */
    VIRTUAL,
    /** A CPU device exposed through a GPU API (for example llvmpipe or PoCL). */
    CPU,
    /** The driver did not report the class. */
    UNKNOWN
}
