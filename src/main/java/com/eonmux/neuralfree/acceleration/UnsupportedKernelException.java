package com.eonmux.neuralfree.acceleration;

/**
 * The selected backend or device cannot provide a kernel: the kernel is missing
 * from the backend's kernel library, the artifact kind is not loadable, or the
 * device lacks a capability the kernel requires.
 *
 * <p>The message names the kernel, the backend and the reason. Engine code may
 * catch it to disable an optional feature; a required kernel lets it propagate.</p>
 */
public class UnsupportedKernelException extends RuntimeException {

    public UnsupportedKernelException(String message) {
        super(message);
    }

    public UnsupportedKernelException(String message, Throwable cause) {
        super(message, cause);
    }
}
