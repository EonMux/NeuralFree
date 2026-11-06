package com.eonmux.neuralfree.acceleration;

/**
 * The selected device cannot serve this model: an architecture, mixer kind, FFN kind, layer feature or weight type
 * the device's engine does not implement, refused by name with the fix or the offending artifact in the message.
 * Thrown while a model loads on a GPU backend, before or while its weights upload and its blocks bind; the model
 * factory treats it as a device limitation and falls back to the CPU engine, while failures of other kinds (a kernel
 * argument left unbound, an assertion-style state, a failed kernel launch) fail the load instead.
 *
 * <p>Extends {@link UnsupportedOperationException} because it is one: the same refusal a host-only feature raises,
 * with a type the load path can tell apart from an engine defect.</p>
 */
public final class GpuModelUnsupportedException extends UnsupportedOperationException {

    public GpuModelUnsupportedException(String message) {
        super(message);
    }

    public GpuModelUnsupportedException(String message, Throwable cause) {
        super(message, cause);
    }
}
