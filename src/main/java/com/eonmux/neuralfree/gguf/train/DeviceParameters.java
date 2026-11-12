package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The device trainer's parameters: one device leaf tensor per trainable parameter, which holds the authoritative
 * values and gradients, and the host arrays of {@link TrainableModel.Parameter} kept as a lazily synchronized mirror.
 *
 * <p><b>Coherence.</b> The device copy is the one a window computes with and a device optimizer steps. The arrays
 * the {@link #parameters()} hand out follow it through their {@link TrainableModel.HostMirror}: an array is read back
 * only when an accessor is called and the device holds a newer copy, and an array handed out live
 * ({@link TrainableModel.Parameter#value()}) may be written, so it is uploaded before the next device use
 * ({@link #syncDevice()}). A run that steps the optimizer on the device therefore moves no parameter or gradient
 * between host and device per window; reading the arrays (a save, an evaluation by a host optimizer, a test) moves
 * exactly what is read.</p>
 *
 * <p>Not thread-safe; owned by one {@link GpuTrainableModel}.</p>
 */
final class DeviceParameters implements AutoCloseable {

    private final List<TrainableModel.Parameter> host;
    private final List<TrainableModel.Parameter> mirrored;
    private final Map<String, Integer> slots;
    private final Tensor[] tensors;
    /** The device copy of slot i is newer than its host values. */
    private final boolean[] hostValuesStale;
    /** The host values of slot i may be newer than the device copy. */
    private final boolean[] deviceValuesStale;
    /** The device gradient of slot i is newer than its host gradient array. */
    private final boolean[] hostGradStale;
    /** Bytes of values and gradients moved between host and device since the initial upload. */
    private long transferredBytes;
    private boolean closed;

    private DeviceParameters(List<TrainableModel.Parameter> host, Tensor[] tensors) {
        this.host = Collections.unmodifiableList(new ArrayList<>(host));
        this.tensors = tensors;
        int n = host.size();
        this.hostValuesStale = new boolean[n];
        this.deviceValuesStale = new boolean[n];
        this.hostGradStale = new boolean[n];
        Map<String, Integer> bySlot = new HashMap<>();
        List<TrainableModel.Parameter> handedOut = new ArrayList<>(n);
        TrainableModel.HostMirror mirror = new Mirror();
        for (int i = 0; i < n; i++) {
            bySlot.put(host.get(i).name(), i);
            handedOut.add(host.get(i).mirroredBy(mirror));
        }
        this.slots = Collections.unmodifiableMap(bySlot);
        this.mirrored = Collections.unmodifiableList(handedOut);
    }

    /**
     * Uploads every parameter of {@code host} to {@code backend} as a leaf that requires grad, shaped
     * {@code [rows, cols]} ({@code [cols]} for a vector).
     */
    static DeviceParameters upload(ComputeBackend backend, List<TrainableModel.Parameter> host) {
        Objects.requireNonNull(backend, "backend");
        Tensor[] tensors = new Tensor[host.size()];
        try {
            for (int i = 0; i < tensors.length; i++) {
                tensors[i] = leaf(backend, host.get(i));
            }
        } catch (RuntimeException e) {
            closeAll(tensors, e);
            throw e;
        }
        return new DeviceParameters(host, tensors);
    }

    private static Tensor leaf(ComputeBackend backend, TrainableModel.Parameter p) {
        Shape shape = p.ne().length == 1 ? Shape.of(p.cols()) : Shape.of(p.rows(), p.cols());
        Tensor flat = Tensor.fromArray(backend, p.value());
        Tensor shaped = flat.shape().equals(shape) ? flat : flat.reshape(shape);
        if (shaped != flat) {
            flat.close();
        }
        Tensor leaf = shaped.withRequiresGrad(true);
        if (leaf != shaped) {
            shaped.close();
        }
        return leaf;
    }

    /** The parameters with their mirrored host arrays, in plan order. */
    List<TrainableModel.Parameter> parameters() {
        return mirrored;
    }

    /** The mirrored parameter of a stored tensor name, or null. */
    TrainableModel.Parameter parameter(String name) {
        Integer slot = slots.get(name);
        return slot == null ? null : mirrored.get(slot);
    }

    /** The parameter count. */
    int size() {
        return tensors.length;
    }

    /** The device leaf of slot {@code i}. */
    Tensor tensor(int i) {
        ensureOpen();
        return tensors[i];
    }

    /** The device leaf of a stored tensor name. */
    Tensor tensor(String name) {
        Integer slot = slots.get(name);
        if (slot == null) {
            throw new IllegalArgumentException("the model has no trainable tensor '" + name + "'");
        }
        return tensor(slot);
    }

    /** The host parameter of slot {@code i}, whose arrays are not synchronized on access. */
    TrainableModel.Parameter hostParameter(int i) {
        return host.get(i);
    }

    /**
     * The bytes of values and gradients this mirror moved between host and device after the initial upload: the
     * synchronization traffic a run pays, zero for windows that step on the device and read no array.
     */
    long transferredBytes() {
        return transferredBytes;
    }

    /** Uploads the host values that may have been written since the device copy was last current. */
    void syncDevice() {
        ensureOpen();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (int i = 0; i < tensors.length; i++) {
                if (deviceValuesStale[i]) {
                    GpuEvent done = tensors[i].storage().copyFromHostFloat(host.get(i).value(),
                            tensors[i].backend().transferQueue());
                    if (done != null) {
                        done.close();
                    }
                    transferredBytes += (long) host.get(i).size() * Float.BYTES;
                    deviceValuesStale[i] = false;
                }
            }
        }
    }

    /** Records that the device stepped every parameter: the host values are read back on their next access. */
    void deviceValuesChanged() {
        Arrays.fill(hostValuesStale, true);
    }

    /** Records that a backward pass or {@link #zeroGrad()} changed the device gradients. */
    void deviceGradientsChanged() {
        Arrays.fill(hostGradStale, true);
    }

    /**
     * Clears every parameter's device gradient (a parameter with none reads as zeros), without touching gradients
     * other models hold in the thread's graph.
     */
    void zeroGrad() {
        ensureOpen();
        RuntimeException failure = null;
        for (Tensor t : tensors) {
            Tensor g = t.grad();
            if (g != null) {
                t.setGrad(null);
                try {
                    g.close();
                } catch (RuntimeException e) {
                    failure = collect(failure, e);
                }
            }
        }
        deviceGradientsChanged();
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        for (Tensor t : tensors) {
            Tensor g = t.grad();
            if (g != null) {
                t.setGrad(null);
                failure = closeOne(g, failure);
            }
            failure = closeOne(t, failure);
        }
        if (failure != null) {
            throw failure;
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the device trainer is closed");
        }
    }

    private int slot(String name) {
        Integer slot = slots.get(name);
        if (slot == null) {
            throw new IllegalStateException("no device parameter '" + name + "'");
        }
        return slot;
    }

    private static RuntimeException closeOne(Tensor t, RuntimeException failure) {
        try {
            t.close();
            return failure;
        } catch (RuntimeException e) {
            return collect(failure, e);
        }
    }

    private static RuntimeException collect(RuntimeException first, RuntimeException next) {
        if (first == null) {
            return next;
        }
        first.addSuppressed(next);
        return first;
    }

    private static void closeAll(Tensor[] tensors, RuntimeException primary) {
        for (Tensor t : tensors) {
            if (t != null) {
                try {
                    t.close();
                } catch (RuntimeException e) {
                    primary.addSuppressed(e);
                }
            }
        }
    }

    /** The coherence of the handed-out arrays with the device copy. */
    private final class Mirror implements TrainableModel.HostMirror {

        @Override
        public void beforeValues(String name, boolean mayWrite) {
            int i = slot(name);
            if (hostValuesStale[i]) {
                ensureOpen();
                float[] fresh = tensors[i].toFloatArray();
                float[] target = host.get(i).value();
                System.arraycopy(fresh, 0, target, 0, target.length);
                transferredBytes += (long) target.length * Float.BYTES;
                hostValuesStale[i] = false;
            }
            if (mayWrite) {
                deviceValuesStale[i] = true;
            }
        }

        @Override
        public void beforeGrad(String name) {
            int i = slot(name);
            if (!hostGradStale[i]) {
                return;
            }
            ensureOpen();
            float[] target = host.get(i).grad();
            Tensor g = tensors[i].grad();
            if (g == null) {
                Arrays.fill(target, 0f);
            } else {
                float[] fresh = g.toFloatArray();
                System.arraycopy(fresh, 0, target, 0, target.length);
                transferredBytes += (long) target.length * Float.BYTES;
            }
            hostGradStale[i] = false;
        }
    }
}
