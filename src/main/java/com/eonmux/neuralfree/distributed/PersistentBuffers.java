package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorValues;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The persistent buffers of a module (the buffers its {@link Module#stateDict()} lists, such as running statistics),
 * listed and read as state dict values for the code that broadcasts or averages them. Values of every dtype are read
 * as the state dict reads them ({@link TensorValues#toFloats}), so {@link Module#loadStateDict(Map, boolean)} writes
 * them back in each buffer's own dtype.
 */
final class PersistentBuffers {

    private PersistentBuffers() {
    }

    /** The state dict paths of {@code module}'s persistent buffers, in state dict order, given its {@code state}. */
    static List<String> paths(Module module, Map<String, Object> state) {
        Objects.requireNonNull(module, "Module cannot be null");
        Objects.requireNonNull(state, "State dict cannot be null");
        Map<String, Tensor> buffers = module.namedBuffers();
        if (buffers.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> paths = new ArrayList<>();
        for (String path : state.keySet()) {
            if (buffers.containsKey(path)) {
                paths.add(path);
            }
        }
        return Collections.unmodifiableList(paths);
    }

    /** The state dict paths of {@code module}'s persistent buffers, in state dict order. */
    static List<String> paths(Module module) {
        Objects.requireNonNull(module, "Module cannot be null");
        if (module.namedBuffers().isEmpty()) {
            return Collections.emptyList();
        }
        return paths(module, module.stateDict());
    }

    /**
     * The current values of the buffers at {@code paths}, keyed by path in that order, as {@code float[]} state dict
     * values; nothing is recorded.
     *
     * @throws IllegalStateException when a path no longer names a buffer of {@code module}
     */
    static Map<String, Object> read(Module module, List<String> paths) {
        Map<String, Object> values = new LinkedHashMap<>();
        if (paths.isEmpty()) {
            return values;
        }
        Map<String, Tensor> buffers = module.namedBuffers();
        for (String path : paths) {
            Tensor buffer = buffers.get(path);
            if (buffer == null) {
                throw new IllegalStateException("Persistent buffer '" + path + "' is no longer registered in "
                    + module.moduleName());
            }
            values.put(path, TensorValues.toFloats(buffer));
        }
        return values;
    }
}
