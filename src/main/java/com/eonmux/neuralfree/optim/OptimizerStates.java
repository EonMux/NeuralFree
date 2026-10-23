package com.eonmux.neuralfree.optim;

import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuCoreOps;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;

import java.util.List;
import java.util.Map;

/**
 * The state-dict mechanics of {@link Optimizer}: snapshot copies of live buffers, key resolution, validation and
 * copying values back in. Values cross through the host only when the train kernels are unavailable or the source
 * lives on another backend.
 */
final class OptimizerStates {

    private OptimizerStates() {
    }

    /** Element {@code index} of {@code buffers}, or null when the list is shorter or the element is null. */
    static Tensor bufferAt(List<Tensor> buffers, int index) {
        return index < buffers.size() ? buffers.get(index) : null;
    }

    /**
     * A new contiguous FLOAT32 tensor holding the current values of {@code live}, with its own storage on the same
     * backend: a device copy when the train kernels are available, otherwise a host round trip.
     */
    static Tensor snapshot(Tensor live) {
        Tensor device = GpuCoreOps.contiguous(live);
        if (device != null) {
            return device;
        }
        TensorStorage storage = TensorStorage.fromHostFloat(live.backend(), live.toFloatArray());
        try {
            return Tensor.fromStorage(live.backend(), live.shape(), DType.FLOAT32, storage);
        } catch (RuntimeException e) {
            storage.close();
            throw e;
        }
    }

    /** Closes the tensors of a partly built state dict after {@code failure}, recording close failures on it. */
    static void closeAfterFailure(Map<String, Object> state, RuntimeException failure) {
        try {
            Optimizer.closeStateDict(state);
        } catch (RuntimeException closeFailure) {
            failure.addSuppressed(closeFailure);
        }
    }

    /**
     * The live buffer {@code key} names ({@code <family prefix><parameter index>}), or null when the key matches no
     * family (a step count, hyper-parameter or unknown key).
     *
     * @throws IllegalArgumentException when the key matches a family but the optimizer keeps no such buffer
     */
    static Tensor target(String owner, Map<String, List<Tensor>> buffers, int parameterCount, String key) {
        for (Map.Entry<String, List<Tensor>> family : buffers.entrySet()) {
            String prefix = family.getKey();
            if (!key.startsWith(prefix) || !isIndex(key, prefix.length())) {
                continue;
            }
            String digits = key.substring(prefix.length());
            int index = digits.length() > 9 ? Integer.MAX_VALUE : Integer.parseInt(digits);
            Tensor buffer = index < parameterCount ? bufferAt(family.getValue(), index) : null;
            if (buffer == null) {
                throw new IllegalArgumentException(owner + " keeps no state '" + key + "': "
                    + (index >= parameterCount
                        ? "it optimizes " + parameterCount + " parameter(s)"
                        : "its hyper-parameters use no '" + prefix + "' buffer for parameter " + index));
            }
            return buffer;
        }
        return null;
    }

    /** Whether {@code key} ends, from {@code from}, in a decimal index without leading zeros. */
    private static boolean isIndex(String key, int from) {
        if (from >= key.length() || (key.charAt(from) == '0' && key.length() - from > 1)) {
            return false;
        }
        for (int i = from; i < key.length(); i++) {
            char c = key.charAt(i);
            if (c < '0' || c > '9') {
                return false;
            }
        }
        return true;
    }

    /**
     * Checks that {@code value} can be copied into {@code target}: a FLOAT32 tensor of the same shape, or a
     * {@code float[]} of the same element count.
     *
     * @throws IllegalArgumentException otherwise, naming the key and both shapes
     */
    static void requireCompatible(String owner, String key, Tensor target, Object value) {
        if (value instanceof Tensor) {
            Tensor source = (Tensor) value;
            if (source.dtype() != DType.FLOAT32) {
                throw new IllegalArgumentException(owner + " state '" + key + "' must be FLOAT32, got "
                    + source.dtype());
            }
            if (!source.shape().equals(target.shape())) {
                throw new IllegalArgumentException(owner + " state '" + key + "' has shape " + source.shape()
                    + ", but the optimizer's buffer has shape " + target.shape());
            }
        } else if (value instanceof float[]) {
            int length = ((float[]) value).length;
            if (length != target.numElements()) {
                throw new IllegalArgumentException(owner + " state '" + key + "' holds " + length
                    + " values, but the optimizer's buffer of shape " + target.shape() + " holds "
                    + target.numElements());
            }
        } else {
            throw new IllegalArgumentException(owner + " state '" + key + "' must be a Tensor or a float[], got "
                + (value == null ? "null" : value.getClass().getName()));
        }
    }

    /**
     * Copies a value accepted by {@link #requireCompatible} into {@code target}: on the device when the source is a
     * tensor on the same backend and the train kernels are available, otherwise through the host. The caller runs
     * this under {@code ComputationGraph.noGrad()}.
     */
    static void copyInto(Tensor target, Object value) {
        if (value instanceof float[]) {
            GpuOptimizerOps.storeFloats(target, (float[]) value);
            return;
        }
        Tensor source = (Tensor) value;
        if (source == target) {
            return;
        }
        if (source.backend() == target.backend() && TrainGpu.forBackend(target.backend()) != null) {
            GpuCoreOps.copyInto(source, target);
        } else {
            GpuOptimizerOps.storeFloats(target, source.toFloatArray());
        }
    }

    /**
     * The step count a state value holds: a whole number in [0, 2^31-1] given as an integral {@link Number}
     * ({@link Byte}, {@link Short}, {@link Integer}, {@link Long}), a {@link Float} or {@link Double} with an integral
     * value, or a one-element FLOAT32 tensor or {@code float[]} (the format of earlier checkpoints).
     *
     * @throws IllegalArgumentException for any other value
     */
    static int stepCount(Object value) {
        double count;
        if (value instanceof Byte || value instanceof Short || value instanceof Integer || value instanceof Long) {
            long whole = ((Number) value).longValue();
            if (whole < 0 || whole > Integer.MAX_VALUE) {
                throw invalidStepCount(value);
            }
            return (int) whole;
        } else if (value instanceof Float || value instanceof Double) {
            count = ((Number) value).doubleValue();
        } else if (value instanceof Tensor && ((Tensor) value).numElements() == 1
                && ((Tensor) value).dtype() == DType.FLOAT32) {
            count = ((Tensor) value).toFloatArray()[0];
        } else if (value instanceof float[] && ((float[]) value).length == 1) {
            count = ((float[]) value)[0];
        } else {
            throw invalidStepCount(value);
        }
        if (!(count >= 0 && count <= Integer.MAX_VALUE && count == Math.rint(count))) {
            throw invalidStepCount(value);
        }
        return (int) count;
    }

    private static IllegalArgumentException invalidStepCount(Object value) {
        String shown = value instanceof Tensor ? "a tensor of shape " + ((Tensor) value).shape() + " and dtype "
            + ((Tensor) value).dtype()
            : value instanceof float[] ? "a float[" + ((float[]) value).length + "]"
            : String.valueOf(value);
        return new IllegalArgumentException("'" + Optimizer.STEP_COUNT_KEY
            + "' must be a whole number in [0, 2^31-1] (a Number, or a one-element FLOAT32 tensor or float[]), got "
            + shown);
    }
}
