package com.eonmux.neuralfree.gguf.hf.torchao;

import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * One torchao tensor subclass of a checkpoint, after its serialized form has been read: the tensor's checkpoint name,
 * the subclass name ({@code Int8Tensor}, {@code NVFP4Tensor}, ...), the subclass attributes the serialization stored
 * (block size, shape, swizzle flag, ...), the logical shape when the form carries one, and the inner tensors the
 * subclass decomposes into. The readers build it from safetensors {@code __metadata__} descriptors or from a
 * PyTorch pickle's {@code _rebuild_wrapper_subclass} state. Immutable.
 */
public final class TorchaoWeight {

    /** One inner tensor of the subclass: its stored dtype, shape and byte source. */
    public static final class Inner {
        private final StDType dtype;
        private final long[] shape;
        private final RowSource source;
        private final String rawName;

        public Inner(StDType dtype, long[] shape, RowSource source, String rawName) {
            this.dtype = Objects.requireNonNull(dtype, "dtype");
            this.shape = Objects.requireNonNull(shape, "shape").clone();
            this.source = Objects.requireNonNull(source, "source");
            this.rawName = rawName;
        }

        /** The dtype the inner tensor is stored as. */
        public StDType dtype() { return dtype; }

        /** The inner tensor's shape, outermost dimension first (a copy). */
        public long[] shape() { return shape.clone(); }

        /** The bytes of the inner tensor. */
        public RowSource source() { return source; }

        /** The checkpoint tensor the inner bytes are stored in, or null for a pickle form that stores no such name. */
        public String rawName() { return rawName; }
    }

    private final String name;
    private final String type;
    private final long[] shape;
    private final Map<String, Object> attributes;
    private final Map<String, Inner> tensors;

    /**
     * @param name       the tensor's checkpoint name (for example {@code model.layers.0.q_proj.weight})
     * @param type       the torchao subclass simple name
     * @param shape      the logical tensor shape from the serialized form, or null when it derives from the weights
     * @param attributes the subclass attributes the form stored
     * @param tensors    the inner tensors by their torchao data name
     */
    public TorchaoWeight(String name, String type, long[] shape, Map<String, Object> attributes,
                         Map<String, Inner> tensors) {
        this.name = Objects.requireNonNull(name, "name");
        this.type = Objects.requireNonNull(type, "type");
        this.shape = shape == null ? null : shape.clone();
        this.attributes = Collections.unmodifiableMap(new LinkedHashMap<>(attributes));
        this.tensors = Collections.unmodifiableMap(new LinkedHashMap<>(tensors));
    }

    /** The tensor's checkpoint name. */
    public String name() { return name; }

    /** The torchao subclass simple name. */
    public String type() { return type; }

    /** The logical tensor shape from the serialized form, or null when the form does not carry one. */
    public long[] shape() { return shape == null ? null : shape.clone(); }

    /** The subclass attribute {@code key}, or null when the form did not store it. */
    public Object attribute(String key) { return attributes.get(key); }

    /** One inner tensor by its torchao data name, or null when the subclass has no such tensor. */
    public Inner tensor(String key) { return tensors.get(key); }

    /** The inner tensors by data name, in the order the form listed them. Unmodifiable. */
    public Map<String, Inner> tensors() { return tensors; }

    /** The names of the attributes the form stored. Unmodifiable. */
    public List<String> attributeNames() { return Collections.unmodifiableList(
            new java.util.ArrayList<>(attributes.keySet())); }

    @Override
    public String toString() {
        return "TorchaoWeight{" + name + " " + type + " " + tensors.keySet() + "}";
    }
}
