package com.eonmux.neuralfree.gguf.hf.torchao;

import com.eonmux.neuralfree.io.safetensors.StDType;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * The torchao serialization vocabulary the restricted PyTorch unpickler accepts: the tensor subclasses a checkpoint
 * may wrap, the classes of their serialized attributes, the enum classes those attributes name, and the torch
 * constants (dtypes, device, layout) a wrapper rebuild carries. Nothing else of torchao is allowed, and the
 * rebuilt objects are plain immutable descriptors, never the Python classes.
 */
public final class TorchaoPickle {

    /** Torchao tensor subclasses, by fully qualified class name. */
    public static final Set<String> TENSOR_CLASSES = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
            "torchao.quantization.Int8Tensor",
            "torchao.quantization.Int4Tensor",
            "torchao.quantization.Int4TilePackedTo4dTensor",
            "torchao.quantization.Int4PreshuffledTensor",
            "torchao.quantization.Int4PlainInt32Tensor",
            "torchao.quantization.IntxUnpackedToInt8Tensor",
            "torchao.quantization.Float8Tensor",
            "torchao.prototype.mx_formats.mx_tensor.MXTensor",
            "torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor")));

    /** The simple names the safetensors metadata may store as {@code _type}. */
    public static final Set<String> TENSOR_SIMPLE_NAMES;
    /** Attribute dataclasses of the tensor subclasses, by simple class name (module must be under torchao). */
    public static final Set<String> ATTRIBUTE_SIMPLE_NAMES = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList(
                    "PerRow", "PerTensor", "PerGroup", "PerAxis",
                    "QuantizeTensorToInt8Kwargs", "QuantizeTensorToFloat8Kwargs",
                    "QuantizeTensorToMXKwargs", "QuantizeTensorToNVFP4Kwargs",
                    "QuantizeTensorToInt4Kwargs", "Float8MMConfig")));

    /** Enum classes the attributes name, by simple class name (module must be under torchao). */
    public static final Set<String> ENUM_SIMPLE_NAMES = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList(
                    "KernelPreference", "MappingType", "ScaleCalculationMode", "Int4PackingFormat",
                    "Int4ChooseQParamsAlgorithm", "Float8PackingFormat", "QuantizationStep",
                    "ZeroPointDomain", "AffineQuantizationType", "IntxPackingFormat",
                    "IntxChooseQParamsAlgorithm", "IntxUnpackedToInt8TensorActivationQuantization")));

    /** torch dtype globals a wrapper rebuild carries, by name under {@code torch.}. */
    public static final Set<String> DTYPE_NAMES = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList(
                    "bfloat16", "float16", "float32", "float64", "int1", "int2", "int3", "int4", "int5", "int6",
                    "int7", "int8", "uint8", "int16", "int32", "int64",
                    "bool", "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz",
                    "float8_e8m0fnu", "float4_e2m1fn_x2")));

    static {
        Set<String> simple = new LinkedHashSet<>();
        for (String fq : TENSOR_CLASSES) {
            simple.add(fq.substring(fq.lastIndexOf('.') + 1));
        }
        TENSOR_SIMPLE_NAMES = Collections.unmodifiableSet(simple);
    }

    private TorchaoPickle() {
    }

    /** True when {@code fq} is a torchao tensor subclass this loader decodes. */
    public static boolean isTensorClass(String fq) {
        return TENSOR_CLASSES.contains(fq);
    }

    /** True when {@code fq} names an attribute dataclass of a tensor subclass ({@code torchao.*}). */
    public static boolean isAttributeClass(String fq) {
        int dot = fq.lastIndexOf('.');
        return dot > 0 && fq.startsWith("torchao.") && ATTRIBUTE_SIMPLE_NAMES.contains(fq.substring(dot + 1));
    }

    /** True when {@code fq} names an enum class the attributes use ({@code torchao.*}). */
    public static boolean isEnumClass(String fq) {
        int dot = fq.lastIndexOf('.');
        return dot > 0 && fq.startsWith("torchao.") && ENUM_SIMPLE_NAMES.contains(fq.substring(dot + 1));
    }

    /** True when {@code name} is a torch dtype constant. */
    public static boolean isDtypeName(String name) {
        return DTYPE_NAMES.contains(name);
    }

    /** The safetensors dtype of a torch dtype marker (for example {@code torch.float8_e4m3fn}). */
    public static StDType dtype(String marker) {
        String name = marker;
        int colon = name.indexOf(':');
        if (colon >= 0) {
            name = name.substring(colon + 1);
        }
        if (name.startsWith("torch.")) {
            name = name.substring("torch.".length());
        }
        switch (name) {
            case "bool": return StDType.BOOL;
            case "int8": return StDType.I8;
            case "uint8": return StDType.U8;
            case "int16": return StDType.I16;
            case "int32": return StDType.I32;
            case "int64": return StDType.I64;
            case "float16": return StDType.F16;
            case "bfloat16": return StDType.BF16;
            case "float32": return StDType.F32;
            case "float64": return StDType.F64;
            case "float8_e4m3fn": return StDType.F8_E4M3;
            case "float8_e5m2": return StDType.F8_E5M2;
            case "float8_e4m3fnuz": return StDType.F8_E4M3FNUZ;
            case "float8_e5m2fnuz": return StDType.F8_E5M2FNUZ;
            case "float8_e8m0fnu": return StDType.F8_E8M0;
            default: return null;
        }
    }

    /** An enum value a serialized attribute carries. Immutable. */
    public static final class EnumValue {
        private final String type;
        private final String name;

        public EnumValue(String type, String name) {
            this.type = type;
            this.name = name;
        }

        /** The enum's fully qualified class name. */
        public String type() { return type; }

        /** The member name. */
        public String name() { return name; }

        @Override
        public String toString() { return type + "." + name; }
    }

    /** A serialized attribute object's state, built by {@code NEWOBJ}/{@code BUILD}. Immutable after BUILD. */
    public static final class AttributeObject {
        private final String type;
        private final Map<String, Object> state;

        public AttributeObject(String type, Map<String, Object> state) {
            this.type = type;
            this.state = state;
        }

        /** The class's fully qualified name. */
        public String type() { return type; }

        /** The state fields. */
        public Map<String, Object> state() { return Collections.unmodifiableMap(state); }

        /** Merges the fields a {@code BUILD} opcode carries. */
        public void putState(Map<String, Object> fields) {
            state.putAll(fields);
        }

        @Override
        public String toString() { return type + state; }
    }

    /** What {@code _rebuild_wrapper_subclass} rebuilds: the wrapper's class, dtype and geometry. */
    public static final class WrapperView {
        private final String className;
        private final String dtype;
        private final long[] size;
        private final long[] stride;
        private final long storageOffset;
        private final String layout;
        private final String device;
        private final boolean requiresGrad;

        public WrapperView(String className, String dtype, long[] size, long[] stride, long storageOffset,
                           String layout, String device, boolean requiresGrad) {
            this.className = className;
            this.dtype = dtype;
            this.size = size.clone();
            this.stride = stride.clone();
            this.storageOffset = storageOffset;
            this.layout = layout;
            this.device = device;
            this.requiresGrad = requiresGrad;
        }

        public String className() { return className; }
        public String dtype() { return dtype; }
        public long[] size() { return size.clone(); }
        public long[] stride() { return stride.clone(); }
        public long storageOffset() { return storageOffset; }
        public String layout() { return layout; }
        public String device() { return device; }
        public boolean requiresGrad() { return requiresGrad; }
    }

    /** What {@code _rebuild_from_type_v2} rebuilds: a wrapper and its state fields. */
    public static final class TensorObject {
        private final String type;
        private final WrapperView view;
        private final Map<String, Object> state;

        public TensorObject(String type, WrapperView view, Map<String, Object> state) {
            this.type = type;
            this.view = view;
            this.state = state;
        }

        /** The tensor subclass's fully qualified class name. */
        public String type() { return type; }

        /** The wrapper geometry. */
        public WrapperView view() { return view; }

        /** The serialized state fields (inner tensors and attributes). */
        public Map<String, Object> state() { return state; }

        @Override
        public String toString() { return "TensorObject{" + type + "}"; }
    }
}
