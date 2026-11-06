package com.eonmux.neuralfree.gguf.hf.torchao;

import com.eonmux.neuralfree.gguf.hf.quant.IntPacking;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;

import java.io.IOException;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Turns the torchao tensor subclasses a checkpoint stores into the ordinary checkpoint tensors the quantization
 * schemes read: the packed weight, its scales and zero points, and an NVFP4 global scale or activation scale. Each
 * subclass name maps to one layout family:
 * <ul>
 *   <li>{@code Int8Tensor}: {@code weight} I8, {@code weight_scale} and an optional {@code weight_zero_point}
 *       ({@link com.eonmux.neuralfree.gguf.hf.config.QuantFormat#INT8_CHANNEL});</li>
 *   <li>{@code Int4Tensor}: {@code weight} U8 (two signed int4 per byte), {@code weight_scale} and
 *       {@code weight_zero_point} in {@code [G, N]} order
 *       ({@link com.eonmux.neuralfree.gguf.hf.config.QuantFormat#INT4_TORCHAO});</li>
 *   <li>{@code Int4TilePackedTo4dTensor}: the tile-packed weight decoded to the same plain U8 layout, and its
 *       {@code scale_and_zero} pack split into the two group-major tensors, for a 2-D or per-expert weight;</li>
 *   <li>{@code Int4PreshuffledTensor}: the mslk preshuffled weight decoded through the inverse of the CUTLASS
 *       reordering atom (16x16 for bf16 activations, 16x32 and {@code unified_encode_int4b}-encoded for FP8
 *       activations), with the bf16 group scale and zero point as stored or the FP8 group scale recovered from its
 *       eight-value lookup and multiplied by the row scale;</li>
 *   <li>{@code Int4PlainInt32Tensor}: XPU's int32 words of four packed int4 bytes decoded to the plain U8 layout,
 *       with the affine bias {@code (8 - z) * s} in the engine's zero-point slot;</li>
 *   <li>{@code IntxUnpackedToInt8Tensor}: one code per int8 byte, packed into nibbles for int4 targets and kept as
 *       byte codes for every other {@code torch.int1}..{@code torch.int8} target, with {@code -z * s} (int4) or the
 *       shifted zero point (the byte targets) in the engine's zero-point slot;</li>
 *   <li>{@code Float8Tensor}: {@code weight} in its FP8 dtype and {@code weight_scale}
 *       ({@link com.eonmux.neuralfree.gguf.hf.config.QuantFormat#FP8});</li>
 *   <li>{@code MXTensor}: {@code weight_packed} and an E8M0 {@code weight_scale} of 32-element blocks, unswizzled
 *       when the checkpoint stores the blocked layout ({@link com.eonmux.neuralfree.gguf.hf.config.QuantFormat#MXFP4_CT}
 *       or {@code MXFP8});</li>
 *   <li>{@code NVFP4Tensor}: {@code weight}, an E4M3 {@code weight_scale} of 16-element blocks (unswizzled when
 *       needed), the global {@code weight_scale_2} (1 when the checkpoint has no per-tensor scale) and the static
 *       activation {@code input_scale} when stored
 *       ({@link com.eonmux.neuralfree.gguf.hf.config.QuantFormat#NVFP4_MODELOPT}).</li>
 * </ul>
 * Other subclasses are refused by name. A subclass's {@code act_pre_scale} becomes a per-column multiplier the
 * rewrite folds into the weight, and its {@code act_quant_scale} / {@code act_quant_zero_point} become the static
 * {@code input_scale} / {@code input_zero_point} the runtime's activation quantization reads. The decoded tensors
 * replace the flattened inner tensors, which the container hides. Immutable.
 */
public final class TorchaoDecoder {

    /** The decoded checkpoint tensors and the flattened tensor names they replace. */
    public static final class Decoded {
        private final Map<String, RowSource> sources;
        private final Map<String, StTensor> tensors;
        private final Set<String> hidden;

        Decoded(Map<String, RowSource> sources, Map<String, StTensor> tensors, Set<String> hidden) {
            this.sources = Collections.unmodifiableMap(sources);
            this.tensors = Collections.unmodifiableMap(tensors);
            this.hidden = Collections.unmodifiableSet(hidden);
        }

        /** The decoded tensor sources by canonical name. */
        public Map<String, RowSource> sources() { return sources; }

        /** The decoded tensors' entries by canonical name. */
        public Map<String, StTensor> tensors() { return tensors; }

        /** The flattened inner tensor names the decoded tensors replace. */
        public Set<String> hidden() { return hidden; }
    }

    private TorchaoDecoder() {
    }

    /**
     * Decodes every weight.
     *
     * @param where the checkpoint description, for error messages
     * @throws IOException when a weight names an unsupported subclass, lacks a required tensor or attribute, or
     *                     stores shapes and fields that disagree
     */
    public static Decoded decode(List<TorchaoWeight> weights, String where) throws IOException {
        Map<String, RowSource> sources = new LinkedHashMap<>();
        Map<String, StTensor> tensors = new LinkedHashMap<>();
        Set<String> hidden = new LinkedHashSet<>();
        for (TorchaoWeight w : weights) {
            Map<String, TorchaoWeight.Inner> data = w.tensors();
            for (TorchaoWeight.Inner inner : data.values()) {
                if (inner.rawName() != null) {
                    hidden.add(inner.rawName());
                }
            }
            switch (w.type()) {
                case "Int8Tensor": int8(w, sources, tensors, where); break;
                case "Int4Tensor": int4(w, sources, tensors, where); break;
                case "Int4TilePackedTo4dTensor": int4TilePacked(w, sources, tensors, where); break;
                case "Int4PreshuffledTensor": int4Preshuffled(w, sources, tensors, where); break;
                case "Int4PlainInt32Tensor": int4PlainInt32(w, sources, tensors, where); break;
                case "IntxUnpackedToInt8Tensor": intxUnpacked(w, sources, tensors, where); break;
                case "Float8Tensor": float8(w, sources, tensors, where); break;
                case "MXTensor": mx(w, sources, tensors, where); break;
                case "NVFP4Tensor": nvfp4(w, sources, tensors, where); break;
                default:
                    throw error(where, w, "is a torchao tensor subclass NeuralFree does not decode; supported"
                            + " subclasses are Int8Tensor, Int4Tensor, Int4TilePackedTo4dTensor,"
                            + " Int4PreshuffledTensor, Int4PlainInt32Tensor, IntxUnpackedToInt8Tensor, Float8Tensor,"
                            + " MXTensor and NVFP4Tensor");
            }
        }
        return new Decoded(sources, tensors, hidden);
    }

    // ------------------------------------------------------------------ subclasses

    private static void int8(TorchaoWeight w, Map<String, RowSource> sources, Map<String, StTensor> tensors,
                             String where) throws IOException {
        requireOnly(w, where, "qdata", "scale", "zero_point", "act_pre_scale", "act_quant_scale",
                "act_quant_zero_point");
        RowSource qdata = inner(w, "qdata", where);
        requireDtype(qdata, w, "qdata", where, StDType.I8);
        RowSource scale = inner(w, "scale", where);
        requireFloat(scale, w, "scale", where);
        long[] q = qdata.shape();
        int experts = q.length == 3 ? Math.toIntExact(q[0]) : 0;
        long k = q[q.length - 1];
        putWeight(sources, tensors, w, qdata, experts, k, where);
        put(sources, tensors, suffix(w, "weight_scale"), scale, where);
        RowSource zero = optional(w, "zero_point");
        if (zero != null) {
            requireIntegerOrFloat(zero, w, "zero_point", where);
            put(sources, tensors, suffix(w, "weight_zero_point"), zero, where);
        }
        activation(w, experts, sources, tensors, where);
    }

    private static void int4(TorchaoWeight w, Map<String, RowSource> sources, Map<String, StTensor> tensors,
                             String where) throws IOException {
        requireOnly(w, where, "qdata", "scale", "zero_point", "act_pre_scale", "act_quant_scale",
                "act_quant_zero_point");
        RowSource qdata = inner(w, "qdata", where);
        requireSignedOrUnsignedBytes(qdata, w, "qdata", where);
        requireEvenInner(qdata, w, "qdata", where);
        RowSource scale = inner(w, "scale", where);
        requireFloat(scale, w, "scale", where);
        long[] q = qdata.shape();
        int experts = q.length == 3 ? Math.toIntExact(q[0]) : 0;
        long k = q[q.length - 1] * 2;
        putWeight(sources, tensors, w, qdata, experts, k, where);
        put(sources, tensors, suffix(w, "weight_scale"), scale, where);
        RowSource zero = optional(w, "zero_point");
        if (zero != null) {
            requireFloat(zero, w, "zero_point", where);
            put(sources, tensors, suffix(w, "weight_zero_point"), zero, where);
        }
        activation(w, experts, sources, tensors, where);
    }

    private static void int4TilePacked(TorchaoWeight w, Map<String, RowSource> sources,
                                       Map<String, StTensor> tensors, String where) throws IOException {
        requireOnly(w, where, "qdata", "scale_and_zero", "act_pre_scale", "act_quant_scale",
                "act_quant_zero_point");
        RowSource qdata = inner(w, "qdata", where);
        requireDtype(qdata, w, "qdata", where, StDType.I32);
        RowSource packed = inner(w, "scale_and_zero", where);
        requireFloat(packed, w, "scale_and_zero", where);
        long[] shape = logicalShape(w, where);
        if (shape.length != 2 && shape.length != 3) {
            throw error(where, w, "has shape " + Arrays.toString(shape) + "; expected [N, K] or [E, N, K]");
        }
        int experts = shape.length == 3 ? Math.toIntExact(shape[0]) : 0;
        long n = shape[shape.length - 2];
        long k = shape[shape.length - 1];
        int group = groupSize(w, shape.length, where);
        long groups = (k + group - 1) / group;
        long[] raw = packed.shape();
        int lead = experts > 0 ? 1 : 0;
        if (raw.length != lead + 3 || raw[raw.length - 1] != 2 || raw[lead] < groups || raw[lead + 1] < n) {
            throw error(where, w, "has scale_and_zero of shape " + Arrays.toString(raw) + "; expected [at least "
                    + groups + ", at least " + n + ", 2] for " + n + " rows and " + groups + " groups"
                    + (experts > 0 ? " behind a leading expert dimension" : ""));
        }
        putWeight(sources, tensors, w, TorchaoSources.tilePacked(qdata, n, k, experts,
                w.name() + " tile-packed weight"), experts, k, where);
        put(sources, tensors, suffix(w, "weight_scale"),
                TorchaoSources.scaleOrZero(packed, false, groups, n, experts, w.name() + " scale"), where);
        put(sources, tensors, suffix(w, "weight_zero_point"),
                TorchaoSources.scaleOrZero(packed, true, groups, n, experts, w.name() + " zero point"), where);
        activation(w, experts, sources, tensors, where);
    }

    /** An mslk preshuffled int4 weight: the permutation is undone into the plain two-nibble layout. */
    private static void int4Preshuffled(TorchaoWeight w, Map<String, RowSource> sources,
                                        Map<String, StTensor> tensors, String where) throws IOException {
        requireOnly(w, where, "qdata", "group_scale", "group_zero", "row_scale");
        RowSource qdata = inner(w, "qdata", where);
        requireSignedOrUnsignedBytes(qdata, w, "qdata", where);
        requireEvenInner(qdata, w, "qdata", where);
        RowSource scale = inner(w, "group_scale", where);
        long[] shape = logicalShape(w, where);
        if (shape.length != 2 && shape.length != 3) {
            throw error(where, w, "has shape " + Arrays.toString(shape) + "; expected [N, K] or [E, N, K]");
        }
        int experts = shape.length == 3 ? Math.toIntExact(shape[0]) : 0;
        long n = shape[shape.length - 2];
        long k = shape[shape.length - 1];
        int group = groupSize(w, shape.length, where);
        if (k % group != 0) {
            throw error(where, w, "has K = " + k + ", not a whole number of " + group + "-element groups");
        }
        int groups = Math.toIntExact(k / group);
        RowSource zero = optional(w, "group_zero");
        RowSource rowScale = optional(w, "row_scale");
        if (zero != null && rowScale != null) {
            throw error(where, w, "stores both group_zero and row_scale, but exactly one activation dtype applies");
        }
        boolean fp8 = rowScale != null;
        if (fp8) {
            requireDtype(scale, w, "group_scale", where, StDType.F8_E4M3);
            requireFloat(rowScale, w, "row_scale", where);
            requireShape(scale, w, "group_scale", where, rank(experts, 3), groups, 8, n);
            requireShape(rowScale, w, "row_scale", where, rank(experts, 1), n);
        } else {
            requireFloat(scale, w, "group_scale", where);
            requireShape(scale, w, "group_scale", where, rank(experts, 2), groups, n);
            if (zero == null) {
                throw error(where, w, "is a bf16-activation preshuffle without its group_zero tensor");
            }
            requireFloat(zero, w, "group_zero", where);
            requireShape(zero, w, "group_zero", where, rank(experts, 2), groups, n);
        }
        putWeight(sources, tensors, w, TorchaoSources.preshuffled(qdata, n, k, experts, fp8,
                w.name() + " preshuffled weight"), experts, k, where);
        put(sources, tensors, suffix(w, "weight_scale"), fp8
                ? TorchaoSources.preshuffleFp8Scale(scale, rowScale, n, groups, experts,
                        w.name() + " group scale")
                : scale, where);
        if (!fp8) {
            put(sources, tensors, suffix(w, "weight_zero_point"), zero, where);
        }
    }

    /** An int4 weight stored as int32 words of four packed bytes (XPU's plain int32 format). */
    private static void int4PlainInt32(TorchaoWeight w, Map<String, RowSource> sources,
                                       Map<String, StTensor> tensors, String where) throws IOException {
        requireOnly(w, where, "qdata", "scale", "zero_point", "act_pre_scale");
        RowSource qdata = inner(w, "qdata", where);
        requireDtype(qdata, w, "qdata", where, StDType.I32);
        RowSource scale = inner(w, "scale", where);
        requireFloat(scale, w, "scale", where);
        RowSource zero = inner(w, "zero_point", where);
        long[] shape = logicalShape(w, where);
        if (shape.length != 2) {
            throw error(where, w, "has shape " + Arrays.toString(shape) + "; expected [N, K]");
        }
        long n = shape[0];
        long k = shape[1];
        int group = groupSize(w, 2, where);
        int groups = Math.toIntExact((k + group - 1) / group);
        requireShape(scale, w, "scale", where, 2, groups, n);
        requireShape(zero, w, "zero_point", where, 2, groups, n);
        if (IntPacking.isFloat(zero.format().stDType())) {
            throw error(where, w, "stores a float zero point, the NPU plain-int32 packing; NeuralFree decodes"
                    + " XPU's int8 zero points with int32 words of four packed int4 bytes");
        }
        requireIntegerOrFloat(zero, w, "zero_point", where);
        putWeight(sources, tensors, w, TorchaoSources.plainInt32(qdata, n, k, new long[]{n, k / 2},
                w.name() + " plain int32 weight"), 0, k, where);
        put(sources, tensors, suffix(w, "weight_scale"), scale, where);
        put(sources, tensors, suffix(w, "weight_zero_point"),
                TorchaoSources.plainInt32Bias(scale, zero, n, groups, 0, w.name() + " zero point"), where);
        activation(w, 0, sources, tensors, where);
    }

    /** A sub-byte weight stored unpacked, one code per int8: int4 packs to nibbles, int8 stays as is. */
    private static void intxUnpacked(TorchaoWeight w, Map<String, RowSource> sources,
                                     Map<String, StTensor> tensors, String where) throws IOException {
        requireOnly(w, where, "qdata", "scale", "zero_point");
        RowSource qdata = inner(w, "qdata", where);
        requireDtype(qdata, w, "qdata", where, StDType.I8);
        RowSource scale = inner(w, "scale", where);
        requireFloat(scale, w, "scale", where);
        RowSource zero = inner(w, "zero_point", where);
        requireIntegerOrFloat(zero, w, "zero_point", where);
        String target = dtypeName(w.attribute("target_dtype"), w, "target_dtype", where);
        int bits = targetBits(target, w, where);
        long[] shape = qdata.shape();
        if (shape.length != 2) {
            throw error(where, w, "has qdata of shape " + Arrays.toString(shape) + "; expected [N, K]");
        }
        long n = shape[0];
        long k = shape[1];
        int group = groupSize(w, 2, where);
        if (k % group != 0) {
            throw error(where, w, "has block_size " + Arrays.toString(ints(w.attribute("block_size"), w,
                    "block_size", where)) + " for K = " + k + "; expected a whole number of groups");
        }
        int groups = Math.toIntExact(k / group);
        requireShape(scale, w, "scale", where, 2, n, groups);
        requireShape(zero, w, "zero_point", where, 2, n, groups);
        switch (bits) {
            case 4:
                putWeight(sources, tensors, w, TorchaoSources.packNibbles(qdata, n, k, 0,
                        w.name() + " unpacked int4 weight"), 0, k, where);
                put(sources, tensors, suffix(w, "weight_scale"),
                        TorchaoSources.groupMajorFloats(scale, n, groups, 0, w.name() + " scale"), where);
                put(sources, tensors, suffix(w, "weight_zero_point"),
                        TorchaoSources.intxBias(scale, zero, n, groups, 0, w.name() + " zero point"), where);
                break;
            default:
                // int1..int3 and int5..int7 keep their int8 byte codes: the int8 channel scheme carries any
                // integer code with its scale and zero point, so the affine dequantization is exact.
                putWeight(sources, tensors, w, qdata, 0, k, where);
                put(sources, tensors, suffix(w, "weight_scale"), scale, where);
                put(sources, tensors, suffix(w, "weight_zero_point"), zero, where);
                break;
        }
    }

    private static int targetBits(String target, TorchaoWeight w, String where) throws IOException {
        if (target == null || !target.startsWith("int") || target.length() <= 3) {
            throw error(where, w, "has target_dtype '" + target + "'; expected torch.int1 through torch.int8");
        }
        try {
            int bits = Integer.parseInt(target.substring(3));
            if (bits < 1 || bits > 8) {
                throw error(where, w, "has target_dtype '" + target + "'; expected torch.int1 through torch.int8");
            }
            return bits;
        } catch (NumberFormatException e) {
            throw error(where, w, "has target_dtype '" + target + "'; expected torch.int1 through torch.int8");
        }
    }

    /**
     * The {@code block_size} group size of a tensor of {@code rank} dimensions: the last entry, with every leading
     * entry 1 (torchao writes {@code [1, group_size]}, or {@code [1, 1, group_size]} for a per-expert weight).
     */
    private static int groupSize(TorchaoWeight w, int rank, String where) throws IOException {
        int[] block = ints(w.attribute("block_size"), w, "block_size", where);
        if (block.length < 1 || block.length > rank || block[block.length - 1] <= 0) {
            throw error(where, w, "has block_size " + Arrays.toString(block) + "; expected at most " + rank
                    + " entries with a positive group size last");
        }
        for (int i = 0; i < block.length - 1; i++) {
            if (block[i] != 1) {
                throw error(where, w, "has block_size " + Arrays.toString(block) + "; only per-group"
                        + " quantization along K is decoded, so every leading entry must be 1");
            }
        }
        return block[block.length - 1];
    }

    /** The rank of a scale tensor: {@code dims} with a leading expert dimension when there is one. */
    private static int rank(int experts, int dims) {
        return experts > 0 ? dims + 1 : dims;
    }

    /** Requires a tensor of rank {@code rank} whose dimensions after an optional expert are {@code tail}. */
    private static void requireShape(RowSource s, TorchaoWeight w, String field, String where, int rank,
                                     long... tail) throws IOException {
        long[] shape = s.shape();
        if (shape.length != rank) {
            throw error(where, w, "has " + field + " of shape " + Arrays.toString(shape) + "; expected rank "
                    + rank + " with the tail " + Arrays.toString(tail));
        }
        int lead = shape.length - tail.length;
        for (int i = 0; i < tail.length; i++) {
            if (shape[lead + i] != tail[i]) {
                throw error(where, w, "has " + field + " of shape " + Arrays.toString(shape) + "; expected "
                        + (lead == 1 ? "[" + shape[0] + ", " : "[") + joined(tail) + "]");
            }
        }
    }

    private static String joined(long[] values) {
        StringBuilder s = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                s.append(", ");
            }
            s.append(values[i]);
        }
        return s.toString();
    }

    private static void float8(TorchaoWeight w, Map<String, RowSource> sources, Map<String, StTensor> tensors,
                               String where) throws IOException {
        requireOnly(w, where, "qdata", "scale");
        RowSource qdata = inner(w, "qdata", where);
        StDType t = qdata.format().stDType();
        if (t != StDType.F8_E4M3 && t != StDType.F8_E5M2 && t != StDType.F8_E4M3FNUZ
                && t != StDType.F8_E5M2FNUZ) {
            throw error(where, w, "has qdata of dtype " + qdata.format() + "; expected an FP8 dtype");
        }
        RowSource scale = inner(w, "scale", where);
        requireFloat(scale, w, "scale", where);
        put(sources, tensors, w.name(), qdata, where);
        put(sources, tensors, suffix(w, "weight_scale"), scale, where);
    }

    private static void mx(TorchaoWeight w, Map<String, RowSource> sources, Map<String, StTensor> tensors,
                           String where) throws IOException {
        requireOnly(w, where, "qdata", "scale");
        RowSource qdata = inner(w, "qdata", where);
        RowSource scale = inner(w, "scale", where);
        requireDtype(scale, w, "scale", where, StDType.F8_E8M0);
        String elem = dtypeName(w.attribute("elem_dtype"), w, "elem_dtype", where);
        int block = blockSize(w, where, 32);
        long[] shape;
        StDType t = qdata.format().stDType();
        if ("float4_e2m1fn_x2".equals(elem)) {
            requireDtype(qdata, w, "qdata", where, StDType.U8);
            long[] q = qdata.shape();
            if (q.length == 0) {
                throw error(where, w, "has a scalar qdata");
            }
            shape = q.clone();
            shape[shape.length - 1] *= 2;
        } else if ("float8_e5m2".equals(elem)) {
            throw error(where, w, "has elem_dtype 'float8_e5m2'; the MXFP8 layout is E4M3 only");
        } else if ("float8_e4m3fn".equals(elem)) {
            if (t != StDType.F8_E4M3) {
                throw error(where, w, "has qdata of dtype " + qdata.format() + " but elem_dtype '" + elem + "'");
            }
            shape = qdata.shape();
        } else {
            throw error(where, w, "has elem_dtype '" + elem + "'; only float4_e2m1fn_x2 and float8_e4m3fn have"
                    + " NeuralFree mappings");
        }
        if (shape.length < 2) {
            throw error(where, w, "has shape " + Arrays.toString(shape) + "; MX weights are matrices");
        }
        long[] lead = Arrays.copyOf(shape, shape.length - 1);
        long k = shape[shape.length - 1];
        if (k % block != 0) {
            throw error(where, w, "has K = " + k + ", not a whole number of " + block + "-element MX blocks");
        }
        RowSource scales = scale(w, scale, lead, k / block, where);
        requireDtype(scales, w, "scale", where, StDType.F8_E8M0);
        // MXFP8 keeps the compressed-tensors weight name; MXFP4's packed fp4 pairs are weight_packed.
        put(sources, tensors, suffix(w, "float4_e2m1fn_x2".equals(elem) ? "weight_packed" : "weight"), qdata, where);
        put(sources, tensors, suffix(w, "weight_scale"), scales, where);
    }

    private static void nvfp4(TorchaoWeight w, Map<String, RowSource> sources, Map<String, StTensor> tensors,
                              String where) throws IOException {
        requireOnly(w, where, "qdata", "scale", "per_tensor_scale", "act_per_tensor_scale");
        RowSource qdata = inner(w, "qdata", where);
        requireDtype(qdata, w, "qdata", where, StDType.U8);
        RowSource scale = inner(w, "scale", where);
        requireDtype(scale, w, "scale", where, StDType.F8_E4M3);
        int block = blockSize(w, where, 16);
        if (block != 16) {
            throw error(where, w, "has block_size " + block + "; NVFP4 scales cover 16 elements");
        }
        long[] q = qdata.shape();
        if (q.length < 2) {
            throw error(where, w, "has qdata of shape " + Arrays.toString(q) + "; NVFP4 weights are matrices");
        }
        long[] shape = q.clone();
        shape[shape.length - 1] *= 2;
        long[] lead = Arrays.copyOf(shape, shape.length - 1);
        long k = shape[shape.length - 1];
        if (k % block != 0) {
            throw error(where, w, "has K = " + k + ", not a whole number of 16-element NVFP4 blocks");
        }
        RowSource scales = scale(w, scale, lead, k / block, where);
        requireDtype(scales, w, "scale", where, StDType.F8_E4M3);
        put(sources, tensors, w.name(), qdata, where);
        put(sources, tensors, suffix(w, "weight_scale"), scales, where);
        RowSource global = optional(w, "per_tensor_scale");
        put(sources, tensors, suffix(w, "weight_scale_2"),
                global != null ? global : TorchaoSources.floats(new float[]{1.0f}, new long[]{1},
                        w.name() + " global scale"), where);
        RowSource act = optional(w, "act_per_tensor_scale");
        if (act != null) {
            put(sources, tensors, suffix(w, "input_scale"), act, where);
        }
    }

    // ------------------------------------------------------------------ scale layouts

    /** The scale source unblocked to the leading dimensions {@code lead} and {@code groups} columns, or the stored
     *  grid as is. */
    private static RowSource scale(TorchaoWeight w, RowSource raw, long[] lead, long groups, String where)
            throws IOException {
        long n = 1;
        for (long d : lead) {
            n = Math.multiplyExact(n, d);
        }
        Object flag = w.attribute("is_swizzled_scales");
        if (flag != null && !(flag instanceof Boolean)) {
            throw error(where, w, "has a non-boolean is_swizzled_scales field");
        }
        boolean swizzled = Boolean.TRUE.equals(flag);
        long rowsPerExpert = lead[lead.length - 1];
        long rowBlocks = (rowsPerExpert + 127) / 128;
        long colBlocks = (groups + 3) / 4;
        long[] blockGrid = new long[]{rowBlocks * 32, colBlocks * 16};
        long[] expected = lead.length == 1 ? blockGrid
                : concat(Arrays.copyOf(lead, lead.length - 1), blockGrid);
        if (swizzled) {
            if (!Arrays.equals(raw.shape(), expected)) {
                throw error(where, w, "has a swizzled scale of shape " + Arrays.toString(raw.shape())
                        + "; expected " + Arrays.toString(expected) + " for " + n + " rows and " + groups
                        + " groups");
            }
            return TorchaoSources.unblocked(raw, lead, groups, w.name() + " unswizzled scale");
        }
        long[] expectedGrid = Arrays.copyOf(lead, lead.length + 1);
        expectedGrid[lead.length] = groups;
        if (!Arrays.equals(raw.shape(), expectedGrid)) {
            throw error(where, w, "has a scale of shape " + Arrays.toString(raw.shape()) + "; expected "
                    + Arrays.toString(expectedGrid));
        }
        return raw;
    }

    // ------------------------------------------------------------------ helpers

    /**
     * Puts the weight's canonical tensor, carrying the subclass's {@code act_pre_scale} as a per-column multiplier
     * the rewrite folds into a float weight.
     */
    private static void putWeight(Map<String, RowSource> sources, Map<String, StTensor> tensors, TorchaoWeight w,
                                  RowSource source, int experts, long k, String where) throws IOException {
        float[] pre = preScale(w, experts, k, where);
        if (pre != null) {
            source = TorchaoSources.preScaled(source, pre, w.name() + " weight");
        }
        put(sources, tensors, w.name(), source, where);
    }

    /** The {@code act_pre_scale} inner tensor as per-column values, or null. */
    private static float[] preScale(TorchaoWeight w, int experts, long k, String where) throws IOException {
        TorchaoWeight.Inner inner = w.tensor("act_pre_scale");
        if (inner == null) {
            return null;
        }
        RowSource src = inner.source();
        requireFloat(src, w, "act_pre_scale", where);
        long count = 1;
        for (long d : src.shape()) {
            count = Math.multiplyExact(count, d);
        }
        long want = Math.multiplyExact(Math.max(1, experts), k);
        if (count != want && !(experts > 0 && count == k)) {
            throw error(where, w, "has act_pre_scale of shape " + Arrays.toString(src.shape()) + "; expected "
                    + k + " columns" + (experts > 0 ? " or one per expert (" + want + " values)" : ""));
        }
        return TorchaoSources.floats(src, w.name() + " act_pre_scale");
    }

    /** The static activation quantization siblings a subclass may carry. */
    private static void activation(TorchaoWeight w, int experts, Map<String, RowSource> sources,
                                   Map<String, StTensor> tensors, String where) throws IOException {
        RowSource scale = optional(w, "act_quant_scale");
        if (scale != null) {
            requireFloat(scale, w, "act_quant_scale", where);
            requirePerTensorOrExpert(scale, w, "act_quant_scale", experts, where);
            put(sources, tensors, suffix(w, "input_scale"), scale, where);
        }
        RowSource zero = optional(w, "act_quant_zero_point");
        if (zero != null) {
            requireIntegerOrFloat(zero, w, "act_quant_zero_point", where);
            requirePerTensorOrExpert(zero, w, "act_quant_zero_point", experts, where);
            put(sources, tensors, suffix(w, "input_zero_point"), zero, where);
        }
    }

    private static void requirePerTensorOrExpert(RowSource s, TorchaoWeight w, String field, int experts,
                                                 String where) throws IOException {
        long count = 1;
        for (long d : s.shape()) {
            count = Math.multiplyExact(count, d);
        }
        if (count != 1 && (experts <= 0 || count != experts)) {
            throw error(where, w, "has " + field + " of shape " + Arrays.toString(s.shape())
                    + "; expected one value, or one per expert");
        }
    }

    private static RowSource inner(TorchaoWeight w, String name, String where) throws IOException {
        TorchaoWeight.Inner inner = w.tensor(name);
        if (inner == null) {
            throw error(where, w, "has no " + name + " tensor (it has " + w.tensors().keySet() + ")");
        }
        return inner.source();
    }

    private static RowSource optional(TorchaoWeight w, String name) {
        TorchaoWeight.Inner inner = w.tensor(name);
        return inner == null ? null : inner.source();
    }

    /** Rejects an inner tensor the subclass does not define. */
    private static void requireOnly(TorchaoWeight w, String where, String... allowed) throws IOException {
        outer:
        for (String name : w.tensors().keySet()) {
            for (String a : allowed) {
                if (a.equals(name)) {
                    continue outer;
                }
            }
            throw error(where, w, "has the inner tensor '" + name + "', which the NeuralFree " + w.type()
                    + " decode does not support");
        }
    }

    private static void requireDtype(RowSource s, TorchaoWeight w, String field, String where, StDType dtype)
            throws IOException {
        StDType t = s.format().stDType();
        if (t != dtype) {
            throw error(where, w, "has " + field + " of dtype " + s.format() + "; expected " + dtype.headerName());
        }
    }

    /** Two int4 values per byte may be stored as a signed byte ({@code Int4Tensor}) or an unsigned one. */
    private static void requireSignedOrUnsignedBytes(RowSource s, TorchaoWeight w, String field, String where)
            throws IOException {
        StDType t = s.format().stDType();
        if (t != StDType.I8 && t != StDType.U8) {
            throw error(where, w, "has " + field + " of dtype " + s.format() + "; expected I8 or U8");
        }
    }

    private static void requireFloat(RowSource s, TorchaoWeight w, String field, String where) throws IOException {
        StDType t = s.format().stDType();
        if (t != StDType.F16 && t != StDType.BF16 && t != StDType.F32 && t != StDType.F64) {
            throw error(where, w, "has " + field + " of dtype " + s.format() + "; expected a float dtype");
        }
    }

    private static void requireIntegerOrFloat(RowSource s, TorchaoWeight w, String field, String where)
            throws IOException {
        StDType t = s.format().stDType();
        if (t == null) {
            throw error(where, w, "has " + field + " of dtype " + s.format() + "; expected an integer or float"
                    + " dtype");
        }
        switch (t) {
            case I8: case U8: case I16: case U16: case I32: case U32: case I64: case U64:
            case F16: case BF16: case F32: case F64:
                return;
            default:
                throw error(where, w, "has " + field + " of dtype " + s.format() + "; expected an integer or"
                        + " float dtype");
        }
    }

    private static void requireEvenInner(RowSource qdata, TorchaoWeight w, String field, String where)
            throws IOException {
        long[] s = qdata.shape();
        if (s.length == 0 || s[s.length - 1] % 2 != 0) {
            throw error(where, w, "has " + field + " of shape " + Arrays.toString(s)
                    + "; two int4 values per byte need an even innermost dimension");
        }
    }

    private static void put(Map<String, RowSource> sources, Map<String, StTensor> tensors, String name,
                            RowSource source, String where) throws IOException {
        StDType dtype = source.format().stDType();
        if (dtype == null || dtype.bits() % 8 != 0) {
            throw new IOException("Invalid torchao tensor '" + name + "' in " + where + ": it decodes to "
                    + source.format() + ", which is not a byte-addressable tensor");
        }
        long numel = 1;
        for (long d : source.shape()) {
            numel = Math.multiplyExact(numel, d);
        }
        long bytes = Math.multiplyExact(numel, dtype.bits() / 8L);
        if (sources.put(name, source) != null) {
            throw new IOException("Invalid torchao tensor '" + name + "' in " + where + ": two decoded tensors"
                    + " have that name");
        }
        tensors.put(name, new StTensor(name, dtype, source.shape(), 0, bytes));
    }

    /** The logical shape: the serialized form's own, else the decoded one the reader derived. */
    private static long[] logicalShape(TorchaoWeight w, String where) throws IOException {
        Object shape = w.attribute("shape");
        if (shape != null) {
            return longs(shape, w, "shape", where);
        }
        long[] inferred = w.shape();
        if (inferred == null) {
            throw error(where, w, "stores no shape and its form carries none");
        }
        return inferred;
    }

    private static int blockSize(TorchaoWeight w, String where, int dflt) throws IOException {
        Object v = w.attribute("block_size");
        if (v == null) {
            return dflt;
        }
        if (v instanceof Number) {
            long n = ((Number) v).longValue();
            if (n <= 0 || n > Integer.MAX_VALUE) {
                throw error(where, w, "has block_size " + n);
            }
            return (int) n;
        }
        int[] v2 = ints(v, w, "block_size", where);
        if (v2.length == 1) {
            return v2[0];
        }
        if (v2.length == 2 && v2[0] == 1) {
            return v2[1];
        }
        throw error(where, w, "has block_size " + Arrays.toString(v2) + "; expected a positive size");
    }

    private static String dtypeName(Object v, TorchaoWeight w, String field, String where) throws IOException {
        if (v instanceof String) {
            String s = (String) v;
            int dot = s.lastIndexOf('.');
            return dot >= 0 ? s.substring(dot + 1) : s;
        }
        if (v instanceof Map) {
            Object data = ((Map<?, ?>) v).get("_data");
            if (data instanceof String) {
                return (String) data;
            }
        }
        if (v instanceof TorchaoPickle.EnumValue) {
            String s = ((TorchaoPickle.EnumValue) v).name();
            int dot = s.lastIndexOf('.');
            return dot >= 0 ? s.substring(dot + 1) : s;
        }
        throw error(where, w, "has " + field + " that is not a torch dtype");
    }

    private static long[] longs(Object v, TorchaoWeight w, String field, String where) throws IOException {
        if (v instanceof List) {
            List<?> l = (List<?>) v;
            long[] out = new long[l.size()];
            for (int i = 0; i < out.length; i++) {
                Object o = l.get(i);
                if (!(o instanceof Number)) {
                    throw error(where, w, "has " + field + "[" + i + "] that is not an integer");
                }
                out[i] = ((Number) o).longValue();
            }
            return out;
        }
        if (v instanceof Object[]) {
            Object[] a = (Object[]) v;
            long[] out = new long[a.length];
            for (int i = 0; i < out.length; i++) {
                if (!(a[i] instanceof Long)) {
                    throw error(where, w, "has " + field + "[" + i + "] that is not an integer");
                }
                out[i] = (Long) a[i];
            }
            return out;
        }
        throw error(where, w, "has " + field + " that is not an integer list");
    }

    private static int[] ints(Object v, TorchaoWeight w, String field, String where) throws IOException {
        long[] l = longs(v, w, field, where);
        int[] out = new int[l.length];
        for (int i = 0; i < out.length; i++) {
            if (l[i] < Integer.MIN_VALUE || l[i] > Integer.MAX_VALUE) {
                throw error(where, w, "has " + field + "[" + i + "] = " + l[i] + ", outside the int range");
            }
            out[i] = (int) l[i];
        }
        return out;
    }

    private static long[] concat(long[] a, long[] b) {
        long[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static long product(long[] shape, int from, int to, TorchaoWeight w, String where) throws IOException {
        long n = 1;
        for (int i = from; i < to; i++) {
            if (shape[i] < 0) {
                throw error(where, w, "has a negative dimension in shape " + Arrays.toString(shape));
            }
            n = Math.multiplyExact(n, shape[i]);
        }
        return n;
    }

    /** The module-relative auxiliary name: dotted for a {@code .weight} tensor, underscore for a fused parameter. */
    private static String suffix(TorchaoWeight w, String suffix) {
        String name = w.name();
        if (name.endsWith(".weight")) {
            return name.substring(0, name.length() - ".weight".length()) + "." + suffix;
        }
        return name + "_" + suffix;
    }

    private static IOException error(String where, TorchaoWeight w, String message) {
        return new IOException("Invalid torchao tensor '" + w.name() + "' (" + w.type() + ") in " + where + ": "
                + message);
    }
}
