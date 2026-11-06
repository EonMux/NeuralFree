package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.tensor.PlannedTensor;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlan;

import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The {@code general.file_type} of a Hugging Face model view: the llama.cpp {@code LlamaFileType}
 * (LC/gguf-py/gguf/constants.py:6295-6337) of the tensor type that holds the most bytes among the planned weights of
 * rank 2 or more ({@code .weight} tensors; every tensor of rank 2 or more when no such weight exists, every tensor when
 * none has rank 2). {@link #ALL_F32} when that type is F32, and {@link #GUESSED} when the type has no file type
 * (NF extension types, integer and intermediate block types). Ties go to the type planned first. A float-dominant
 * view with preserved FP8 weights ({@code GGMLType.F8_E4M3FN}) is {@link #MOSTLY_F8_E4M3}, as the converter sets
 * MOSTLY_F8_E4M3 over its f16/bf16/f32 guess (LC/conversion/base.py:1600-1604). A conversion with an explicit output
 * type writes that type's file type instead ({@link #forOutType}); FP8 weights are only preserved under AUTO.
 */
public final class HfFileType {

    /** {@code LlamaFileType.ALL_F32}. */
    public static final int ALL_F32 = 0;
    /** {@code LlamaFileType.MOSTLY_F16}. */
    private static final int F16 = 1;
    /** {@code LlamaFileType.MOSTLY_BF16}. */
    private static final int BF16 = 32;
    /** {@code LlamaFileType.MOSTLY_F8_E4M3} (llama.cpp include/llama.h:161): FP8 E4M3 weights. */
    public static final int MOSTLY_F8_E4M3 = 42;
    /** {@code LlamaFileType.GUESSED}: not specified by the model file. */
    public static final int GUESSED = 1024;

    private static final Map<GGMLType, Integer> FILE_TYPES;

    static {
        Map<GGMLType, Integer> m = new EnumMap<>(GGMLType.class);
        m.put(GGMLType.F32, ALL_F32);
        m.put(GGMLType.F16, 1);
        m.put(GGMLType.Q4_0, 2);
        m.put(GGMLType.Q4_1, 3);
        m.put(GGMLType.Q8_0, 7);
        m.put(GGMLType.Q5_0, 8);
        m.put(GGMLType.Q5_1, 9);
        m.put(GGMLType.Q2_K, 10);
        m.put(GGMLType.Q3_K, 12);
        m.put(GGMLType.Q4_K, 15);
        m.put(GGMLType.Q5_K, 17);
        m.put(GGMLType.Q6_K, 18);
        m.put(GGMLType.IQ2_XXS, 19);
        m.put(GGMLType.IQ2_XS, 20);
        m.put(GGMLType.IQ3_XXS, 23);
        m.put(GGMLType.IQ1_S, 24);
        m.put(GGMLType.IQ4_NL, 25);
        m.put(GGMLType.IQ3_S, 26);
        m.put(GGMLType.IQ2_S, 28);
        m.put(GGMLType.IQ4_XS, 30);
        m.put(GGMLType.IQ1_M, 31);
        m.put(GGMLType.BF16, 32);
        m.put(GGMLType.TQ1_0, 36);
        m.put(GGMLType.TQ2_0, 37);
        m.put(GGMLType.MXFP4, 38);
        m.put(GGMLType.NVFP4, 39);
        m.put(GGMLType.Q1_0, 40);
        m.put(GGMLType.Q2_0, 41);
        m.put(GGMLType.F8_E4M3FN, MOSTLY_F8_E4M3);
        FILE_TYPES = Collections.unmodifiableMap(m);
    }

    private HfFileType() {
    }

    /** The file type of {@code plan}; {@link #ALL_F32} for a plan without tensors. */
    public static int guess(TensorPlan plan) {
        Objects.requireNonNull(plan, "plan");
        return guess(plan.tensors());
    }

    /** The file type of the planned {@code tensors}; {@link #ALL_F32} when there are none. */
    public static int guess(List<PlannedTensor> tensors) {
        Objects.requireNonNull(tensors, "tensors");
        GGMLType dominant = dominant(tensors, true, true);
        if (dominant == null) {
            dominant = dominant(tensors, true, false);
        }
        if (dominant == null) {
            dominant = dominant(tensors, false, false);
        }
        if (dominant == null) {
            return ALL_F32;
        }
        int type = of(dominant);
        // The converter sets MOSTLY_F8_E4M3 over a float guess whenever it preserved FP8 weights
        // (LC/conversion/base.py:1600-1604); a quantized dominant type keeps its own file type.
        if ((type == ALL_F32 || type == F16 || type == BF16) && preservesF8E4m3(tensors)) {
            return MOSTLY_F8_E4M3;
        }
        return type;
    }

    /** True when a planned weight carries the preserved FP8 codes ({@code GGMLType.F8_E4M3FN}). */
    private static boolean preservesF8E4m3(List<PlannedTensor> tensors) {
        for (PlannedTensor t : tensors) {
            if (t.type() == GGMLType.F8_E4M3FN && t.ggufName().endsWith(".weight")) {
                return true;
            }
        }
        return false;
    }

    /**
     * The file type a conversion with {@code outType} writes: the requested type's {@code LlamaFileType} for an
     * explicit {@code --outtype} (F32, F16, BF16, Q8_0, TQ1_0, TQ2_0), as the converter writes
     * {@code add_file_type(self.ftype)} whatever types its forced-F32 and F16-kept tensors take; {@link #guess} of
     * {@code plan} for SOURCE and AUTO, which name no single type.
     */
    public static int forOutType(ModelSourceOptions.OutType outType, TensorPlan plan) {
        Objects.requireNonNull(outType, "outType");
        switch (outType) {
            case F32:
                return ALL_F32;
            case F16:
                return of(GGMLType.F16);
            case BF16:
                return of(GGMLType.BF16);
            case Q8_0:
                return of(GGMLType.Q8_0);
            case TQ1_0:
                return of(GGMLType.TQ1_0);
            case TQ2_0:
                return of(GGMLType.TQ2_0);
            default:
                return guess(plan);
        }
    }

    /** The {@code LlamaFileType} of a model whose weights are mostly {@code type}. */
    public static int of(GGMLType type) {
        Integer id = FILE_TYPES.get(Objects.requireNonNull(type, "type"));
        return id == null ? GUESSED : id;
    }

    /**
     * The tensor type {@code fileType} names ({@code LlamaFileType} {@code ALL_F32} / {@code MOSTLY_<type>}), or
     * null for {@link #GUESSED} and an id without a type.
     */
    public static GGMLType typeOf(long fileType) {
        for (Map.Entry<GGMLType, Integer> e : FILE_TYPES.entrySet()) {
            if (e.getValue() == fileType) {
                return e.getKey();
            }
        }
        return null;
    }

    /** The type with the most bytes among the tensors that pass the filters, or null when none passes. */
    private static GGMLType dominant(List<PlannedTensor> tensors, boolean matrices, boolean weights) {
        Map<GGMLType, Long> bytes = new LinkedHashMap<>();
        for (PlannedTensor t : tensors) {
            if (matrices && t.ne().length < 2) {
                continue;
            }
            if (weights && !t.ggufName().endsWith(".weight")) {
                continue;
            }
            bytes.merge(t.type(), t.byteSize(), Long::sum);
        }
        GGMLType best = null;
        long most = -1;
        for (Map.Entry<GGMLType, Long> e : bytes.entrySet()) {
            if (e.getValue() > most) {
                best = e.getKey();
                most = e.getValue();
            }
        }
        return best;
    }
}
