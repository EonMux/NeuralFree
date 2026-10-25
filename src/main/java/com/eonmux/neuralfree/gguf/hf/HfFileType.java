package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
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
 * (NF extension types, integer and intermediate block types). Ties go to the type planned first.
 */
public final class HfFileType {

    /** {@code LlamaFileType.ALL_F32}. */
    public static final int ALL_F32 = 0;
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
        return dominant == null ? ALL_F32 : of(dominant);
    }

    /** The {@code LlamaFileType} of a model whose weights are mostly {@code type}. */
    public static int of(GGMLType type) {
        Integer id = FILE_TYPES.get(Objects.requireNonNull(type, "type"));
        return id == null ? GUESSED : id;
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
