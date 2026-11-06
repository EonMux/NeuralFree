package com.eonmux.neuralfree.gguf.quant.write;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Locale;
import java.util.Objects;

/**
 * The quantization recipes of the quantize tool ({@code QUANT_OPTIONS} and
 * {@code llama_ftype_get_default_type}, llama.cpp tools/quantize/quantize.cpp:34-75 and
 * src/llama-quant.cpp:927-970): the file type names of the command line, the {@code LLAMA_FTYPE_*} value
 * {@code general.file_type} carries in the output, and the tensor type the mixture starts from. {@link
 * #COPY} is the copy-only recipe; its id is F32's, as in the reference list that places COPY after F32 so
 * the id 0 never matches it first.
 */
public enum QuantizeFtype {

    Q1_0("Q1_0", 40, GGMLType.Q1_0),
    Q2_0("Q2_0", 41, GGMLType.Q2_0),
    Q4_0("Q4_0", 2, GGMLType.Q4_0),
    Q4_1("Q4_1", 3, GGMLType.Q4_1),
    MXFP4_MOE("MXFP4_MOE", 38, GGMLType.MXFP4),
    Q5_0("Q5_0", 8, GGMLType.Q5_0),
    Q5_1("Q5_1", 9, GGMLType.Q5_1),
    IQ2_XXS("IQ2_XXS", 19, GGMLType.IQ2_XXS),
    IQ2_XS("IQ2_XS", 20, GGMLType.IQ2_XS),
    IQ2_S("IQ2_S", 28, GGMLType.IQ2_XS),
    IQ2_M("IQ2_M", 29, GGMLType.IQ2_S),
    IQ1_S("IQ1_S", 24, GGMLType.IQ1_S),
    IQ1_M("IQ1_M", 31, GGMLType.IQ1_M),
    TQ1_0("TQ1_0", 36, GGMLType.TQ1_0),
    TQ2_0("TQ2_0", 37, GGMLType.TQ2_0),
    Q2_K("Q2_K", 10, GGMLType.Q2_K),
    Q2_K_S("Q2_K_S", 21, GGMLType.Q2_K),
    IQ3_XXS("IQ3_XXS", 23, GGMLType.IQ3_XXS),
    IQ3_S("IQ3_S", 26, GGMLType.IQ3_S),
    IQ3_M("IQ3_M", 27, GGMLType.IQ3_S),
    Q3_K("Q3_K", 12, GGMLType.Q3_K),
    IQ3_XS("IQ3_XS", 22, GGMLType.IQ3_S),
    Q3_K_S("Q3_K_S", 11, GGMLType.Q3_K),
    Q3_K_M("Q3_K_M", 12, GGMLType.Q3_K),
    Q3_K_L("Q3_K_L", 13, GGMLType.Q3_K),
    IQ4_NL("IQ4_NL", 25, GGMLType.IQ4_NL),
    IQ4_XS("IQ4_XS", 30, GGMLType.IQ4_XS),
    Q4_K("Q4_K", 15, GGMLType.Q4_K),
    Q4_K_S("Q4_K_S", 14, GGMLType.Q4_K),
    Q4_K_M("Q4_K_M", 15, GGMLType.Q4_K),
    Q5_K("Q5_K", 17, GGMLType.Q5_K),
    Q5_K_S("Q5_K_S", 16, GGMLType.Q5_K),
    Q5_K_M("Q5_K_M", 17, GGMLType.Q5_K),
    Q6_K("Q6_K", 18, GGMLType.Q6_K),
    Q8_0("Q8_0", 7, GGMLType.Q8_0),
    F16("F16", 1, GGMLType.F16),
    BF16("BF16", 32, GGMLType.BF16),
    F32("F32", 0, GGMLType.F32),
    COPY("COPY", 0, GGMLType.F32);

    private final String name;
    private final int id;
    private final GGMLType defaultType;

    QuantizeFtype(String name, int id, GGMLType defaultType) {
        this.name = name;
        this.id = id;
        this.defaultType = defaultType;
    }

    /** The recipe name as the command line and the output name spell it. */
    public String recipeName() {
        return name;
    }

    /** The {@code LLAMA_FTYPE_*} value {@code general.file_type} carries. */
    public int id() {
        return id;
    }

    /** The tensor type the mixture starts from ({@code llama_ftype_get_default_type}). */
    public GGMLType defaultType() {
        return defaultType;
    }

    /** True for {@link #COPY}: every tensor keeps its stored type and data. */
    public boolean onlyCopy() {
        return this == COPY;
    }

    /**
     * The recipe of {@code value}: a recipe name, case-insensitive, or a numeric {@code LLAMA_FTYPE_*}
     * id ({@code try_parse_ftype}).
     *
     * @throws IllegalArgumentException naming the value when it is neither
     */
    public static QuantizeFtype parse(String value) {
        Objects.requireNonNull(value, "value");
        String s = value.trim().toUpperCase(Locale.ROOT);
        for (QuantizeFtype f : values()) {
            if (f.name.equals(s)) {
                return f;
            }
        }
        try {
            int id = Integer.parseInt(s);
            for (QuantizeFtype f : values()) {
                if (f.id == id && f != COPY) {
                    return f;
                }
            }
        } catch (NumberFormatException e) {
            // not a numeric ftype
        }
        throw new IllegalArgumentException("'" + value + "' is not a quantization type; use one of "
                + names());
    }

    /** The recipe names in command-line order, for usage text and errors. */
    public static String names() {
        StringBuilder b = new StringBuilder();
        for (QuantizeFtype f : values()) {
            if (b.length() > 0) {
                b.append(", ");
            }
            b.append(f.name);
        }
        return b.toString();
    }
}
