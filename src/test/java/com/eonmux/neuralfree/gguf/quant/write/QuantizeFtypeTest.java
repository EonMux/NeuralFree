package com.eonmux.neuralfree.gguf.quant.write;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The recipe table: the {@code LLAMA_FTYPE_*} ids that {@code general.file_type} carries, the default
 * tensor types (including the IQ2_S/IQ2_M aliases onto IQ2_XS/IQ2_S), and the parse by name and numeric
 * id with the reference's first-match order.
 */
public class QuantizeFtypeTest {

    @Test
    public void idsMatchLlamaCpp() {
        assertEquals(0, QuantizeFtype.F32.id());
        assertEquals(1, QuantizeFtype.F16.id());
        assertEquals(2, QuantizeFtype.Q4_0.id());
        assertEquals(3, QuantizeFtype.Q4_1.id());
        assertEquals(7, QuantizeFtype.Q8_0.id());
        assertEquals(8, QuantizeFtype.Q5_0.id());
        assertEquals(9, QuantizeFtype.Q5_1.id());
        assertEquals(10, QuantizeFtype.Q2_K.id());
        assertEquals(11, QuantizeFtype.Q3_K_S.id());
        assertEquals(12, QuantizeFtype.Q3_K_M.id());
        assertEquals(13, QuantizeFtype.Q3_K_L.id());
        assertEquals(14, QuantizeFtype.Q4_K_S.id());
        assertEquals(15, QuantizeFtype.Q4_K_M.id());
        assertEquals(16, QuantizeFtype.Q5_K_S.id());
        assertEquals(17, QuantizeFtype.Q5_K_M.id());
        assertEquals(18, QuantizeFtype.Q6_K.id());
        assertEquals(19, QuantizeFtype.IQ2_XXS.id());
        assertEquals(20, QuantizeFtype.IQ2_XS.id());
        assertEquals(21, QuantizeFtype.Q2_K_S.id());
        assertEquals(22, QuantizeFtype.IQ3_XS.id());
        assertEquals(23, QuantizeFtype.IQ3_XXS.id());
        assertEquals(24, QuantizeFtype.IQ1_S.id());
        assertEquals(25, QuantizeFtype.IQ4_NL.id());
        assertEquals(26, QuantizeFtype.IQ3_S.id());
        assertEquals(27, QuantizeFtype.IQ3_M.id());
        assertEquals(28, QuantizeFtype.IQ2_S.id());
        assertEquals(29, QuantizeFtype.IQ2_M.id());
        assertEquals(30, QuantizeFtype.IQ4_XS.id());
        assertEquals(31, QuantizeFtype.IQ1_M.id());
        assertEquals(32, QuantizeFtype.BF16.id());
        assertEquals(36, QuantizeFtype.TQ1_0.id());
        assertEquals(37, QuantizeFtype.TQ2_0.id());
        assertEquals(38, QuantizeFtype.MXFP4_MOE.id());
        assertEquals(40, QuantizeFtype.Q1_0.id());
        assertEquals(41, QuantizeFtype.Q2_0.id());
    }

    @Test
    public void defaultTypes() {
        assertEquals(GGMLType.Q4_K, QuantizeFtype.Q4_K_M.defaultType());
        assertEquals(GGMLType.Q4_K, QuantizeFtype.Q4_K.defaultType());
        assertEquals(GGMLType.IQ2_XS, QuantizeFtype.IQ2_S.defaultType());
        assertEquals(GGMLType.IQ2_S, QuantizeFtype.IQ2_M.defaultType());
        assertEquals(GGMLType.IQ3_S, QuantizeFtype.IQ3_XS.defaultType());
        assertEquals(GGMLType.Q3_K, QuantizeFtype.Q3_K.defaultType());
        assertEquals(GGMLType.MXFP4, QuantizeFtype.MXFP4_MOE.defaultType());
        assertEquals(GGMLType.Q2_K, QuantizeFtype.Q2_K_S.defaultType());
        assertTrue(QuantizeFtype.COPY.onlyCopy());
        assertTrue(QuantizeFtype.COPY.onlyCopy() && QuantizeFtype.COPY.id() == 0);
    }

    @Test
    public void parseByNameIgnoreCaseAndById() {
        assertEquals(QuantizeFtype.Q4_K_M, QuantizeFtype.parse("q4_k_m"));
        assertEquals(QuantizeFtype.MXFP4_MOE, QuantizeFtype.parse("mxfp4_moe"));
        assertEquals(QuantizeFtype.Q4_K, QuantizeFtype.parse("15"));
        assertEquals(QuantizeFtype.Q3_K, QuantizeFtype.parse("12"));
        assertEquals(QuantizeFtype.F32, QuantizeFtype.parse("0"));
        try {
            QuantizeFtype.parse("42");
            fail("42 is not a recipe id");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("is not a quantization type"));
        }
    }
}
