package com.eonmux.neuralfree.acceleration.backend.metal;

import org.junit.Test;

import java.util.SortedMap;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The function-constant contract of generated MSL: ids 0-2 carry the local size and id 3 carries
 * {@code NF_FA_SMEM_FLOATS}; any other id or a mismatched type is refused by name.
 */
public class SpecConstantsTest {

    private static final String KERNEL = "constant uint _419_tmp [[function_constant(0)]];\n"
            + "constant uint _419 = is_function_constant_defined(_419_tmp) ? _419_tmp : 1u;\n"
            + "constant uint _420_tmp [[function_constant(1)]];\n"
            + "constant uint _421_tmp [[function_constant(2)]];\n"
            + "constant int NF_FA_SMEM_FLOATS_tmp [[function_constant(3)]];\n"
            + "kernel void LLM_Unary(uint gid [[thread_position_in_grid]]) { }\n";

    @Test
    public void theLocalSizeAndTheSharedMemorySizeMapToTheirConstants() {
        SpecConstants constants = SpecConstants.parse(KERNEL, "LLM_Unary");
        assertFalse(constants.isEmpty());
        SortedMap<Integer, Long> values = constants.values(new long[]{64, 2, 1}, 7168);
        assertEquals(Long.valueOf(64), values.get(0));
        assertEquals(Long.valueOf(2), values.get(1));
        assertEquals(Long.valueOf(1), values.get(2));
        assertEquals(Long.valueOf(7168), values.get(Integer.valueOf(SpecConstants.FA_SMEM_SPEC_ID)));
        assertEquals(4, values.size());
        assertEquals(Character.valueOf('u'), constants.kinds().get(0));
        assertEquals(Character.valueOf('i'),
                constants.kinds().get(Integer.valueOf(SpecConstants.FA_SMEM_SPEC_ID)));
    }

    @Test
    public void aKernelWithoutSpecConstantsHasOnePipelineForEverySize() {
        SpecConstants constants = SpecConstants.parse(
                "kernel void plain(uint gid [[thread_position_in_grid]]) { }\n", "plain");
        assertTrue(constants.isEmpty());
        assertFalse(constants.hasMacros());
        assertTrue(constants.values(new long[]{256, 1, 1}, 3072).isEmpty());
    }

    @Test
    public void constantsUsedInConstantExpressionsBecomeLibraryMacros() {
        String msl = "#ifndef SPIRV_CROSS_CONSTANT_ID_0\n#define SPIRV_CROSS_CONSTANT_ID_0 1u\n#endif\n"
                + "constant uint _553 = SPIRV_CROSS_CONSTANT_ID_0;\n"
                + "#ifndef SPIRV_CROSS_CONSTANT_ID_3\n#define SPIRV_CROSS_CONSTANT_ID_3 3072\n#endif\n"
                + "constant int NF_FA_SMEM_FLOATS = SPIRV_CROSS_CONSTANT_ID_3;\n"
                + "kernel void flash(uint gid [[thread_position_in_grid]]) { }\n";
        SpecConstants constants = SpecConstants.parse(msl, "flash");
        assertTrue(constants.hasMacros());
        SortedMap<Integer, Long> values = constants.values(new long[]{64, 1, 1}, 7168);
        assertEquals("-DSPIRV_CROSS_CONSTANT_ID_0=64 -DSPIRV_CROSS_CONSTANT_ID_3=7168",
                constants.macroOptions(values));
        assertTrue("macro-style constants are not pipeline function constants",
                constants.functionValues(values).isEmpty());
    }

    @Test
    public void aKernelCanUseBothForms() {
        String msl = "constant uint _419_tmp [[function_constant(0)]];\n"
                + "#ifndef SPIRV_CROSS_CONSTANT_ID_3\n#define SPIRV_CROSS_CONSTANT_ID_3 3072\n#endif\n"
                + "constant int NF_FA_SMEM_FLOATS = SPIRV_CROSS_CONSTANT_ID_3;\n"
                + "kernel void mixed(uint gid [[thread_position_in_grid]]) { }\n";
        SpecConstants constants = SpecConstants.parse(msl, "mixed");
        SortedMap<Integer, Long> values = constants.values(new long[]{128, 1, 1}, 3072);
        assertEquals("-DSPIRV_CROSS_CONSTANT_ID_3=3072", constants.macroOptions(values));
        SortedMap<Integer, Long> function = constants.functionValues(values);
        assertEquals(1, function.size());
        assertEquals(Long.valueOf(128), function.get(0));
    }

    @Test
    public void anUnknownSpecialisationIdIsRefusedByName() {
        try {
            SpecConstants.parse("constant uint x [[function_constant(4)]];\n", "LLM_X");
            fail("a function constant the backend cannot supply must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("LLM_X"));
            assertTrue(expected.getMessage(), expected.getMessage().contains("4"));
        }
    }

    @Test
    public void anUnknownMacroIdIsRefusedByName() {
        try {
            SpecConstants.parse("#define SPIRV_CROSS_CONSTANT_ID_4 1\n", "LLM_X");
            fail("a macro constant the backend cannot supply must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("LLM_X"));
            assertTrue(expected.getMessage(), expected.getMessage().contains("4"));
        }
    }

    @Test
    public void aLocalSizeConstantOfTheWrongTypeIsRefused() {
        try {
            SpecConstants.parse("constant int x [[function_constant(0)]];\n", "LLM_X");
            fail("a local-size constant must be uint");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("expected uint"));
        }
    }

    @Test
    public void theSharedMemoryConstantMustBeInt() {
        try {
            SpecConstants.parse("constant uint x [[function_constant(3)]];\n", "LLM_X");
            fail("NF_FA_SMEM_FLOATS must be int");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("expected int"));
        }
    }

    @Test
    public void valuesRequireThreeDimensions() {
        SpecConstants constants = SpecConstants.parse(KERNEL, "LLM_Unary");
        try {
            constants.values(new long[]{64, 1}, 3072);
            fail("the local size must be padded to three dimensions");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("three dimensions"));
        }
    }
}
