package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link HfQuantAliases}: every alias row, the online and rejected names, and normalization. */
public class HfQuantAliasesTest {

    private static void assertRow(String name, HfQuantMethod method) {
        HfQuantAliases.Resolution r = HfQuantAliases.resolve(name);
        assertEquals(name, method, r.method());
        assertEquals(HfQuantAliases.normalize(name), r.name());
    }

    @Test
    public void kernelAndEngineAliases() {
        assertRow("gptq", HfQuantMethod.GPTQ);
        assertRow("gptq_marlin", HfQuantMethod.GPTQ);
        assertRow("moe_wna16", HfQuantMethod.GPTQ);
        assertRow("awq", HfQuantMethod.AWQ);
        assertRow("awq_marlin", HfQuantMethod.AWQ);
        assertRow("humming", HfQuantMethod.AUTO_ROUND);
        assertRow("inc", HfQuantMethod.AUTO_ROUND);
        assertRow("auto-round", HfQuantMethod.AUTO_ROUND);
        for (String m : new String[]{"modelopt", "modelopt_fp4", "modelopt_fp8", "modelopt_mxfp8", "modelopt_mixed",
                "petit_nvfp4"}) {
            assertRow(m, HfQuantMethod.MODELOPT);
        }
        assertRow("quark", HfQuantMethod.QUARK);
        assertRow("quark_mxfp4", HfQuantMethod.QUARK);
        assertRow("mxfp8", HfQuantMethod.MXFP8);
        assertRow("mxfp4", HfQuantMethod.MXFP4);
        assertRow("gpt_oss_mxfp4", HfQuantMethod.MXFP4);
        assertRow("auto-round-int8", HfQuantMethod.W8A8_INT8);
        assertRow("w8a8_int8", HfQuantMethod.W8A8_INT8);
        assertRow("w8a8_fp8", HfQuantMethod.W8A8_FP8);
        assertRow("w4afp8", HfQuantMethod.W4AFP8);
        assertRow("mlx_q4", HfQuantMethod.MLX);
        assertRow("mlx_q8", HfQuantMethod.MLX);
        assertRow("fp8", HfQuantMethod.FP8);
        assertRow("fbgemm_fp8", HfQuantMethod.FBGEMM_FP8);
        assertRow("deepseek_v4_fp8", HfQuantMethod.DEEPSEEK_V4_FP8);
        assertRow("compressed-tensors", HfQuantMethod.COMPRESSED_TENSORS);
        assertRow("bitsandbytes", HfQuantMethod.BITSANDBYTES);
        assertRow("bitnet", HfQuantMethod.BITNET);
        assertRow("blockwise_int8", HfQuantMethod.BLOCKWISE_INT8);
        assertRow("hqq", HfQuantMethod.HQQ);
        assertRow("fp_quant", HfQuantMethod.FP_QUANT);
    }

    @Test
    public void onlineQuantizationNames() {
        for (String m : new String[]{"experts_int8", "quark_int4fp8_moe", "mxfp_w4a8", "ptpc_fp8"}) {
            assertRow(m, HfQuantMethod.ONLINE);
            assertTrue(m, HfQuantAliases.isOnline(m));
        }
        assertFalse(HfQuantAliases.isOnline("fp8"));
        assertFalse(HfQuantAliases.isOnline("bitsandbytes"));
    }

    @Test
    public void rejectedNamesCarryAReason() {
        for (String m : new String[]{"torchao", "modelslim", "aqlm", "eetq", "vptq", "spqr", "higgs", "quanto"}) {
            HfQuantAliases.Resolution r = HfQuantAliases.resolve(m);
            assertEquals(m, HfQuantMethod.REJECTED, r.method());
            assertNotNull(m, r.reason());
        }
        assertNull(HfQuantAliases.resolve("fp8").reason());
    }

    @Test
    public void unknownNamesAreErrors() {
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfQuantAliases.resolve("Foo_Q"));
        assertEquals("unsupported quant_method 'Foo_Q'", e.getMessage());
        assertThrows(GgufFormatException.class, () -> HfQuantAliases.resolve(""));
    }

    @Test
    public void namesAreNormalized() {
        assertEquals("compressed-tensors", HfQuantAliases.normalize("Compressed_Tensors"));
        assertEquals(HfQuantMethod.GPTQ, HfQuantAliases.resolve("GPTQ").method());
        assertEquals("gptq", HfQuantAliases.resolve("GPTQ").name());
    }

    @Test
    public void tableCoversEveryMethodThatReadsAConfig() {
        Map<String, HfQuantMethod> table = HfQuantAliases.table();
        for (HfQuantMethod m : HfQuantMethod.values()) {
            if (m != HfQuantMethod.NONE) {
                assertTrue(m.name(), table.containsValue(m));
            }
        }
        assertThrows(UnsupportedOperationException.class, () -> table.put("x", HfQuantMethod.FP8));
    }
}
