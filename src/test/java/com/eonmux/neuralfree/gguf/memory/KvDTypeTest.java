package com.eonmux.neuralfree.gguf.memory;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The KV cache types: the kernel codes shared with kv_types.h / kv_types.glsl, the ggml block geometry, kernel variant
 * names and the llama.cpp {@code -ctk} / {@code -ctv} name parsing (llama.cpp common/arg.cpp:304-314).
 */
public class KvDTypeTest {

    @Test
    public void codesMatchTheKernelConstants() {
        assertEquals(0, KvDType.F32.code());
        assertEquals(1, KvDType.F16.code());
        assertEquals(2, KvDType.Q8_0.code());
        assertEquals(3, KvDType.BF16.code());
        assertEquals(4, KvDType.Q4_0.code());
        assertEquals(5, KvDType.Q4_1.code());
        assertEquals(6, KvDType.IQ4_NL.code());
        assertEquals(7, KvDType.Q5_0.code());
        assertEquals(8, KvDType.Q5_1.code());
        for (int i = 0; i < KvDType.all().size(); i++) {
            assertEquals(i, KvDType.all().get(i).code());
        }
    }

    @Test
    public void blockGeometryFollowsGgml() {
        assertFalse(KvDType.F32.quantized());
        assertFalse(KvDType.F16.quantized());
        assertFalse(KvDType.BF16.quantized());
        assertEquals(1, KvDType.BF16.blockSize());
        assertEquals(2, KvDType.BF16.blockBytes());
        int[] bytes = {34, 18, 20, 18, 22, 24};
        KvDType[] blocks = {KvDType.Q8_0, KvDType.Q4_0, KvDType.Q4_1, KvDType.IQ4_NL, KvDType.Q5_0, KvDType.Q5_1};
        for (int i = 0; i < blocks.length; i++) {
            KvDType t = blocks[i];
            assertTrue(t.quantized());
            assertEquals(32, t.blockSize());
            assertEquals(t + " block bytes", bytes[i], t.blockBytes());
            assertEquals(4L * bytes[i], t.rowBytes(128));
            assertTrue(t.supportsWidth(96));
            assertFalse(t.supportsWidth(48));
            assertThrows(IllegalArgumentException.class, () -> t.rowBytes(48));
        }
        assertEquals(2L * 48, KvDType.BF16.rowBytes(48));
        assertThrows(IllegalArgumentException.class, () -> KvDType.F16.rowBytes(-1));
    }

    @Test
    public void kernelVariantsCoverEveryTypeInCodeOrder() {
        List<String> v = KvDType.kernelVariants("LLM_KV_Store_");
        assertEquals(Arrays.asList("LLM_KV_Store_F32", "LLM_KV_Store_F16", "LLM_KV_Store_Q8_0", "LLM_KV_Store_BF16",
                "LLM_KV_Store_Q4_0", "LLM_KV_Store_Q4_1", "LLM_KV_Store_IQ4_NL", "LLM_KV_Store_Q5_0",
                "LLM_KV_Store_Q5_1"), v);
        assertEquals("LLM_FlashDecode_IQ4_NL", KvDType.IQ4_NL.kernel("LLM_FlashDecode_"));
        assertThrows(UnsupportedOperationException.class, () -> v.add("x"));
    }

    @Test
    public void parsesTheLlamaCppNames() {
        assertEquals("f32, f16, q8_0, bf16, q4_0, q4_1, iq4_nl, q5_0, q5_1", KvDType.names());
        for (KvDType t : KvDType.all()) {
            assertEquals(t, KvDType.parse(t.name().toLowerCase(java.util.Locale.ROOT), "p"));
            assertEquals(t, KvDType.parse(" " + t.name() + " ", "p"));
        }
        assertEquals(KvDType.Q8_0, KvDType.parse("q8", "p"));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> KvDType.parse("q4_k", "--kv-type"));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--kv-type=q4_k "));
        assertTrue(e.getMessage(), e.getMessage().endsWith(KvDType.names()));
        assertThrows(IllegalArgumentException.class, () -> KvDType.parse(null, "p"));
    }
}
