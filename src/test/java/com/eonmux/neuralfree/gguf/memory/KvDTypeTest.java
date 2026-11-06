package com.eonmux.neuralfree.gguf.memory;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The KV cache types: the kernel codes shared with kv_types.h / kv_types.glsl, the ggml block geometry, kernel variant
 * names and the llama.cpp {@code -ctk} / {@code -ctv} name parsing (llama.cpp common/arg.cpp:304-314) plus the FP8
 * {@code CacheDType} names of vLLM (vllm/config/cache.py:44-67).
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
        assertEquals(9, KvDType.FP8_E4M3.code());
        assertEquals(10, KvDType.FP8_E5M2.code());
        assertEquals(11, KvDType.FP8_PER_TOKEN_HEAD.code());
        assertEquals(12, KvDType.INT8_PER_TOKEN_HEAD.code());
        assertEquals(13, KvDType.INT4_PER_TOKEN_HEAD.code());
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
    public void fp8StoresOneCodePerValue() {
        assertNull(KvDType.FP8_E4M3.ggmlType());
        assertNull(KvDType.FP8_E5M2.ggmlType());
        for (KvDType t : Arrays.asList(KvDType.FP8_E4M3, KvDType.FP8_E5M2)) {
            assertFalse(t.quantized());
            assertTrue(t.fp8());
            assertFalse(t.perTokenHead());
            assertEquals(1, t.blockSize());
            assertEquals(1, t.blockBytes());
            assertEquals(48, t.rowBytes(48));
            assertTrue(t.supportsWidth(1));
            assertTrue(t.supportsWidth(48));
            assertFalse(t.supportsWidth(-1));
            assertTrue(t.sidePlanes().isEmpty());
        }
    }

    /** Per-token-head rows keep one code per value (two per byte for INT4) and a per-head F32 scale plane. */
    @Test
    public void perTokenHeadStoresCodesAndOneScalePerHead() {
        for (KvDType t : Arrays.asList(KvDType.FP8_PER_TOKEN_HEAD, KvDType.INT8_PER_TOKEN_HEAD,
                KvDType.INT4_PER_TOKEN_HEAD)) {
            assertNull(t.ggmlType());
            assertFalse(t.fp8());
            assertTrue(t.perTokenHead());
            assertFalse(t.quantized());
            assertEquals(1, t.sidePlanes().size());
            KvDType.SidePlane plane = t.sidePlanes().get(0);
            assertTrue(plane.perHead());
            assertEquals(KvDType.F32, plane.dtype());
            assertEquals(1, plane.width());
            assertEquals(1, plane.widthFor(1));
            assertEquals(2, plane.widthFor(2));
            assertEquals(8, plane.rowBytesFor(2));
        }
        assertEquals(48, KvDType.FP8_PER_TOKEN_HEAD.rowBytes(48));
        assertEquals(48, KvDType.INT8_PER_TOKEN_HEAD.rowBytes(48));
        assertEquals(24, KvDType.INT4_PER_TOKEN_HEAD.rowBytes(48));
        assertEquals(2, KvDType.INT4_PER_TOKEN_HEAD.blockSize());
        assertTrue(KvDType.INT4_PER_TOKEN_HEAD.supportsWidth(48));
        assertFalse(KvDType.INT4_PER_TOKEN_HEAD.supportsWidth(49));
        assertThrows(IllegalArgumentException.class, () -> KvDType.INT4_PER_TOKEN_HEAD.rowBytes(49));
    }

    @Test
    public void kernelVariantsCoverEveryTypeInCodeOrder() {
        List<String> v = KvDType.kernelVariants("LLM_KV_Store_");
        assertEquals(Arrays.asList("LLM_KV_Store_F32", "LLM_KV_Store_F16", "LLM_KV_Store_Q8_0", "LLM_KV_Store_BF16",
                "LLM_KV_Store_Q4_0", "LLM_KV_Store_Q4_1", "LLM_KV_Store_IQ4_NL", "LLM_KV_Store_Q5_0",
                "LLM_KV_Store_Q5_1", "LLM_KV_Store_FP8_E4M3", "LLM_KV_Store_FP8_E5M2",
                "LLM_KV_Store_FP8_PER_TOKEN_HEAD", "LLM_KV_Store_INT8_PER_TOKEN_HEAD",
                "LLM_KV_Store_INT4_PER_TOKEN_HEAD", "LLM_KV_Store_NVFP4", "LLM_KV_Store_NVFP4_4OVER6",
                "LLM_KV_Store_TQ_K8_KEYS", "LLM_KV_Store_TQ_4BIT_KEYS", "LLM_KV_Store_TQ_3BIT_KEYS",
                "LLM_KV_Store_TQ_4BIT_VALUES", "LLM_KV_Store_TQ_3BIT_VALUES", "LLM_KV_Store_FP8_DS_MLA",
                "LLM_KV_Store_NVFP4_DS_MLA"), v);
        assertEquals("LLM_FlashDecode_IQ4_NL", KvDType.IQ4_NL.kernel("LLM_FlashDecode_"));
        assertEquals("LLM_FlashDecode_FP8_E4M3", KvDType.FP8_E4M3.kernel("LLM_FlashDecode_"));
        assertThrows(UnsupportedOperationException.class, () -> v.add("x"));
    }

    @Test
    public void parsesTheLlamaCppAndVllmNames() {
        assertEquals("f32, f16, q8_0, bf16, q4_0, q4_1, iq4_nl, q5_0, q5_1, fp8_e4m3, fp8_e5m2,"
                + " fp8_per_token_head, int8_per_token_head, int4_per_token_head, nvfp4, nvfp4_4over6,"
                + " fp8_ds_mla, nvfp4_ds_mla, fp8_inc, turboquant_k8v4, turboquant_4bit_nc,"
                + " turboquant_k3v4_nc, turboquant_3bit_nc", KvDType.names());
        for (KvDType t : KvDType.all()) {
            assertEquals(t, KvDType.parse(t.name().toLowerCase(java.util.Locale.ROOT), "p"));
            assertEquals(t, KvDType.parse(" " + t.name() + " ", "p"));
        }
        assertEquals(KvDType.Q8_0, KvDType.parse("q8", "p"));
        assertEquals(KvDType.FP8_E4M3, KvDType.parse("fp8", "p"));
        assertEquals(KvDType.FP8_E4M3, KvDType.parse("FP8_E4M3", "p"));
        assertEquals(KvDType.FP8_E5M2, KvDType.parse("fp8_e5m2", "p"));
        assertEquals(KvDType.FP8_PER_TOKEN_HEAD, KvDType.parse("fp8_per_token_head", "p"));
        assertEquals(KvDType.INT8_PER_TOKEN_HEAD, KvDType.parse("INT8_PER_TOKEN_HEAD", "p"));
        assertEquals(KvDType.INT4_PER_TOKEN_HEAD, KvDType.parse("int4_per_token_head", "p"));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> KvDType.parse("q4_k", "--kv-type"));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--kv-type=q4_k "));
        assertTrue(e.getMessage(), e.getMessage().endsWith(KvDType.names()));
        assertThrows(IllegalArgumentException.class, () -> KvDType.parse(null, "p"));
    }
}
