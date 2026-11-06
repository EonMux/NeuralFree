package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Kernel selection of {@link AttnKernels} for K and V caches of equal and of different types: the same-type variants
 * for equal types, the mixed kernels with the packed {@code kv_types} argument otherwise, the packed {@code flags}
 * value of a tree verify step, and the argument positions of the mixed kernels against their signatures in
 * {@code kernels/llm/flash_decode.cl} and {@code flash_prefill.cl} (read from the source tree, as the Gradle test
 * working directory is the project directory).
 */
public class AttnKernelsTest {

    private static final Path KERNELS = Paths.get("src", "main", "resources", "kernels", "llm");

    @Test
    public void equalTypesSelectTheTypeVariants() {
        for (KvDType t : KvDType.kernelTypes()) {
            assertEquals(AttnKernels.flashDecode(t), AttnKernels.flashDecode(t, t));
            assertEquals(AttnKernels.flashAttn(t), AttnKernels.flashAttn(t, t));
        }
    }

    @Test
    public void differentTypesSelectTheMixedKernels() {
        for (KvDType k : KvDType.values()) {
            for (KvDType v : KvDType.values()) {
                if (k != v) {
                    assertEquals(AttnKernels.FLASH_DECODE_MIXED, AttnKernels.flashDecode(k, v));
                    assertEquals(AttnKernels.FLASH_ATTN_MIXED, AttnKernels.flashAttn(k, v));
                }
            }
        }
        assertTrue(AttnKernels.all().contains(AttnKernels.FLASH_DECODE_MIXED));
        assertTrue(AttnKernels.all().contains(AttnKernels.FLASH_ATTN_MIXED));
    }

    @Test
    public void kvTypesPacksTheKAndVCodes() {
        for (KvDType k : KvDType.values()) {
            for (KvDType v : KvDType.values()) {
                int packed = AttnKernels.kvTypes(k, v);
                assertEquals(k.code(), packed & 0xFF);
                assertEquals(v.code(), (packed >> AttnKernels.KV_TYPE_BITS) & 0xFF);
                assertEquals(0, packed >> (2 * AttnKernels.KV_TYPE_BITS));
            }
        }
    }

    /** The fresh-row count of a tree verify step rides in flags bits 5-31 next to the tree bit. */
    @Test
    public void treeFlagsPacksTheFreshRowCountBesideTheTreeBit() {
        int base = AttnKernels.attnFlags(true, true, false) | AttnKernels.FLAG_V_ALIAS;
        int flags = AttnKernels.treeFlags(base, 31);
        assertEquals(base | AttnKernels.FLAG_TREE | (31 << AttnKernels.FLAG_FRESH_SHIFT), flags);
        assertEquals(31, AttnKernels.flagFresh(flags));
        // a step without fresh rows packs 0, and without the tree bit no count is read at all
        assertEquals(base | AttnKernels.FLAG_TREE, AttnKernels.treeFlags(base, 0));
        assertEquals(0, AttnKernels.flagFresh(base));
        // the count fills exactly bits 5-31: the 33 fresh rows of a width-32 verify step pack, while larger
        // counts, negative ones and stale bits do not fit
        assertEquals(33, AttnKernels.flagFresh(AttnKernels.treeFlags(base, 33)));
        assertThrows(IllegalArgumentException.class, () -> AttnKernels.treeFlags(base, 1 << 27));
        assertThrows(IllegalArgumentException.class, () -> AttnKernels.treeFlags(base, -1));
        assertThrows(IllegalArgumentException.class,
                () -> AttnKernels.treeFlags(base | (7 << AttnKernels.FLAG_FRESH_SHIFT), 2));
    }

    @Test
    public void withKvTypesKeepsSameTypeArgumentsAndAppendsForMixed() {
        Object[] decode = new Object[AttnKernels.FD_ARG_COUNT];
        assertSame(decode, AttnKernels.withKvTypes(decode, KvDType.F16, KvDType.F16));
        Object[] mixedDecode = AttnKernels.withKvTypes(decode, KvDType.F16, KvDType.Q4_0);
        assertNotSame(decode, mixedDecode);
        assertEquals(AttnKernels.FD_MIXED_ARG_COUNT, mixedDecode.length);
        assertEquals(AttnKernels.kvTypes(KvDType.F16, KvDType.Q4_0), mixedDecode[AttnKernels.ARG_FD_KV_TYPES]);

        Object[] prefill = new Object[AttnKernels.FA_ARG_COUNT];
        Object[] mixedPrefill = AttnKernels.withKvTypes(prefill, KvDType.Q8_0, KvDType.BF16);
        assertEquals(AttnKernels.FA_MIXED_ARG_COUNT, mixedPrefill.length);
        assertEquals(AttnKernels.kvTypes(KvDType.Q8_0, KvDType.BF16), mixedPrefill[AttnKernels.ARG_FA_KV_TYPES]);

        assertThrows(IllegalArgumentException.class,
                () -> AttnKernels.withKvTypes(new Object[AttnKernels.FD_MIXED_ARG_COUNT], KvDType.F16, KvDType.F32));
        assertThrows(NullPointerException.class,
                () -> AttnKernels.withKvTypes(decode, null, KvDType.F32));
    }

    @Test
    public void mixedKernelSignaturesMatchTheArgumentIndices() throws IOException {
        checkSignature("flash_decode.cl", AttnKernels.FLASH_DECODE_MIXED, AttnKernels.FD_MIXED_ARG_COUNT,
                AttnKernels.FD_MIXED_SCALAR_COUNT, AttnKernels.ARG_FD_KV_TYPES);
        checkSignature("flash_prefill.cl", AttnKernels.FLASH_ATTN_MIXED, AttnKernels.FA_MIXED_ARG_COUNT,
                AttnKernels.FA_MIXED_SCALAR_COUNT, AttnKernels.ARG_FA_KV_TYPES);
    }

    /** The parameters of {@code kernel} in {@code file}: count, scalar count and the position of {@code kv_types}. */
    private static void checkSignature(String file, String kernel, int args, int scalars, int kvTypesIndex)
            throws IOException {
        String src = new String(Files.readAllBytes(KERNELS.resolve(file)), StandardCharsets.UTF_8);
        String head = "NF_KERNEL void " + kernel + "(";
        int open = src.indexOf(head);
        assertTrue(kernel + " is declared in " + file, open >= 0);
        int close = src.indexOf(')', open);
        List<String> params = new ArrayList<>();
        for (String p : src.substring(open + head.length(), close).split(",")) {
            params.add(p.trim());
        }
        assertEquals(kernel + " parameters", args, params.size());
        int nScalars = 0;
        for (String p : params) {
            if (!p.contains("*")) {
                nScalars++;
            }
        }
        assertEquals(kernel + " scalar parameters", scalars, nScalars);
        assertEquals("const int kv_types", params.get(kvTypesIndex));
    }
}
