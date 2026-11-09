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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Paged flash-attention plans and kernels ({@code kernels/llm/flash_decode.cl} LLM_FlashDecodePaged_* and
 * {@code flash_prefill.cl} LLM_FlashAttnPaged_*): the plan constraints of {@link FlashAttnTiling#decodePaged} over
 * the per-sequence slot range (the split length chosen over maxKvLen), the paged grid that folds the sequence in
 * front of the KV head and its tiles, the partial budget of n_seq rows, the kernel selection and {@code kv_types}
 * arguments of the paged same-type and mixed kernels, and every paged entry point's parameter layout (count and
 * scalar count against the {@code ARG_FDP_*} / {@code ARG_FAP_*} tables, read from the source tree, as the Gradle
 * test working directory is the project directory).
 */
public class FlashAttnTilingPagedTest {

    private static final Path KERNELS = Paths.get("src", "main", "resources", "kernels", "llm");

    private static final int SMEM_MIN = 3072;
    private static final int SMEM_48K = 12288 - 1024;
    private static final int CU = 40;
    private static final int MAX_KV = 2048;

    @Test
    public void decodePagedPlansMatchTheDecodeConstraints() {
        // The plan constraints are those of decode, with the split length chosen over maxKvLen: g_tile,
        // min/maxSplitLen and maxSplits equal a decode plan whose key range is maxKvLen.
        int[][] shapes = {{32, 32, 128, 128}, {32, 8, 128, 128}, {8, 4, 256, 256}, {128, 1, 576, 512},
                {16, 2, 512, 512}};
        for (int[] sh : shapes) {
            for (int smem : new int[]{SMEM_MIN, SMEM_48K}) {
                for (boolean fast : new boolean[]{false, true}) {
                    FlashAttnTiling paged = FlashAttnTiling.decodePaged(4, sh[0], sh[1], sh[2], sh[3], MAX_KV,
                            smem, CU, fast);
                    FlashAttnTiling base = FlashAttnTiling.decode(sh[0], sh[1], sh[2], sh[3], MAX_KV, smem, CU,
                            fast);
                    assertEquals(FlashAttnTiling.Plan.DECODE, paged.plan());
                    assertEquals(paged.gTile(), base.gTile());
                    assertEquals(paged.minSplitLen(), base.minSplitLen());
                    assertEquals(paged.maxSplitLen(), base.maxSplitLen());
                    assertEquals(paged.maxSplits(), base.maxSplits());
                    assertEquals(MAX_KV, paged.keyCapacity());
                    assertEquals(fast, paged.decodeFastPath());
                    checkDecodeConstraints(paged);
                }
            }
        }
    }

    @Test
    public void decodePagedSplitsFollowAPerSequenceKeyCount() {
        FlashAttnTiling t = FlashAttnTiling.decodePaged(4, 32, 8, 128, 128, MAX_KV, SMEM_48K, CU, false);
        assertEquals(MAX_KV, t.keyCapacity());
        assertSplit(t.decodeSplit(0), 1, 32);
        assertSplit(t.decodeSplit(128), 2, 64);
        assertSplit(t.decodeSplit(1000), 3, 352);
        assertSplit(t.decodeSplit(MAX_KV), 3, 704);
        // A step may never exceed the per-sequence slot range a sequence is planned over.
        assertThrows(IllegalArgumentException.class, () -> t.decodeSplit(MAX_KV + 1));
        assertThrows(IllegalArgumentException.class, () -> t.decodeSplit(-1));
        // The fast path's device: the same wanted target, capped at the 128-key workgroup length.
        FlashAttnTiling f = FlashAttnTiling.decodePaged(4, 32, 8, 128, 128, MAX_KV, SMEM_48K, CU, true);
        assertSplit(f.decodeSplit(0), 1, 32);
        assertSplit(f.decodeSplit(128), 2, 64);
        assertSplit(f.decodeSplit(1000), 8, 128);
        assertSplit(f.decodeSplit(MAX_KV), 16, 128);
    }

    @Test
    public void decodePagedGlobalFoldsTheSequenceInFrontOfTheKvHeads() {
        FlashAttnTiling t = FlashAttnTiling.decodePaged(6, 32, 8, 128, 128, MAX_KV, SMEM_48K, CU, false);
        // One query head per workgroup: the 32/8 = 4 heads per KV head planned as four one-head tiles.
        assertEquals(1, t.gTile());
        assertEquals(4, t.groupTiles());
        long[] global = t.decodePagedGlobal(6, 9);
        assertArrayEquals(new long[]{9L * FlashAttnTiling.WG, 6L * 8 * t.groupTiles()}, global);
        assertArrayEquals(new long[]{FlashAttnTiling.WG, 1}, FlashAttnTiling.decodeLocal());
        // The one unfolding the kernel does before the KV head index: the sequence block of gid1 is
        // n_head_kv * groupTiles wide, so every workgroup resolves exactly one (seq, kvh, tile).
        final int seqBlock = 8 * t.groupTiles();
        for (long gid1 = 0; gid1 < global[1]; gid1++) {
            long seq = gid1 / seqBlock;
            long gid = gid1 - seq * seqBlock;
            assertTrue(gid >= 0 && gid < seqBlock);
            assertEquals(seq, gid1 / seqBlock);
        }
        assertThrows(IllegalArgumentException.class, () -> t.decodePagedGlobal(0, 1));
        assertThrows(IllegalStateException.class, () -> FlashAttnTiling
                .prefill(32, 8, 128, 128, MAX_KV, SMEM_48K, CU).decodePagedGlobal(2, 1));
    }

    @Test
    public void decodePagedPartialFloatsHoldEverySequence() {
        FlashAttnTiling t = FlashAttnTiling.decodePaged(6, 32, 8, 128, 128, MAX_KV, SMEM_48K, CU, false);
        assertEquals(6L * 32 * t.maxSplits() * 130, t.decodePagedPartialFloats(6));
        // One sequence's rows stay the base decode budget at the same key capacity.
        assertEquals(t.decodePartialFloats(), t.decodePagedPartialFloats(1));
        assertThrows(IllegalArgumentException.class, () -> t.decodePagedPartialFloats(0));
    }

    @Test
    public void invalidPagedPlansAreRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> FlashAttnTiling.decodePaged(0, 32, 8, 128, 128, MAX_KV, SMEM_MIN, CU, false));
        assertThrows(IllegalArgumentException.class,
                () -> FlashAttnTiling.decodePaged(-4, 32, 8, 128, 128, MAX_KV, SMEM_MIN, CU, true));
        // The shape checks are those of decode.
        assertThrows(IllegalArgumentException.class,
                () -> FlashAttnTiling.decodePaged(2, 32, 6, 128, 128, MAX_KV, SMEM_MIN, CU, false));
        assertThrows(IllegalArgumentException.class,
                () -> FlashAttnTiling.decodePaged(2, 32, 8, 130, 128, MAX_KV, SMEM_MIN, CU, true));
        assertThrows(IllegalArgumentException.class,
                () -> FlashAttnTiling.decodePaged(2, 32, 8, 128, 128, 0, SMEM_MIN, CU, false));
        assertThrows(IllegalArgumentException.class,
                () -> FlashAttnTiling.decodePaged(2, 32, 8, 128, 128, MAX_KV, 100, CU, true));
    }

    @Test
    public void pagedKernelsFollowTheSameTypeAndMixedSelection() {
        for (KvDType t : KvDType.kernelTypes()) {
            assertEquals("LLM_FlashDecodePaged_" + t.kernelSuffix(), AttnKernels.flashDecodePaged(t));
            assertEquals(AttnKernels.flashDecodePaged(t), AttnKernels.flashDecodePaged(t, t));
            assertEquals("LLM_FlashAttnPaged_" + t.kernelSuffix(), AttnKernels.flashAttnPaged(t));
            assertEquals(AttnKernels.flashAttnPaged(t), AttnKernels.flashAttnPaged(t, t));
        }
        assertSame(AttnKernels.FLASH_DECODE_PAGED_MIXED, AttnKernels.flashDecodePaged(KvDType.F16, KvDType.Q4_0));
        assertSame(AttnKernels.FLASH_ATTN_PAGED_MIXED, AttnKernels.flashAttnPaged(KvDType.Q8_0, KvDType.BF16));
        // Every paged kernel is in the family list, for LlmKernels.prewarm.
        List<String> all = AttnKernels.all();
        for (KvDType t : KvDType.kernelTypes()) {
            assertTrue(all.contains(AttnKernels.flashDecodePaged(t)));
            assertTrue(all.contains(AttnKernels.flashAttnPaged(t)));
        }
        assertTrue(all.contains(AttnKernels.FLASH_DECODE_PAGED_MIXED));
        assertTrue(all.contains(AttnKernels.FLASH_ATTN_PAGED_MIXED));
    }

    @Test
    public void pagedWithKvTypesKeepsSameTypeArgumentsAndAppendsForMixed() {
        Object[] decode = new Object[AttnKernels.FDP_ARG_COUNT];
        assertSame(decode, AttnKernels.pagedWithKvTypes(decode, KvDType.F16, KvDType.F16));
        Object[] mixedDecode = AttnKernels.pagedWithKvTypes(decode, KvDType.F16, KvDType.Q4_0);
        assertNotSame(decode, mixedDecode);
        assertEquals(AttnKernels.FDP_MIXED_ARG_COUNT, mixedDecode.length);
        assertEquals(AttnKernels.kvTypes(KvDType.F16, KvDType.Q4_0), mixedDecode[AttnKernels.ARG_FDP_KV_TYPES]);

        Object[] prefill = new Object[AttnKernels.FAP_ARG_COUNT];
        Object[] mixedPrefill = AttnKernels.pagedWithKvTypes(prefill, KvDType.Q8_0, KvDType.BF16);
        assertEquals(AttnKernels.FAP_MIXED_ARG_COUNT, mixedPrefill.length);
        assertEquals(AttnKernels.kvTypes(KvDType.Q8_0, KvDType.BF16), mixedPrefill[AttnKernels.ARG_FAP_KV_TYPES]);

        assertThrows(IllegalArgumentException.class,
                () -> AttnKernels.pagedWithKvTypes(new Object[AttnKernels.FDP_MIXED_ARG_COUNT],
                        KvDType.F16, KvDType.F32));
        assertThrows(NullPointerException.class,
                () -> AttnKernels.pagedWithKvTypes(decode, null, KvDType.F32));
    }

    @Test
    public void pagedKernelSignaturesMatchTheArgumentIndices() throws IOException {
        // Signature parity of the 20 paged entry points against the ARG_FDP_* / ARG_FAP_* tables. The typed
        // kernels are instantiated by generator macros and the decode ones declare their parameters through the
        // run-args macro; parameters() resolves both.
        String decodeSrc = new String(Files.readAllBytes(KERNELS.resolve("flash_decode.cl")), StandardCharsets.UTF_8);
        for (KvDType t : KvDType.kernelTypes()) {
            checkKernel(decodeSrc, AttnKernels.flashDecodePaged(t), AttnKernels.FDP_ARG_COUNT,
                    AttnKernels.FDP_SCALAR_COUNT);
        }
        checkKernel(decodeSrc, AttnKernels.FLASH_DECODE_PAGED_MIXED, AttnKernels.FDP_MIXED_ARG_COUNT,
                AttnKernels.FDP_MIXED_SCALAR_COUNT);
        String prefillSrc = new String(Files.readAllBytes(KERNELS.resolve("flash_prefill.cl")),
                StandardCharsets.UTF_8);
        for (KvDType t : KvDType.kernelTypes()) {
            checkKernel(prefillSrc, AttnKernels.flashAttnPaged(t), AttnKernels.FAP_ARG_COUNT,
                    AttnKernels.FAP_SCALAR_COUNT);
        }
        checkKernel(prefillSrc, AttnKernels.FLASH_ATTN_PAGED_MIXED, AttnKernels.FAP_MIXED_ARG_COUNT,
                AttnKernels.FAP_MIXED_SCALAR_COUNT);
    }

    /** The paged argument lists extend the base lists in order: the page scalars shift the buffers, then add theirs. */
    @Test
    public void pagedArgumentListIsTheBaseListPlusThePageGeometry() {
        // Paged decode: 25 base + 5 page scalars, then the 8 base buffers (shifted by 5) and 3 sequence buffers.
        assertEquals(AttnKernels.FD_SCALAR_COUNT + 5, AttnKernels.FDP_SCALAR_COUNT);
        assertEquals(AttnKernels.FD_ARG_COUNT + 8, AttnKernels.FDP_ARG_COUNT);
        assertEquals(AttnKernels.ARG_FD_BUF_Q + 5, AttnKernels.ARG_FDP_BUF_Q);
        assertEquals(AttnKernels.ARG_FD_BUF_PARTIAL + 5, AttnKernels.ARG_FDP_BUF_PARTIAL);
        assertEquals(AttnKernels.ARG_FDP_BUF_PARTIAL + 1, AttnKernels.ARG_FDP_BUF_PAGE_TABLE);
        assertEquals(AttnKernels.ARG_FDP_BUF_PAGE_TABLE + 2, AttnKernels.ARG_FDP_BUF_Q_POS_SEQ);
        // Paged prefill: 29 base + 2 page scalars, then the 9 base buffers (shifted by 2) and the page table.
        assertEquals(AttnKernels.FA_SCALAR_COUNT + 2, AttnKernels.FAP_SCALAR_COUNT);
        assertEquals(AttnKernels.FA_ARG_COUNT + 3, AttnKernels.FAP_ARG_COUNT);
        assertEquals(AttnKernels.ARG_FA_BUF_Q + 2, AttnKernels.ARG_FAP_BUF_Q);
        assertEquals(AttnKernels.ARG_FA_BUF_PARTIAL + 2, AttnKernels.ARG_FAP_BUF_PARTIAL);
        assertEquals(AttnKernels.ARG_FAP_BUF_PARTIAL + 1, AttnKernels.ARG_FAP_BUF_PAGE_TABLE);
    }

    private static void assertSplit(FlashAttnTiling.Split s, int count, int length) {
        assertEquals("split count of " + s, count, s.count());
        assertEquals("split length of " + s, length, s.length());
    }

    /** The checks of nf_fd_plan_ok plus the page geometry of nf_fd_paged_ok in flash_decode.cl. */
    private static void checkDecodeConstraints(FlashAttnTiling t) {
        assertTrue(t.toString(), t.gTile() >= 1 && t.gTile() <= FlashAttnTiling.MAX_G_TILE);
        assertTrue(t.toString(), t.maxSplitLen() >= 32 && t.maxSplitLen() % 32 == 0);
        assertTrue(t.toString(), t.minSplitLen() >= 32 && t.minSplitLen() % 32 == 0
                && t.minSplitLen() <= t.maxSplitLen());
        assertTrue(t.toString(), (long) t.gTile() * (t.dK() + t.maxSplitLen()) <= t.smemFloats());
        assertTrue(t.toString(), (long) t.gTile() * t.dV() <= FlashAttnTiling.ACC_PER_GROUP);
        assertTrue(t.toString(), (long) t.maxSplits() * t.maxSplitLen() >= t.keyCapacity());
    }

    /** The parameters of {@code kernel} in {@code src}: count and scalar count against the argument-index table. */
    private static void checkKernel(String src, String kernel, int args, int scalars) {
        assertEquals(kernel + " parameters", args, parameters(src, kernel).size());
        int nScalars = 0;
        for (String p : parameters(src, kernel)) {
            if (!p.contains("*")) {
                nScalars++;
            }
        }
        assertEquals(kernel + " scalar parameters", scalars, nScalars);
    }

    /** The parameter list of {@code kernel}'s signature, with the decode run-args macro expanded. */
    private static List<String> parameters(String src, String kernel) {
        String inner = declaredParameters(src, kernel);
        final String runArgs = "NF_FLASH_DECODE_PAGED_RUN_ARGS_PAGED";
        int mac = inner.indexOf(runArgs + "(");
        if (mac >= 0) {
            int open = mac + runArgs.length();
            int close = closingParen(inner, open);
            String type = inner.substring(open + 1, close).trim();
            inner = inner.substring(0, mac) + withKvType(macroBody(src, runArgs), type) + inner.substring(close + 1);
        }
        List<String> params = new ArrayList<>();
        for (String p : inner.split(",")) {
            if (!p.trim().isEmpty()) {
                params.add(p.trim());
            }
        }
        return params;
    }

    /**
     * The text between the parentheses of {@code kernel}'s declaration: a literal {@code NF_KERNEL void kernel(...)},
     * or the declaration a kernel-generating macro {@code #define NAME(SFX, KV_T, KV_CODE)} instantiates for
     * {@code NAME(<suffix>, <type>, <code>)}, with {@code KV_T} replaced by the instantiation's type.
     */
    private static String declaredParameters(String src, String kernel) {
        String head = "NF_KERNEL void " + kernel + "(";
        int open = src.indexOf(head);
        if (open >= 0) {
            int paren = open + head.length() - 1;
            return src.substring(paren + 1, closingParen(src, paren));
        }
        final String generated = "##SFX(";
        for (int at = src.indexOf(generated); at >= 0; at = src.indexOf(generated, at + 1)) {
            int decl = src.lastIndexOf("NF_KERNEL void ", at);
            if (decl < 0) {
                continue;
            }
            String prefix = src.substring(decl + "NF_KERNEL void ".length(), at);
            if (!kernel.startsWith(prefix)) {
                continue;
            }
            int def = src.lastIndexOf("#define ", at);
            String macro = src.substring(def + "#define ".length(), src.indexOf('(', def));
            String[] args = instantiation(src, macro, kernel.substring(prefix.length()));
            if (args == null) {
                continue;
            }
            String body = macroBody(src, macro);
            int paren = body.indexOf("NF_KERNEL void " + prefix + generated) + ("NF_KERNEL void " + prefix).length()
                    + generated.length() - 1;
            return withKvType(body.substring(paren + 1, closingParen(body, paren)), args[1].trim());
        }
        throw new AssertionError(kernel + " is declared in the kernel sources");
    }

    /** The arguments of the line {@code macro(suffix, ...)} at the start of a line, or null when none instantiates it. */
    private static String[] instantiation(String src, String macro, String suffix) {
        for (String line : src.split("\n", -1)) {
            String trimmed = line.trim();
            if (trimmed.startsWith(macro + "(" + suffix + ",") && trimmed.endsWith(")")) {
                return trimmed.substring(macro.length() + 1, trimmed.length() - 1).split(",");
            }
        }
        return null;
    }

    /** The index of the parenthesis closing the one at {@code open}. */
    private static int closingParen(String s, int open) {
        int depth = 0;
        for (int i = open; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                depth++;
            } else if (s.charAt(i) == ')' && --depth == 0) {
                return i;
            }
        }
        throw new AssertionError("unbalanced parentheses after index " + open);
    }

    /** {@code text} with the macro parameter {@code KV_T} replaced by {@code type}. */
    private static String withKvType(String text, String type) {
        return text.replaceAll("\\bKV_T\\b", java.util.regex.Matcher.quoteReplacement(type));
    }

    /** The function-macro body of {@code name} in {@code src}, its line continuations joined with spaces. */
    private static String macroBody(String src, String name) {
        String[] lines = src.split("\n", -1);
        int def = -1;
        for (int i = 0; i < lines.length; i++) {
            if (lines[i].startsWith("#define " + name + "(")) {
                def = i;
                break;
            }
        }
        assertTrue("macro " + name + " is defined", def >= 0);
        StringBuilder body = new StringBuilder();
        for (int i = def + 1; i < lines.length; i++) {
            String line = lines[i];
            int cont = line.lastIndexOf('\\');
            if (cont >= 0 && cont == line.length() - 1) {
                body.append(line, 0, cont).append(' ');
            } else {
                body.append(line);
                break;
            }
        }
        return body.toString();
    }
}
