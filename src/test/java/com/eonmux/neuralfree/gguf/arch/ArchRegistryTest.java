package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Arch} names and the {@link ArchRegistry} merge of the architecture tables.
 */
public class ArchRegistryTest {

    /** {@code LLM_ARCH_NAMES} in llama.cpp src/llama-arch.cpp has 168 entries including "(unknown)"; NeuralFree
     * keeps its eagle-1 row ({@link Arch#EAGLE}), which the reference list no longer carries, and adds the
     * vLLM-only text architectures GPT-J ({@link Arch#GPTJ}), Step1 ({@link Arch#STEP1}), Step3 text
     * ({@link Arch#STEP3_TEXT}), the openPangu MoE trunk ({@link Arch#PANGU_MOE}), the four families without a
     * llama.cpp converter ({@link Arch#HYPERCLOVAX}, {@link Arch#DOMYN_EDGE}, {@link Arch#RNJ1},
     * {@link Arch#AXK1}), OPT ({@link Arch#OPT}), Step5 ({@link Arch#STEP5}), the Gated DeltaNet hybrid OLMo Hybrid
     * ({@link Arch#OLMO_HYBRID}), the Mamba2 hybrid with shared attention blocks Zamba2 ({@link Arch#ZAMBA2}), LongCat-Flash ({@link Arch#LONGCAT_FLASH}) and its n-gram variant ({@link Arch#LONGCAT_FLASH_NGRAM}), the KDA hybrid AliceAI ({@link Arch#ALICE_AI}), the lightning/MLA hybrid Bailing MoE V2.5 ({@link Arch#BAILINGMOE_V25}), Inkling ({@link Arch#INKLING}), the looped-trunk Dory ({@link Arch#DORY}) and IQuestLoopCoder ({@link Arch#IQUEST_LOOPCODER}), and the four concat-fc draft rows (eagle-mistral-large3, eagle-llama4,
     * eagle-minicpm, eagle-cohere), the InternS1-Pro text backbone ({@link Arch#INTERN_S1_PRO}), for 192 constants. */
    @Test
    public void archHasEveryLlamaCppName() {
        assertEquals(192, Arch.values().length);
        Set<String> names = new HashSet<>();
        for (Arch a : Arch.values()) {
            assertTrue("duplicate name " + a.ggufName(), names.add(a.ggufName()));
            assertSame(a, Arch.fromGgufName(a.ggufName()));
        }
        assertSame(Arch.OPENAI_MOE, Arch.fromGgufName("gpt-oss"));
        assertSame(Arch.UNKNOWN, Arch.fromGgufName("no-such-arch"));
        assertSame(Arch.UNKNOWN, Arch.fromGgufName(null));
    }

    @Test
    public void clipAndUnknownAreRejected() {
        for (Arch a : Arrays.asList(Arch.CLIP, Arch.UNKNOWN)) {
            ArchTraits t = ArchRegistry.traits(a);
            assertEquals(SupportClass.NA, t.support());
            assertNotNull(t.unsupportedReason());
            assertFalse(t.unsupportedReason().trim().isEmpty());
        }
        assertTrue(ArchRegistry.traits(Arch.CLIP).unsupportedReason().contains("clip"));
    }

    /** Every architecture must be registered by exactly one of the four tables (or as rejected). */
    @Test
    public void everyArchIsRegistered() {
        assertEquals("architectures without a table entry: " + ArchRegistry.missing(),
                Collections.emptySet(), ArchRegistry.missing());
    }

    @Test
    public void unregisteredArchThrowsNamingIt() {
        for (Arch a : ArchRegistry.missing()) {
            UnsupportedArchException e = assertThrows(UnsupportedArchException.class, () -> ArchRegistry.get(a));
            assertSame(a, e.arch());
            assertTrue(e.getMessage().contains(a.ggufName()));
        }
    }

    /** deepseek41 runs: its V4.1 sparse attention is served by the CPU DeepSeek-V4 engine. */
    @Test
    public void deepseek41IsSupported() {
        ArchTraits t = ArchRegistry.traits(Arch.DEEPSEEK41);
        assertEquals(SupportClass.GEN, t.support());
        assertNull(t.unsupportedReason());
        assertEquals(MemoryKind.DSV4, t.memoryKind());
        assertEquals(ResidualKind.HYPER_CONNECTIONS, t.residualKind());
    }

    @Test
    public void tablesDoNotRegisterRejectedArchs() {
        Set<Arch> rejected = EnumSet.of(Arch.CLIP, Arch.UNKNOWN);
        for (List<ArchEntry> table : Arrays.asList(ArchTableDense.entries(), ArchTableMoe.entries(),
                ArchTableHybrid.entries(), ArchTableSpecial.entries())) {
            for (ArchEntry e : table) {
                assertFalse(e.arch() + " is registered by ArchRegistry itself", rejected.contains(e.arch()));
            }
        }
    }

    @Test
    public void mergeRejectsDuplicates() {
        ArchEntry a = entry(Arch.LLAMA);
        ArchEntry b = entry(Arch.LLAMA);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> ArchRegistry.merge(Arrays.asList(Collections.singletonList(a), Collections.singletonList(b))));
        assertTrue(e.getMessage().contains("llama"));
    }

    @Test
    public void mergeKeepsEveryEntry() {
        ArchEntry llama = entry(Arch.LLAMA);
        ArchEntry qwen = entry(Arch.QWEN2);
        Map<Arch, ArchEntry> m = ArchRegistry.merge(Arrays.asList(Collections.singletonList(llama),
                Collections.singletonList(qwen), ArchRegistry.rejectedEntries()));
        assertEquals(4, m.size());
        assertSame(llama, m.get(Arch.LLAMA));
        assertSame(qwen, m.get(Arch.QWEN2));
    }

    @Test
    public void traitsBuilderRequiresCoreFields() {
        assertThrows(IllegalStateException.class, () -> ArchTraits.builder(Arch.LLAMA).build());
        assertThrows(IllegalStateException.class,
                () -> ArchTraits.builder(Arch.LLAMA).support(SupportClass.NA).build());
        ArchTraits t = ArchTraits.builder(Arch.NOMIC_BERT).ropeType(RopeType.NEOX).memoryKind(MemoryKind.NONE)
                .support(SupportClass.EMBED).causalDefault(false).graphFamily(Arch.BERT).build();
        assertSame(Arch.BERT, t.graphFamily());
        assertFalse(t.causalDefault());
        assertSame(Arch.LLAMA, entry(Arch.LLAMA).traits().graphFamily());
    }

    private static ArchEntry entry(Arch arch) {
        ArchTraits t = ArchTraits.builder(arch).ropeType(RopeType.NORM).memoryKind(MemoryKind.KV)
                .support(SupportClass.GEN).build();
        return ArchEntry.of(t, ArchHparamsHook.NONE);
    }
}
