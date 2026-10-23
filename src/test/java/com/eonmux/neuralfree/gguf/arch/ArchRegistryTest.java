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
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Arch} names and the {@link ArchRegistry} merge of the architecture tables.
 */
public class ArchRegistryTest {

    /** {@code LLM_ARCH_NAMES} in llama.cpp src/llama-arch.cpp:8-172 has 164 entries including "(unknown)". */
    @Test
    public void archHasEveryLlamaCppName() {
        assertEquals(164, Arch.values().length);
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
    public void clipGptjAndUnknownAreRejected() {
        for (Arch a : Arrays.asList(Arch.CLIP, Arch.GPTJ, Arch.UNKNOWN)) {
            ArchTraits t = ArchRegistry.traits(a);
            assertEquals(SupportClass.NA, t.support());
            assertNotNull(t.unsupportedReason());
            assertFalse(t.unsupportedReason().trim().isEmpty());
        }
        assertTrue(ArchRegistry.traits(Arch.CLIP).unsupportedReason().contains("clip"));
        assertTrue(ArchRegistry.traits(Arch.GPTJ).unsupportedReason().contains("gptj"));
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

    @Test
    public void tablesDoNotRegisterRejectedArchs() {
        Set<Arch> rejected = EnumSet.of(Arch.CLIP, Arch.GPTJ, Arch.UNKNOWN);
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
        assertEquals(5, m.size());
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
