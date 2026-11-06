package com.eonmux.neuralfree.gguf.plugin;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchEntry;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfMetadataHook;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.arch.HfVocabHook;
import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertThrows;

/**
 * The plugin SPI: {@link Plugins#load} finds the providers declared in
 * {@code META-INF/services} once, sorted by class name, and the registries merge the provider rows with the
 * collision rule "providers never replace built-ins": every collision is rejected at registry-table build
 * with both sources named.
 *
 * <p>{@code META-INF/services/com.eonmux.neuralfree.gguf.plugin.ArchProvider} and its
 * {@code com.eonmux.neuralfree.gguf.plugin.HfArchProvider} sibling register the providers of this class; their
 * on-classpath names are listed deliberately in reverse sorted order, so the sort test fails if
 * {@link Plugins#load} returned the ServiceLoader's file order. The merge and collision tests call the
 * package's {@code mergeProviders} helpers with synthetic providers instead, so they do not depend on which
 * thread of the test JVM builds the registries' tables first.
 */
public class PluginsTest {

    /** The GGUF names the providers of this class contribute; no built-in table registers them. */
    private static final String PLUGIN_HF_NAME = "MfTestPluginArchForCausalLM";

    /** Service provider, registered first in the provider file, sorted second by class name. */
    public static final class ProviderZulu implements ArchProvider {
        public ProviderZulu() {
        }

        @Override
        public List<ArchEntry> archEntries() {
            return Collections.emptyList();
        }
    }

    /** Service provider, registered second in the provider file, sorted first by class name. */
    public static final class ProviderAlpha implements ArchProvider {
        public ProviderAlpha() {
        }

        @Override
        public List<ArchEntry> archEntries() {
            return Collections.emptyList();
        }
    }

    /** Service provider, registered first in the provider file, sorted second by class name. */
    public static final class HfProviderZulu implements HfArchProvider {
        public HfProviderZulu() {
        }

        @Override
        public List<HfArchEntry> hfArchEntries() {
            return Collections.singletonList(textEntry(PLUGIN_HF_NAME));
        }
    }

    /** Service provider, registered second in the provider file, sorted first by class name. */
    public static final class HfProviderAlpha implements HfArchProvider {
        public HfProviderAlpha() {
        }

        @Override
        public List<HfArchEntry> hfArchEntries() {
            return Collections.emptyList();
        }
    }

    /** A one-row plugin provider for the merge and collision tests. */
    static final class SingleRowProvider implements ArchProvider {
        private final ArchEntry row;

        SingleRowProvider(ArchEntry row) {
            this.row = row;
        }

        @Override
        public List<ArchEntry> archEntries() {
            return Collections.singletonList(row);
        }
    }

    /** A second one-row plugin provider; its name is larger than {@code SingleRowProvider}'s. */
    static final class SecondRowProvider implements ArchProvider {
        private final ArchEntry row;

        SecondRowProvider(ArchEntry row) {
            this.row = row;
        }

        @Override
        public List<ArchEntry> archEntries() {
            return Collections.singletonList(row);
        }
    }

    /** A plugin provider that fails its contract. */
    static final class NullRowsProvider implements ArchProvider {
        @Override
        public List<ArchEntry> archEntries() {
            return null;
        }
    }

    /** A one-row plugin provider for the HF merge and collision tests. */
    static final class SingleHfRowProvider implements HfArchProvider {
        private final HfArchEntry row;

        SingleHfRowProvider(HfArchEntry row) {
            this.row = row;
        }

        @Override
        public List<HfArchEntry> hfArchEntries() {
            return Collections.singletonList(row);
        }
    }

    /** A second one-row plugin provider; its name is larger than {@code SingleHfRowProvider}'s. */
    static final class SecondHfRowProvider implements HfArchProvider {
        private final HfArchEntry row;

        SecondHfRowProvider(HfArchEntry row) {
            this.row = row;
        }

        @Override
        public List<HfArchEntry> hfArchEntries() {
            return Collections.singletonList(row);
        }
    }

    /** A plugin provider that fails its contract. */
    static final class NullHfRowsProvider implements HfArchProvider {
        @Override
        public List<HfArchEntry> hfArchEntries() {
            return null;
        }
    }

    /** The kind of an HF row fixed per test case; {@code HfArchEntry.of} builds only text rows. */
    static final class KindHfRowProvider implements HfArchProvider {
        private final HfArchEntry row;

        KindHfRowProvider(HfArchEntry row) {
            this.row = row;
        }

        @Override
        public List<HfArchEntry> hfArchEntries() {
            return Collections.singletonList(row);
        }
    }

    @Test
    public void loadsServiceProvidersSortedByClassNameNotByFileOrder() {
        List<String> archNames = classNames(Plugins.load(ArchProvider.class));
        assertEquals(Arrays.asList(PluginsTest.class.getName() + "$ProviderAlpha",
                PluginsTest.class.getName() + "$ProviderZulu"), archNames);
        List<String> hfNames = classNames(Plugins.load(HfArchProvider.class));
        assertEquals(Arrays.asList(PluginsTest.class.getName() + "$HfProviderAlpha",
                PluginsTest.class.getName() + "$HfProviderZulu"), hfNames);
    }

    @Test
    public void loadsProvidersOnce() {
        List<ArchProvider> arch = Plugins.load(ArchProvider.class);
        assertSame(arch, Plugins.load(ArchProvider.class));
        List<HfArchProvider> hf = Plugins.load(HfArchProvider.class);
        assertSame(hf, Plugins.load(HfArchProvider.class));
    }

    /** The rows the service providers return reach the HF registry when it builds its table. */
    @Test
    public void providerRowsMergeIntoTheHfRegistry() {
        assertTrue(HfArchRegistry.has(PLUGIN_HF_NAME, HfArchEntry.Kind.TEXT));
        assertFalse(HfArchRegistry.has(PLUGIN_HF_NAME, HfArchEntry.Kind.MMPROJ));
    }

    /** The SPIs whose registries ship without providers load empty until someone registers. */
    @Test
    public void projectorsToolParsersAndReasoningParsersLoadEmpty() {
        assertTrue(Plugins.load(ProjectorProvider.class).isEmpty());
        assertTrue(Plugins.load(ToolParserProvider.class).isEmpty());
        assertTrue(Plugins.load(ReasoningParserProvider.class).isEmpty());
    }

    /** The logits-processor SPI loads its service provider (this test's one contributes no factory). */
    @Test
    public void logitsProcessorProvidersLoad() {
        List<LogitsProcessorProvider> providers = Plugins.load(LogitsProcessorProvider.class);
        assertEquals(1, providers.size());
        assertEquals(EmptyLogitsProcessorProvider.class.getName(), providers.get(0).getClass().getName());
    }

    @Test
    public void archProviderRowsMergeAndBuiltInRowsHold() {
        Map<Arch, ArchEntry> base = new EnumMap<>(Arch.class);
        ArchEntry llama = entry(Arch.LLAMA);
        ArchEntry mamba = entry(Arch.MAMBA);
        base.put(Arch.LLAMA, llama);
        Map<Arch, ArchEntry> m = ArchRegistry.mergeProviders(base,
                Arrays.asList(new SingleRowProvider(mamba), new SingleRowProvider(entry(Arch.BLOOM))));
        assertEquals(3, m.size());
        assertSame(llama, m.get(Arch.LLAMA));
        assertSame(mamba, m.get(Arch.MAMBA));
    }

    @Test
    public void archProviderRowCollidingWithBuiltInIsRejected() {
        Map<Arch, ArchEntry> base = new EnumMap<>(Arch.class);
        base.put(Arch.LLAMA, entry(Arch.LLAMA));
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> ArchRegistry.mergeProviders(base,
                        Collections.singletonList(new SingleRowProvider(entry(Arch.LLAMA)))));
        assertTrue(e.getMessage(), e.getMessage().contains("'llama'"));
        assertTrue(e.getMessage(), e.getMessage().contains("built-in architecture tables"));
        assertTrue(e.getMessage(), e.getMessage().contains("$SingleRowProvider"));
    }

    @Test
    public void archProviderRowCollidingWithAnotherProviderIsRejected() {
        Map<Arch, ArchEntry> base = new EnumMap<>(Arch.class);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> ArchRegistry.mergeProviders(base,
                        Arrays.asList(new SingleRowProvider(entry(Arch.MAMBA)),
                                new SecondRowProvider(entry(Arch.MAMBA)))));
        assertTrue(e.getMessage(), e.getMessage().contains("'mamba'"));
        assertTrue(e.getMessage(), e.getMessage().contains("$SingleRowProvider"));
        assertTrue(e.getMessage(), e.getMessage().contains("$SecondRowProvider"));
    }

    @Test
    public void archProviderFailingItsContractIsRejected() {
        Map<Arch, ArchEntry> base = new EnumMap<>(Arch.class);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> ArchRegistry.mergeProviders(base, Collections.singletonList(new NullRowsProvider())));
        assertTrue(e.getMessage(), e.getMessage().contains("$NullRowsProvider"));
    }

    @Test
    public void hfProviderRowsMergeAcrossKindsAndBuiltInRowsHold() {
        Map<HfArchEntry.Kind, Map<String, HfArchEntry>> base = baseHf();
        HfArchEntry mmproj = textEntry("MfTestPluginMmproj").withKind(HfArchEntry.Kind.MMPROJ);
        Map<HfArchEntry.Kind, Map<String, HfArchEntry>> m = HfArchRegistry.mergeProviders(base,
                Arrays.asList(new SingleHfRowProvider(textEntry("MfTestPluginText")),
                        new KindHfRowProvider(mmproj)));
        assertSame(mmproj, m.get(HfArchEntry.Kind.MMPROJ).get("MfTestPluginMmproj"));
    }

    @Test
    public void hfProviderRowCollidingWithBuiltInIsRejected() {
        Map<HfArchEntry.Kind, Map<String, HfArchEntry>> base = baseHf();
        base.get(HfArchEntry.Kind.TEXT).put("LlamaForCausalLM", textEntry("LlamaForCausalLM"));
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> HfArchRegistry.mergeProviders(base,
                        Collections.singletonList(new SingleHfRowProvider(textEntry("LlamaForCausalLM")))));
        assertTrue(e.getMessage(), e.getMessage().contains("'LlamaForCausalLM'"));
        assertTrue(e.getMessage(), e.getMessage().contains("built-in HF architecture tables"));
        assertTrue(e.getMessage(), e.getMessage().contains("$SingleHfRowProvider"));
    }

    @Test
    public void hfProviderRowCollidingWithAnotherProviderIsRejected() {
        Map<HfArchEntry.Kind, Map<String, HfArchEntry>> base = baseHf();
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> HfArchRegistry.mergeProviders(base,
                        Arrays.asList(new SingleHfRowProvider(textEntry("MfTestPluginText")),
                                new SecondHfRowProvider(textEntry("MfTestPluginText")))));
        assertTrue(e.getMessage(), e.getMessage().contains("'MfTestPluginText'"));
        assertTrue(e.getMessage(), e.getMessage().contains("$SingleHfRowProvider"));
        assertTrue(e.getMessage(), e.getMessage().contains("$SecondHfRowProvider"));
    }

    @Test
    public void hfProviderFailingItsContractIsRejected() {
        Map<HfArchEntry.Kind, Map<String, HfArchEntry>> base = baseHf();
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> HfArchRegistry.mergeProviders(base, Collections.singletonList(new NullHfRowsProvider())));
        assertTrue(e.getMessage(), e.getMessage().contains("$NullHfRowsProvider"));
    }

    private static Map<HfArchEntry.Kind, Map<String, HfArchEntry>> baseHf() {
        Map<HfArchEntry.Kind, Map<String, HfArchEntry>> base = new EnumMap<>(HfArchEntry.Kind.class);
        for (HfArchEntry.Kind k : HfArchEntry.Kind.values()) {
            base.put(k, new LinkedHashMap<String, HfArchEntry>());
        }
        return base;
    }

    private static List<String> classNames(List<?> providers) {
        List<String> out = new ArrayList<>(providers.size());
        for (Object p : providers) {
            out.add(p.getClass().getName());
        }
        return out;
    }

    private static ArchEntry entry(Arch arch) {
        ArchTraits t = ArchTraits.builder(arch).ropeType(RopeType.NORM).memoryKind(MemoryKind.KV)
                .support(SupportClass.GEN).build();
        return ArchEntry.of(t, ArchHparamsHook.NONE);
    }

    /** A minimal text-model row: an invented HF name backed by the llama family's conversion seams. */
    private static HfArchEntry textEntry(String hfName) {
        HfVocabHook noVocab = (ctx, pre) -> Collections.emptyMap();
        return HfArchEntry.of(Collections.singletonList(hfName), HfArchResolver.of(Arch.LLAMA),
                HfMetadataHook.GENERIC, HfTensorHook.DEFAULT, noVocab);
    }
}
