package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.plugin.LogitsProcessorProvider;
import com.eonmux.neuralfree.gguf.plugin.Plugins;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The custom logits processor registry: plugin providers and configured factory classes merge with the "a name
 * registered twice is rejected" rule, a request's {@code vllm_xargs} decide which processors apply, and a factory
 * refusing an argument fails admission naming it.
 */
public class LogitsProcessorsTest {

    /** A factory contributed by a provider in these tests. */
    static final class ProviderFactory implements LogitsProcessorFactory {
        @Override
        public String name() {
            return "provider_test";
        }

        @Override
        public List<String> aliases() {
            return Collections.singletonList("provider-alias");
        }

        @Override
        public void validate(LogitsProcessorArgs args) {
            if (args.has("token")) {
                args.integer("token");
            }
        }

        @Override
        public SamplerStage create(LogitsProcessorArgs args) {
            return args.has("token") ? new MaskStage(args.integer("token")) : null;
        }
    }

    /** A factory named on the command line; public so the registry can construct it reflectively. */
    public static final class NamedFactory implements LogitsProcessorFactory {
        public NamedFactory() {
        }

        @Override
        public String name() {
            return "named_test";
        }

        @Override
        public SamplerStage create(LogitsProcessorArgs args) {
            return null;
        }
    }

    /** A class that is not a factory. */
    public static final class NotAFactory {
        public NotAFactory() {
        }
    }

    /** A factory without a public no-argument constructor. */
    public static final class NoDefaultConstructor implements LogitsProcessorFactory {
        private NoDefaultConstructor() {
        }

        @Override
        public String name() {
            return "no_default";
        }

        @Override
        public SamplerStage create(LogitsProcessorArgs args) {
            return null;
        }
    }

    /** A provider over fixed rows; null models a provider failing its contract. */
    static final class RowProvider implements LogitsProcessorProvider {
        private final List<LogitsProcessorFactory> rows;

        RowProvider(List<LogitsProcessorFactory> rows) {
            this.rows = rows;
        }

        @Override
        public List<LogitsProcessorFactory> logitsProcessorFactories() {
            return rows;
        }
    }

    /** Masks every candidate but one token. */
    static final class MaskStage extends SamplerStage {
        private final int token;

        MaskStage(int token) {
            this.token = token;
        }

        @Override
        public void apply(Sampler.Candidates candidates) {
            for (int i = 0; i < candidates.size(); i++) {
                if (candidates.id(i) != token) {
                    candidates.setLogit(i, Float.NEGATIVE_INFINITY);
                }
            }
            candidates.setSorted(false);
        }
    }

    @Test
    public void providerRowsMergeAndNamesResolve() {
        LogitsProcessors registry = LogitsProcessors.build(
                Collections.singletonList(new RowProvider(Collections.singletonList(new ProviderFactory()))),
                Collections.singletonList(NamedFactory.class.getName()));
        assertEquals(2, registry.factories().size());
        assertTrue(registry.names().contains("provider_test"));
        assertTrue("aliases are keys too", registry.names().contains("provider_alias"));
        assertTrue(registry.names().contains("named_test"));
        assertFalse(registry.isEmpty());
    }

    @Test
    public void aNameRegisteredTwiceIsRejectedNamingBothSources() {
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> LogitsProcessors.build(
                Collections.singletonList(new RowProvider(Collections.singletonList(new NamedFactory()))),
                Collections.singletonList(NamedFactory.class.getName())));
        assertTrue(e.getMessage(), e.getMessage().contains("named_test"));
        assertTrue(e.getMessage(), e.getMessage().contains("$RowProvider"));
        assertTrue(e.getMessage(), e.getMessage().contains("class " + NamedFactory.class.getName()));
    }

    @Test
    @SuppressWarnings({"unchecked", "rawtypes"})
    public void aProviderFailingItsContractIsRejected() {
        IllegalStateException nullRows = assertThrows(IllegalStateException.class,
                () -> LogitsProcessors.build(
                        Collections.singletonList(new RowProvider(null)), Collections.emptyList()));
        assertTrue(nullRows.getMessage(), nullRows.getMessage().contains("$RowProvider"));

        List<LogitsProcessorFactory> raw = (List) Collections.singletonList("not a factory");
        IllegalStateException wrongRow = assertThrows(IllegalStateException.class,
                () -> LogitsProcessors.build(
                        Collections.singletonList(new RowProvider(raw)), Collections.emptyList()));
        assertTrue(wrongRow.getMessage(), wrongRow.getMessage().contains("java.lang.String"));
        assertTrue(wrongRow.getMessage(), wrongRow.getMessage().contains("$RowProvider"));
    }

    @Test
    public void configuredClassesAreLoadedAndChecked() {
        List<LogitsProcessorFactory> loaded = LogitsProcessors.loadClasses(
                Collections.singletonList(NamedFactory.class.getName()));
        assertEquals(1, loaded.size());
        assertTrue(loaded.get(0) instanceof NamedFactory);

        assertMessage(() -> LogitsProcessors.loadClasses(Collections.singletonList("no.such.Factory")),
                LogitsProcessors.PROP_LOGITS_PROCESSORS + ": no.such.Factory is not on the class path");
        assertMessage(() -> LogitsProcessors.loadClasses(Collections.singletonList(NotAFactory.class.getName())),
                "does not implement " + LogitsProcessorFactory.class.getName());
        assertMessage(() -> LogitsProcessors.loadClasses(
                        Collections.singletonList(NoDefaultConstructor.class.getName())),
                "needs a public no-argument constructor");
        assertMessage(() -> LogitsProcessors.loadClasses(Collections.singletonList(" ")),
                "blank class name");
    }

    @Test
    public void factoriesCreateStagesPerRequestAndSkipRequestsTheyLeaveAlone() {
        LogitsProcessors registry = LogitsProcessors.of(Collections.singletonList(new ProviderFactory()));

        GenerationRequest plain = GenerationRequest.defaults();
        assertTrue(registry.stages(plain).isEmpty());

        GenerationRequest armed = GenerationRequest.builder()
                .logitsProcessorArgs(LogitsProcessorArgs.of(Collections.singletonMap("token", 5)))
                .build();
        List<SamplerStage> stages = registry.stages(armed);
        assertEquals(1, stages.size());

        List<SamplerStage> again = registry.stages(armed);
        assertFalse(stages.get(0) == again.get(0));
    }

    @Test
    public void anInvalidArgumentFailsAdmissionNamingTheProcessorAndArgument() {
        LogitsProcessors registry = LogitsProcessors.of(Collections.singletonList(new ProviderFactory()));
        GenerationRequest wrong = GenerationRequest.builder()
                .logitsProcessorArgs(LogitsProcessorArgs.of(Collections.singletonMap("token", "five")))
                .build();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> registry.stages(wrong));
        assertTrue(e.getMessage(), e.getMessage().contains("vllm_xargs: provider_test:"));
        assertTrue(e.getMessage(), e.getMessage().contains("vllm_xargs.token must be an integer"));
    }

    @Test
    public void aConfigurationWithNoProcessorAndTheTestProviderBuildEmpty() {
        assertEquals("the test provider file registers one provider", 1,
                Plugins.load(LogitsProcessorProvider.class).size());
        assertTrue(LogitsProcessors.forConfig(EngineConfig.defaults()).isEmpty());
        assertSame(LogitsProcessors.empty(), LogitsProcessors.of(Collections.emptyList()));
    }

    @Test
    public void forConfigLoadsTheConfiguredClasses() {
        EngineConfig config = EngineConfig.builder()
                .logitsProcessors(Collections.singletonList(NamedFactory.class.getName())).build();
        LogitsProcessors registry = LogitsProcessors.forConfig(config);
        assertFalse(registry.isEmpty());
        assertTrue(registry.names().contains("named_test"));
        assertSame("the same configuration is built once", registry, LogitsProcessors.forConfig(config));
        assertMessage(() -> LogitsProcessors.forConfig(EngineConfig.builder()
                        .logitsProcessors(Collections.singletonList("no.such.Factory")).build()),
                "no.such.Factory");
    }

    @Test
    public void duplicateConfiguredClassesAndBlankEntriesRefuse() {
        assertMessage(() -> EngineConfig.builder().logitsProcessors(
                List.of(NamedFactory.class.getName(), NamedFactory.class.getName())), "repeats");
        assertMessage(() -> EngineConfig.builder().logitsProcessors(List.of(" ")), "blank class name");
        List<String> copy = new ArrayList<>(List.of(NamedFactory.class.getName()));
        EngineConfig config = EngineConfig.builder().logitsProcessors(copy).build();
        copy.add("other");
        assertEquals("the list is copied", List.of(NamedFactory.class.getName()), config.logitsProcessors());
        assertThrows(UnsupportedOperationException.class,
                () -> config.logitsProcessors().add("other"));
    }

    @Test
    public void propertiesParseTheCommaSeparatedClasses() {
        java.util.Properties p = new java.util.Properties();
        p.setProperty(EngineConfig.PROP_LOGITS_PROCESSORS,
                " a.b.Factory , c.d.Factory ");
        assertEquals(List.of("a.b.Factory", "c.d.Factory"), EngineConfig.fromProperties(p).logitsProcessors());
        p.setProperty(EngineConfig.PROP_LOGITS_PROCESSORS, "a.b.Factory,,c.d.Factory");
        assertMessage(() -> EngineConfig.fromProperties(p), "empty entry");
    }

    private static void assertMessage(Runnable call, String expected) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, call::run);
        assertTrue(e.getMessage(), e.getMessage().contains(expected));
    }
}
