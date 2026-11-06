package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The control-vector flags of {@link LLMDemoOptions} (llama.cpp {@code --control-vector},
 * {@code --control-vector-scaled} and {@code --control-vector-layer-range}): both value forms, the layer range's
 * two arguments, the flag override of the {@code -Dneuralfree.*} property list, and the refusals naming the flag.
 */
public class LLMDemoOptionsControlVectorTest {

    private static final String[] PROPS = {
            EngineConfig.PROP_CONTROL_VECTOR, EngineConfig.PROP_CONTROL_VECTOR_LAYER_RANGE,
    };

    private final Map<String, String> saved = new HashMap<>();

    @Before
    public void clearProperties() {
        for (String key : PROPS) {
            saved.put(key, System.getProperty(key));
            System.clearProperty(key);
        }
    }

    @After
    public void restoreProperties() {
        for (Map.Entry<String, String> e : saved.entrySet()) {
            if (e.getValue() == null) {
                System.clearProperty(e.getKey());
            } else {
                System.setProperty(e.getKey(), e.getValue());
            }
        }
    }

    private static LLMDemoOptions parse(String... args) throws UsageException {
        return LLMDemoOptions.parse(args);
    }

    @Test
    public void theVectorFlagsCarryPathsScalesAndRange() throws UsageException {
        EngineConfig e = parse("--control-vector", "a.gguf,b.gguf:0.5",
                "--control-vector-scaled", "c.gguf", "2.0",
                "--control-vector-layer-range", "2", "5").engine;
        assertEquals(3, e.controlVectors().size());
        assertEquals("a.gguf", e.controlVectors().get(0).path());
        assertEquals(1.0f, e.controlVectors().get(0).scale(), 0f);
        assertEquals("b.gguf", e.controlVectors().get(1).path());
        assertEquals(0.5f, e.controlVectors().get(1).scale(), 0f);
        assertEquals("c.gguf", e.controlVectors().get(2).path());
        assertEquals(2.0f, e.controlVectors().get(2).scale(), 0f);
        assertEquals(2, e.controlVectorLayerStart());
        assertEquals(5, e.controlVectorLayerEnd());
    }

    @Test
    public void scaledTakesTheCommaListForm() throws UsageException {
        EngineConfig e = parse("--control-vector-scaled", "a.gguf:1.5,b.gguf:0.25").engine;
        assertEquals(2, e.controlVectors().size());
        assertEquals(1.5f, e.controlVectors().get(0).scale(), 0f);
        assertEquals(0.25f, e.controlVectors().get(1).scale(), 0f);
    }

    @Test
    public void flagsOverrideThePropertyList() throws UsageException {
        System.setProperty(EngineConfig.PROP_CONTROL_VECTOR, "p1.gguf,p2.gguf:3");
        System.setProperty(EngineConfig.PROP_CONTROL_VECTOR_LAYER_RANGE, "1:2");
        EngineConfig e = parse("--control-vector", "cli.gguf", "--control-vector-layer-range", "3", "4").engine;
        assertEquals(1, e.controlVectors().size());
        assertEquals("cli.gguf", e.controlVectors().get(0).path());
        assertEquals(3, e.controlVectorLayerStart());
        assertEquals(4, e.controlVectorLayerEnd());
    }

    @Test
    public void thePropertyBaseSurvivesWithoutFlags() throws UsageException {
        System.setProperty(EngineConfig.PROP_CONTROL_VECTOR, "p1.gguf:7");
        assertEquals(7f, parse().engine.controlVectors().get(0).scale(), 0f);
    }

    @Test
    public void invalidValuesAreRefusedNamingTheFlag() {
        UsageException e = assertThrows(UsageException.class,
                () -> parse("--control-vector-scaled", "a.gguf"));
        assertTrue(e.getMessage(), e.getMessage().contains("--control-vector-scaled"));
        UsageException f = assertThrows(UsageException.class,
                () -> parse("--control-vector-scaled", "a.gguf:x,b.gguf:2"));
        assertTrue(f.getMessage(), f.getMessage().contains("--control-vector-scaled"));
        UsageException r = assertThrows(UsageException.class,
                () -> parse("--control-vector-layer-range", "2"));
        assertTrue(r.getMessage(), r.getMessage().contains("--control-vector-layer-range"));
        UsageException n = assertThrows(UsageException.class,
                () -> parse("--control-vector"));
        assertTrue(n.getMessage(), n.getMessage().contains("--control-vector"));
    }

    @Test
    public void usageListsTheControlVectorFlags() {
        String usage = String.join("\n", LLMDemoOptions.usage());
        assertTrue(usage.contains("--control-vector"));
        assertTrue(usage.contains("--control-vector-scaled"));
        assertTrue(usage.contains("--control-vector-layer-range"));
    }
}
