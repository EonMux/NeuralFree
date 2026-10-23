package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineConfig.LoraSpec;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The adapter list of {@link EngineConfig} ({@code neuralfree.lora = path[:scale][,path[:scale]...]}, the
 * {@code --lora} / {@code --lora-scaled} entries of llama.cpp's common_adapter_lora_info).
 */
public class LoraSpecTest {

    @Test
    public void anEntryWithoutAScaleHasScaleOne() {
        LoraSpec s = LoraSpec.parse(" adapters/q.gguf ");
        assertEquals("adapters/q.gguf", s.path());
        assertEquals(1.0f, s.scale(), 0f);
    }

    @Test
    public void theTextAfterTheLastColonIsTheScaleWhenItIsANumber() {
        assertEquals(new LoraSpec("a.gguf", 0.5f), LoraSpec.parse("a.gguf:0.5"));
        assertEquals(new LoraSpec("C:/lora/a.gguf", -1.25f), LoraSpec.parse("C:/lora/a.gguf:-1.25"));
        assertEquals(LoraSpec.of("C:/lora/a.gguf"), LoraSpec.parse("C:/lora/a.gguf"));
    }

    @Test
    public void aListKeepsItsOrderAndSkipsEmptyEntries() {
        assertEquals(Arrays.asList(new LoraSpec("a.gguf", 2.0f), LoraSpec.of("b.gguf")),
                LoraSpec.parseList("a.gguf:2, ,b.gguf,"));
        assertTrue(LoraSpec.parseList(null).isEmpty());
    }

    @Test
    public void thePropertyAndTheBuilderFillTheConfig() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_LORA, "x.gguf:0.75,y.gguf");
        EngineConfig fromProps = EngineConfig.fromProperties(p);
        assertEquals(Arrays.asList(new LoraSpec("x.gguf", 0.75f), LoraSpec.of("y.gguf")), fromProps.loras());
        assertEquals(fromProps.loras(), fromProps.toBuilder().build().loras());

        EngineConfig built = EngineConfig.builder().addLora("z.gguf", 1.5f).build();
        assertEquals(Collections.singletonList(new LoraSpec("z.gguf", 1.5f)), built.loras());
        assertTrue(EngineConfig.defaults().loras().isEmpty());
    }

    @Test
    public void aBlankPathOrANonFiniteScaleIsRejected() {
        try {
            new LoraSpec(" ", 1.0f);
            fail("a blank path must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("path"));
        }
        try {
            new LoraSpec("a.gguf", Float.NaN);
            fail("a NaN scale must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("a.gguf"));
        }
    }
}
