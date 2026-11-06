package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The strict floor: the names it takes, the {@code neuralfree.tool.strict} property, and refusals by name. */
public class ToolStrictTest {

    @Test
    public void theLevelComesFromItsName() {
        assertEquals(ToolStrict.AUTO, ToolStrict.fromName(null));
        assertEquals(ToolStrict.AUTO, ToolStrict.fromName(""));
        assertEquals(ToolStrict.AUTO, ToolStrict.fromName(" auto "));
        assertEquals(ToolStrict.FUNCTION, ToolStrict.fromName("Function"));
        assertEquals(ToolStrict.PARAMETER, ToolStrict.fromName("parameter"));
        assertTrue(ToolStrict.PARAMETER.atLeast(ToolStrict.FUNCTION));
        assertTrue(ToolStrict.FUNCTION.atLeast(ToolStrict.AUTO));
        assertTrue(!ToolStrict.AUTO.atLeast(ToolStrict.FUNCTION));
    }

    @Test
    public void anUnknownLevelIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ToolStrict.fromName("strict"));
        assertTrue(e.getMessage(), e.getMessage().contains("strict"));
        assertTrue(e.getMessage(), e.getMessage().contains("auto, function, parameter"));
    }

    @Test
    public void thePropertySetsTheLevel() {
        Properties p = new Properties();
        assertEquals(ToolStrict.AUTO, EngineConfig.fromProperties(p).toolStrict());
        p.setProperty(EngineConfig.PROP_TOOL_STRICT, "function");
        assertEquals(ToolStrict.FUNCTION, EngineConfig.fromProperties(p).toolStrict());
        assertEquals("the builder carries the level over",
                EngineConfig.fromProperties(p).toolStrict(), EngineConfig.fromProperties(p).toBuilder().build()
                        .toolStrict());
        p.setProperty(EngineConfig.PROP_TOOL_STRICT, "parameter");
        assertEquals(ToolStrict.PARAMETER, EngineConfig.fromProperties(p).toolStrict());
        p.setProperty(EngineConfig.PROP_TOOL_STRICT, "sometimes");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.tool.strict"));
        assertTrue(e.getMessage(), e.getMessage().contains("sometimes"));
    }
}
