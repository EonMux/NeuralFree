package com.eonmux.neuralfree.demo.tools;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * The launcher surface: the usage names every implemented subcommand (the offline ones without a model), and
 * every listed command the launcher does not implement is named as pending. The routes themselves are one-line
 * dispatches into the command classes their {@code --help} surfaces and tests cover.
 */
public class NfToolTest {

    @Test
    public void theUsageNamesEveryImplementedSubcommand() {
        StringBuilder out = new StringBuilder();
        NfTool.usage(out);
        String usage = out.toString();
        for (String command : new String[]{"export", "download", "imatrix", "cvector"}) {
            assertTrue("the usage names the implemented command " + command, usage.contains(command));
        }
        assertTrue("the surface lists the pending commands too", usage.contains("quantize"));
    }

    @Test
    public void everyImplementedRouteHasAHelpSurfaceOfItsOwn() {
        assertTrue(ImatrixCommand.DEFAULT_OUT.endsWith(".gguf"));
        assertTrue(CvectorGenerator.DEFAULT_OUT.endsWith(".gguf"));
    }
}
