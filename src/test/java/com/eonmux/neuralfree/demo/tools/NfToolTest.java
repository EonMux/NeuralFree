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
        for (String command : new String[]{"export", "download", "slice", "imatrix", "cvector", "perplexity", "passkey",
                "results", "debug", "eval-callback", "bench", "embedding", "retrieval", "lookup-create", "lookup-merge",
                "lookup-stats", "batched-bench", "pack-experts", "tokenize", "dump", "hash", "set-metadata",
                "new-metadata", "convert-endian", "chat", "complete", "eval", "bench-serve",
                "bench-throughput", "bench-latency", "bench-sweep", "bench-startup", "bench-compare"}) {
            assertTrue("the usage names the implemented command " + command, usage.contains(command));
        }
        assertTrue("the surface lists the pending commands too", usage.contains("quantize"));
        assertTrue("batched-bench is routed", NfTool.IMPLEMENTED.contains("batched-bench"));
        assertTrue("pack-experts is routed", NfTool.IMPLEMENTED.contains("pack-experts"));
        assertTrue("fit-params is routed", NfTool.IMPLEMENTED.contains("fit-params"));
        for (String command : new String[]{"results", "debug", "eval-callback"}) {
            assertTrue(command + " is routed", NfTool.IMPLEMENTED.contains(command));
        }
        for (String command : new String[]{"bench-serve", "bench-throughput", "bench-latency",
                "bench-sweep", "bench-startup", "bench-compare"}) {
            assertTrue(command + " is routed", NfTool.IMPLEMENTED.contains(command));
        }
        assertTrue("eval is routed", NfTool.IMPLEMENTED.contains("eval"));
    }

    @Test
    public void everyImplementedRouteHasAHelpSurfaceOfItsOwn() {
        assertTrue(ImatrixCommand.DEFAULT_OUT.endsWith(".gguf"));
        assertTrue(CvectorGenerator.DEFAULT_OUT.endsWith(".gguf"));
        assertTrue("the bench usage documents its sweep flags", BenchCommand.USAGE.contains("--n-prompt"));
    }
}
