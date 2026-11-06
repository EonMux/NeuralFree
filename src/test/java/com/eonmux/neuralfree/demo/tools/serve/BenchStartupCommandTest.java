package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@code bench-startup}: the defaults of the reference, the health URL derivation and the refusal of
 * an invocation that could not measure anything.
 */
public class BenchStartupCommandTest {

    @Test
    public void theDefaultsMatchTheReference() throws Exception {
        BenchStartupCommand.Args args = BenchStartupCommand.parse(
                new String[]{"--serve-cmd", "neuralfree serve model.gguf"});
        assertEquals(Arrays.asList("neuralfree", "serve", "model.gguf"), args.serveCmd);
        assertEquals(3, args.numItersCold);
        assertEquals(1, args.numItersWarmup);
        assertEquals(3, args.numItersWarm);
        assertEquals(300, args.serverReadyTimeout);
        assertEquals("http://127.0.0.1:8080/health", args.healthUrl);
        assertNull(args.logFile);
        assertEquals(null, args.outputJson);
    }

    @Test
    public void theHealthUrlFollowsTheServerFlagsOrTheExplicitValue() throws Exception {
        BenchStartupCommand.Args derived = BenchStartupCommand.parse(new String[]{
                "--serve-cmd", "neuralfree serve --host 0.0.0.0 --port 9001"});
        assertEquals("http://0.0.0.0:9001/health", derived.healthUrl);
        BenchStartupCommand.Args explicit = BenchStartupCommand.parse(new String[]{
                "--serve-cmd", "neuralfree serve", "--health-url", "http://example.test/health"});
        assertEquals("http://example.test/health", explicit.healthUrl);
    }

    @Test
    public void theFlagsParse() throws Exception {
        BenchStartupCommand.Args args = BenchStartupCommand.parse(new String[]{
                "--serve-cmd", "s", "--num-iters-cold", "2", "--num-iters-warmup", "0",
                "--num-iters-warm", "4", "--server-ready-timeout", "9", "--log-file", "log.txt",
                "--show-stdout", "--output-json", "out.json"});
        assertEquals(2, args.numItersCold);
        assertEquals(0, args.numItersWarmup);
        assertEquals(4, args.numItersWarm);
        assertEquals(9, args.serverReadyTimeout);
        assertEquals("log.txt", args.logFile.toString());
        assertTrue(args.showStdout);
        assertEquals("out.json", args.outputJson);
    }

    @Test
    public void aSweepWithoutIterationsIsRefused() {
        try {
            BenchStartupCommand.parse(new String[]{"--serve-cmd", "s", "--num-iters-cold", "0",
                    "--num-iters-warm", "0"});
            fail("at least one iteration is needed");
        } catch (BenchStartupCommand.UsageException expected) {
            assertTrue(expected.getMessage().contains("cannot both be 0"));
        }
    }

    @Test
    public void aMissingCommandOrAnUnknownFlagIsRefused() {
        try {
            BenchStartupCommand.parse(new String[]{});
            fail("--serve-cmd is required");
        } catch (BenchStartupCommand.UsageException expected) {
            assertTrue(expected.getMessage().contains("--serve-cmd"));
        }
        try {
            BenchStartupCommand.parse(new String[]{"--serve-cmd", "s", "--no-such-flag"});
            fail("unknown flags are refused");
        } catch (BenchStartupCommand.UsageException expected) {
            assertTrue(expected.getMessage().contains("unknown option"));
        }
    }
}
