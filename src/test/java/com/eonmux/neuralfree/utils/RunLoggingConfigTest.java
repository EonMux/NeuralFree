package com.eonmux.neuralfree.utils;

import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.LoggerContext;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The run tasks' logging configuration ({@code src/main/config/log4j2.xml}) writes its log lines to standard error, so
 * a tool's standard output holds only what the tool prints: {@code nftool bench -o jsonl} rows piped into
 * {@code nftool bench-compare}, which refuses any line that is not JSON.
 */
public class RunLoggingConfigTest {

    private static Path runConfig() {
        Path dir = Paths.get(System.getProperty("user.dir")).toAbsolutePath();
        for (Path d = dir; d != null; d = d.getParent()) {
            Path file = d.resolve("src/main/config/log4j2.xml");
            if (Files.isRegularFile(file)) {
                return file;
            }
        }
        throw new IllegalStateException("src/main/config/log4j2.xml not found above " + dir);
    }

    @Test
    public void infoLinesGoToStandardErrorAndLeaveStandardOutputEmpty() {
        PrintStream out = System.out;
        PrintStream err = System.err;
        ByteArrayOutputStream capturedOut = new ByteArrayOutputStream();
        ByteArrayOutputStream capturedErr = new ByteArrayOutputStream();
        LoggerContext ctx = new LoggerContext("run-logging-config-test", null, runConfig().toUri());
        try {
            System.setOut(new PrintStream(capturedOut, true, StandardCharsets.UTF_8));
            System.setErr(new PrintStream(capturedErr, true, StandardCharsets.UTF_8));
            ctx.start();
            Logger logger = ctx.getLogger(RunLoggingConfigTest.class.getName());
            logger.info("weights uploaded");
            ctx.stop();
        } finally {
            System.setOut(out);
            System.setErr(err);
        }
        assertEquals("", capturedOut.toString(StandardCharsets.UTF_8));
        String logged = capturedErr.toString(StandardCharsets.UTF_8);
        assertTrue(logged, logged.contains("INFO") && logged.contains("weights uploaded"));
    }
}
