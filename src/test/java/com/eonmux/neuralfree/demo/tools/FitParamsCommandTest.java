package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The command line of {@code nftool fit-params}: the model comes positionally or through {@code -m}, the fit
 * flags take their values ({@code --fit-target} in MiB, {@code --fit-ctx} the smallest context, {@code
 * --fit-print} the estimate), the context and batch flags reach the memory configuration, and every flag whose
 * placement the fit itself has to choose ({@code -ngl}, {@code -ot}, {@code --cpu-moe}, {@code --fit}) is
 * refused by name. {@code --help} prints the usage without a model, and a CPU backend has no device memory to
 * fit, so it is refused too.
 */
public class FitParamsCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static EngineConfig base() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build();
    }

    @Test
    public void theModelComesPositionallyOrThroughTheModelFlag() throws Exception {
        Path model = tmp.newFile("model.gguf").toPath();
        FitParamsCommand.Options positional =
                FitParamsCommand.parse(new String[]{model.toString()}, base());
        assertEquals(model, positional.model);
        FitParamsCommand.Options flagged =
                FitParamsCommand.parse(new String[]{"-m", model.toString(), "--fit-print"}, base());
        assertEquals(model, flagged.model);
        assertTrue(flagged.fitPrint);
    }

    @Test
    public void theFitFlagsTakeTheirValues() throws Exception {
        Path model = tmp.newFile("model.gguf").toPath();
        FitParamsCommand.Options o = FitParamsCommand.parse(new String[]{model.toString(),
                "--fit-target", "256", "-fitc", "8192", "-fitp"}, base());
        assertEquals(256, o.fitTargetMb);
        assertEquals(8192, o.fitCtx);
        assertTrue(o.fitPrint);
    }

    @Test
    public void theFitDefaultsMatchTheReference() throws Exception {
        Path model = tmp.newFile("model.gguf").toPath();
        FitParamsCommand.Options o = FitParamsCommand.parse(new String[]{model.toString()}, base());
        assertEquals(FitParamsCommand.DEFAULT_FIT_TARGET_MIB, o.fitTargetMb);
        assertEquals(FitParamsCommand.DEFAULT_FIT_CTX, o.fitCtx);
    }

    @Test
    public void theContextAndBatchFlagsReachTheMemoryConfiguration() throws Exception {
        Path model = tmp.newFile("model.gguf").toPath();
        FitParamsCommand.Options o = FitParamsCommand.parse(new String[]{model.toString(),
                "-c", "2048", "--ubatch-size", "64"}, base());
        assertEquals(2048, o.cfg.memory().nCtx());
        assertEquals(64, o.cfg.memory().ubatch());
    }

    @Test
    public void aPlacementFlagIsRefusedByName() throws Exception {
        Path model = tmp.newFile("model.gguf").toPath();
        for (String[] refusal : new String[][]{
                {"-ngl", "99"}, {"--gpu-layers", "8"},
                {"-ot", "blk\\..*=cpu"}, {"--override-tensor", "blk\\..*=cpu"},
                {"--cpu-moe"}, {"-cmoe"}, {"--n-cpu-moe", "1"}, {"-ncffn", "1"},
                {"--fit"}}) {
            String[] args = new String[refusal.length + 1];
            args[0] = model.toString();
            System.arraycopy(refusal, 0, args, 1, refusal.length);
            FitParamsCommand.UsageException e = assertThrows(FitParamsCommand.UsageException.class,
                    () -> FitParamsCommand.parse(args, base()));
            assertTrue(refusal[0] + " is named: " + e.getMessage(),
                    e.getMessage().contains(refusal[0]));
        }
    }

    @Test
    public void anUnknownOptionOrAMissingModelIsRefused() throws Exception {
        Path model = tmp.newFile("model.gguf").toPath();
        FitParamsCommand.UsageException unknown = assertThrows(FitParamsCommand.UsageException.class,
                () -> FitParamsCommand.parse(new String[]{model.toString(), "--no-such-flag"}, base()));
        assertTrue(unknown.getMessage(), unknown.getMessage().contains("--no-such-flag"));
        FitParamsCommand.UsageException noModel = assertThrows(FitParamsCommand.UsageException.class,
                () -> FitParamsCommand.parse(new String[]{"--fit-print"}, base()));
        assertTrue(noModel.getMessage(), noModel.getMessage().contains("model"));
        FitParamsCommand.UsageException twice = assertThrows(FitParamsCommand.UsageException.class,
                () -> FitParamsCommand.parse(new String[]{model.toString(), model.toString()}, base()));
        assertTrue(twice.getMessage(), twice.getMessage().contains("too many file arguments"));
    }

    @Test
    public void invalidValuesAreRefusedWithTheFlag() throws Exception {
        Path model = tmp.newFile("model.gguf").toPath();
        for (String[] bad : new String[][]{
                {"--fit-target", "-5"}, {"--fit-target", "abc"},
                {"--fit-ctx", "0"}, {"--fit-ctx", "later"},
                {"-c", "-1"}, {"--ubatch-size", "0"}, {"-ub", "big"}}) {
            String[] args = new String[bad.length + 1];
            args[0] = model.toString();
            System.arraycopy(bad, 0, args, 1, bad.length);
            FitParamsCommand.UsageException e = assertThrows(FitParamsCommand.UsageException.class,
                    () -> FitParamsCommand.parse(args, base()));
            assertTrue(bad[0] + " is named: " + e.getMessage(), e.getMessage().contains(bad[0]));
        }
    }

    @Test
    public void helpPrintsTheUsageAndNeedsNoModel() throws Exception {
        PrintStream realOut = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(captured, true, "UTF-8"));
            assertNull(FitParamsCommand.parse(new String[]{"--help"}, base()));
            assertNull(FitParamsCommand.parse(new String[]{"-h"}, base()));
        } finally {
            System.setOut(realOut);
        }
        String usage = captured.toString("UTF-8");
        assertTrue(usage, usage.contains("nftool fit-params"));
        assertTrue(usage, usage.contains("--fit-target"));
        assertTrue(usage, usage.contains("--fit-ctx"));
        assertTrue(usage, usage.contains("--fit-print"));
    }

    @Test
    public void aCpuBackendHasNoDeviceMemoryToFit() throws Exception {
        Path model = tmp.newFile("model.gguf").toPath();
        FitParamsCommand.Options o = FitParamsCommand.parse(new String[]{model.toString()}, base());
        FitParamsCommand.UsageException e = assertThrows(FitParamsCommand.UsageException.class,
                () -> FitParamsCommand.execute(o));
        assertTrue(e.getMessage(), e.getMessage().contains("--backend cpu"));
    }

    /** The tool over the machine's OpenCL device: the tiny synthetic model fits resident at its full
     * training context, the estimate prints with {@code --fit-print} and the arguments reproduce the load. */
    @Test
    public void theToolPlansTheOpenClDeviceAndPrintsTheFittedArguments() throws Exception {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = SyntheticModels.llama().writeTempFile();
        PrintStream realOut = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(captured, true, "UTF-8"));
            EngineConfig openCl = EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).build();
            FitParamsCommand.execute(FitParamsCommand.parse(
                    new String[]{model.toString(), "--fit-print"}, openCl));
        } finally {
            System.setOut(realOut);
            Files.deleteIfExists(model);
        }
        String stdout = captured.toString("UTF-8");
        assertTrue(stdout, stdout.contains("fit-params " + model + ": RESIDENT placement"));
        assertTrue(stdout, stdout.contains("-c 256 -ngl all"));
        assertTrue(stdout, stdout.contains("usable device memory"));
    }
}
