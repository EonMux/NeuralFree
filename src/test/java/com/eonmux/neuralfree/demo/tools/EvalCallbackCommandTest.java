package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;

import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool eval-callback} over the synthetic llama model on OpenCL: the prompt's tokens print, the stage trace
 * reports the stages the filter selects (and only those), and a CPU model, which has no stage trace, is refused by
 * name. Skipped when OpenCL is unavailable.
 */
public class EvalCallbackCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path model;

    @Before
    public void writeModel() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).build()).writeTempFile();
    }

    private static EngineConfig gpu() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(8).build())
                .build();
    }

    private static EngineConfig cpu() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(8).build())
                .build();
    }

    private static EvalCallbackCommand.Options options(Path model, String... extra) throws Exception {
        String[] args = new String[4 + extra.length];
        args[0] = "-m";
        args[1] = model.toString();
        args[2] = "-p";
        args[3] = "Hello my name is";
        System.arraycopy(extra, 0, args, 4, extra.length);
        return EvalCallbackCommand.parse(args);
    }

    private String trace(String filter) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        EvalCallbackCommand.dispatch(options(model, "--tensor-filter", filter), gpu(),
                new PrintStream(bytes, true));
        return bytes.toString();
    }

    @Test
    public void theTraceReportsEverySelectedStage() throws Exception {
        String text = trace("mixer-0");
        assertTrue(text, text.contains("number of input tokens ="));
        assertTrue(text, text.contains("mixer-0"));
        assertTrue(text, text.contains("sum = "));
        assertTrue("only the selected stage prints", !text.contains("embedding ["));
        assertTrue("only the selected stage prints", !text.contains("mixer-1"));
    }

    @Test
    public void anEmbeddingFilterSelectsTheEmbeddingStage() throws Exception {
        String text = trace("embedding");
        assertTrue(text, text.contains("embedding ["));
        assertTrue(!text.contains("mixer-0"));
    }

    @Test
    public void aCpuModelHasNoStageTraceAndIsRefusedByName() throws Exception {
        IOException e = assertThrows(IOException.class,
                () -> EvalCallbackCommand.dispatch(options(model), cpu(), new PrintStream(new ByteArrayOutputStream())));
        assertTrue(e.getMessage(), e.getMessage().contains("GPU"));
        assertTrue(e.getMessage(), e.getMessage().contains("CPU"));
    }
}
