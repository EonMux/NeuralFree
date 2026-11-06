package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool debug} over the synthetic models on the CPU backend: saving the logits writes the reference's four
 * files with the model's own values, saving embeddings writes one normalized row per prompt for a pooled model, the
 * per-op trace refuses a CPU model by name, and the trace-only flags are refused with {@code --save-logits}.
 */
public class DebugCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static EngineConfig config() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(32).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    private static ByteArrayOutputStream bytes() {
        return new ByteArrayOutputStream();
    }

    private static PrintStream stream(ByteArrayOutputStream bytes) {
        return new PrintStream(bytes, true);
    }

    private static DebugCommand.Options options(Path model, Path dir, String... extra) throws Exception {
        String[] args = new String[6 + extra.length];
        args[0] = "-m";
        args[1] = model.toString();
        args[2] = "-p";
        args[3] = "Hello my name is";
        args[4] = "--logits-output-dir";
        args[5] = dir.toString();
        System.arraycopy(extra, 0, args, 6, extra.length);
        return DebugCommand.parse(args);
    }

    @Test
    public void savingLogitsWritesTheReferenceFiles() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path dir = tmp.newFolder().toPath();
        ByteArrayOutputStream out = bytes();
        ByteArrayOutputStream err = bytes();
        DebugCommand.dispatch(options(model, dir, "--save-logits"), config(), stream(out), stream(err));

        int[] expectedTokens;
        float[] expectedLogits;
        int nVocab;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            expectedTokens = m.tokenizer().encode("Hello my name is", true);
            expectedLogits = DebugCommand.lastLogits(m, expectedTokens);
            nVocab = m.vocabSize();
        }
        String name = DebugCommand.modelName(model);
        Path base = dir.resolve("llamacpp-" + name);
        assertTrue(out.toString(), out.toString().contains("Token ids (" + expectedTokens.length + "):"));

        float[] saved = readFloats(Files.readAllBytes(base.resolveSibling(name(base) + ".bin")));
        assertArrayEquals(expectedLogits, saved, 0.0f);
        String text = new String(Files.readAllBytes(base.resolveSibling(name(base) + ".txt")),
                StandardCharsets.UTF_8);
        assertEquals("one line per logit", nVocab, text.split("\n").length);
        assertTrue(text, text.startsWith("0: "));
        String prompt = new String(Files.readAllBytes(base.resolveSibling(name(base) + "-prompt.txt")),
                StandardCharsets.UTF_8);
        assertTrue(prompt, prompt.startsWith("prompt: Hello my name is\n"));
        assertTrue(prompt, prompt.contains("n_tokens: " + expectedTokens.length + "\n"));
        assertTrue(prompt, prompt.contains("token ids: "));
        int[] savedTokens = readInts(Files.readAllBytes(base.resolveSibling(name(base) + "-tokens.bin")));
        assertArrayEquals(expectedTokens, savedTokens);
    }

    @Test
    public void savingEmbeddingsWritesOneNormalizedRowForAPooledModel() throws Exception {
        Path model = SyntheticModels.bert().writeTempFile();
        Path dir = tmp.newFolder().toPath();
        DebugCommand.dispatch(options(model, dir, "--save-logits", "--embedding", "--embd-normalize", "2"),
                config(), stream(bytes()), stream(bytes()));
        int nEmbd;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            nEmbd = m.spec().embedding().nEmbd();
        }
        Path base = dir.resolve("llamacpp-" + DebugCommand.modelName(model) + "-embeddings");
        float[] saved = readFloats(Files.readAllBytes(
                base.resolveSibling(base.getFileName() + ".bin")));
        assertEquals("one pooled row", nEmbd, saved.length);
        double norm = 0;
        for (float v : saved) {
            norm += (double) v * v;
        }
        assertEquals(1.0, Math.sqrt(norm), 1e-4);
    }

    @Test
    public void theTraceRefusesACpuModelByName() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        DebugCommand.Options o = DebugCommand.parse(new String[]{"-m", model.toString(), "-p", "Hello my name is",
                "--tensor-filter", "mixer"});
        IOException e = assertThrows(IOException.class,
                () -> DebugCommand.dispatch(o, config(), stream(bytes()), stream(bytes())));
        assertTrue(e.getMessage(), e.getMessage().contains("GPU"));
        assertTrue(e.getMessage(), e.getMessage().contains("--save-logits"));
    }

    @Test
    public void saveOnlyFlagsAreRefusedWithoutSaveLogits() {
        DebugCommand.UsageException dir = assertThrows(DebugCommand.UsageException.class,
                () -> DebugCommand.parse(new String[]{"-m", "model.gguf", "-p", "x", "--logits-output-dir", "out"}));
        assertTrue(dir.getMessage(), dir.getMessage().contains("--logits-output-dir"));
        DebugCommand.UsageException embedding = assertThrows(DebugCommand.UsageException.class,
                () -> DebugCommand.parse(new String[]{"-m", "model.gguf", "-p", "x", "--embedding"}));
        assertTrue(embedding.getMessage(), embedding.getMessage().contains("--embedding"));
    }

    @Test
    public void theTraceOnlyFlagsAreRefusedWithSaveLogits() throws Exception {
        DebugCommand.UsageException filter = assertThrows(DebugCommand.UsageException.class,
                () -> DebugCommand.parse(new String[]{"-m", "model.gguf", "-p", "x", "--save-logits",
                        "--tensor-filter", "mixer"}));
        assertTrue(filter.getMessage(), filter.getMessage().contains("--tensor-filter"));
        DebugCommand.UsageException abort = assertThrows(DebugCommand.UsageException.class,
                () -> DebugCommand.parse(new String[]{"-m", "model.gguf", "-p", "x", "--save-logits",
                        "--abort-on-nan"}));
        assertTrue(abort.getMessage(), abort.getMessage().contains("--abort-on-nan"));
    }

    @Test
    public void normalizationNeedsEmbeddingsAndStepsAreCounted() {
        DebugCommand.UsageException norm = assertThrows(DebugCommand.UsageException.class,
                () -> DebugCommand.parse(new String[]{"-m", "model.gguf", "-p", "x", "--embd-normalize", "1"}));
        assertTrue(norm.getMessage(), norm.getMessage().contains("--embd-normalize"));
        DebugCommand.UsageException steps = assertThrows(DebugCommand.UsageException.class,
                () -> DebugCommand.parse(new String[]{"-m", "model.gguf", "-p", "x", "--steps", "-1"}));
        assertTrue(steps.getMessage(), steps.getMessage().contains("--steps"));
    }

    @Test
    public void theModelNameDropsTheExtension() {
        assertEquals("llama-3.2-1B", DebugCommand.modelName(java.nio.file.Paths.get("/models/llama-3.2-1B.gguf")));
        assertEquals("model", DebugCommand.modelName(java.nio.file.Paths.get("model")));
    }

    private static String name(Path base) {
        return base.getFileName().toString();
    }

    private static float[] readFloats(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[bytes.length / Float.BYTES];
        for (int i = 0; i < values.length; i++) {
            values[i] = buffer.getFloat();
        }
        return values;
    }

    private static int[] readInts(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int[] values = new int[bytes.length / Integer.BYTES];
        for (int i = 0; i < values.length; i++) {
            values[i] = buffer.getInt();
        }
        return values;
    }
}
