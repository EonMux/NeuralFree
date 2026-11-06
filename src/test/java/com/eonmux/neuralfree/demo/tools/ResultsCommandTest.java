package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
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
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool results} over the synthetic llama model on the CPU backend: the written file is llama.cpp's GGUF
 * layout ({@code path_model}, an I32 token tensor and an F32 token-major logits tensor), a check of a fresh run
 * passes with zero NMSE, a changed logit fails with a large NMSE, and a changed token is refused before any logit
 * is compared.
 */
public class ResultsCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static EngineConfig config() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(128).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    private static PrintStream stream(ByteArrayOutputStream bytes) {
        return new PrintStream(bytes, true);
    }

    private int run(Path model, Path results, String... extra) throws Exception {
        String[] args = new String[6 + extra.length];
        args[0] = "-m";
        args[1] = model.toString();
        args[2] = "-p";
        args[3] = "The quick brown fox";
        args[4] = "-o";
        args[5] = results.toString();
        System.arraycopy(extra, 0, args, 6, extra.length);
        return ResultsCommand.dispatch(args, config(), System.out, System.err);
    }

    @Test
    public void theWrittenFileHoldsTheReferenceLayout() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path results = tmp.newFolder().toPath().resolve("results.gguf");
        int[] expectedTokens;
        float[] expectedLogits;
        int nVocab;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            expectedTokens = m.tokenizer().encode("The quick brown fox", true);
            expectedLogits = ResultsCommand.logits(m, expectedTokens, -1);
            nVocab = m.vocabSize();
        }
        assertEquals(0, run(model, results));

        try (GGUFFileParser file = new GGUFFileParser(results.toString())) {
            assertEquals(model.toString(), file.getString(ResultsCommand.KEY_PATH_MODEL));
            GGUFTensorInfo tokens = file.getTensor(ResultsCommand.TOKENS_TENSOR);
            assertNotNull(tokens);
            assertEquals(GGMLType.I32, tokens.type());
            assertArrayEquals(new long[]{expectedTokens.length}, tokens.dims());
            GGUFTensorInfo logits = file.getTensor(ResultsCommand.LOGITS_TENSOR);
            assertNotNull(logits);
            assertEquals(GGMLType.F32, logits.type());
            assertArrayEquals(new long[]{expectedTokens.length, nVocab}, logits.dims());
            assertArrayEquals(expectedTokens, readInts(file, tokens));
            assertArrayEquals(expectedLogits, readFloats(file, logits), 0.0f);
        }
    }

    @Test
    public void checkingAFreshRunPassesWithZeroNmse() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path results = tmp.newFolder().toPath().resolve("results.gguf");
        assertEquals(0, run(model, results));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        int code = ResultsCommand.dispatch(new String[]{"-m", model.toString(), "-p", "The quick brown fox",
                "-o", results.toString(), "--check"}, config(), stream(out), stream(err));
        assertEquals(0, code);
        assertTrue(out.toString(), out.toString().contains("OK"));
        assertTrue(err.toString(), err.toString().contains("NMSE=0.000e+00"));
    }

    @Test
    public void aChangedLogitFailsTheCheck() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path results = tmp.newFolder().toPath().resolve("results.gguf");
        assertEquals(0, run(model, results));
        try (GGUFFileParser file = new GGUFFileParser(results.toString())) {
            GGUFTensorInfo logits = file.getTensor(ResultsCommand.LOGITS_TENSOR);
            long offset = file.getDataOffset() + logits.offset();
            try (FileChannel channel = FileChannel.open(results, StandardOpenOption.READ,
                    StandardOpenOption.WRITE)) {
                ByteBuffer first = ByteBuffer.allocate(Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
                channel.read(first, offset);
                first.flip();
                float value = first.getFloat() + 100.0f;
                ByteBuffer changed = ByteBuffer.allocate(Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
                changed.putFloat(value).flip();
                channel.write(changed, offset);
            }
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        int code = ResultsCommand.dispatch(new String[]{"-m", model.toString(), "-p", "The quick brown fox",
                "-o", results.toString(), "--check"}, config(), stream(out), stream(err));
        assertEquals(1, code);
        assertTrue(out.toString(), out.toString().contains("FAIL"));
        assertTrue(err.toString(), err.toString().contains("NMSE="));
        assertTrue(err.toString(), !err.toString().contains("NMSE=0.000e+00"));
    }

    @Test
    public void aChangedTokenIsRefusedBeforeAnyLogitComparison() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path results = tmp.newFolder().toPath().resolve("results.gguf");
        assertEquals(0, run(model, results));
        try (GGUFFileParser file = new GGUFFileParser(results.toString())) {
            GGUFTensorInfo tokens = file.getTensor(ResultsCommand.TOKENS_TENSOR);
            long offset = file.getDataOffset() + tokens.offset();
            try (FileChannel channel = FileChannel.open(results, StandardOpenOption.READ,
                    StandardOpenOption.WRITE)) {
                ByteBuffer changed = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
                changed.putInt(7).flip();
                channel.write(changed, offset);
            }
        }
        IOException e = assertThrows(IOException.class, () -> ResultsCommand.dispatch(new String[]{"-m",
                model.toString(), "-p", "The quick brown fox", "-o", results.toString(), "--check"}, config(),
                System.out, System.err));
        assertTrue(e.getMessage(), e.getMessage().contains("position 0"));
    }

    @Test
    public void aFileOfAnotherModelIsRefusedByName() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path other = SyntheticModels.llama().writeTempFile();
        Path results = tmp.newFolder().toPath().resolve("results.gguf");
        assertEquals(0, run(model, results));
        IOException e = assertThrows(IOException.class, () -> ResultsCommand.dispatch(new String[]{"-m",
                other.toString(), "-p", "The quick brown fox", "-o", results.toString(), "--check"}, config(),
                System.out, System.err));
        assertTrue(e.getMessage(), e.getMessage().contains("holds the results of"));
    }

    @Test
    public void anOutputFileIsRequired() {
        ResultsCommand.UsageException e = assertThrows(ResultsCommand.UsageException.class,
                () -> ResultsCommand.parse(new String[]{"-m", "model.gguf", "-p", "x"}));
        assertTrue(e.getMessage(), e.getMessage().contains("--output"));
    }

    @Test
    public void theNmseIsTheReferenceMetric() {
        assertEquals(0.0, ResultsCommand.nmse(new float[]{1, 2}, new float[]{1, 2}), 0.0);
        assertEquals(0.2, ResultsCommand.nmse(new float[]{1, 2}, new float[]{1, 3}), 1e-12);
        assertEquals(0.25, ResultsCommand.nmse(new float[]{2}, new float[]{1}), 1e-12);
    }

    private static int[] readInts(GGUFFileParser file, GGUFTensorInfo info) throws IOException {
        ByteBuffer bytes = ByteBuffer.wrap(file.readTensorData(info)).order(ByteOrder.LITTLE_ENDIAN);
        int[] values = new int[(int) info.numElements()];
        for (int i = 0; i < values.length; i++) {
            values[i] = bytes.getInt();
        }
        return values;
    }

    private static float[] readFloats(GGUFFileParser file, GGUFTensorInfo info) throws IOException {
        ByteBuffer bytes = ByteBuffer.wrap(file.readTensorData(info)).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[(int) info.numElements()];
        for (int i = 0; i < values.length; i++) {
            values[i] = bytes.getFloat();
        }
        return values;
    }
}
