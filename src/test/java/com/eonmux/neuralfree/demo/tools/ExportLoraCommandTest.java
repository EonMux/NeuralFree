package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The command line of {@code nftool export-lora} ({@link ExportLoraCommand}): the usage, the named refusals of
 * the flag surface (missing arguments, an unknown option, a merge without adapters, {@code --peft} beside
 * {@code --lora}, a non-numeric scale), an end-to-end merge of a GGUF base with a GGUF adapter file and the
 * end-to-end PEFT conversion ({@code --peft}) of a synthetic adapter directory over a synthetic Hugging Face
 * base, whose output is the adapter GGUF {@code convert_lora_to_gguf.py} writes.
 */
public class ExportLoraCommandTest {

    private static final int IN = 32;
    private static final int OUT = 32;
    private static final int RANK = 4;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ------------------------------------------------------------------ fixtures

    private Path baseFile() throws IOException {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.tensorF32("blk.0.attn_q.weight", new long[]{IN, OUT}, new float[IN * OUT]);
        return w.writeTempFile();
    }

    /** The A factor values: {@code A[k][i]} at {@code k*IN + i}. */
    private static float[] aValues() {
        float[] a = new float[RANK * IN];
        for (int j = 0; j < a.length; j++) {
            a[j] = 0.25f * (j / IN) + 0.01f * j;
        }
        return a;
    }

    /** The B factor values: {@code B[o][k]} at {@code o*RANK + k}. */
    private static float[] bValues() {
        float[] b = new float[OUT * RANK];
        for (int j = 0; j < b.length; j++) {
            b[j] = 0.5f - 0.05f * j;
        }
        return b;
    }

    private Path adapterFile() throws IOException {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.str("general.type", "adapter");
        w.str("adapter.type", "lora");
        w.f32("adapter.lora.alpha", 4.0f);
        w.tensorF32("blk.0.attn_q.weight.lora_a", new long[]{IN, RANK}, aValues());
        w.tensorF32("blk.0.attn_q.weight.lora_b", new long[]{RANK, OUT}, bValues());
        return w.writeTempFile();
    }

    // ------------------------------------------------------------------ the flag surface

    @Test
    public void helpPrintsTheUsageAndWritesNothing() throws Exception {
        assertNull(ExportLoraCommand.execute(new String[]{"--help"}));
    }

    @Test
    public void missingArgumentsAreNamed() {
        ExportLoraCommand.UsageException e = assertThrows(ExportLoraCommand.UsageException.class,
                () -> ExportLoraCommand.execute(new String[0]));
        assertTrue(e.getMessage(), e.getMessage().contains("needs a base model path"));

        e = assertThrows(ExportLoraCommand.UsageException.class,
                () -> ExportLoraCommand.execute(new String[]{"base.gguf"}));
        assertTrue(e.getMessage(), e.getMessage().contains("needs an output path"));
    }

    @Test
    public void anUnknownOptionIsNamed() {
        ExportLoraCommand.UsageException e = assertThrows(ExportLoraCommand.UsageException.class,
                () -> ExportLoraCommand.execute(new String[]{"--quant", "a.gguf", "b.gguf"}));
        assertTrue(e.getMessage(), e.getMessage().contains("--quant"));
    }

    @Test
    public void aMergeWithoutAdaptersIsAUsageError() throws Exception {
        Path base = baseFile();
        Path out = tmp.getRoot().toPath().resolve("out.gguf");
        ExportLoraCommand.UsageException e = assertThrows(ExportLoraCommand.UsageException.class,
                () -> ExportLoraCommand.execute(new String[]{base.toString(), out.toString()}));
        assertTrue(e.getMessage(), e.getMessage().contains("--lora"));
    }

    @Test
    public void peftTakesNoLoraFlags() throws Exception {
        Path base = baseFile();
        Path out = tmp.getRoot().toPath().resolve("out.gguf");
        ExportLoraCommand.UsageException e = assertThrows(ExportLoraCommand.UsageException.class,
                () -> ExportLoraCommand.execute(new String[]{base.toString(), out.toString(),
                        "--peft", "adapter", "--lora", "x.gguf"}));
        assertTrue(e.getMessage(), e.getMessage().contains("--peft"));
    }

    @Test
    public void aNonNumericOrNonFiniteScaleIsNamed() throws Exception {
        Path base = baseFile();
        Path out = tmp.getRoot().toPath().resolve("out.gguf");
        Path adapter = adapterFile();
        ExportLoraCommand.UsageException e = assertThrows(ExportLoraCommand.UsageException.class,
                () -> ExportLoraCommand.execute(new String[]{base.toString(), out.toString(),
                        "--lora-scaled", adapter.toString(), "abc"}));
        assertTrue(e.getMessage(), e.getMessage().contains("needs a number"));
        e = assertThrows(ExportLoraCommand.UsageException.class,
                () -> ExportLoraCommand.execute(new String[]{base.toString(), out.toString(),
                        "--lora-scaled", adapter.toString(), "NaN"}));
        assertTrue(e.getMessage(), e.getMessage().contains("finite"));
        e = assertThrows(ExportLoraCommand.UsageException.class,
                () -> ExportLoraCommand.execute(new String[]{base.toString(), out.toString(),
                        "--lora-scaled", adapter.toString()}));
        assertTrue(e.getMessage(), e.getMessage().contains("--lora-scaled needs a value"));
    }

    // ------------------------------------------------------------------ the merges

    @Test
    public void executeMergesAGgufBaseWithAnAdapter() throws Exception {
        Path base = baseFile();
        Path adapter = adapterFile();
        Path out = tmp.getRoot().toPath().resolve("merged.gguf");
        Path written = ExportLoraCommand.execute(new String[]{base.toString(), out.toString(),
                "--lora", adapter.toString()});
        assertEquals(out, written);
        try (GGUFFileParser m = new GGUFFileParser(out.toString())) {
            assertEquals(GGMLType.F32, m.getTensor("blk.0.attn_q.weight").type());
            assertEquals(1L, ((Number) m.getMetadata().get(GgufKeys.GENERAL_FILE_TYPE)).longValue());
            float[] q = m.readTensorDataAsFloat(m.getTensor("blk.0.attn_q.weight"));
            // alpha 4 of rank 4 at scale 1 folds to 1: the merged value of element (0, 0) is B A[0][0]
            float[] a = aValues();
            float[] b = bValues();
            double delta = 0;
            for (int k = 0; k < RANK; k++) {
                delta += (double) b[k] * a[k * IN];
            }
            assertEquals(delta, q[0], 1e-4);
        }
    }

    @Test
    public void executeConvertsAPeftDirectoryToAnAdapterGguf() throws Exception {
        Path base = SyntheticHf.llama(2, 64, 4, 2, 320, StDType.F32)
                .write(Files.createDirectory(tmp.getRoot().toPath().resolve("hfbase")));
        Path dir = Files.createDirectory(tmp.getRoot().toPath().resolve("peft"));
        SafetensorsWriter.Builder w = SafetensorsWriter.builder();
        String prefix = "base_model.model.";
        float[] a = new float[RANK * 64];
        for (int j = 0; j < a.length; j++) {
            a[j] = 0.1f * (j % RANK) + 0.02f * j;
        }
        float[] b = new float[64 * RANK];
        for (int j = 0; j < b.length; j++) {
            b[j] = 0.3f - 0.02f * j;
        }
        w.add(prefix + "model.layers.0.self_attn.q_proj.lora_A.weight", StDType.F32, new long[]{RANK, 64},
                SyntheticHf.f32(a));
        w.add(prefix + "model.layers.0.self_attn.q_proj.lora_B.weight", StDType.F32, new long[]{64, RANK},
                SyntheticHf.f32(b));
        w.metadata("format", "pt");
        w.build().write(dir.resolve("adapter_model.safetensors"));
        Files.write(dir.resolve("adapter_config.json"), SyntheticHf
                .json("{'peft_type':'lora','r':4,'lora_alpha':8}").getBytes(StandardCharsets.UTF_8));
        Path out = tmp.getRoot().toPath().resolve("adapter.gguf");
        Path written = ExportLoraCommand.execute(new String[]{base.toString(), out.toString(),
                "--peft", dir.toString()});
        assertEquals(out, written);
        try (GGUFFileParser m = new GGUFFileParser(out.toString())) {
            assertEquals("adapter", m.getString(GgufKeys.GENERAL_TYPE));
            assertEquals("lora", m.getString(GgufKeys.ADAPTER_TYPE));
            assertEquals(8.0f, ((Number) m.getMetadata().get(GgufKeys.ADAPTER_LORA_ALPHA)).floatValue(), 1e-6f);
            assertEquals(GGMLType.F32, m.getTensor("blk.0.attn_q.weight.lora_a").type());
            assertArrayEquals(new long[]{64, RANK}, m.getTensor("blk.0.attn_q.weight.lora_a").dims());
            assertEquals(GGMLType.F32, m.getTensor("blk.0.attn_q.weight.lora_b").type());
        }
    }
}
