package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.function.ThrowingRunnable;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Format detection and opening in {@link ModelSources} on real files: every rule of the detection order, the
 * normative messages, that GGUF models open as {@link GGUFFileParser}, and that Hugging Face checkpoints are handed
 * to the safetensors loader (whose own behaviour HfModelSourceTest covers).
 */
public class ModelSourcesTest {

    private static final ModelSourceOptions OPTIONS = ModelSourceOptions.defaults();
    private static final SourceOrigin BASE = new SourceOrigin(Paths.get("base.gguf"), ModelFormat.GGUF);

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static GgufTestWriter model() {
        return new GgufTestWriter()
                .architecture("llama")
                .tensorF32("output_norm.weight", new long[]{4}, 1f, 2f, 3f, 4f);
    }

    private static Path touch(Path file, String content) throws IOException {
        Files.write(file, content.getBytes(StandardCharsets.UTF_8));
        return file;
    }

    private Path hfDir(String configName) throws IOException {
        Path dir = tmp.newFolder().toPath();
        touch(dir.resolve("model.safetensors"), "not read by detection");
        touch(dir.resolve(configName), "{}");
        return dir;
    }

    private Path peftDir() throws IOException {
        Path dir = tmp.newFolder().toPath();
        touch(dir.resolve("adapter_config.json"), "{}");
        touch(dir.resolve("adapter_model.safetensors"), "not read by detection");
        return dir;
    }

    private static void assertMessageContains(Throwable e, String expected) {
        assertTrue(e.getMessage(), e.getMessage().contains(expected));
    }

    /**
     * Opens {@code path} through the Hugging Face loader, which rejects the placeholder files of these tests as an
     * invalid checkpoint (an unreadable safetensors header or a missing configuration) rather than refusing the format.
     */
    private static IOException assertRejectedAsInvalidCheckpoint(ThrowingRunnable open) {
        IOException e = assertThrows(IOException.class, open);
        assertTrue(e.getMessage(), e.getMessage().startsWith("Invalid "));
        return e;
    }

    // ------------------------------------------------------------------ GGUF

    @Test
    public void fileWithGgufMagicOpensAsGguf() throws IOException {
        Path file = tmp.getRoot().toPath().resolve("weights.bin");
        model().writeTo(file);
        assertEquals(ModelFormat.GGUF, ModelSources.detect(file));
        try (ModelSource source = ModelSources.open(file.toString(), OPTIONS)) {
            assertTrue(source instanceof GGUFFileParser);
            assertEquals(ModelFormat.GGUF, source.format());
            assertEquals(file.toString(), source.getFilePath());
            assertEquals("llama", source.getString("general.architecture"));
            assertEquals(1, source.getTensorCount());
        }
    }

    @Test
    public void ggufNamedFileWithoutMagicKeepsTheParserMessage() throws IOException {
        Path file = touch(tmp.getRoot().toPath().resolve("broken.gguf"), "GGML....");
        assertEquals(ModelFormat.GGUF, ModelSources.detect(file));
        IOException e = assertThrows(IOException.class, () -> ModelSources.open(file.toString(), OPTIONS));
        assertMessageContains(e, "bad magic");
    }

    @Test
    public void directoryWithOneGgufOpensThatFile() throws IOException {
        Path dir = tmp.newFolder().toPath();
        model().writeTo(dir.resolve("model.gguf"));
        touch(dir.resolve("README.md"), "card");
        assertEquals(ModelFormat.GGUF, ModelSources.detect(dir));
        try (ModelSource source = ModelSources.open(dir.toString(), OPTIONS)) {
            assertEquals(dir.resolve("model.gguf").toString(), source.getFilePath());
        }
    }

    @Test
    public void directoryWithOneSplitSetOpensItsFirstSplit() throws IOException {
        File dir = tmp.newFolder();
        for (int no = 1; no >= 0; no--) {
            GGUFFileParserSplitTest.split(no).writeTo(Paths.get(GGUFFileParserSplitTest.splitName(dir, no)));
        }
        try (ModelSource source = ModelSources.open(dir.getPath(), OPTIONS)) {
            assertEquals(GGUFFileParserSplitTest.splitName(dir, 0), source.getFilePath());
            assertEquals(2, ((GGUFFileParser) source).getShardCount());
            assertEquals(2, source.getTensorCount());
            assertEquals(3, source.getTensors().size());
        }
    }

    @Test
    public void directoryWithUnrelatedGgufsIsRefused() throws IOException {
        Path dir = tmp.newFolder().toPath();
        model().writeTo(dir.resolve("a.gguf"));
        model().writeTo(dir.resolve("b.gguf"));
        FileNotFoundException e = assertThrows(FileNotFoundException.class, () -> ModelSources.detect(dir));
        assertMessageContains(e, "holds 2 GGUF files that are not one split set (a.gguf, b.gguf); pass one of them");
    }

    @Test
    public void splitSetWithoutItsFirstSplitIsRefused() throws IOException {
        File dir = tmp.newFolder();
        GGUFFileParserSplitTest.split(1).writeTo(Paths.get(GGUFFileParserSplitTest.splitName(dir, 1)));
        model().writeTo(new File(dir, "model-00003-of-00002.gguf").toPath());
        assertThrows(FileNotFoundException.class, () -> ModelSources.detect(dir.toPath()));
    }

    // ------------------------------------------------------------------ safetensors

    @Test
    public void safetensorsFileIsAHuggingFaceCheckpoint() throws IOException {
        Path file = touch(tmp.getRoot().toPath().resolve("model.safetensors"), "header");
        assertEquals(ModelFormat.SAFETENSORS, ModelSources.detect(file));
        assertMessageContains(assertRejectedAsInvalidCheckpoint(() -> ModelSources.open(file.toString(), OPTIONS)),
                file.getFileName().toString());
    }

    @Test
    public void safetensorsShardListedByAnIndexIsRefusedWithItsDirectory() throws IOException {
        Path dir = tmp.newFolder().toPath();
        touch(dir.resolve("config.json"), "{}");
        Path shard = touch(dir.resolve("model-00001-of-00002.safetensors"), "header");
        touch(dir.resolve("model-00002-of-00002.safetensors"), "header");
        touch(dir.resolve("model.safetensors.index.json"), "{\"metadata\": {\"total_size\": 8}, \"weight_map\": {"
                + "\"a.weight\": \"model\\u002d00001-of-00002.safetensors\","
                + " \"b.weight\": \"model-00002-of-00002.safetensors\"}}");
        assertEquals(ModelFormat.SAFETENSORS, ModelSources.detect(shard));
        IOException e = assertThrows(IOException.class, () -> ModelSources.open(shard.toString(), OPTIONS));
        assertEquals("'" + shard + "' is one shard of a sharded checkpoint; pass the directory '"
                + dir.toAbsolutePath() + "'", e.getMessage());
    }

    @Test
    public void safetensorsFileNotListedByTheIndexOpensAlone() throws IOException {
        Path dir = tmp.newFolder().toPath();
        Path extra = touch(dir.resolve("consolidated.safetensors"), "header");
        touch(dir.resolve("model.safetensors.index.json"),
                "{\"weight_map\": {\"a.weight\": \"model-00001-of-00001.safetensors\"}}");
        IOException e = assertRejectedAsInvalidCheckpoint(() -> ModelSources.open(extra.toString(), OPTIONS));
        assertFalse(e.getMessage(), e.getMessage().contains("is one shard"));
        assertMessageContains(e, "config.json");
    }

    @Test
    public void malformedIndexIsReported() throws IOException {
        Path dir = tmp.newFolder().toPath();
        Path file = touch(dir.resolve("model.safetensors"), "header");
        Path index = touch(dir.resolve("model.safetensors.index.json"), "{\"weight_map\": {\"a\": \"bad\\q\"}}");
        IOException e = assertThrows(IOException.class, () -> ModelSources.open(file.toString(), OPTIONS));
        assertEquals("Invalid safetensors file '" + index + "': a string holds the invalid escape \\q",
                e.getMessage());
    }

    @Test
    public void directoryWithConfigJsonIsAHuggingFaceCheckpoint() throws IOException {
        Path dir = hfDir("config.json");
        assertEquals(ModelFormat.SAFETENSORS, ModelSources.detect(dir));
        assertMessageContains(assertRejectedAsInvalidCheckpoint(() -> ModelSources.open(dir.toString(), OPTIONS)),
                "model.safetensors");
    }

    @Test
    public void directoryWithParamsJsonIsAMistralNativeCheckpoint() throws IOException {
        Path dir = hfDir("params.json");
        assertEquals(ModelFormat.SAFETENSORS, ModelSources.detect(dir));
        assertRejectedAsInvalidCheckpoint(() -> ModelSources.open(dir.toString(), OPTIONS));
    }

    @Test
    public void safetensorsWinOverGgufInOneDirectory() throws IOException {
        Path dir = hfDir("config.json");
        model().writeTo(dir.resolve("model.gguf"));
        assertEquals(ModelFormat.SAFETENSORS, ModelSources.detect(dir));
    }

    @Test
    public void safetensorsWithoutConfigNameWhatWasLookedFor() throws IOException {
        Path dir = tmp.newFolder().toPath();
        touch(dir.resolve("model.safetensors"), "header");
        FileNotFoundException e = assertThrows(FileNotFoundException.class, () -> ModelSources.detect(dir));
        assertMessageContains(e, "config.json");
        assertMessageContains(e, "params.json");
    }

    @Test
    public void projectorViewOpensThroughTheSafetensorsLoader() throws IOException {
        Path dir = hfDir("config.json");
        assertRejectedAsInvalidCheckpoint(() -> ModelSources.openProjector(dir, OPTIONS));
        Path gguf = tmp.newFolder().toPath();
        model().writeTo(gguf.resolve("model.gguf"));
        IOException e = assertThrows(IOException.class, () -> ModelSources.openProjector(gguf, OPTIONS));
        assertMessageContains(e, "is not a Hugging Face checkpoint directory");
    }

    // ------------------------------------------------------------------ adapters

    @Test
    public void peftDirectoryIsRefusedAsAModel() throws IOException {
        Path dir = peftDir();
        assertEquals(ModelFormat.SAFETENSORS, ModelSources.detect(dir));
        IOException e = assertThrows(IOException.class, () -> ModelSources.open(dir.toString(), OPTIONS));
        assertFalse(e instanceof FileNotFoundException);
        assertEquals("'" + dir + "' is a PEFT adapter; load it with --lora on its base model", e.getMessage());
    }

    @Test
    public void peftDirectoryOpensAsAnAdapterOnlyWithTheSafetensorsLoader() throws IOException {
        Path dir = peftDir();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> ModelSources.openAdapter(dir.toString(), BASE, OPTIONS));
        assertEquals(ModelSources.PEFT_UNSUPPORTED, e.getMessage());
    }

    @Test
    public void peftDirectoryWithABaseConfigIsStillAnAdapter() throws IOException {
        Path dir = peftDir();
        touch(dir.resolve("config.json"), "{}");
        IOException e = assertThrows(IOException.class, () -> ModelSources.open(dir.toString(), OPTIONS));
        assertEquals("'" + dir + "' is a PEFT adapter; load it with --lora on its base model", e.getMessage());
    }

    @Test
    public void checkpointBesideAnAdapterIsTheModel() throws IOException {
        Path dir = hfDir("config.json");
        touch(dir.resolve("adapter_config.json"), "{}");
        touch(dir.resolve("adapter_model.safetensors"), "not read by detection");
        assertEquals(ModelFormat.SAFETENSORS, ModelSources.detect(dir));
        IOException model = assertRejectedAsInvalidCheckpoint(() -> ModelSources.open(dir.toString(), OPTIONS));
        assertFalse(model.getMessage(), model.getMessage().contains("PEFT adapter"));
        UnsupportedOperationException adapter = assertThrows(UnsupportedOperationException.class,
                () -> ModelSources.openAdapter(dir.toString(), BASE, OPTIONS));
        assertEquals(ModelSources.PEFT_UNSUPPORTED, adapter.getMessage());
    }

    @Test
    public void ggufAdapterOpensAsGguf() throws IOException {
        Path file = tmp.getRoot().toPath().resolve("adapter.gguf");
        model().str("general.type", "adapter").writeTo(file);
        try (ModelSource source = ModelSources.openAdapter(file.toString(), BASE, OPTIONS)) {
            assertSame(GGUFFileParser.class, source.getClass());
            assertEquals("adapter", source.getString("general.type"));
        }
    }

    @Test
    public void modelDirectoryIsNotAnAdapter() throws IOException {
        Path dir = hfDir("config.json");
        FileNotFoundException e = assertThrows(FileNotFoundException.class,
                () -> ModelSources.openAdapter(dir.toString(), BASE, OPTIONS));
        assertMessageContains(e, "is not a LoRA adapter");
    }

    // ------------------------------------------------------------------ refused and missing

    @Test
    public void pickleOnlyDirectoryIsRefused() throws IOException {
        Path dir = tmp.newFolder().toPath();
        touch(dir.resolve("pytorch_model.bin"), "pickle");
        touch(dir.resolve("config.json"), "{}");
        IOException e = assertThrows(IOException.class, () -> ModelSources.detect(dir));
        assertFalse(e instanceof FileNotFoundException);
        assertEquals("Invalid checkpoint '" + dir + "': PyTorch pickle checkpoints are not supported; use the"
                + " model's safetensors files", e.getMessage());
    }

    @Test
    public void pickleFilesAreRefused() throws IOException {
        for (String name : new String[]{"pytorch_model-00001-of-00002.bin", "consolidated.00.pth", "model.pt"}) {
            Path file = touch(tmp.getRoot().toPath().resolve(name), "pickle");
            IOException e = assertThrows(IOException.class, () -> ModelSources.open(file.toString(), OPTIONS));
            assertFalse(name, e instanceof FileNotFoundException);
            assertMessageContains(e, "PyTorch pickle checkpoints are not supported");
        }
    }

    @Test
    public void emptyDirectoryNamesWhatWasLookedFor() throws IOException {
        Path dir = tmp.newFolder().toPath();
        FileNotFoundException e = assertThrows(FileNotFoundException.class, () -> ModelSources.detect(dir));
        assertMessageContains(e, dir.toString());
        assertMessageContains(e, "GGUF");
        assertMessageContains(e, "*.safetensors with config.json or params.json");
        assertMessageContains(e, "adapter_config.json");
    }

    @Test
    public void unknownFileNamesWhatWasLookedFor() throws IOException {
        Path file = touch(tmp.getRoot().toPath().resolve("notes.txt"), "text");
        FileNotFoundException e = assertThrows(FileNotFoundException.class,
                () -> ModelSources.open(file.toString(), OPTIONS));
        assertMessageContains(e, "GGUF magic");
    }

    @Test
    public void missingPathIsNotFound() {
        String missing = new File(tmp.getRoot(), "absent.gguf").getPath();
        FileNotFoundException e = assertThrows(FileNotFoundException.class,
                () -> ModelSources.open(missing, OPTIONS));
        assertMessageContains(e, "does not exist");
    }

    @Test
    public void requireOpenableAcceptsFilesAndDirectories() throws IOException {
        Path file = touch(tmp.getRoot().toPath().resolve("any.bin"), "x");
        ModelSources.requireOpenable(file.toString());
        ModelSources.requireOpenable(tmp.newFolder().getPath());
        FileNotFoundException e = assertThrows(FileNotFoundException.class,
                () -> ModelSources.requireOpenable(new File(tmp.getRoot(), "absent").getPath()));
        assertMessageContains(e, "model not found or not readable");
    }
}
