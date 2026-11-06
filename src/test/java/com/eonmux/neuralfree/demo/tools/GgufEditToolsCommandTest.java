package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.edit.GgufEndianConverter;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool set-metadata}, {@code new-metadata} and {@code convert-endian}: the command lines, the in-place
 * value change, the metadata rewrite with its convenience flags and refusals, the dry runs, and the endianness
 * conversion of a real file.
 */
public class GgufEditToolsCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ================================================================== set-metadata

    @Test
    public void setMetadataParsesThreePositionalsAndDryRun() throws Exception {
        GgufSetMetadataCommand.Parsed p = GgufSetMetadataCommand.parse(
                new String[] {"--dry-run", "m.gguf", "tokenizer.ggml.bos_token_id", "2"});
        assertEquals("m.gguf", p.model);
        assertEquals("tokenizer.ggml.bos_token_id", p.key);
        assertEquals("2", p.value);
        assertTrue(p.dryRun);
    }

    @Test
    public void setMetadataRefusals() {
        assertSetUsage(new String[] {"m.gguf", "k"}, "needs a model, a metadata key and a value");
        assertSetUsage(new String[] {"--nope", "m.gguf", "k", "v"}, "unknown option");
        assertSetUsage(new String[] {"m.gguf", "k", "v", "extra"}, "too many arguments");
    }

    @Test
    public void setMetadataChangesTheValueInPlace() throws Exception {
        Path file = fixture().writeTempFile();
        long size = Files.size(file);
        StringBuilder out = new StringBuilder();
        GgufSetMetadataCommand.execute(GgufSetMetadataCommand.parse(
                new String[] {"--dry-run", file.toString(), "tokenizer.ggml.bos_token_id", "2"}), out);
        assertTrue(out.toString(), out.toString().contains("from 1 to 2"));
        assertTrue(out.toString(), out.toString().contains("Dry run"));
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            assertEquals(Integer.valueOf(1), src.getMetadata().get("tokenizer.ggml.bos_token_id"));
        }

        StringBuilder written = new StringBuilder();
        GgufSetMetadataCommand.execute(GgufSetMetadataCommand.parse(
                new String[] {file.toString(), "tokenizer.ggml.bos_token_id", "2"}), written);
        assertEquals(size, Files.size(file));
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            assertEquals(Integer.valueOf(2), src.getMetadata().get("tokenizer.ggml.bos_token_id"));
            assertArrayEquals(fixture().tensorData("t1"), src.readTensorData(src.getTensor("t1")));
        }
        assertTrue(written.toString(), written.toString().contains("Field changed"));
    }

    @Test
    public void setMetadataWithoutAChangeSaysSo() throws Exception {
        Path file = fixture().writeTempFile();
        StringBuilder out = new StringBuilder();
        GgufSetMetadataCommand.execute(GgufSetMetadataCommand.parse(
                new String[] {file.toString(), "tokenizer.ggml.bos_token_id", "1"}), out);
        assertTrue(out.toString(), out.toString().contains("already set to requested value 1"));
        assertFalse(out.toString(), out.toString().contains("Field changed"));
    }

    // ================================================================== new-metadata

    @Test
    public void newMetadataParsesEveryFlag() throws Exception {
        GgufNewMetadataCommand.Parsed p = GgufNewMetadataCommand.parse(new String[] {
                "--set", "general.name=str:new", "--remove", "general.description",
                "--general-name", "n", "--general-description", "d", "--pre-tokenizer", "llama3",
                "--chat-template", "{{ x }}", "--special-token", "eos", "</s>",
                "--special-token-by-id", "bos", "2", "--overwrite", "--dry-run", "in.gguf", "out.gguf"});
        assertEquals("in.gguf", p.input);
        assertEquals("out.gguf", p.output);
        assertEquals(4, p.overrides.size());
        assertTrue(p.remove.contains("general.description"));
        assertEquals(2, p.specialTokens.size());
        assertEquals("{{ x }}", p.chatTemplate);
        assertTrue(p.overwrite);
        assertTrue(p.dryRun);
    }

    @Test
    public void newMetadataRefusals() {
        assertNewUsage(new String[] {"in.gguf"}, "needs an input and an output");
        assertNewUsage(new String[] {"--set", "no-equals", "in.gguf", "out.gguf"}, "--set");
        assertNewUsage(new String[] {"in.gguf", "out.gguf", "--special-token"}, "--special-token needs a value");
        assertNewUsage(new String[] {"in.gguf", "out.gguf", "extra"}, "too many file arguments");
        assertNewUsage(new String[] {"--chat-template", "[{bad}]", "in.gguf", "out.gguf"}, "not valid JSON");
    }

    @Test
    public void newMetadataSetsRemovesAndAddsValues() throws Exception {
        Path in = fixture().writeTempFile();
        Path out = tmp.getRoot().toPath().resolve("out.gguf");
        StringBuilder printed = new StringBuilder();
        GgufNewMetadataCommand.execute(GgufNewMetadataCommand.parse(new String[] {
                "--set", "general.name=str:new name",
                "--set", "k.uint8=uint8:255",
                "--remove", "tokenizer.ggml.add_bos_token",
                "--general-description", "described",
                "--pre-tokenizer", "llama3",
                in.toString(), out.toString()}), printed);
        assertTrue(printed.toString(), printed.toString().contains("wrote " + out));

        try (GGUFFileParser src = ModelSources.parseGguf(out.toString())) {
            assertEquals("new name", src.getString("general.name"));
            assertEquals("described", src.getString("general.description"));
            assertEquals("llama3", src.getString("tokenizer.ggml.pre"));
            assertEquals(GGUFValueType.UINT8, src.getValueType("k.uint8"));
            assertEquals(Integer.valueOf(255), src.getMetadata().get("k.uint8"));
            assertNull(src.getMetadata().get("tokenizer.ggml.add_bos_token"));
            assertArrayEquals(fixture().tensorData("t1"), src.readTensorData(src.getTensor("t1")));
            assertEquals(32L, src.getAlignment());
        }
    }

    @Test
    public void newMetadataResolvesSpecialTokensByTextAndId() throws Exception {
        Path in = fixture().writeTempFile();
        Path out = tmp.getRoot().toPath().resolve("special.gguf");
        GgufNewMetadataCommand.execute(GgufNewMetadataCommand.parse(new String[] {
                "--special-token", "eos", "</s>", "--special-token-by-id", "bos", "2",
                in.toString(), out.toString()}), new StringBuilder());
        try (GGUFFileParser src = ModelSources.parseGguf(out.toString())) {
            assertEquals(Integer.valueOf(2), src.getMetadata().get("tokenizer.ggml.bos_token_id"));
            assertEquals(Integer.valueOf(3), src.getMetadata().get("tokenizer.ggml.eos_token_id"));
        }
    }

    @Test
    public void newMetadataReplacesTheWholeChatTemplateFamily() throws Exception {
        Path in = fixture().writeTempFile();
        Path out = tmp.getRoot().toPath().resolve("chat.gguf");
        String templates = "[{\"name\":\"default\",\"template\":\"D\"},"
                + "{\"name\":\"tool use\",\"template\":\"T\"}]";
        GgufNewMetadataCommand.execute(GgufNewMetadataCommand.parse(new String[] {
                "--chat-template", templates, in.toString(), out.toString()}), new StringBuilder());
        try (GGUFFileParser src = ModelSources.parseGguf(out.toString())) {
            assertEquals("D", src.getString("tokenizer.chat_template"));
            assertEquals("T", src.getString("tokenizer.chat_template.tool_use"));
            assertEquals(Arrays.asList("tool_use"), src.getMetadata().get("tokenizer.chat_templates"));
        }
    }

    @Test
    public void newMetadataReadsChatTemplatesFromFiles() throws Exception {
        Path in = fixture().writeTempFile();
        Path jinja = tmp.newFile("chat_template.jinja").toPath();
        Files.write(jinja, "FROM FILE".getBytes(StandardCharsets.UTF_8));
        Path fromFile = tmp.getRoot().toPath().resolve("jinja.gguf");
        GgufNewMetadataCommand.execute(GgufNewMetadataCommand.parse(new String[] {
                "--chat-template-file", jinja.toString(), in.toString(), fromFile.toString()}), new StringBuilder());
        try (GGUFFileParser src = ModelSources.parseGguf(fromFile.toString())) {
            assertEquals("FROM FILE", src.getString("tokenizer.chat_template"));
        }

        Path config = tmp.newFile("tokenizer_config.json").toPath();
        Files.write(config, "{\"chat_template\":\"FROM CONFIG\"}".getBytes(StandardCharsets.UTF_8));
        Path fromConfig = tmp.getRoot().toPath().resolve("config.gguf");
        GgufNewMetadataCommand.execute(GgufNewMetadataCommand.parse(new String[] {
                "--chat-template-config", config.toString(), in.toString(), fromConfig.toString()}),
                new StringBuilder());
        try (GGUFFileParser src = ModelSources.parseGguf(fromConfig.toString())) {
            assertEquals("FROM CONFIG", src.getString("tokenizer.chat_template"));
        }
    }

    @Test
    public void newMetadataReadsAJsonMetadataFile() throws Exception {
        Path in = fixture().writeTempFile();
        Path json = tmp.newFile("meta.json").toPath();
        Files.write(json, ("{\"general.description\":\"from json\",\"general.tags\":[\"a\",\"b\"],"
                + "\"k.count\":7}").getBytes(StandardCharsets.UTF_8));
        Path out = tmp.getRoot().toPath().resolve("json.gguf");
        GgufNewMetadataCommand.execute(GgufNewMetadataCommand.parse(new String[] {
                "--metadata", json.toString(), in.toString(), out.toString()}), new StringBuilder());
        try (GGUFFileParser src = ModelSources.parseGguf(out.toString())) {
            assertEquals("from json", src.getString("general.description"));
            assertEquals(Arrays.asList("a", "b"), src.getMetadata().get("general.tags"));
            assertEquals(Integer.valueOf(7), src.getMetadata().get("k.count"));
        }
    }

    @Test
    public void newMetadataDryRunWritesNothing() throws Exception {
        Path in = fixture().writeTempFile();
        Path out = tmp.getRoot().toPath().resolve("dry.gguf");
        StringBuilder printed = new StringBuilder();
        GgufNewMetadataCommand.execute(GgufNewMetadataCommand.parse(new String[] {
                "--dry-run", "--set", "general.name=str:x", in.toString(), out.toString()}), printed);
        assertFalse(Files.exists(out));
        assertTrue(printed.toString(), printed.toString().contains("dry run"));
    }

    @Test
    public void newMetadataRefusesUnknownAndUnresolvableSpecialTokens() throws Exception {
        Path in = fixture().writeTempFile();
        Path out = tmp.getRoot().toPath().resolve("x.gguf");
        IOException unknown = assertThrows(IOException.class, () -> GgufNewMetadataCommand.execute(
                GgufNewMetadataCommand.parse(new String[] {"--special-token", "nope", "x", in.toString(),
                        out.toString()}), new StringBuilder()));
        assertTrue(unknown.getMessage(), unknown.getMessage().contains("unknown special token name 'nope'"));

        IOException missing = assertThrows(IOException.class, () -> GgufNewMetadataCommand.execute(
                GgufNewMetadataCommand.parse(new String[] {"--special-token", "eos", "absent", in.toString(),
                        out.toString()}), new StringBuilder()));
        assertTrue(missing.getMessage(), missing.getMessage().contains("'absent'"));

        IOException range = assertThrows(IOException.class, () -> GgufNewMetadataCommand.execute(
                GgufNewMetadataCommand.parse(new String[] {"--special-token-by-id", "eos", "99", in.toString(),
                        out.toString()}), new StringBuilder()));
        assertTrue(range.getMessage(), range.getMessage().contains("outside"));
    }

    @Test
    public void newMetadataRefusesASplitModelAndANonGgufInput() throws Exception {
        Path in = fixture().writeTempFile();
        Path prefix = tmp.getRoot().toPath().resolve("split.gguf");
        try (GGUFFileParser src = ModelSources.parseGguf(in.toString())) {
            GgufWriter.write(src, prefix, new GgufWriter.Options(GgufWriter.Mode.NATIVE, false, 32, false, 1, 0));
        }
        Path first = tmp.getRoot().toPath().resolve("split-00001-of-00002.gguf");
        assertTrue(Files.exists(first));
        IOException split = assertThrows(IOException.class, () -> GgufNewMetadataCommand.execute(
                GgufNewMetadataCommand.parse(new String[] {first.toString(),
                        tmp.getRoot().toPath().resolve("s.gguf").toString()}), new StringBuilder()));
        assertTrue(split.getMessage(), split.getMessage().contains("split model"));

        Path safetensors = tmp.newFile("model.safetensors").toPath();
        IOException hf = assertThrows(IOException.class, () -> GgufNewMetadataCommand.execute(
                GgufNewMetadataCommand.parse(new String[] {safetensors.toString(),
                        tmp.getRoot().toPath().resolve("h.gguf").toString()}), new StringBuilder()));
        assertTrue(hf.getMessage(), hf.getMessage().contains("Hugging Face"));
    }

    // ================================================================== convert-endian

    @Test
    public void convertEndianParsesItsFlags() throws Exception {
        GgufConvertEndianCommand.Parsed p = GgufConvertEndianCommand.parse(
                new String[] {"--out", "o.gguf", "--overwrite", "m.gguf", "big"});
        assertEquals("o.gguf", p.out.toString());
        assertEquals("big", p.order);
        assertTrue(p.overwrite);
        assertTrue(assertThrows(GgufConvertEndianCommand.UsageException.class,
                () -> GgufConvertEndianCommand.parse(new String[] {"m.gguf", "middle"})).getMessage()
                .contains("not big, little or native"));
        assertTrue(assertThrows(GgufConvertEndianCommand.UsageException.class,
                () -> GgufConvertEndianCommand.parse(new String[] {"m.gguf"})).getMessage()
                .contains("needs a GGUF file and an order"));
    }

    @Test
    public void convertEndianConvertsInPlace() throws Exception {
        Path file = tmp.getRoot().toPath().resolve("model.gguf");
        Files.copy(fixture().writeTempFile(), file);
        byte[] original = Files.readAllBytes(file);
        StringBuilder out = new StringBuilder();
        GgufConvertEndianCommand.execute(GgufConvertEndianCommand.parse(
                new String[] {file.toString(), "big"}), out);
        assertTrue(out.toString(), out.toString().contains("file seems to be LITTLE endian"));
        assertTrue(out.toString(), out.toString().contains("Preparing to convert from LITTLE to BIG"));
        assertTrue(out.toString(), out.toString().contains("Completion: wrote " + file + " in BIG endian"));
        assertEquals(GgufEndianConverter.Endian.BIG, GgufEndianConverter.inspect(file).fileEndian());
        assertFalse(Arrays.equals(original, Files.readAllBytes(file)));
    }

    @Test
    public void convertEndianDryRunWritesNothing() throws Exception {
        Path file = tmp.getRoot().toPath().resolve("model.gguf");
        Files.copy(fixture().writeTempFile(), file);
        byte[] original = Files.readAllBytes(file);
        StringBuilder out = new StringBuilder();
        GgufConvertEndianCommand.execute(GgufConvertEndianCommand.parse(
                new String[] {"--dry-run", file.toString(), "big"}), out);
        assertTrue(out.toString(), out.toString().contains("dry run"));
        assertArrayEquals(original, Files.readAllBytes(file));
    }

    @Test
    public void convertEndianReportsAFileAlreadyInTheTargetOrder() throws Exception {
        Path file = fixture().writeTempFile();
        StringBuilder out = new StringBuilder();
        GgufConvertEndianCommand.execute(GgufConvertEndianCommand.parse(
                new String[] {file.toString(), "little"}), out);
        assertTrue(out.toString(), out.toString().contains("already LITTLE endian. Nothing to do."));
    }

    // ================================================================== helpers

    private static void assertSetUsage(String[] args, String message) {
        GgufSetMetadataCommand.UsageException e = assertThrows(GgufSetMetadataCommand.UsageException.class,
                () -> GgufSetMetadataCommand.parse(args));
        assertTrue(e.getMessage(), e.getMessage().contains(message));
    }

    private static void assertNewUsage(String[] args, String message) {
        GgufNewMetadataCommand.UsageException e = assertThrows(GgufNewMetadataCommand.UsageException.class,
                () -> GgufNewMetadataCommand.parse(args));
        assertTrue(e.getMessage(), e.getMessage().contains(message));
    }

    private static GgufTestWriter fixture() {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("synthetic");
        w.u32("general.alignment", 32);
        w.str("general.name", "original");
        w.u32("tokenizer.ggml.bos_token_id", 1);
        w.u32("tokenizer.ggml.eos_token_id", 2);
        w.bool("tokenizer.ggml.add_bos_token", true);
        w.str("tokenizer.chat_template", "OLD");
        w.strArray("tokenizer.ggml.tokens", "<unk>", "<s>", "hello", "</s>");
        w.tensorF32("t1", new long[]{4}, 1.0f, 2.0f, 3.0f, 4.0f);
        w.tensorF32("t2", new long[]{2, 2}, 5.0f, 6.0f, 7.0f, 8.0f);
        return w;
    }
}
