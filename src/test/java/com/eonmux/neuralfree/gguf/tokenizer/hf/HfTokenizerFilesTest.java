package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfTokenizerFiles}: locating every tokenizer file, the Hugging Face cache snapshot layout (symbolic links),
 * the content-name rule, the {@code tokenizerDir} override as a directory and as a GGUF file, and the typed reads
 * of the located files, including the RWKV ({@link RwkvVocabReader}) and PLaMo ({@link PlamoVocabReader}) formats.
 */
public class HfTokenizerFilesTest {

    private static final String TINY_TOKENIZER =
            "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1},\"merges\":[\"a b\"]}}";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static Path write(Path file, String content) throws IOException {
        Files.createDirectories(file.getParent());
        Files.write(file, content.getBytes(StandardCharsets.UTF_8));
        return file;
    }

    private static void assertMessageContains(Throwable e, String expected) {
        assertTrue(e.getMessage(), e.getMessage().contains(expected));
    }

    private Path dirWith(String... names) throws IOException {
        Path dir = tmp.newFolder().toPath();
        for (String name : names) {
            write(dir.resolve(name), "x");
        }
        return dir;
    }

    // ------------------------------------------------------------------ locating

    @Test
    public void locateFindsEveryTokenizerFile() throws IOException {
        Path dir = dirWith("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "vocab.json", "merges.txt", "vocab.txt",
                "qwen.tiktoken", "tiktoken.model", "other.tiktoken", "tekken.json", "rwkv_vocab_v20230424.txt",
                "tokenizer.jsonl", "chat_template.jinja", "chat_template.json", "processor_config.json",
                "generation_config.json", "config.json");
        write(dir.resolve("additional_chat_templates/tool_use.jinja"), "t");
        write(dir.resolve("additional_chat_templates/rag.jinja"), "r");
        write(dir.resolve("additional_chat_templates/notes.txt"), "n");
        HfTokenizerFiles f = HfTokenizerFiles.locate(dir);
        assertEquals(dir, f.dir());
        assertFalse(f.isEmpty());
        assertEquals(dir.resolve("tokenizer.json"), f.tokenizerJson());
        assertEquals(dir.resolve("tokenizer_config.json"), f.tokenizerConfigJson());
        assertEquals(dir.resolve("special_tokens_map.json"), f.specialTokensMapJson());
        assertEquals(dir.resolve("added_tokens.json"), f.addedTokensJson());
        assertEquals(dir.resolve("tokenizer.model"), f.sentencePieceModel());
        assertEquals(dir.resolve("sentencepiece.bpe.model"), f.sentencePieceBpeModel());
        assertEquals(dir.resolve("vocab.json"), f.vocabJson());
        assertEquals(dir.resolve("merges.txt"), f.mergesTxt());
        assertEquals(dir.resolve("vocab.txt"), f.vocabTxt());
        assertEquals(Arrays.asList(dir.resolve("other.tiktoken"), dir.resolve("qwen.tiktoken"),
                dir.resolve("tiktoken.model")), f.tiktokenFiles());
        assertEquals(dir.resolve("qwen.tiktoken"), f.tiktokenFile());
        assertEquals(dir.resolve("tekken.json"), f.tekkenJson());
        assertEquals(dir.resolve("tekken.json"), f.mistralTokenizerFile());
        assertEquals(dir.resolve("rwkv_vocab_v20230424.txt"), f.rwkvVocabFile());
        assertEquals(dir.resolve("tokenizer.jsonl"), f.plamoJsonl());
        assertEquals(dir.resolve("chat_template.jinja"), f.chatTemplateJinja());
        assertEquals(dir.resolve("chat_template.json"), f.chatTemplateJson());
        assertEquals(dir.resolve("processor_config.json"), f.processorConfigJson());
        assertEquals(dir.resolve("generation_config.json"), f.generationConfigJson());
        assertEquals(dir.resolve("config.json"), f.file("config.json"));
        assertEquals(Arrays.asList(dir.resolve("additional_chat_templates/rag.jinja"),
                dir.resolve("additional_chat_templates/tool_use.jinja")), f.additionalChatTemplates());
        Map<String, String> named = f.readAdditionalChatTemplates();
        assertEquals(Arrays.asList("rag", "tool_use"), Arrays.asList(named.keySet().toArray()));
        assertEquals("t", named.get("tool_use"));
        assertFalse(f.hasGgufTokenizer());
    }

    @Test
    public void fallbacksFollowTheConverterOrder() throws IOException {
        HfTokenizerFiles f = HfTokenizerFiles.locate(dirWith("spiece.model", "tokenizer.model.v3",
                "tokenizer.model.v7", "rwkv_vocab_v20240101.txt", "tiktoken.model"));
        assertEquals("spiece.model", f.sentencePieceModel().getFileName().toString());
        assertEquals("tokenizer.model.v7", f.mistralTokenizerFile().getFileName().toString());
        assertEquals("rwkv_vocab_v20240101.txt", f.rwkvVocabFile().getFileName().toString());
        assertEquals("tiktoken.model", f.tiktokenFile().getFileName().toString());
        HfTokenizerFiles none = HfTokenizerFiles.locate(dirWith("config.json"));
        assertNull(none.tokenizerJson());
        assertNull(none.sentencePieceModel());
        assertNull(none.tiktokenFile());
        assertNull(none.mistralTokenizerFile());
        assertNull(none.rwkvVocabFile());
        assertNull(none.readTokenizerJson());
        assertNull(none.readChatTemplate());
        assertTrue(none.readTokenizerConfig().isEmpty());
        assertTrue(none.readMergesTxt().isEmpty());
        assertTrue(none.readAddedTokens().isEmpty());
        HfTokenizerFiles xlmr = HfTokenizerFiles.locate(dirWith("sentencepiece.bpe.model"));
        assertNull("a dotted stem is not a Mistral tokenizer file", xlmr.mistralTokenizerFile());
        assertEquals("sentencepiece.bpe.model", xlmr.sentencePieceBpeModel().getFileName().toString());
    }

    @Test
    public void locateRejectsAMissingDirectory() {
        Path missing = tmp.getRoot().toPath().resolve("missing");
        assertThrows(FileNotFoundException.class, () -> HfTokenizerFiles.locate(missing));
    }

    @Test
    public void emptyHasNoFiles() throws IOException {
        HfTokenizerFiles f = HfTokenizerFiles.empty();
        assertTrue(f.isEmpty());
        assertNull(f.dir());
        assertNull(f.tokenizerJson());
        assertTrue(f.tiktokenFiles().isEmpty());
        assertTrue(f.additionalChatTemplates().isEmpty());
        assertTrue(f.ggufTokenizerKeys().isEmpty());
        assertTrue(f.readGenerationConfig().isEmpty());
        assertNull(f.configuredFile("vocab_file"));
    }

    @Test
    public void aSymlinkedSnapshotLayoutIsFollowed() throws IOException {
        Path repo = tmp.newFolder("models--org--name").toPath();
        Path blobs = repo.resolve("blobs");
        Path snapshot = repo.resolve("snapshots").resolve("0123abcd");
        Files.createDirectories(blobs);
        Files.createDirectories(snapshot.resolve("additional_chat_templates"));
        write(blobs.resolve("1111"), TINY_TOKENIZER);
        write(blobs.resolve("2222"), "{\"bos_token\":\"<s>\"}");
        write(blobs.resolve("3333"), "{{ messages }}");
        try {
            Files.createSymbolicLink(snapshot.resolve("tokenizer.json"), Paths.get("../../blobs/1111"));
            Files.createSymbolicLink(snapshot.resolve("tokenizer_config.json"), Paths.get("../../blobs/2222"));
            Files.createSymbolicLink(snapshot.resolve("additional_chat_templates").resolve("tool_use.jinja"),
                    Paths.get("../../../blobs/3333"));
        } catch (UnsupportedOperationException | IOException e) {
            Assume.assumeNoException("symbolic links are not supported here", e);
        }
        Files.createSymbolicLink(snapshot.resolve("dangling.json"), Paths.get("../../blobs/missing"));

        HfTokenizerFiles f = HfTokenizerFiles.locate(snapshot);
        assertEquals(snapshot.resolve("tokenizer.json"), f.tokenizerJson());
        assertNull("a dangling link is not a file", f.file("dangling.json"));
        assertArrayEquals(new String[]{"a b"}, f.readTokenizerJson().model().merges());
        assertEquals("<s>", f.readTokenizerConfig().str("bos_token"));
        assertEquals(Collections.singletonMap("tool_use", "{{ messages }}"), f.readAdditionalChatTemplates());
    }

    // ------------------------------------------------------------------ content names

    @Test
    public void contentNamesMustStayInsideTheModelDirectory() throws IOException {
        Path dir = tmp.newFolder().toPath();
        String where = "tokenizer_config.json:vocab_file";
        assertEquals(dir.resolve("sub").resolve("qwen.tiktoken"),
                HfTokenizerFiles.resolveContentName(dir, "sub/qwen.tiktoken", where));
        assertEquals(dir.resolve("qwen.tiktoken"), HfTokenizerFiles.resolveContentName(dir, "./qwen.tiktoken", where));
        for (String bad : new String[]{"", "../outside.tiktoken", "sub/../../outside", "sub\\..\\..\\x",
                "/etc/passwd", "\\\\server\\share", "C:\\x", "a\0b"}) {
            IOException e = assertThrows(bad, IOException.class,
                    () -> HfTokenizerFiles.resolveContentName(dir, bad, where));
            assertTrue(e.getMessage(), e.getMessage().startsWith("Invalid Hugging Face checkpoint '" + dir + "': "
                    + where + ": file name '"));
        }
    }

    @Test
    public void configuredFilesUseTheContentNameRule() throws IOException {
        Path dir = tmp.newFolder().toPath();
        write(dir.resolve("sub/qwen.tiktoken"), "x");
        Path config = write(dir.resolve("tokenizer_config.json"), "{\"vocab_file\":\"sub/qwen.tiktoken\","
                + "\"merges_file\":null,\"bad\":\"../x\",\"absent\":\"missing.txt\",\"number\":3}");
        HfTokenizerFiles f = HfTokenizerFiles.locate(dir);
        assertEquals(dir.resolve("sub/qwen.tiktoken"), f.configuredFile("vocab_file"));
        assertNull(f.configuredFile("merges_file"));
        assertNull(f.configuredFile("tokenizer_file"));
        assertMessageContains(assertThrows(IOException.class, () -> f.configuredFile("bad")),
                "contains a '..' segment");
        assertMessageContains(assertThrows(IOException.class, () -> f.configuredFile("absent")),
                "not a file of the checkpoint");
        IOException number = assertThrows(IOException.class, () -> f.configuredFile("number"));
        assertMessageContains(number, "Invalid Hugging Face checkpoint '" + dir + "': " + config);
    }

    // ------------------------------------------------------------------ tokenizerDir

    @Test
    public void aTokenizerDirectoryReplacesTheModelFiles() throws IOException {
        Path model = dirWith("tokenizer.json", "chat_template.jinja");
        Path other = tmp.newFolder().toPath();
        write(other.resolve("tokenizer.json"), TINY_TOKENIZER);
        HfTokenizerFiles f = HfTokenizerFiles.locate(model).withTokenizerDir(other);
        assertEquals(other, f.dir());
        assertEquals(other.resolve("tokenizer.json"), f.tokenizerJson());
        assertNull(f.chatTemplateJinja());
        assertFalse(f.hasGgufTokenizer());
        HfTokenizerFiles same = HfTokenizerFiles.locate(model);
        assertTrue(same.withTokenizerDir(null) == same);
    }

    @Test
    public void aTokenizerDirectoryKeepsTheModelsGenerationAndProcessorConfigs() throws IOException {
        Path model = dirWith("tokenizer.json", "generation_config.json", "processor_config.json");
        Path other = dirWith("tokenizer.json", "generation_config.json");
        HfTokenizerFiles f = HfTokenizerFiles.locate(model).withTokenizerDir(other);
        assertEquals(other.resolve("tokenizer.json"), f.tokenizerJson());
        assertEquals(model.resolve("generation_config.json"), f.generationConfigJson());
        assertEquals(model.resolve("processor_config.json"), f.processorConfigJson());
        HfTokenizerFiles bare = HfTokenizerFiles.locate(dirWith("tokenizer.json")).withTokenizerDir(other);
        assertNull("a tokenizer directory supplies no generation config", bare.generationConfigJson());
    }

    @Test
    public void aGgufTokenizerSourceSuppliesItsTokenizerKeys() throws IOException {
        Path model = dirWith("tokenizer.json");
        Path gguf = tmp.getRoot().toPath().resolve("vocab.gguf");
        new GgufTestWriter()
                .architecture("llama")
                .str("tokenizer.ggml.model", "gpt2")
                .strArray("tokenizer.ggml.tokens", "a", "b", "ab")
                .i32Array("tokenizer.ggml.token_type", 1, 1, 1)
                .u32("tokenizer.ggml.bos_token_id", 0)
                .str("general.name", "vocab")
                .tensorF32("output_norm.weight", new long[]{4}, 1f, 2f, 3f, 4f)
                .writeTo(gguf);
        HfTokenizerFiles f = HfTokenizerFiles.locate(model).withTokenizerDir(gguf);
        assertTrue(f.hasGgufTokenizer());
        assertEquals(gguf, f.ggufTokenizerFile());
        assertEquals("the model directory files stay located", model.resolve("tokenizer.json"), f.tokenizerJson());
        Map<String, Object> keys = f.ggufTokenizerKeys();
        assertEquals(Arrays.asList("tokenizer.ggml.bos_token_id", "tokenizer.ggml.model", "tokenizer.ggml.token_type",
                "tokenizer.ggml.tokens"), Arrays.asList(keys.keySet().toArray()));
        assertEquals("gpt2", keys.get("tokenizer.ggml.model"));
        assertEquals(Arrays.asList("a", "b", "ab"), keys.get("tokenizer.ggml.tokens"));
        assertEquals(GGUFValueType.STRING, f.ggufValueType("tokenizer.ggml.model"));
        assertEquals(GGUFValueType.ARRAY, f.ggufValueType("tokenizer.ggml.tokens"));
        assertEquals(GGUFValueType.STRING, f.ggufArrayElementType("tokenizer.ggml.tokens"));
        assertEquals(GGUFValueType.INT32, f.ggufArrayElementType("tokenizer.ggml.token_type"));
        assertEquals(GGUFValueType.UINT32, f.ggufValueType("tokenizer.ggml.bos_token_id"));
        assertNull(f.ggufArrayElementType("tokenizer.ggml.model"));
        assertThrows(UnsupportedOperationException.class, () -> keys.remove("tokenizer.ggml.model"));
        @SuppressWarnings("unchecked")
        List<Object> tokens = (List<Object>) keys.get("tokenizer.ggml.tokens");
        assertThrows(UnsupportedOperationException.class, () -> tokens.add("c"));
    }

    @Test
    public void otherTokenizerSourcesAreRejected() throws IOException {
        Path model = dirWith("tokenizer.json");
        HfTokenizerFiles f = HfTokenizerFiles.locate(model);
        Path noTokenizer = tmp.getRoot().toPath().resolve("weights.gguf");
        new GgufTestWriter().architecture("llama").tensorF32("output_norm.weight", new long[]{2}, 1f, 2f)
                .writeTo(noTokenizer);
        assertMessageContains(assertThrows(IOException.class, () -> f.withTokenizerDir(noTokenizer)),
                "holds no tokenizer.* metadata");
        Path text = write(tmp.getRoot().toPath().resolve("notes.txt"), "hello");
        assertThrows(IOException.class, () -> f.withTokenizerDir(text));
        Path safetensors = write(tmp.getRoot().toPath().resolve("model.safetensors"), "{}");
        assertMessageContains(assertThrows(IOException.class, () -> f.withTokenizerDir(safetensors)),
                "SAFETENSORS");
        assertThrows(FileNotFoundException.class, () -> f.withTokenizerDir(tmp.getRoot().toPath().resolve("nope")));
    }

    // ------------------------------------------------------------------ typed reads

    @Test
    public void textFilesReadAsPythonTextMode() throws IOException {
        Path dir = tmp.newFolder().toPath();
        write(dir.resolve("chat_template.jinja"), "{% if x %}\r\nyes\rno\n");
        write(dir.resolve("merges.txt"), "#version: 0.2\na b\n\n  c   d  \nbad\ne f g\n\u0120 \u0120\n");
        write(dir.resolve("vocab.txt"), "[PAD]\r\n[UNK]\n##a\r\n\nlast");
        write(dir.resolve("vocab.json"), "{\"a\":0,\"b\":1}");
        write(dir.resolve("added_tokens.json"), "{\"<extra>\":5}");
        HfTokenizerFiles f = HfTokenizerFiles.locate(dir);
        assertEquals("{% if x %}\nyes\nno\n", f.readChatTemplate());
        assertEquals(Arrays.asList("a b", "c d", "\u0120 \u0120"), f.readMergesTxt());
        assertEquals(Arrays.asList("[PAD]", "[UNK]", "##a", "", "last"), f.readVocabTxt());
        assertEquals(Integer.valueOf(1), f.readVocabJson().get("b"));
        assertEquals(Collections.singletonMap("<extra>", 5), f.readAddedTokens());
    }

    @Test
    public void mergesWithoutAHeaderKeepTheFirstLine() throws IOException {
        Path dir = tmp.newFolder().toPath();
        write(dir.resolve("merges.txt"), "a b\nc d");
        assertEquals(Arrays.asList("a b", "c d"), HfTokenizerFiles.locate(dir).readMergesTxt());
    }

    @Test
    public void malformedJsonFilesAreInvalidCheckpoints() throws IOException {
        Path dir = tmp.newFolder().toPath();
        Path config = write(dir.resolve("tokenizer_config.json"), "[1, 2]");
        write(dir.resolve("added_tokens.json"), "{\"<extra>\":-5}");
        write(dir.resolve("vocab.json"), "{\"a\":\"zero\"}");
        HfTokenizerFiles f = HfTokenizerFiles.locate(dir);
        IOException e = assertThrows(IOException.class, f::readTokenizerConfig);
        assertTrue(e.getMessage(), e.getMessage().startsWith("Invalid Hugging Face checkpoint '" + dir + "': "
                + config));
        assertMessageContains(assertThrows(IOException.class, f::readAddedTokens), "outside [0, ");
        assertMessageContains(assertThrows(IOException.class, f::readVocabJson), "expected integer");
        Files.write(dir.resolve("chat_template.jinja"), new byte[]{'a', (byte) 0xFF});
        assertMessageContains(assertThrows(IOException.class, HfTokenizerFiles.locate(dir)::readChatTemplate),
                "invalid UTF-8 byte sequence at byte 1");
        JsonView view = HfTokenizerFiles.readJsonObject(write(dir.resolve("generation_config.json"),
                "{\"top_k\":20,\"top_k\":40}"));
        assertEquals("the last duplicate wins", 40, view.i32("top_k"));
    }

    // ------------------------------------------------------------------ RWKV and PLaMo vocabularies

    @Test
    public void rwkvVocabularyLinesAreParsed() throws IOException {
        String text = "1 '\\x00' 1\n"
                + "2 b'\\xe4\\xb8' 2\n"
                + "3 ' ' 1\n"
                + "4 '\u4e2d' 3\n"
                + "5 \"'\" 1\n"
                + "6 'a b' 3\n"
                + "7 '\\u00e9\\n\\\\' 4\n"
                + "8 b'\\'\"' 2\r\n";
        RwkvVocabReader r = RwkvVocabReader.parse(text);
        assertEquals(8, r.size());
        assertArrayEquals(new byte[]{0}, r.bytes(0));
        assertEquals("\\x00", r.reprText(0));
        assertArrayEquals(new byte[]{(byte) 0xE4, (byte) 0xB8}, r.bytes(1));
        assertEquals("\\xe4\\xb8", r.reprText(1));
        assertEquals(" ", r.reprText(2));
        assertEquals("\\xe4\\xb8\\xad", r.reprText(3));
        assertEquals("'", r.reprText(4));
        assertEquals("a b", r.reprText(5));
        assertArrayEquals(new byte[]{(byte) 0xC3, (byte) 0xA9, '\n', '\\'}, r.bytes(6));
        assertEquals("\\xc3\\xa9\\n\\\\", r.reprText(6));
        assertEquals("\\'\"", r.reprText(7));
    }

    @Test
    public void malformedRwkvLinesAreRejected() {
        assertMessageContains(assertThrows(IOException.class, () -> RwkvVocabReader.parse("1 'ab' 3\n")),
                "line 1: the token has 2 bytes");
        assertMessageContains(assertThrows(IOException.class, () -> RwkvVocabReader.parse("1 'a' 1\n3 'b' 1\n")),
                "line 2: id 3 is not 2");
        assertThrows(IOException.class, () -> RwkvVocabReader.parse("1 'a'\n"));
        assertThrows(IOException.class, () -> RwkvVocabReader.parse("1 'a 1\n"));
        assertThrows(IOException.class, () -> RwkvVocabReader.parse("1 b'\u00e9' 2\n"));
        assertThrows(IOException.class, () -> RwkvVocabReader.parse("1 '\\ud800' 3\n"));
        assertThrows(IOException.class, () -> RwkvVocabReader.parse("1 '\\x4' 1\n"));
        assertThrows(IOException.class, () -> RwkvVocabReader.parse("1 'a' 'b' 1\n"));
        assertThrows(IOException.class, () -> RwkvVocabReader.parse("\n"));
    }

    @Test
    public void plamoRowsAreParsed() throws IOException {
        PlamoVocabReader p = PlamoVocabReader.parse("[\"<unk>\", 0.0, \"UNKNOWN\", 0]\n\n"
                + "[\"<|plamo:op|>\", 0, \"CONTROL\"]\r\n[\"\u2581a\", -1.5]\n   \n");
        assertEquals(3, p.size());
        assertEquals("<unk>", p.token(0));
        assertEquals("UNKNOWN", p.type(0));
        assertEquals("CONTROL", p.type(1));
        assertEquals("\u2581a", p.token(2));
        assertEquals(-1.5, p.score(2), 0.0);
        assertEquals(PlamoVocabReader.DEFAULT_TYPE, p.type(2));
        assertMessageContains(assertThrows(IOException.class, () -> PlamoVocabReader.parse("{\"a\": 1}\n")),
                "line 1: expected a JSON array");
        assertMessageContains(assertThrows(IOException.class, () -> PlamoVocabReader.parse("[\"a\", 1]\n[1, 2]\n")),
                "line 2: the token");
        assertThrows(IOException.class, () -> PlamoVocabReader.parse("[\"a\", \"x\"]\n"));
        assertThrows(IOException.class, () -> PlamoVocabReader.parse("[\"a\", 1, 7]\n"));
        assertThrows(IOException.class, () -> PlamoVocabReader.parse("[\"a\", 1\n"));
    }

    @Test
    public void vocabularyFilesNameTheCheckpointInErrors() throws IOException {
        Path dir = tmp.newFolder().toPath();
        Path rwkv = write(dir.resolve("rwkv_vocab_v20230424.txt"), "1 'a' 1\n2 'b' 2\n");
        IOException e = assertThrows(IOException.class, () -> RwkvVocabReader.read(rwkv));
        assertTrue(e.getMessage(), e.getMessage().startsWith("Invalid Hugging Face checkpoint '" + dir + "': " + rwkv
                + ": line 2: "));
        Path plamo = write(dir.resolve("tokenizer.jsonl"), "[\"a\", 1]\n");
        assertEquals(1, PlamoVocabReader.read(plamo).size());
    }
}
