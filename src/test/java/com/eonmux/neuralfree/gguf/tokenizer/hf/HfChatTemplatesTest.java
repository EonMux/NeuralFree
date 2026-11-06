package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfChatTemplates}: the transformers precedence (template files, tokenizer_config.json, processor_config.json,
 * chat_template.json, then the writer's fallback), the GGUF keys of named templates, the bundled templates and the
 * Mistral community template choice, and the tokenizer_config response template carried beside them.
 */
public class HfChatTemplatesTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path dir(String... namesAndContents) throws IOException {
        Path dir = tmp.newFolder().toPath();
        for (int i = 0; i < namesAndContents.length; i += 2) {
            Path file = dir.resolve(namesAndContents[i]);
            Files.createDirectories(file.getParent());
            Files.write(file, namesAndContents[i + 1].replace('\'', '"').getBytes(StandardCharsets.UTF_8));
        }
        return dir;
    }

    private static Map<String, Object> resolve(Path dir, String fallback) throws IOException {
        return HfChatTemplates.resolve(HfTokenizerFiles.locate(dir), fallback);
    }

    private static Map<String, String> choice(String name, String template) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("name", name);
        m.put("template", template);
        return m;
    }

    // ------------------------------------------------------------------ response template

    @Test
    public void theTokenizerConfigResponseTemplateRidesAlongAsCompactJson() throws IOException {
        Path dir = dir("tokenizer_config.json", "{'chat_template':'CONFIG','response_template':"
                + "{'start_anchor':'<a>','fields':{'content':{'close':'</é>'}}}}");
        Map<String, Object> md = resolve(dir, null);
        assertEquals("CONFIG", md.get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));
        assertEquals("{\"start_anchor\":\"<a>\",\"fields\":{\"content\":{\"close\":\"</é>\"}}}",
                md.get(GgufKeys.TOKENIZER_RESPONSE_TEMPLATE));
        assertEquals("a response template without any chat template still rides along",
                Collections.singleton(GgufKeys.TOKENIZER_RESPONSE_TEMPLATE),
                resolve(dir("tokenizer_config.json", "{'response_template':{'fields':{}}}"), null).keySet());
        assertFalse("a non-object response template is ignored", resolve(dir("tokenizer_config.json",
                "{'chat_template':'C','response_template':'text'}"), null).containsKey(GgufKeys.TOKENIZER_RESPONSE_TEMPLATE));
    }

    // ------------------------------------------------------------------ precedence

    @Test
    public void templateFilesWinOverTokenizerConfig() throws IOException {
        Path dir = dir("chat_template.jinja", "FILE", "additional_chat_templates/tool_use.jinja", "TOOLS",
                "tokenizer_config.json", "{'chat_template':'CONFIG'}");
        Map<String, Object> md = resolve(dir, "FALLBACK");
        assertEquals("FILE", md.get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));
        assertEquals("TOOLS", md.get(ChatTemplateSource.KEY_TOOL_USE));
        assertEquals(Collections.singletonList("tool_use"), md.get(ChatTemplateSource.KEY_NAMES));
    }

    @Test
    public void aLoneChatTemplateFileIsTheDefaultTemplate() throws IOException {
        Map<String, Object> md = resolve(dir("chat_template.jinja", "FILE",
                "tokenizer_config.json", "{'chat_template':'CONFIG'}"), null);
        assertEquals(Collections.singletonMap(GgufKeys.TOKENIZER_CHAT_TEMPLATE, "FILE"), md);
    }

    @Test
    public void tokenizerConfigWinsOverTheProcessorFiles() throws IOException {
        Path dir = dir("tokenizer_config.json", "{'chat_template':'CONFIG'}",
                "processor_config.json", "{'chat_template':'PROCESSOR'}",
                "chat_template.json", "{'chat_template':'JSON'}");
        assertEquals("CONFIG", resolve(dir, null).get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));
    }

    @Test
    public void processorConfigWinsOverChatTemplateJson() throws IOException {
        Path dir = dir("processor_config.json", "{'chat_template':'PROCESSOR'}",
                "chat_template.json", "{'chat_template':'JSON'}");
        assertEquals("PROCESSOR", resolve(dir, null).get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));
        assertEquals("JSON", resolve(dir("chat_template.json", "{'chat_template':'JSON'}"), null)
                .get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));
    }

    @Test
    public void theFallbackAppliesOnlyWithoutAnyTemplate() throws IOException {
        assertEquals("FALLBACK", resolve(dir(), "FALLBACK").get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));
        assertTrue(resolve(dir(), null).isEmpty());
        assertTrue("a template of the wrong type is ignored",
                resolve(dir("tokenizer_config.json", "{'chat_template':42}"), null).isEmpty());
    }

    // ------------------------------------------------------------------ named templates

    @Test
    public void aTemplateListWritesSanitizedNamedKeys() {
        List<Object> list = Arrays.asList(choice("default", "D"), choice("rag-x", "R"), choice("tool use", "T"),
                choice("", "skipped"), Collections.singletonMap("name", "no-template"), "not a map");
        Map<String, Object> md = HfChatTemplates.toGguf(list);
        assertEquals("D", md.get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));
        assertEquals("R", md.get(ChatTemplateSource.namedKey("rag_x")));
        assertEquals("T", md.get(ChatTemplateSource.namedKey("tool_use")));
        assertEquals(Arrays.asList("rag_x", "tool_use"), md.get(ChatTemplateSource.KEY_NAMES));
        assertEquals(4, md.size());
        assertThrows(UnsupportedOperationException.class, () -> md.put("k", "v"));
    }

    @Test
    public void aTemplateListFromTokenizerConfigIsKept() throws IOException {
        Map<String, Object> md = resolve(dir("tokenizer_config.json",
                "{'chat_template':[{'name':'default','template':'D'},{'name':'rag','template':'R'}]}"), null);
        assertEquals("D", md.get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));
        assertEquals("R", md.get(ChatTemplateSource.namedKey("rag")));
    }

    @Test
    public void toGgufRejectsOtherTypes() {
        assertThrows(IllegalArgumentException.class, () -> HfChatTemplates.toGguf(42L));
    }

    // ------------------------------------------------------------------ bundled and community templates

    @Test
    public void bundledTemplatesAreReadFromTheClasspath() {
        String rwkv = HfChatTemplates.bundled("llama-cpp-rwkv-world.jinja");
        assertFalse(rwkv.isEmpty());
        assertTrue(rwkv.contains("messages"));
        assertThrows(IllegalArgumentException.class, () -> HfChatTemplates.bundled("../templates/x.jinja"));
        assertThrows(IllegalArgumentException.class, () -> HfChatTemplates.bundled("missing.jinja"));
        assertThrows(IllegalArgumentException.class, () -> HfChatTemplates.bundled("x.txt"));
    }

    @Test
    public void mistralCommunityTemplatesFollowTheTokenizerVersion() {
        assertEquals("mistral-v1", HfChatTemplates.mistralCommunity(1, false));
        assertEquals("mistral-v3", HfChatTemplates.mistralCommunity(3, false));
        assertEquals("mistral-v3-tekken", HfChatTemplates.mistralCommunity(3, true));
        assertEquals("mistral-v7", HfChatTemplates.mistralCommunity(7, false));
        assertEquals("mistral-v7-tekken", HfChatTemplates.mistralCommunity(7, true));
        assertEquals(HfChatTemplates.bundled("Mistral-Small-3.2-24B-Instruct-2506.jinja"),
                HfChatTemplates.mistralCommunity(11, true));
        assertEquals(HfChatTemplates.bundled("unsloth-mistral-Devstral-Small-2507.jinja"),
                HfChatTemplates.mistralCommunity(13, true));
        assertThrows(GgufFormatException.class, () -> HfChatTemplates.mistralCommunity(2, false));
        for (int v : new int[]{1, 3, 7}) {
            assertTrue(ChatTemplateSource.isName(HfChatTemplates.mistralCommunity(v, v != 1)));
        }
    }
}
