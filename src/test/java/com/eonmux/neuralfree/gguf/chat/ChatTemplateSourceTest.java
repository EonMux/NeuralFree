package com.eonmux.neuralfree.gguf.chat;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ChatTemplateSource#fromFile}: the file's text resolves like a {@code tokenizer.chat_template} value (a
 * built-in name included), a byte-order mark is dropped, and a missing file, an empty file, oversized contents and
 * malformed UTF-8 are errors. {@link ChatTemplateSource#fromOverride} reads a {@code --chat-template} value as
 * llama.cpp and vLLM do: a built-in name, Jinja source, or a template file path.
 */
public class ChatTemplateSourceTest {

    @Rule
    public final TemporaryFolder temp = new TemporaryFolder();

    @Test
    public void theFileTextResolvesLikeAMetadataValue() throws Exception {
        File file = temp.newFile("t.jinja");
        Files.writeString(file.toPath(), "{% for m in messages %}{{m.content}}{% endfor %}");
        assertEquals("{% for m in messages %}{{m.content}}{% endfor %}", ChatTemplateSource.fromFile(file.toPath()));
    }

    @Test
    public void aBuiltInNameSelectsTheBuiltInTemplateAndABomIsDropped() throws Exception {
        File named = temp.newFile("chatml.jinja");
        Files.writeString(named.toPath(), "\uFEFFchatml");
        assertEquals(ChatTemplateSource.CHATML, ChatTemplateSource.fromFile(named.toPath()));
    }

    @Test
    public void anOverrideIsANameJinjaSourceOrATemplateFile() throws Exception {
        assertEquals(ChatTemplateSource.named("llama3"), ChatTemplateSource.fromOverride("llama3"));
        assertEquals("a name ignores the model's tool-use template, as llama.cpp's override does",
                ChatTemplateSource.CHATML, ChatTemplateSource.fromOverride("chatml"));
        assertEquals("{{ messages[0].content }}", ChatTemplateSource.fromOverride("{{ messages[0].content }}"));
        assertEquals("a newline marks Jinja source", "User:\nAssistant:",
                ChatTemplateSource.fromOverride("User:\nAssistant:"));
        File file = temp.newFile("tool_chat_template.jinja");
        Files.writeString(file.toPath(), "{% for m in messages %}[{{m.content}}]{% endfor %}");
        assertEquals("{% for m in messages %}[{{m.content}}]{% endfor %}",
                ChatTemplateSource.fromOverride(file.getPath()));
        File named = temp.newFile("named.jinja");
        Files.writeString(named.toPath(), "gemma");
        assertEquals(ChatTemplateSource.named("gemma"), ChatTemplateSource.fromOverride(named.getPath()));
    }

    @Test
    public void anOverrideNamingNoTemplateIsRefused() {
        String absent = new File(temp.getRoot(), "absent.jinja").getPath();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ChatTemplateSource.fromOverride(absent));
        assertTrue(e.getMessage(), e.getMessage().contains("'" + absent + "'"));
        assertTrue(e.getMessage(), e.getMessage().contains("names no regular file"));
        assertThrows(IllegalArgumentException.class, () -> ChatTemplateSource.fromOverride("llama9"));
        assertThrows(IllegalArgumentException.class, () -> ChatTemplateSource.fromOverride(temp.getRoot().getPath()));
        assertThrows(IllegalArgumentException.class, () -> ChatTemplateSource.fromOverride(" "));
        assertThrows(IllegalArgumentException.class, () -> ChatTemplateSource.fromOverride(null));
    }

    @Test
    public void refusals() throws Exception {
        assertThrows(IllegalArgumentException.class, () -> ChatTemplateSource.fromFile(null));
        assertThrows(IllegalArgumentException.class,
                () -> ChatTemplateSource.fromFile(new File(temp.getRoot(), "absent.jinja").toPath()));
        File empty = temp.newFile("empty.jinja");
        Files.writeString(empty.toPath(), "  \n");
        assertThrows(java.io.UncheckedIOException.class, () -> ChatTemplateSource.fromFile(empty.toPath()));
        File bad = temp.newFile("bad.jinja");
        Files.write(bad.toPath(), new byte[]{(byte) 0xFF, (byte) 0xFE});
        assertThrows(java.io.UncheckedIOException.class, () -> ChatTemplateSource.fromFile(bad.toPath()));
    }
}
