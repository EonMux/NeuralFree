package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Resolves the Jinja source a model's prompts are rendered with, as llama.cpp does when it initializes
 * a model's chat templates (common/chat.cpp common_chat_templates_init, :729-780).
 *
 * <p>The default template is {@code tokenizer.chat_template}. When it is absent, empty or the literal
 * {@code chatml}, the {@code tokenizer.chat_template.tool_use} template replaces it, or the built-in
 * {@link #CHATML} template when there is none (common/chat.cpp:752-758). The resolved source then
 * receives llama.cpp's two source patches, which disable checks that would otherwise abort rendering
 * of valid conversations (common/chat.cpp:760-780).</p>
 *
 * <p>A template that is exactly one of {@link #names()} selects a built-in template, as llama.cpp's
 * named templates do (src/llama-chat.cpp:28-79): {@code chatml}, and the Mistral styles
 * {@code mistral-v1}, {@code mistral-v3}, {@code mistral-v3-tekken}, {@code mistral-v7} and
 * {@code mistral-v7-tekken}, whose Jinja sources under {@code /chat/templates/} reproduce
 * src/llama-chat.cpp:258-298. The llama.cpp converter writes these names into
 * {@code tokenizer.chat_template} for Mistral-format checkpoints (conversion/base.py:2697-2709).</p>
 */
public final class ChatTemplateSource {

    /** GGUF key of the default chat template. */
    public static final String KEY_DEFAULT = GgufKeys.TOKENIZER_CHAT_TEMPLATE;
    /** GGUF key of the tool-use chat template ({@code llama_model_chat_template(model, "tool_use")}). */
    public static final String KEY_TOOL_USE = namedKey("tool_use");
    /** GGUF key listing the names of the named chat templates (gguf-py/gguf/gguf_writer.py:1355-1356). */
    public static final String KEY_NAMES = GgufKeys.TOKENIZER_CHAT_TEMPLATES;

    /** llama.cpp's built-in ChatML template (common/chat.cpp:687-693, CHATML_TEMPLATE_SRC). */
    public static final String CHATML =
            "{%- for message in messages -%}\n"
            + "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n"
            + "{%- endfor -%}\n"
            + "{%- if add_generation_prompt -%}\n"
            + "  {{- '<|im_start|>assistant\n' -}}\n"
            + "{%- endif -%}";

    /** Name that selects the built-in ChatML template in place of a template source. */
    public static final String CHATML_NAME = "chatml";

    /** Classpath directory of the named templates other than {@link #CHATML_NAME}. */
    private static final String RESOURCE_DIR = "/chat/templates/";
    private static final String RESOURCE_SUFFIX = ".jinja";

    /** Named templates read from {@link #RESOURCE_DIR}, in llama.cpp's table order. */
    private static final Set<String> RESOURCE_NAMES = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
            "mistral-v1", "mistral-v3", "mistral-v3-tekken", "mistral-v7", "mistral-v7-tekken")));

    private static final Set<String> NAMES;

    static {
        Set<String> names = new LinkedHashSet<>();
        names.add(CHATML_NAME);
        names.addAll(RESOURCE_NAMES);
        NAMES = Collections.unmodifiableSet(names);
    }

    /** Named template sources, each read from its resource on first use. */
    private static final ConcurrentMap<String, String> LOADED = new ConcurrentHashMap<>();

    private static final String HARMONY_CHANNEL = "<|channel|>";
    private static final String HARMONY_CHECK_HINT = "in message.content or";
    private static final String HARMONY_CHECK =
            "{%- if \"<|channel|>analysis<|message|>\" in message.content or "
            + "\"<|channel|>final<|message|>\" in message.content %}";

    private static final String MISTRAL_TOOL_CALLS = "[TOOL_CALLS]";
    private static final String MISTRAL_CHECK_HINT = "if (message['content'] is none or";
    private static final String MISTRAL_CHECK =
            "{%- if (message['content'] is none or message['content'] == '' or "
            + "message['content']|length == 0) and (message['tool_calls'] is not defined or "
            + "message['tool_calls'] is none or message['tool_calls']|length == 0) %}";

    private static final String DISABLED_CHECK = "{%- if false %}";

    private ChatTemplateSource() {}

    /**
     * The template source prompts are rendered with.
     *
     * @param defaultTemplate the {@code tokenizer.chat_template} value, or null when absent
     * @param toolUseTemplate the {@code tokenizer.chat_template.tool_use} value, or null when absent
     * @return the resolved and patched Jinja source; never null or empty
     */
    public static String resolve(String defaultTemplate, String toolUseTemplate) {
        String src = defaultTemplate == null ? "" : defaultTemplate;
        String toolUse = toolUseTemplate == null ? "" : toolUseTemplate;
        if (src.isEmpty() || CHATML_NAME.equals(src)) {
            src = toolUse.isEmpty() ? CHATML : toolUse;
        }
        String named = named(src);
        return patch(named != null ? named : src);
    }

    /**
     * The template source prompts are rendered with when a model has only {@code template} as its
     * {@code tokenizer.chat_template}: {@link #resolve(String, String)} without a tool-use template.
     */
    public static String resolve(String template) {
        return resolve(template, null);
    }

    /**
     * GGUF key of the chat template named {@code name}: {@code tokenizer.chat_template.<name>}
     * (gguf-py/gguf/gguf_writer.py:1353), of which {@link #KEY_TOOL_USE} is one.
     */
    public static String namedKey(String name) {
        if (name == null || name.isEmpty()) throw new IllegalArgumentException("chat template name is empty");
        return String.format(GgufKeys.TOKENIZER_CHAT_TEMPLATE_N, name);
    }

    /** The template names that select a built-in template, in llama.cpp's table order (unmodifiable). */
    public static Set<String> names() {
        return NAMES;
    }

    /** Whether {@code template} is exactly one of {@link #names()}. */
    public static boolean isName(String template) {
        return template != null && NAMES.contains(template);
    }

    /**
     * The Jinja source of the built-in template {@code name}, or null when {@code name} is not one of
     * {@link #names()}.
     *
     * @throws IllegalStateException when the template's resource is missing from the classpath
     * @throws UncheckedIOException  when the template's resource cannot be read
     */
    public static String named(String name) {
        if (CHATML_NAME.equals(name)) return CHATML;
        if (name == null || !RESOURCE_NAMES.contains(name)) return null;
        return LOADED.computeIfAbsent(name, ChatTemplateSource::load);
    }

    private static String load(String name) {
        String path = RESOURCE_DIR + name + RESOURCE_SUFFIX;
        try (InputStream in = ChatTemplateSource.class.getResourceAsStream(path)) {
            if (in == null) {
                throw new IllegalStateException("built-in chat template '" + name + "' is missing its resource " + path);
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read built-in chat template resource " + path, e);
        }
    }

    /**
     * Applies llama.cpp's source patches: the gpt-oss check that rejects assistant history containing
     * channel markers, and the Mistral check that rejects assistant turns without content or tool calls,
     * both become {@code {%- if false %}} (common/chat.cpp:760-780).
     */
    static String patch(String src) {
        String out = src;
        if (out.contains(HARMONY_CHANNEL) && out.contains(HARMONY_CHECK_HINT)) {
            out = out.replace(HARMONY_CHECK, DISABLED_CHECK);
        }
        if (out.contains(MISTRAL_TOOL_CALLS) && out.contains(MISTRAL_CHECK_HINT)) {
            out = out.replace(MISTRAL_CHECK, DISABLED_CHECK);
        }
        return out;
    }
}
