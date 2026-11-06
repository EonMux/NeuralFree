package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.CharacterCodingException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
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
 * named templates do (src/llama-chat.cpp:28-83): {@code chatml} and the 53 other styles of llama.cpp's
 * table ({@code llama2}, {@code llama3}, {@code gemma}, {@code phi3}, {@code mistral-v7}, ...), whose
 * Jinja sources under {@code /chat/templates/} each render exactly what llama.cpp's
 * {@code llm_chat_apply_template} builds for that style (src/llama-chat.cpp:250-939). The llama.cpp
 * converter writes the Mistral names into {@code tokenizer.chat_template} for Mistral-format
 * checkpoints (conversion/base.py:2697-2709); {@code --chat-template} and a per-request
 * {@code chat_template} take any of them.</p>
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
            "llama2", "llama2-sys", "llama2-sys-bos", "llama2-sys-strip",
            "mistral-v1", "mistral-v3", "mistral-v3-tekken", "mistral-v7", "mistral-v7-tekken",
            "phi3", "phi4", "falcon3", "zephyr", "monarch", "gemma", "orion", "openchat", "vicuna", "vicuna-orca",
            "deepseek", "deepseek2", "deepseek3", "deepseek-ocr", "command-r", "llama3", "chatglm3", "chatglm4",
            "glmedge", "minicpm", "exaone3", "exaone4", "exaone-moe", "rwkv-world", "granite", "granite-4.0",
            "granite-4.1", "gigachat", "megrez", "yandex", "bailing", "bailing-think", "bailing2", "llama4",
            "smolvlm", "hunyuan-moe", "gpt-oss", "hunyuan-dense", "hunyuan-vl", "kimi-k2", "seed_oss", "grok-2",
            "pangu-embedded", "solar-open")));

    private static final Set<String> NAMES;

    static {
        Set<String> names = new LinkedHashSet<>();
        names.add(CHATML_NAME);
        names.addAll(RESOURCE_NAMES);
        NAMES = Collections.unmodifiableSet(names);
    }

    /**
     * The template name only detection guesses, never the name table: llama.cpp's {@code dots1}
     * (src/llama-chat.cpp:218-219) has no {@code LLM_CHAT_TEMPLATES} entry, so a template naming it
     * detects nothing and {@code dots1} is refused like any other unknown name.
     */
    static final String DOTS1_NAME = "dots1";

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

    /**
     * Cap for a template override file ({@code --chat-template-file}, server-chat.cpp:593): a
     * file whose size exceeds it is refused before any allocation.
     */
    public static final long TEMPLATE_FILE_MAX_BYTES = 4_194_304;

    /** System property naming a template override, read as {@link #fromOverride} reads it: one of
     *  {@link #names()}, a Jinja source or a template file (llama.cpp's and vLLM's {@code --chat-template}). */
    public static final String OVERRIDE_PROPERTY = "neuralfree.chat.template";

    /** System property naming a template override file, under {@link #fromFile}'s rules
     *  (llama.cpp's {@code --chat-template-file}). */
    public static final String OVERRIDE_FILE_PROPERTY = "neuralfree.chat.template.file";

    /**
     * System property switching llama.cpp's Jinja engine on ({@code true}, the default) or onto the
     * non-Jinja route ({@code false}), as llama.cpp's {@code --jinja} and {@code --no-jinja} do: on
     * the non-Jinja route the model's template is replaced by the built-in
     * {@link ChatTemplateDetector} guesses from its text, and a template guessing nothing is refused.
     */
    public static final String JINJA_PROPERTY = "neuralfree.chat.jinja";

    /** Characters that mark an override value as Jinja source rather than a file path (vLLM's {@code JINJA_CHARS}). */
    private static final String JINJA_CHARS = "{}\n";

    private ChatTemplateSource() {}

    /**
     * The template source the {@link #OVERRIDE_PROPERTY} and {@link #OVERRIDE_FILE_PROPERTY}
     * properties select, or null when neither is set: the override replaces the model's
     * {@code tokenizer.chat_template} wherever a tokenizer is built. Setting both properties
     * is refused, as llama.cpp refuses {@code --chat-template} with {@code --chat-template-file}.
     *
     * @return the resolved Jinja source; never null or empty when the properties name anything
     * @throws IllegalArgumentException when both properties are set, or the file named by
     *                                  {@link #OVERRIDE_FILE_PROPERTY} is not a regular file
     * @throws UncheckedIOException     when the override file cannot be read or is not UTF-8
     */
    public static String fromProperties() {
        Override override = resolveFromProperties();
        return override == null ? null : override.source;
    }

    /**
     * The template text the properties name before any resolution: an override property's value, or
     * the text of its template file (what llama.cpp's {@code --chat-template} and
     * {@code --chat-template-file} keep, and what {@link ChatTemplateDetector} reads on the
     * non-Jinja route). Null when neither property is set.
     *
     * @throws IllegalArgumentException when both properties are set, or the file named by
     *                                  {@link #OVERRIDE_FILE_PROPERTY} is not a regular file
     * @throws UncheckedIOException     when the override file cannot be read or is not UTF-8
     */
    public static String rawFromProperties() {
        Override override = resolveFromProperties();
        return override == null ? null : override.raw;
    }

    /** One override: the text it contributes, and the resolved Jinja source it selects. */
    private static final class Override {
        final String raw;
        final String source;

        Override(String raw, String source) {
            this.raw = raw;
            this.source = source;
        }
    }

    private static Override resolveFromProperties() {
        String inline = System.getProperty(OVERRIDE_PROPERTY);
        String file = System.getProperty(OVERRIDE_FILE_PROPERTY);
        boolean hasInline = inline != null && !inline.isBlank();
        boolean hasFile = file != null && !file.isBlank();
        if (hasInline && hasFile) {
            throw new IllegalArgumentException(OVERRIDE_PROPERTY + " and " + OVERRIDE_FILE_PROPERTY
                    + " are both set; name one template source, not two");
        }
        try {
            if (hasInline) {
                String raw = inline.trim();
                return new Override(raw, fromOverride(raw));
            }
            if (hasFile) {
                Path path = Path.of(file.trim());
                String raw = overrideText(path);
                return new Override(raw, resolve(raw, null));
            }
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(propertySource(hasFile) + ": " + e.getMessage(), e);
        } catch (UncheckedIOException e) {
            throw new UncheckedIOException(propertySource(hasFile) + ": " + e.getMessage(), e.getCause());
        }
        return null;
    }

    /**
     * The {@link #JINJA_PROPERTY} switch: {@code true} when unset.
     *
     * @throws IllegalArgumentException when the property is set to anything else
     */
    public static boolean jinjaFromProperties() {
        String value = System.getProperty(JINJA_PROPERTY);
        if (value == null) return true;
        String v = value.trim();
        if ("true".equalsIgnoreCase(v)) return true;
        if ("false".equalsIgnoreCase(v)) return false;
        throw new IllegalArgumentException(JINJA_PROPERTY + " must be true or false, got '" + value + "'");
    }

    /** The property the last {@link #fromProperties} failure came from. */
    private static String propertySource(boolean file) {
        return file ? OVERRIDE_FILE_PROPERTY : OVERRIDE_PROPERTY;
    }

    /**
     * The template an override value selects, as {@code --chat-template} reads it: one of {@link #names()}
     * selects that built-in template (llama.cpp), a value holding Jinja syntax (a curly brace or a newline)
     * is the template source, and any other value names a template file, read under
     * {@link #fromFile}'s rules, as vLLM's {@code load_chat_template} reads a path. The model's
     * {@code tokenizer.chat_template.tool_use} plays no part, as in llama.cpp's override.
     *
     * @return the resolved Jinja source; never null or empty
     * @throws IllegalArgumentException when the value is null or blank, or is neither a name nor Jinja
     *                                  source and names no regular file
     * @throws UncheckedIOException     when the named file cannot be read, is too large, or is not UTF-8
     */
    public static String fromOverride(String value) {
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException("chat template override is empty");
        }
        if (isName(value) || looksLikeJinja(value)) {
            return resolve(value, null);
        }
        Path file;
        try {
            file = Path.of(value);
        } catch (InvalidPathException e) {
            throw new IllegalArgumentException(notATemplate(value), e);
        }
        if (!Files.isRegularFile(file)) {
            throw new IllegalArgumentException(notATemplate(value));
        }
        return fromFile(file);
    }

    private static boolean looksLikeJinja(String value) {
        for (int i = 0; i < JINJA_CHARS.length(); i++) {
            if (value.indexOf(JINJA_CHARS.charAt(i)) >= 0) return true;
        }
        return false;
    }

    private static String notATemplate(String value) {
        return "chat template '" + value + "' is not one of the " + NAMES.size() + " built-in template names,"
                + " holds no Jinja syntax ({, } or a newline), and names no regular file";
    }

    /**
     * The template source prompts are rendered with.
     *
     * @param defaultTemplate the {@code tokenizer.chat_template} value, or null when absent
     * @param toolUseTemplate the {@code tokenizer.chat_template.tool_use} value, or null when absent
     * @return the resolved and patched Jinja source; never null or empty
     */
    public static String resolve(String defaultTemplate, String toolUseTemplate) {
        String src = legacyTemplate(defaultTemplate, toolUseTemplate);
        String named = named(src);
        return patch(named != null ? named : src);
    }

    /**
     * The template text the non-Jinja route reads, as llama.cpp's {@code common_chat_templates_init}
     * keeps it before any Jinja rendering (common/chat.cpp:729-758): a default that is empty or the
     * literal {@code chatml} falls back to the tool-use template, or to the built-in
     * {@link #CHATML} source when there is none. An override replaces the model's templates and
     * plays no part in that fallback — pass {@code null} for {@code toolUseTemplate} along with it.
     *
     * @param defaultTemplate the {@code tokenizer.chat_template} value, or null when absent
     * @param toolUseTemplate the {@code tokenizer.chat_template.tool_use} value, or null when absent
     * @return the unpatched template text; never null or empty
     */
    public static String legacyTemplate(String defaultTemplate, String toolUseTemplate) {
        String src = defaultTemplate == null ? "" : defaultTemplate;
        String toolUse = toolUseTemplate == null ? "" : toolUseTemplate;
        if (src.isEmpty() || CHATML_NAME.equals(src)) {
            src = toolUse.isEmpty() ? CHATML : toolUse;
        }
        return src;
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

    /**
     * A template override file, llama.cpp's {@code --chat-template-file}: the file's contents are
     * the template source, or one of the {@link #names()} to select a built-in template, resolved
     * exactly like a GGUF's {@code tokenizer.chat_template} value. The file's UTF-8 text may
     * carry a byte-order mark; malformed bytes are an error, not replacements; the size is capped
     * by {@link #TEMPLATE_FILE_MAX_BYTES} before the file is read.
     *
     * @return the Jinja source to render prompts with; never null or empty
     * @throws IllegalArgumentException when {@code file} is null or not a regular file
     * @throws UncheckedIOException     when the file cannot be read, is too large, or is not UTF-8
     */
    public static String fromFile(Path file) {
        return resolve(overrideText(file), null);
    }

    /**
     * The UTF-8 text a template override file contributes, before any resolution: the file's
     * contents, byte-order mark stripped, capped by {@link #TEMPLATE_FILE_MAX_BYTES} before the
     * file is read (\u00A73.6). This is the text llama.cpp's {@code --chat-template-file} keeps and
     * {@link ChatTemplateDetector} reads on the non-Jinja route.
     *
     * @throws IllegalArgumentException when {@code file} is null or not a regular file
     * @throws UncheckedIOException     when the file cannot be read, is too large, or is not UTF-8
     */
    public static String overrideText(Path file) {
        if (file == null) throw new IllegalArgumentException("chat template file is null");
        long size;
        try {
            if (!Files.isRegularFile(file)) {
                throw new IllegalArgumentException("chat template file is not a regular file: " + file);
            }
            size = Files.size(file);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot stat chat template file " + file, e);
        }
        if (size > TEMPLATE_FILE_MAX_BYTES) {
            String message = "chat template file " + file + " is " + size
                    + " bytes, over the " + TEMPLATE_FILE_MAX_BYTES + " cap";
            throw new UncheckedIOException(message, new IOException(message));
        }
        byte[] bytes;
        try {
            bytes = Files.readAllBytes(file);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read chat template file " + file, e);
        }
        String text;
        try {
            text = StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(java.nio.ByteBuffer.wrap(bytes)).toString();
        } catch (CharacterCodingException e) {
            throw new UncheckedIOException("chat template file " + file + " is not UTF-8", e);
        }
        int start = !text.isEmpty() && text.charAt(0) == '\uFEFF' ? 1 : 0;
        String source = start == 0 ? text : text.substring(start);
        if (source.trim().isEmpty()) {
            String message = "chat template file " + file + " is empty";
            throw new UncheckedIOException(message, new IOException(message));
        }
        return source;
    }

    /**
     * The template text an override value contributes, as llama.cpp keeps it for
     * {@code --chat-template} ({@code --chat-template-file} reads the file at parse): the value
     * itself when it is one of {@link #names()} or Jinja source, else the text of the template
     * file it names ({@link #overrideText}). The value is validated exactly as
     * {@link #fromOverride} validates it, so a value that is none of the three is refused here too.
     *
     * @throws IllegalArgumentException when the value is null or blank, or is neither a name nor
     *                                  Jinja source and names no regular file
     * @throws UncheckedIOException     when the named file cannot be read, is too large, or is not UTF-8
     */
    public static String overrideValueText(String value) {
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException("chat template override is empty");
        }
        fromOverride(value);
        if (isName(value) || looksLikeJinja(value)) {
            return value;
        }
        try {
            return overrideText(Path.of(value));
        } catch (InvalidPathException e) {
            throw new IllegalArgumentException(notATemplate(value), e);
        }
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
     * Whether {@code source} is the Jinja source of one of the built-in templates ({@link #named},
     * plus the detection-only {@link #DOTS1_NAME}). Each reproduces llama.cpp's own formatter for
     * its style, which places (or leaves out) system messages itself, so a renderer passes the
     * conversation through unchanged.
     */
    public static boolean isBuiltIn(String source) {
        if (source == null) return false;
        if (CHATML.equals(source)) return true;
        for (String name : RESOURCE_NAMES) {
            if (named(name).equals(source)) return true;
        }
        return patch(dots1()).equals(source);
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

    /**
     * The unpatched Jinja source of the detection-only {@link #DOTS1_NAME} template, which
     * {@link #named} does not return: {@code dots1} has no {@link #names()} entry, so nothing
     * selects it but a {@link ChatTemplateDetector} guess.
     *
     * @throws IllegalStateException when the template's resource is missing from the classpath
     * @throws UncheckedIOException  when the template's resource cannot be read
     */
    static String dots1() {
        return LOADED.computeIfAbsent(DOTS1_NAME, ChatTemplateSource::load);
    }

    /**
     * The built-in template {@link ChatTemplateDetector} guessed, patched like any resolved source.
     * The name may be the detection-only {@link #DOTS1_NAME}, which {@link #fromOverride} refuses.
     *
     * @throws IllegalArgumentException when {@code name} is not one of {@link #names()} and not
     *                                  {@link #DOTS1_NAME}
     */
    public static String detectedSource(String name) {
        if (isName(name)) return resolve(name, null);
        if (DOTS1_NAME.equals(name)) return patch(dots1());
        throw new IllegalArgumentException("no built-in chat template named '" + name + "'");
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
