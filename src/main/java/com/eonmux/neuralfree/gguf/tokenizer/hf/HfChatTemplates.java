package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * The chat templates of a Hugging Face checkpoint as GGUF metadata, in the transformers order
 * (tokenization_utils_base.py:1812-1830, 1084-1088):
 * <ol>
 *   <li>template files: {@code chat_template.jinja} is the default template and each
 *       {@code additional_chat_templates/<name>.jinja} a named one; when any exists they replace the
 *       tokenizer_config entry;</li>
 *   <li>else tokenizer_config.json {@code chat_template}: a string, or a list of {@code {name, template}};</li>
 *   <li>else the processor's template: processor_config.json, then chat_template.json {@code chat_template};</li>
 *   <li>else the fallback a vocabulary writer passes, such as a bundled template ({@link #bundled}).</li>
 * </ol>
 * The keys are those {@code GGUFWriter.add_chat_template} writes (gguf-py/gguf/gguf_writer.py:1332-1363): the default
 * template as {@code tokenizer.chat_template}, every other named template as {@code tokenizer.chat_template.<name>}
 * with its name sanitized to ASCII letters, digits and {@code _}, and the names in {@code tokenizer.chat_templates}.
 * Unlike llama.cpp's converter, which prefers the tokenizer_config entry (vocab.py:340-348), the files win, as they
 * do for transformers.
 */
public final class HfChatTemplates {

    private static final Logger LOGGER = LogManager.getLogger(HfChatTemplates.class);

    /** Classpath directory of the converter's bundled templates (llama.cpp models/templates). */
    private static final String BUNDLED_DIR = "/hf/templates/";
    private static final String DEFAULT_NAME = "default";
    private static final String KEY_CHAT_TEMPLATE = "chat_template";
    private static final Pattern NOT_NAME_CHAR = Pattern.compile("[^A-Za-z0-9]");
    private static final Pattern BUNDLED_FILE = Pattern.compile("[A-Za-z0-9][A-Za-z0-9._-]*\\.jinja");

    private HfChatTemplates() {}

    /**
     * The chat template keys of the checkpoint whose tokenizer files are {@code files}; {@code fallback} is used when
     * no file names a template. Empty when there is no template at all.
     *
     * @param files    the tokenizer files
     * @param fallback a template source to use when the checkpoint has none, or null
     * @throws IOException when a template file cannot be read
     */
    public static Map<String, Object> resolve(HfTokenizerFiles files, String fallback) throws IOException {
        Objects.requireNonNull(files, "files");
        Object template = fileTemplates(files);
        if (template == null) template = configTemplate(files.readTokenizerConfig());
        if (template == null) template = configTemplate(files.readProcessorConfig());
        if (template == null) template = configTemplate(files.readChatTemplateJson());
        if (template == null) template = fallback;
        return template == null ? Collections.<String, Object>emptyMap() : toGguf(template);
    }

    /**
     * A template bundled from llama.cpp's {@code models/templates} directory, such as
     * {@code llama-cpp-rwkv-world.jinja}.
     *
     * @throws IllegalArgumentException when {@code file} is not a bundled template's file name
     * @throws UncheckedIOException     when the bundled resource cannot be read
     */
    public static String bundled(String file) {
        Objects.requireNonNull(file, "file");
        if (!BUNDLED_FILE.matcher(file).matches()) {
            throw new IllegalArgumentException("'" + file + "' is not a bundled chat template file name");
        }
        String path = BUNDLED_DIR + file;
        try (InputStream in = HfChatTemplates.class.getResourceAsStream(path)) {
            if (in == null) {
                throw new IllegalArgumentException("no bundled chat template '" + file + "' (" + path + ")");
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read bundled chat template " + path, e);
        }
    }

    /**
     * The keys {@code add_chat_template(value)} writes for a template string or a list of {@code {name, template}}
     * maps: entries without a name or template are skipped, and {@code default} is the default template.
     *
     * @throws IllegalArgumentException when {@code template} is neither a string nor a list
     */
    public static Map<String, Object> toGguf(Object template) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (template instanceof String) {
            out.put(GgufKeys.TOKENIZER_CHAT_TEMPLATE, template);
            return Collections.unmodifiableMap(out);
        }
        if (!(template instanceof List)) {
            throw new IllegalArgumentException("a chat template is a string or a list of {name, template}, not "
                    + template.getClass().getSimpleName());
        }
        String defaultTemplate = null;
        Set<String> names = new LinkedHashSet<>();
        for (Object choice : (List<?>) template) {
            if (!(choice instanceof Map)) continue;
            Object rawName = ((Map<?, ?>) choice).get("name");
            Object body = ((Map<?, ?>) choice).get("template");
            String name = NOT_NAME_CHAR.matcher(rawName instanceof String ? (String) rawName : "").replaceAll("_");
            if (name.isEmpty() || !(body instanceof String)) continue;
            if (DEFAULT_NAME.equals(name)) {
                defaultTemplate = (String) body;
            } else {
                names.add(name);
                out.put(ChatTemplateSource.namedKey(name), body);
            }
        }
        if (!names.isEmpty()) {
            out.put(ChatTemplateSource.KEY_NAMES, Collections.unmodifiableList(new ArrayList<>(names)));
        }
        if (defaultTemplate != null) out.put(GgufKeys.TOKENIZER_CHAT_TEMPLATE, defaultTemplate);
        return Collections.unmodifiableMap(out);
    }

    /**
     * {@code MistralModel.get_community_chat_template} (mistral.py:54-90): the llama.cpp template names for v1, v3
     * and v7 (rendered by {@link ChatTemplateSource}), the bundled Mistral Small 3.2 template for v11 and the
     * Devstral template for v13.
     *
     * @throws GgufFormatException for any other version
     */
    public static String mistralCommunity(int version, boolean tekken) {
        switch (version) {
            case 1: return "mistral-v1";
            case 3: return tekken ? "mistral-v3-tekken" : "mistral-v3";
            case 7: return tekken ? "mistral-v7-tekken" : "mistral-v7";
            case 11: return bundled("Mistral-Small-3.2-24B-Instruct-2506.jinja");
            case 13: return bundled("unsloth-mistral-Devstral-Small-2507.jinja");
            default:
                throw new GgufFormatException("no chat template for the " + (tekken ? "tekken" : "spm")
                        + " tokenizer version v" + version + "; add a chat_template.jinja to the checkpoint");
        }
    }

    /** The template files as a string (default only) or a named list; null when there are none. */
    private static Object fileTemplates(HfTokenizerFiles files) throws IOException {
        String defaultTemplate = files.readChatTemplate();
        Map<String, String> additional = files.readAdditionalChatTemplates();
        if (additional.isEmpty()) return defaultTemplate;
        List<Map<String, String>> named = new ArrayList<>();
        if (defaultTemplate != null) named.add(choice(DEFAULT_NAME, defaultTemplate));
        for (Map.Entry<String, String> e : additional.entrySet()) named.add(choice(e.getKey(), e.getValue()));
        return named;
    }

    private static Map<String, String> choice(String name, String template) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("name", name);
        m.put("template", template);
        return m;
    }

    /** The {@code chat_template} member of a configuration file: a string or a list; else null. */
    private static Object configTemplate(JsonView config) {
        Object template = config.raw(KEY_CHAT_TEMPLATE);
        if (template == null || template instanceof String || template instanceof List) return template;
        LOGGER.warn("bad type for {} ({}); ignoring it", config.where(KEY_CHAT_TEMPLATE),
                template.getClass().getSimpleName());
        return null;
    }
}
