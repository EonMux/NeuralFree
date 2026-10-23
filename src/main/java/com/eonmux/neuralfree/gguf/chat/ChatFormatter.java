package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.gguf.MiniJinja;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Renders chat prompts with a model's resolved chat template ({@link ChatTemplateSource}), as llama.cpp's
 * Jinja path does (common/chat.cpp common_chat_templates_apply_jinja).
 *
 * <p>At construction the template is parsed once and probed for system-role support the way llama.cpp's
 * capability check does (common/jinja/caps.cpp:205-229): a system and a user message are rendered with
 * llama.cpp's probe variables, and the system role counts as supported when the system text reaches the
 * output; a template that raises, such as Gemma's "System role not supported", does not support it. Prompts
 * for such a template carry the system prompt merged into the next message ({@link ChatMessages#prepare}).</p>
 *
 * <p>A template that fails to parse or render leaves the prompt to a fallback, logged as a warning: the
 * gpt-oss harmony format for gpt-oss models, otherwise the built-in ChatML template.</p>
 *
 * <p>Instances are immutable and thread-safe.</p>
 */
public final class ChatFormatter {

    private static final Logger LOGGER = LogManager.getLogger(ChatFormatter.class);

    private static final MiniJinja.Template CHATML_TEMPLATE = MiniJinja.compile(ChatTemplateSource.CHATML);
    private static final String PROBE_SYSTEM_TEXT = "System message";
    private static final String PROBE_USER_TEXT = "User message";
    /** llama.cpp's {@code datetime} and {@code date_string} formats, {@code %b %d %Y} and {@code %d %b %Y} in the C locale. */
    private static final DateTimeFormatter DATETIME_FORMAT = DateTimeFormatter.ofPattern("MMM dd yyyy", Locale.ENGLISH);
    private static final DateTimeFormatter DATE_STRING_FORMAT = DateTimeFormatter.ofPattern("dd MMM yyyy", Locale.ENGLISH);

    private final String source;
    private final MiniJinja.Template template;
    private final boolean supportsSystemRole;
    private final String bosText;
    private final String eosText;
    private final boolean addBos;
    private final boolean addEos;
    private final boolean harmonyFallback;

    /**
     * @param source       the resolved template source ({@link ChatTemplateSource#resolve}); not null
     * @param bosText      the BOS token text for {@code bos_token}, empty when the vocabulary has none
     * @param eosText      the EOS token text for {@code eos_token}, empty when the vocabulary has none
     * @param addBos       whether tokenization adds BOS ({@code llama_vocab_get_add_bos})
     * @param addEos       whether tokenization adds EOS ({@code llama_vocab_get_add_eos})
     * @param architecture the GGUF architecture name, selecting the fallback format; may be null
     */
    public ChatFormatter(String source, String bosText, String eosText, boolean addBos, boolean addEos,
                         String architecture) {
        if (source == null) throw new IllegalArgumentException("chat template source is null");
        this.source = source;
        this.bosText = bosText == null ? "" : bosText;
        this.eosText = eosText == null ? "" : eosText;
        this.addBos = addBos;
        this.addEos = addEos;
        this.harmonyFallback = "gpt-oss".equals(architecture) || "gpt_oss".equals(architecture);
        this.template = compile(source);
        this.supportsSystemRole = template != null && probeSystemRole(template);
    }

    private static MiniJinja.Template compile(String source) {
        if (ChatTemplateSource.CHATML.equals(source)) return CHATML_TEMPLATE;
        try {
            return MiniJinja.compile(source);
        } catch (RuntimeException e) {
            LOGGER.warn("The chat template ({} chars) does not parse, prompts use the fallback format: {}",
                    source.length(), e.getMessage());
            return null;
        }
    }

    /**
     * llama.cpp's system-role probe: a system and a user message, an empty tool list, empty BOS/EOS
     * and a generation prompt (common/jinja/caps.cpp:44-50, :205-229).
     */
    private static boolean probeSystemRole(MiniJinja.Template t) {
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("messages", Arrays.asList(message("system", PROBE_SYSTEM_TEXT), message("user", PROBE_USER_TEXT)));
        context.put("tools", Collections.emptyList());
        context.put("bos_token", "");
        context.put("eos_token", "");
        context.put("add_generation_prompt", true);
        try {
            return t.render(context).contains(PROBE_SYSTEM_TEXT);
        } catch (RuntimeException e) {
            LOGGER.debug("Chat template system-role probe raised: {}", e.getMessage());
            return false;
        }
    }

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put(ChatMessages.ROLE, role);
        m.put(ChatMessages.CONTENT, content);
        return m;
    }

    /** The template source prompts are rendered with. */
    public String source() { return source; }

    /** Whether the template renders a leading system message; when not, it is merged into the next message. */
    public boolean supportsSystemRole() { return supportsSystemRole; }

    /**
     * Renders a conversation followed by the assistant generation prompt.
     *
     * @param messages       maps with a string {@code role} and a {@code content}; not modified
     * @param enableThinking the template's {@code enable_thinking} variable
     * @throws IllegalArgumentException for malformed messages
     */
    public String format(List<? extends Map<String, ?>> messages, boolean enableThinking) {
        List<Map<String, Object>> copy = ChatMessages.copyOf(messages);
        LocalDate today = LocalDate.now();
        if (template != null) {
            try {
                return render(template, ChatMessages.prepare(copy, source, supportsSystemRole), enableThinking, today);
            } catch (RuntimeException e) {
                LOGGER.warn("Chat template rendering failed ({}-char template, {} messages), using the fallback"
                        + " format: {}", source.length(), copy.size(), e.getMessage());
            }
        }
        if (harmonyFallback) return HarmonyFallback.format(copy, today);
        return render(CHATML_TEMPLATE, ChatMessages.prepare(copy, ChatTemplateSource.CHATML, true), enableThinking, today);
    }

    /**
     * Renders with the llama.cpp context variables (common/chat.cpp:877-911): messages, add_generation_prompt,
     * bos_token, eos_token, enable_thinking, and the local-date {@code datetime} and {@code date_string} of
     * common_chat_extra_context (:1065-1073), which templates such as Llama 3.1 write into their system
     * header. {@code tools} stays undefined because no tools are offered,
     * as llama.cpp does for an empty tool list (common/chat.cpp:894); templates such as llama3.1 test
     * {@code tools is not none} and would otherwise render a tool preamble. With thinking disabled, a
     * template's closed think block is kept: it tells the model to answer directly. The result passes
     * {@link #stripAddedSpecials}, so tokenizing it yields exactly one BOS.
     */
    private String render(MiniJinja.Template t, List<Map<String, Object>> messages, boolean enableThinking,
                          LocalDate today) {
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("messages", messages);
        context.put("add_generation_prompt", true);
        context.put("bos_token", bosText);
        context.put("eos_token", eosText);
        context.put("enable_thinking", enableThinking);
        context.put("datetime", DATETIME_FORMAT.format(today));
        context.put("date_string", DATE_STRING_FORMAT.format(today));
        return stripAddedSpecials(t.render(context), bosText, eosText, addBos, addEos);
    }

    /**
     * Removes the BOS text a template wrote at the start of the prompt when the tokenizer adds BOS itself, and
     * the EOS text it wrote at the end when the tokenizer adds EOS, as llama.cpp does after rendering
     * (common/chat.cpp:938-949, with {@code add_bos}/{@code add_eos} from {@code llama_vocab_get_add_bos/eos},
     * :804-805). Without this the encoded prompt would start with two BOS tokens (src/llama-vocab.cpp:3767-3768).
     */
    public static String stripAddedSpecials(String rendered, String bosText, String eosText, boolean addBos,
                                            boolean addEos) {
        String out = rendered;
        if (addBos && !bosText.isEmpty() && out.startsWith(bosText)) {
            out = out.substring(bosText.length());
        }
        if (addEos && !eosText.isEmpty() && out.endsWith(eosText)) {
            out = out.substring(0, out.length() - eosText.length());
        }
        return out;
    }
}
