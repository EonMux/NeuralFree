package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.response.ResponseParser;
import com.eonmux.neuralfree.gguf.chat.response.ResponseTemplate;
import com.eonmux.neuralfree.gguf.chat.response.TemplateParsing;
import com.eonmux.neuralfree.gguf.chat.response.ToolArguments;
import com.eonmux.neuralfree.utils.json.Json;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * The tool calls of vLLM's {@code hf} / {@code beam} parsers ({@code _ResponseStream} with tool parsing): the model's
 * response template reads the text, each {@code tool_calls} region that closes becomes its calls — a value shaped
 * {@code {"function": {"name", "arguments"}}} or a list of them, arguments typed by the tools' schemas and written as
 * JSON when they are not text already — and every other region's text is content. A region with an incomplete call or
 * a body that does not parse is dropped whole, with a warning on the final parse as vLLM logs it. Under {@code partial}
 * an open region and a possible delimiter at the end wait. Immutable.
 */
final class ResponseTemplateToolParser implements ToolParser {

    private static final Logger LOGGER = LogManager.getLogger(ResponseTemplateToolParser.class);

    /** {@code json.dumps(..., ensure_ascii=False)}. */
    private static final Json.Style ARGUMENTS = new Json.Style(false, null, ", ", ": ", false);

    private final ResponseTemplate template;
    private final ToolArguments arguments;

    ResponseTemplateToolParser(TemplateParsing parsing, List<ChatTool> tools) {
        this.template = parsing.template();
        this.arguments = ToolArguments.of(tools);
    }

    @Override
    public ToolParse parse(String text, boolean partial) {
        Collector c = new Collector(!partial);
        ResponseParser parser = new ResponseParser(template, "", arguments, null);
        parser.feed(text, c);
        if (!partial) {
            parser.feed(parser.closerRest(), c);
            parser.finish(c);
        }
        return new ToolParse(c.content.toString(), c.calls);
    }

    @Override
    public List<String> triggers() {
        List<String> open = template.field(ResponseTemplate.TOOL_CALLS).openLiterals();
        return open == null ? Collections.<String>emptyList() : open;
    }

    /**
     * Content from every non-tool region, calls from every tool region that closes; with {@code report} a dropped tool
     * region is logged (the final parse, not every partial re-parse).
     */
    private static final class Collector implements ResponseParser.Listener {
        final StringBuilder content = new StringBuilder();
        final List<ToolCall> calls = new ArrayList<>();
        final boolean report;

        Collector(boolean report) {
            this.report = report;
        }

        @Override
        public void chunk(String field, String text, boolean raw) {
            if (!ResponseTemplate.TOOL_CALLS.equals(field)) {
                content.append(text);
            }
        }

        @Override
        public void closed(String field, int start, int end, Object value) {
            if (ResponseTemplate.TOOL_CALLS.equals(field)) {
                List<ToolCall> parsed = toolCalls(value);
                if (parsed.isEmpty() && report) {
                    LOGGER.warn("response_template: dropping an incomplete tool call");
                }
                calls.addAll(parsed);
            }
        }

        @Override
        public void malformed(String field, int start, int end, String error) {
            if (ResponseTemplate.TOOL_CALLS.equals(field) && report) {
                LOGGER.warn("response_template: dropping a malformed tool call ({})", error);
            }
        }
    }

    /** The calls of a tool region's value, or none when any of them is incomplete (vLLM {@code _to_tool_calls}). */
    static List<ToolCall> toolCalls(Object value) {
        List<?> values = value instanceof List ? (List<?>) value : Collections.singletonList(value);
        List<ToolCall> out = new ArrayList<>();
        for (Object item : values) {
            Object fn = item instanceof Map ? ((Map<?, ?>) item).get("function") : null;
            if (!(fn instanceof Map)) {
                return Collections.emptyList();
            }
            Object name = ((Map<?, ?>) fn).get("name");
            Object args = ((Map<?, ?>) fn).get("arguments");
            if (!(name instanceof String) || ((String) name).isEmpty() || args == null) {
                return Collections.emptyList();
            }
            out.add(ToolCall.of((String) name, args instanceof String ? (String) args : Json.toString(args, ARGUMENTS)));
        }
        return out;
    }
}
