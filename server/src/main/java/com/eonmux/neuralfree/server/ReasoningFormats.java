package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.utils.json.JsonView;

/**
 * The per-request {@code reasoning_format} of llama.cpp (server-common.cpp:1469-1471) as the
 * reasoning filter the request's generations route through: {@code auto} and {@code deepseek}
 * keep the configured filter (reasoning split into {@code reasoning_content}), {@code none} runs
 * the raw-output variant of it ({@link ReasoningFilter#asRaw}: with thinking tags everything is
 * content, the tags included; on a harmony model tool calls are still parsed and only the
 * reasoning messages stay inline in the content, as llama.cpp's gpt-oss parser does), and
 * {@code deepseek-legacy} splits like deepseek except on a stream, where the reasoning stays
 * inline in the content (llama.cpp {@code reasoning_in_content}, server-task.cpp:78). Unknown
 * names are a 400.
 */
final class ReasoningFormats {

    private ReasoningFormats() {
    }

    /**
     * The filter of a request body's {@code reasoning_format}, or {@code configured} when the
     * field is absent. {@code streamed} decides {@code deepseek-legacy}.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 for a name outside the four
     */
    static ReasoningFilter fromBody(JsonView body, ReasoningFilter configured, boolean streamed) {
        if (!body.has("reasoning_format")) {
            return configured;
        }
        return resolve(body.str("reasoning_format"), configured, streamed);
    }

    /**
     * The filter {@code chat} parses with: the configured filter's per-request selection for the request's chat
     * template variables ({@link ReasoningFilter#forRequest}), such as {@code hy_v3}'s {@code reasoning_effort} or
     * {@code minicpmv}'s {@code enable_thinking}.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 when a variable the parser reads has the wrong type
     */
    static ReasoningFilter forChat(ReasoningFilter configured, ChatRequest chat) {
        try {
            return configured.forRequest(chat.templateVariables());
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** The filter a {@code reasoning_format} name selects over {@code configured}. */
    static ReasoningFilter resolve(String format, ReasoningFilter configured, boolean streamed) {
        String name = format == null || format.trim().isEmpty() ? "auto" : format.trim();
        switch (name) {
            case "auto":
            case "deepseek":
                return configured;
            case "none":
                return configured == null ? ReasoningFilter.none() : configured.asRaw();
            case "deepseek-legacy":
                return streamed && configured != null ? configured.asRaw() : configured;
            default:
                throw ServerJson.badRequest("reasoning_format '" + format
                        + "' is not one of none, auto, deepseek, deepseek-legacy");
        }
    }
}
