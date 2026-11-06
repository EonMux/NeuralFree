package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Maps OpenAI-compatible request bodies onto the engine: the sampling settings (only present keys leave the
 * llama.cpp defaults), the number of choices, the grammar constraint in each of its spellings, the prompts of a
 * completion and the chat request with its tools. Every malformed field is a 400 that names it.
 */
final class OpenAiRequests {

    /** The most choices one request may ask for. */
    static final int MAX_CHOICES = 128;

    /** The most generations one request may queue: its prompts times {@code n}. */
    static final int MAX_GENERATIONS = 1024;

    /** The most alternatives a response lists per token (OpenAI's {@code top_logprobs} bound). */
    static final int MAX_TOP_LOGPROBS = 20;

    private OpenAiRequests() {
    }

    /** One completion prompt: text, or token ids. */
    record Prompt(String text, int[] tokens) { }

    /** {@code n}: the choices per prompt, in [1, {@link #MAX_CHOICES}]. */
    static int choices(JsonView body) {
        int n = body.optI32("n", 1);
        if (n < 1 || n > MAX_CHOICES) {
            throw ServerJson.badRequest("n must be in [1, " + MAX_CHOICES + "], got " + n);
        }
        return n;
    }

    /**
     * The alternatives a response lists per token, or -1 without logprobs: chat's boolean {@code logprobs} with
     * {@code top_logprobs}, the completions' integer {@code logprobs}, and llama.cpp's {@code n_probs}.
     */
    static int topLogprobs(JsonView body, boolean chat) {
        Object raw = body.raw("logprobs");
        int top = -1;
        if (Boolean.TRUE.equals(raw)) {
            top = body.optI32("top_logprobs", 0);
        } else if (!chat && (raw instanceof Long || raw instanceof Integer)) {
            top = ((Number) raw).intValue();
        } else if (raw != null && !(raw instanceof Boolean)) {
            throw ServerJson.badRequest("logprobs must be " + (chat ? "a boolean" : "an integer or a boolean"));
        } else if (chat && body.raw("top_logprobs") != null) {
            throw ServerJson.badRequest("top_logprobs needs logprobs: true");
        }
        int nProbs = body.optI32("n_probs", 0);
        if (nProbs > 0) {
            top = Math.max(top, nProbs);
        }
        if (top < -1 || top > MAX_TOP_LOGPROBS) {
            throw ServerJson.badRequest("the logprobs alternatives must be in [0, " + MAX_TOP_LOGPROBS + "], got "
                    + top);
        }
        return top;
    }

    /**
     * The generation settings of the body (the choices share them; {@link #forChoice} varies the seed). With
     * {@code topLogprobs} at 0 or above every token captures itself and {@code topLogprobs} alternatives.
     */
    static GenerationRequest sampling(JsonView body, int topLogprobs) {
        GenerationRequest.Builder builder = GenerationRequest.builder();
        int maxTokens = body.has("max_completion_tokens") ? body.optI32("max_completion_tokens", -1)
                : body.optI32("max_tokens", -1);
        if (maxTokens >= 0) {
            builder.maxNewTokens(maxTokens);
        }
        if (topLogprobs >= 0) {
            builder.logprobs(topLogprobs + 1);
        }
        builder.stopStrings(strings(body, "stop"));
        builder.badWords(strings(body, "bad_words"));
        builder.minTokens(Math.max(0, body.optI32("min_tokens", 0)));
        if (body.optBool("ignore_eos", false)) {
            builder.ignoreEos(true);
        }
        Map<Integer, Float> bias = logitBias(body);
        if (!bias.isEmpty()) {
            builder.logitBias(bias);
        }
        builder.sampler(SamplingParams.of(body));
        return builder.build();
    }

    /**
     * The request of choice {@code index}: with an explicit seed, choice {@code i} samples with {@code seed + i}
     * (vLLM's parallel sampling); without one every choice draws its own random seed.
     */
    static GenerationRequest forChoice(GenerationRequest base, int index) {
        long seed = base.sampler().seed();
        if (index == 0 || seed == SamplerConfig.RANDOM_SEED) {
            return base;
        }
        return base.toBuilder().sampler(base.sampler().withSeed(seed + index)).build();
    }

    private static Map<Integer, Float> logitBias(JsonView body) {
        Map<Integer, Float> out = new LinkedHashMap<>();
        if (!body.has("logit_bias")) {
            return out;
        }
        if (!(body.raw("logit_bias") instanceof Map<?, ?> map)) {
            throw ServerJson.badRequest("logit_bias must be an object of token id to bias");
        }
        for (Map.Entry<?, ?> e : map.entrySet()) {
            int id;
            try {
                id = Integer.parseInt(String.valueOf(e.getKey()).trim());
            } catch (NumberFormatException parse) {
                throw ServerJson.badRequest("logit_bias key '" + e.getKey() + "' is not a token id");
            }
            if (id < 0) {
                throw ServerJson.badRequest("logit_bias token id " + id + " is negative");
            }
            double bias = e.getValue() instanceof Number num ? num.doubleValue() : Double.NaN;
            if (!Double.isFinite(bias) || bias < -100.0 || bias > 100.0) {
                throw ServerJson.badRequest("logit_bias value of token " + id + " must be a number in [-100, 100]");
            }
            out.put(id, (float) bias);
        }
        return out;
    }

    private static List<String> strings(JsonView body, String key) {
        List<String> out = new ArrayList<>();
        Object raw = body.raw(key);
        if (raw instanceof String str) {
            out.add(str);
        } else if (raw instanceof List<?> list) {
            for (Object o : list) {
                if (!(o instanceof String str)) {
                    throw ServerJson.badRequest(key + " must be a string or an array of strings");
                }
                out.add(str);
            }
        } else if (raw != null) {
            throw ServerJson.badRequest(key + " must be a string or an array of strings");
        }
        return out;
    }

    // ------------------------------------------------------------------ constraints

    /**
     * The grammar constraint of the body, or null: llama.cpp's {@code grammar} and {@code json_schema}, the vLLM
     * {@code guided_*} fields and {@code structured_outputs}, and OpenAI's {@code response_format}. Two of them
     * at once are refused.
     */
    static ConstraintSpec constraint(JsonView body) {
        List<String> given = new ArrayList<>();
        ConstraintSpec spec = null;
        for (String key : new String[]{"grammar", "json_schema", "guided_json", "guided_regex", "guided_choice",
                "guided_grammar", "guided_structural_tag", "structural_tag", "response_format",
                "structured_outputs"}) {
            if (!body.has(key) || body.raw(key) == null) {
                continue;
            }
            ConstraintSpec c = constraintOf(body, key);
            if (c != null) {
                given.add(key);
                spec = c;
            }
        }
        if (given.size() > 1) {
            throw ServerJson.badRequest("only one output constraint may be given, got " + given);
        }
        return spec;
    }

    private static ConstraintSpec constraintOf(JsonView body, String key) {
        switch (key) {
            case "grammar":
            case "guided_grammar":
                return ConstraintSpec.gbnf(body.str(key));
            case "json_schema":
            case "guided_json":
                return ConstraintSpec.jsonSchema(jsonText(body.raw(key), key), null);
            case "guided_regex":
                return ConstraintSpec.regex(body.str(key), false);
            case "guided_choice":
                return ConstraintSpec.choice(stringList(body.raw(key), key));
            case "guided_structural_tag":
            case "structural_tag":
                return ConstraintSpec.structuralTag(jsonText(body.raw(key), key));
            case "response_format":
                return responseFormat(body.obj(key));
            default:
                return structuredOutputs(body.obj(key));
        }
    }

    /** OpenAI's {@code response_format}: {@code text}, {@code json_object} or {@code json_schema}. */
    private static ConstraintSpec responseFormat(JsonView format) {
        String type = format.optStr("type", "text");
        switch (type) {
            case "text":
                return null;
            case "json_object":
                return format.has("schema") ? ConstraintSpec.jsonSchema(jsonText(format.raw("schema"),
                        "response_format.schema"), null) : ConstraintSpec.jsonObject(null);
            case "json_schema": {
                JsonView spec = format.optObj("json_schema");
                Object schema = spec.has("schema") ? spec.raw("schema") : format.raw("schema");
                return schema == null ? ConstraintSpec.jsonObject(null)
                        : ConstraintSpec.jsonSchema(jsonText(schema, "response_format.json_schema.schema"), null);
            }
            default:
                throw ServerJson.badRequest("response_format type '" + type
                        + "' is not one of text, json_object, json_schema");
        }
    }

    /** vLLM's {@code structured_outputs}: exactly one of json, regex, choice, grammar, json_object, structural_tag. */
    private static ConstraintSpec structuredOutputs(JsonView so) {
        List<ConstraintSpec> specs = new ArrayList<>();
        if (so.has("json")) {
            specs.add(ConstraintSpec.jsonSchema(jsonText(so.raw("json"), "structured_outputs.json"), null));
        }
        if (so.has("regex")) {
            specs.add(ConstraintSpec.regex(so.str("regex"), false));
        }
        if (so.has("choice")) {
            specs.add(ConstraintSpec.choice(stringList(so.raw("choice"), "structured_outputs.choice")));
        }
        if (so.has("grammar")) {
            specs.add(ConstraintSpec.gbnf(so.str("grammar")));
        }
        if (so.optBool("json_object", false)) {
            specs.add(ConstraintSpec.jsonObject(null));
        }
        if (so.has("structural_tag")) {
            specs.add(ConstraintSpec.structuralTag(jsonText(so.raw("structural_tag"),
                    "structured_outputs.structural_tag")));
        }
        if (specs.size() > 1) {
            throw ServerJson.badRequest("structured_outputs takes exactly one constraint kind");
        }
        return specs.isEmpty() ? null : specs.get(0);
    }

    /** A JSON field given as an object or as JSON text, as text. */
    private static String jsonText(Object raw, String key) {
        if (raw instanceof String s) {
            return s;
        }
        if (raw instanceof Map<?, ?> || raw instanceof List<?>) {
            return ServerJson.write(raw);
        }
        throw ServerJson.badRequest(key + " must be a JSON object or JSON text");
    }

    private static List<String> stringList(Object raw, String key) {
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest(key + " must be a non-empty array of strings");
        }
        List<String> out = new ArrayList<>();
        for (Object o : list) {
            if (!(o instanceof String s)) {
                throw ServerJson.badRequest(key + " must be a non-empty array of strings");
            }
            out.add(s);
        }
        return out;
    }

    // ------------------------------------------------------------------ prompts

    /**
     * The prompts of a completion: a string, an array of strings, an array of token ids, or an array of token-id
     * arrays (OpenAI's four {@code prompt} forms).
     */
    static List<Prompt> prompts(JsonView body, int vocabSize) {
        Object raw = body.raw("prompt");
        List<Prompt> out = new ArrayList<>();
        if (raw instanceof String s) {
            out.add(new Prompt(s, null));
        } else if (raw instanceof List<?> list && !list.isEmpty()) {
            if (list.get(0) instanceof Number) {
                out.add(new Prompt(null, tokens(list, vocabSize, "prompt")));
            } else {
                for (int i = 0; i < list.size(); i++) {
                    Object item = list.get(i);
                    if (item instanceof String s) {
                        out.add(new Prompt(s, null));
                    } else if (item instanceof List<?> ids && !ids.isEmpty()) {
                        out.add(new Prompt(null, tokens(ids, vocabSize, "prompt[" + i + "]")));
                    } else {
                        throw ServerJson.badRequest("prompt[" + i + "] must be a string or a non-empty array of"
                                + " token ids");
                    }
                }
            }
        } else {
            throw ServerJson.badRequest("prompt is required: a string, an array of strings, an array of token ids"
                    + " or an array of token-id arrays");
        }
        return out;
    }

    /** Refuses a request whose prompts times {@code n} exceed {@link #MAX_GENERATIONS}. */
    static void requireGenerationBudget(int prompts, int n) {
        if ((long) prompts * n > MAX_GENERATIONS) {
            throw ServerJson.badRequest(prompts + " prompts with n = " + n + " ask for more than "
                    + MAX_GENERATIONS + " generations in one request");
        }
    }

    /** Token ids in [0, vocabSize). */
    static int[] tokens(List<?> list, int vocabSize, String key) {
        int[] ids = new int[list.size()];
        for (int i = 0; i < ids.length; i++) {
            Object o = list.get(i);
            if (!(o instanceof Long || o instanceof Integer)) {
                throw ServerJson.badRequest(key + " must hold integer token ids");
            }
            long id = ((Number) o).longValue();
            if (id < 0 || id >= vocabSize) {
                throw ServerJson.badRequest(key + " token id " + id + " is outside the vocabulary [0, " + vocabSize
                        + ")");
            }
            ids[i] = (int) id;
        }
        return ids;
    }

    // ------------------------------------------------------------------ chat

    /** The chat request of the body: the messages, the template knobs and the tools with their choice. */
    static ChatRequest chat(JsonView body) {
        if (!(body.raw("messages") instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest("messages is required and must be a non-empty array");
        }
        ChatRequest.Builder chat = ChatRequest.builder();
        for (Object o : list) {
            if (!(o instanceof Map<?, ?> message)) {
                throw ServerJson.badRequest("every message must be an object with role and content");
            }
            chat.addMessage(stringKeys(message));
        }
        if (body.has("add_generation_prompt")) {
            chat.addGenerationPrompt(body.optBool("add_generation_prompt", true));
        }
        if (body.has("continue_final_message")) {
            chat.continueFinalMessage(body.optBool("continue_final_message", false));
        }
        if (body.has("enable_thinking")) {
            chat.enableThinking(body.optBool("enable_thinking", false));
        }
        if (body.has("reasoning_effort")) {
            chat.reasoningEffort(body.str("reasoning_effort"));
        }
        if (body.raw("chat_template_kwargs") instanceof Map<?, ?> kwargs) {
            for (Map.Entry<?, ?> e : kwargs.entrySet()) {
                chat.templateKwarg(String.valueOf(e.getKey()), e.getValue());
            }
        }
        List<ChatTool> tools = tools(body);
        chat.tools(tools);
        ChatRequest.ToolChoice choice = toolChoice(body);
        if (choice != null) {
            if (tools.isEmpty() && choice.kind() != ChatRequest.ToolChoice.Kind.NONE
                    && choice.kind() != ChatRequest.ToolChoice.Kind.AUTO) {
                throw ServerJson.badRequest("tool_choice " + choice + " needs tools");
            }
            chat.toolChoice(choice);
        }
        try {
            return chat.build();
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** {@code parallel_tool_calls} (default true). */
    static boolean parallelToolCalls(JsonView body) {
        return body.optBool("parallel_tool_calls", true);
    }

    private static List<ChatTool> tools(JsonView body) {
        List<ChatTool> out = new ArrayList<>();
        Object raw = body.raw("tools");
        if (raw == null) {
            return out;
        }
        if (!(raw instanceof List<?> list)) {
            throw ServerJson.badRequest("tools must be an array");
        }
        for (int i = 0; i < list.size(); i++) {
            if (!(list.get(i) instanceof Map<?, ?> tool) || !"function".equals(tool.get("type"))
                    || !(tool.get("function") instanceof Map<?, ?> fn)) {
                throw ServerJson.badRequest("tools[" + i + "] must be {\"type\": \"function\", \"function\": {...}}");
            }
            out.add(tool(stringKeys(fn), "tools[" + i + "].function", "parameters"));
        }
        return out;
    }

    /** A function declaration: name, optional description, optional parameter schema under {@code schemaKey}. */
    static ChatTool tool(Map<String, Object> fn, String where, String schemaKey) {
        if (!(fn.get("name") instanceof String name) || name.isEmpty()) {
            throw ServerJson.badRequest(where + ".name must be a non-empty string");
        }
        ChatTool.Builder b = ChatTool.builder(name);
        if (fn.get("description") instanceof String d) {
            b.description(d);
        }
        Object params = fn.get(schemaKey);
        if (params instanceof Map<?, ?> p) {
            b.parameters(stringKeys(p));
        } else if (params != null) {
            throw ServerJson.badRequest(where + "." + schemaKey + " must be a JSON schema object");
        }
        try {
            return b.build();
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(where + ": " + e.getMessage());
        }
    }

    private static ChatRequest.ToolChoice toolChoice(JsonView body) {
        Object raw = body.raw("tool_choice");
        if (raw == null) {
            return null;
        }
        if (raw instanceof String s) {
            try {
                return ChatRequest.ToolChoice.fromOpenAi(s);
            } catch (IllegalArgumentException e) {
                throw ServerJson.badRequest(e.getMessage());
            }
        }
        if (raw instanceof Map<?, ?> m && m.get("function") instanceof Map<?, ?> fn
                && fn.get("name") instanceof String name && !name.isEmpty()) {
            return ChatRequest.ToolChoice.function(name);
        }
        throw ServerJson.badRequest("tool_choice must be none, auto, required or"
                + " {\"type\": \"function\", \"function\": {\"name\": ...}}");
    }

    /** A JSON object with its keys as strings. */
    static Map<String, Object> stringKeys(Map<?, ?> map) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : map.entrySet()) {
            out.put(String.valueOf(e.getKey()), e.getValue());
        }
        return out;
    }
}
