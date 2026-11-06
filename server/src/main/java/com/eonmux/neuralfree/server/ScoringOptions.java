package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.Map;
import java.util.Set;

/**
 * The fields of vLLM's score and rerank requests beside the two sides (vllm/entrypoints/pooling/scoring): the
 * task {@code instruction} folded with {@code chat_template_kwargs} (the explicit kwarg wins, as vLLM's validator
 * merges one into the other), the per-side token caps {@code max_tokens_per_query} and {@code max_tokens_per_doc},
 * and the pair truncation ({@code truncate_prompt_tokens}, {@code truncation_side}). Parsing refuses by name: a
 * non-string instruction, the reserved renderer keys {@code chat_template}, {@code tools} and {@code tokenize},
 * and any kwarg but {@code instruction} (the only variable a score template takes here beside {@code query} and
 * {@code document}).
 *
 * @param instruction the task instruction, or null when the request carries none
 * @param maxQuery    the query's token cap, 0 for none
 * @param maxDoc      the document's token cap, 0 for none
 * @param truncation  the pair truncation ({@code truncate_prompt_tokens}, {@code truncation_side})
 */
record ScoringOptions(String instruction, int maxQuery, int maxDoc, EmbeddingOptions truncation) {

    /** The kwargs a score template renderer reserves, refused as vLLM refuses them. */
    private static final Set<String> RESERVED_KWARGS = Set.of("chat_template", "tools", "tokenize");

    /** The options of {@code body} for a model whose context holds at most {@code maxModelLength} tokens. */
    static ScoringOptions of(JsonView body, EngineService service) {
        return new ScoringOptions(instruction(body), maxTokens(body, service, "max_tokens_per_query"),
                maxTokens(body, service, "max_tokens_per_doc"),
                EmbeddingOptions.truncation(body, service.maxModelLength()));
    }

    /**
     * The task instruction: {@code chat_template_kwargs['instruction']} when the kwargs carry it, else the
     * top-level {@code instruction}.
     */
    private static String instruction(JsonView body) {
        String instruction = null;
        if (body.raw("chat_template_kwargs") != null) {
            if (!(body.raw("chat_template_kwargs") instanceof Map<?, ?> kwargs)) {
                throw ServerJson.badRequest("chat_template_kwargs must be an object");
            }
            for (Map.Entry<?, ?> kwarg : kwargs.entrySet()) {
                String key = String.valueOf(kwarg.getKey());
                if ("instruction".equals(key)) {
                    if (!(kwarg.getValue() instanceof String s)) {
                        throw ServerJson.badRequest("chat_template_kwargs.instruction must be a string");
                    }
                    instruction = s;
                } else if (RESERVED_KWARGS.contains(key)) {
                    throw ServerJson.badRequest("chat_template_kwargs contains reserved keys that conflict with"
                            + " fixed scorer arguments: ['" + key + "']");
                } else {
                    throw ServerJson.badRequest("chat_template_kwargs." + key + " is not a variable of the score"
                            + " template; a score template takes query, document and instruction");
                }
            }
        }
        if (body.raw("instruction") != null) {
            if (!(body.raw("instruction") instanceof String top)) {
                throw ServerJson.badRequest("instruction must be a string");
            }
            if (instruction == null) {
                instruction = top;
            }
        }
        return instruction;
    }

    /** A per-side cap: 0 keeps every token; a positive value stays under the model's context. */
    private static int maxTokens(JsonView body, EngineService service, String field) {
        if (body.raw(field) == null) {
            return 0;
        }
        int n = body.i32(field);
        if (n < 0) {
            throw ServerJson.badRequest(field + " must be 0 (no truncation) or positive, got " + n);
        }
        int max = service.maxModelLength();
        if (n >= max) {
            throw ServerJson.badRequest(field + " (" + n + ") must be less than max_model_len (" + max + ")");
        }
        return n;
    }
}
