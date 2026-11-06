package com.eonmux.neuralfree.server;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The prompts of a llama.cpp native completion ({@code tokenize_input_prompts}, {@code tools/server/server-common.cpp}):
 * a string; an array of token ids; an array mixing strings and token ids (one prompt, the strings tokenized in
 * place, BOS only before a leading string); an object with {@code prompt_string}; or, when an array holds no token id
 * at all, an array of any of these, one prompt each. Multimodal data refuses: the server's models are text-only.
 */
final class LlamaPrompts {

    private LlamaPrompts() {
    }

    /** The prompts of {@code raw} as tokens; each encodes to at least one token. */
    static List<int[]> of(Object raw, EngineService service) {
        if (raw == null) {
            throw ServerJson.badRequest("prompt is required");
        }
        List<int[]> out = new ArrayList<>();
        if (raw instanceof List<?> list && !containsTokenId(list)) {
            if (list.isEmpty()) {
                throw ServerJson.badRequest("prompt must not be an empty array");
            }
            for (int i = 0; i < list.size(); i++) {
                out.add(nonEmpty(subprompt(list.get(i), service, "prompt[" + i + "]"), "prompt[" + i + "]"));
            }
        } else {
            out.add(nonEmpty(subprompt(raw, service, "prompt"), "prompt"));
        }
        return out;
    }

    /**
     * The tokens of a string or of an array mixing strings and token ids ({@code tokenize_mixed}), as llama.cpp's
     * infill reads {@code input_prefix} and {@code input_suffix}.
     */
    static int[] text(Object raw, EngineService service, boolean addSpecial, boolean parseSpecial, String key) {
        if (raw instanceof String text) {
            return service.encode(text, addSpecial, parseSpecial);
        }
        if (raw instanceof List<?> list) {
            return mixed(list, service, key, addSpecial, parseSpecial);
        }
        throw ServerJson.badRequest(key + " must be a string or an array of token ids and strings");
    }

    private static int[] subprompt(Object raw, EngineService service, String key) {
        if (raw instanceof String text) {
            return service.encode(text, true, true);
        }
        if (raw instanceof List<?> list) {
            return mixed(list, service, key, true, true);
        }
        if (raw instanceof Map<?, ?> map && map.get("prompt_string") instanceof String text) {
            if (map.containsKey("multimodal_data")) {
                throw ServerJson.badRequest(key + ".multimodal_data: the model takes no multimodal input");
            }
            return service.encode(text, true, true);
        }
        throw ServerJson.badRequest(key + " must be a string, an array of token ids and strings, or an object with"
                + " prompt_string");
    }

    /** {@code tokenize_mixed}: strings tokenized in place, BOS only when the first element is a string. */
    private static int[] mixed(List<?> list, EngineService service, String key, boolean addSpecial,
                               boolean parseSpecial) {
        List<Integer> ids = new ArrayList<>();
        int vocab = service.vocabSize();
        for (int i = 0; i < list.size(); i++) {
            Object o = list.get(i);
            if (o instanceof String text) {
                for (int id : service.encode(text, addSpecial && i == 0, parseSpecial)) {
                    ids.add(id);
                }
            } else if (o instanceof Long || o instanceof Integer) {
                long id = ((Number) o).longValue();
                if (id < 0 || id >= vocab) {
                    throw ServerJson.badRequest(key + "[" + i + "] token id " + id + " is outside the vocabulary");
                }
                ids.add((int) id);
            } else {
                throw ServerJson.badRequest(key + "[" + i + "] must be a token id or a string");
            }
        }
        int[] out = new int[ids.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ids.get(i);
        }
        return out;
    }

    private static boolean containsTokenId(List<?> list) {
        for (Object o : list) {
            if (o instanceof Long || o instanceof Integer) {
                return true;
            }
        }
        return false;
    }

    private static int[] nonEmpty(int[] tokens, String key) {
        if (tokens.length == 0) {
            throw ServerJson.badRequest(key + " encodes to no tokens");
        }
        return tokens;
    }
}
