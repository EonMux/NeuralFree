package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One validated Cohere {@code /v2/embed} request (vLLM's {@code CohereEmbedRequest}) and its translation into the
 * OpenAI embeddings body {@link PoolingInputs} reads: {@code texts} become {@code input} strings, {@code images}
 * and {@code inputs} become one user conversation each, {@code output_dimension} becomes {@code dimensions} and
 * {@code truncate} with {@code max_tokens} become {@code truncate_prompt_tokens} with its {@code truncation_side}
 * ({@code END} right, {@code START} left), as vLLM's {@code _resolve_cohere_truncation} maps them. Immutable.
 *
 * @param texts          the {@code texts}, or null when the request gives none
 * @param conversations  the user conversations of {@code images} or {@code inputs}, or null for {@code texts}
 * @param hasImage       whether any input holds an image
 * @param dimensions     {@code output_dimension}, or 0 for the whole embedding
 * @param embeddingTypes the requested types, {@code ["float"]} by default, without repeats
 * @param truncate       {@code END}, {@code START} or {@code NONE}
 * @param maxTokens      {@code max_tokens}, or 0 when absent
 */
record CohereRequest(List<String> texts, List<List<Object>> conversations, boolean hasImage, int dimensions,
                     List<String> embeddingTypes, String truncate, int maxTokens) {

    /**
     * The request of {@code body}.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming the field of a malformed or missing input
     */
    static CohereRequest of(JsonView body) {
        int given = (body.raw("texts") != null ? 1 : 0) + (body.raw("images") != null ? 1 : 0)
                + (body.raw("inputs") != null ? 1 : 0);
        if (given != 1) {
            throw ServerJson.badRequest("Exactly one of texts, images, or inputs must be provided, and it must be"
                    + " non-empty");
        }
        List<String> texts = null;
        List<List<Object>> conversations;
        boolean image;
        if (body.raw("texts") != null) {
            texts = List.copyOf(strings(body, "texts"));
            conversations = null;
            image = false;
        } else if (body.raw("images") != null) {
            conversations = new ArrayList<>();
            for (String url : strings(body, "images")) {
                conversations.add(user(List.of(imagePart(url))));
            }
            image = true;
        } else {
            conversations = inputs(body);
            image = conversations.stream().anyMatch(CohereRequest::holdsImage);
        }
        if (body.optI64("priority", 0) != 0) {
            throw ServerJson.badRequest("priority " + body.raw("priority") + " is not supported: pooling requests"
                    + " run in arrival order");
        }
        String truncate = truncate(body);
        int maxTokens = maxTokens(body);
        if (image && maxTokens > 0 && !truncate.equals("NONE")) {
            throw ServerJson.badRequest("max_tokens with truncate " + truncate + " cuts text prompts only; an"
                    + " image input keeps its rows (use truncate NONE to bound it)");
        }
        return new CohereRequest(texts, conversations == null ? null : List.copyOf(conversations), image,
                dimension(body), types(body), truncate, maxTokens);
    }

    /** The OpenAI embeddings body of this request, its numbers as the JSON parser types them ({@code Long}). */
    Map<String, Object> poolingBody() {
        Map<String, Object> out = new LinkedHashMap<>();
        if (texts != null) {
            out.put("input", texts);
        } else {
            out.put("messages", conversations);
        }
        if (dimensions > 0) {
            out.put("dimensions", (long) dimensions);
        }
        if (!truncate.equals("NONE") && (maxTokens > 0 || texts != null)) {
            out.put("truncate_prompt_tokens", maxTokens > 0 ? (long) maxTokens : -1L);
            out.put("truncation_side", truncate.equals("START") ? "left" : "right");
        }
        return out;
    }

    /**
     * Refuses an input of {@code tokens} prompt rows longer than {@code max_tokens} under {@code truncate NONE}
     * (vLLM's {@code _check_cohere_max_tokens}).
     */
    void checkLength(long tokens, String field) {
        if (truncate.equals("NONE") && maxTokens > 0 && tokens > maxTokens) {
            throw ServerJson.badRequest(field + ": Input of " + tokens + " tokens exceeds max_tokens=" + maxTokens
                    + " with truncate=NONE. Set truncate to END or START to allow truncation.");
        }
    }

    private static List<String> strings(JsonView body, String key) {
        if (!(body.raw(key) instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest(key + " must be a non-empty array of strings");
        }
        List<String> out = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            if (!(list.get(i) instanceof String s)) {
                throw ServerJson.badRequest(key + "[" + i + "] must be a string");
            }
            out.add(s);
        }
        return out;
    }

    /** The conversations of {@code inputs}: each {@code {"content": [parts]}} as one user message. */
    private static List<List<Object>> inputs(JsonView body) {
        if (!(body.raw("inputs") instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest("inputs must be a non-empty array of {\"content\": [...]} objects");
        }
        List<List<Object>> out = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            String at = "inputs[" + i + "]";
            if (!(list.get(i) instanceof Map<?, ?> input) || !(input.get("content") instanceof List<?> content)
                    || content.isEmpty()) {
                throw ServerJson.badRequest(at + " must be an object with a non-empty content array");
            }
            List<Object> parts = new ArrayList<>(content.size());
            for (int p = 0; p < content.size(); p++) {
                parts.add(part(content.get(p), at + ".content[" + p + "]"));
            }
            out.add(user(parts));
        }
        return out;
    }

    /** One {@code CohereEmbedContent} part as a chat content part: {@code text} or {@code image_url}. */
    private static Map<String, Object> part(Object raw, String at) {
        if (!(raw instanceof Map<?, ?> part)) {
            throw ServerJson.badRequest(at + " must be a content part object");
        }
        Object type = part.get("type");
        if ("text".equals(type)) {
            if (!(part.get("text") instanceof String text)) {
                throw ServerJson.badRequest(at + ": CohereEmbedContent with type='text' requires text");
            }
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("type", "text");
            out.put("text", text);
            return out;
        }
        if (ChatMessages.PART_IMAGE.equals(type)) {
            if (!(part.get(ChatMessages.PART_IMAGE) instanceof Map<?, ?> image)
                    || !(image.get("url") instanceof String url) || url.isEmpty()) {
                throw ServerJson.badRequest(at + ": CohereEmbedContent with type='image_url' requires"
                        + " image_url.url");
            }
            return imagePart(url);
        }
        throw ServerJson.badRequest(at + " has unsupported type '" + type + "': Cohere embed content takes text"
                + " and image_url parts");
    }

    private static Map<String, Object> imagePart(String url) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("type", ChatMessages.PART_IMAGE);
        out.put(ChatMessages.PART_IMAGE, Map.of("url", url));
        return out;
    }

    private static List<Object> user(List<Object> parts) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "user");
        message.put("content", parts);
        return List.of(message);
    }

    private static boolean holdsImage(List<Object> conversation) {
        for (Object message : conversation) {
            for (Object part : (List<?>) ((Map<?, ?>) message).get("content")) {
                if (ChatMessages.PART_IMAGE.equals(((Map<?, ?>) part).get("type"))) {
                    return true;
                }
            }
        }
        return false;
    }

    private static int dimension(JsonView body) {
        if (body.raw("output_dimension") == null) {
            return 0;
        }
        int n = body.i32("output_dimension");
        if (n < 1) {
            throw ServerJson.badRequest("output_dimension must be at least 1, got " + n);
        }
        return n;
    }

    private static List<String> types(JsonView body) {
        if (body.raw("embedding_types") == null) {
            return List.of("float");
        }
        if (!(body.raw("embedding_types") instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest("embedding_types must be a non-empty array of float, binary, ubinary and"
                    + " base64");
        }
        List<String> out = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            if (!(list.get(i) instanceof String type) || !CohereEmbedResource.EMBEDDING_TYPES.contains(type)) {
                throw ServerJson.badRequest("embedding_types[" + i + "] must be float, binary, ubinary or base64,"
                        + " got " + list.get(i));
            }
            if (!out.contains(type)) {
                out.add(type);
            }
        }
        return List.copyOf(out);
    }

    private static String truncate(JsonView body) {
        String mode = body.optStr("truncate", "END");
        if (!mode.equals("NONE") && !mode.equals("START") && !mode.equals("END")) {
            throw ServerJson.badRequest("truncate must be NONE, START or END, got '" + mode + "'");
        }
        return mode;
    }

    private static int maxTokens(JsonView body) {
        if (body.raw("max_tokens") == null) {
            return 0;
        }
        int n = body.i32("max_tokens");
        if (n < 1) {
            throw ServerJson.badRequest("max_tokens must be at least 1, got " + n);
        }
        return n;
    }
}
