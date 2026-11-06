package com.eonmux.neuralfree.server;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The input items of a stored response: each gets an id when the request gave none ({@code msg_} for a message,
 * {@code item_} otherwise), and {@code GET /v1/responses/{id}/input_items} pages them as an OpenAI list
 * ({@code data}, {@code first_id}, {@code last_id}, {@code has_more}). Stateless.
 */
final class ResponsesPages {

    private ResponsesPages() {
    }

    /** Copies of {@code items}, each with an {@code id}. */
    static List<Object> withIds(List<Object> items) {
        List<Object> out = new ArrayList<>();
        for (Object o : items) {
            if (o instanceof Map<?, ?> item) {
                Map<String, Object> copy = new LinkedHashMap<>(OpenAiRequests.stringKeys(item));
                if (!(copy.get("id") instanceof String)) {
                    boolean message = copy.get("role") != null || "message".equals(copy.get("type"));
                    copy.put("id", ResponsesEvents.newId(message ? "msg_" : "item_"));
                }
                out.add(copy);
            } else {
                out.add(o);
            }
        }
        return out;
    }

    /**
     * One page of {@code items}: newest first for {@code desc}, starting after the item {@code after} names.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 when {@code after} names no item
     */
    static Map<String, Object> page(List<Object> items, int limit, String order, String after) {
        List<Object> ordered = new ArrayList<>(items);
        if ("desc".equals(order)) {
            Collections.reverse(ordered);
        }
        int from = 0;
        if (after != null) {
            from = indexOf(ordered, after) + 1;
            if (from == 0) {
                throw ServerJson.badRequest("after names no input item: '" + after + "'");
            }
        }
        int to = Math.min(ordered.size(), from + limit);
        List<Object> data = List.copyOf(ordered.subList(from, to));
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("object", "list");
        out.put("data", data);
        out.put("first_id", data.isEmpty() ? null : idOf(data.get(0)));
        out.put("last_id", data.isEmpty() ? null : idOf(data.get(data.size() - 1)));
        out.put("has_more", to < ordered.size());
        return out;
    }

    private static int indexOf(List<Object> items, String id) {
        for (int i = 0; i < items.size(); i++) {
            if (id.equals(idOf(items.get(i)))) {
                return i;
            }
        }
        return -1;
    }

    private static Object idOf(Object item) {
        return item instanceof Map<?, ?> m ? m.get("id") : null;
    }
}
