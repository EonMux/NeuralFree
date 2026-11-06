package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The inputs of vLLM's score and rerank requests ({@code ScoreInput}, vllm/entrypoints/pooling/scoring): each
 * entry is a string, or a {@code ScoreMultiModalParam} {@code {"content": [...]}} of {@code text}, {@code
 * image_url}, {@code input_audio} and {@code audio_url} parts, and a side is one entry or an array of them. Text
 * parts join with newlines, as vLLM's content parser joins them. Parsing is pure: the entries carry the part
 * references, never their bytes; what a model accepts of them — a CLIP dual encoder takes a text-only or an
 * image-only entry, a projector model pools any combination — the serving routes decide.
 */
final class ScoreSides {

    /**
     * One score or rerank entry: its text parts joined (the string itself for a string entry, empty for a content
     * entry without text parts), the references of its image parts, whether it holds an audio part, the raw
     * content parts of a content entry (null for a string), and the request field it came from.
     *
     * @param text   the string, or the content entry's text parts joined with newlines
     * @param images the {@code image_url} references, in part order
     * @param audio  whether the content holds an {@code input_audio} or {@code audio_url} part
     * @param content the raw content parts, or null for a string entry
     * @param field  the request field the entry came from, named in every refusal
     */
    record Entry(String text, List<String> images, boolean audio, List<?> content, String field) {

        /** True when the entry is exactly one image and no text or audio (the dual-encoder image form). */
        boolean isImage() {
            return content != null && images.size() == 1 && !audio && text.isEmpty();
        }

        /** The reference of the entry's one image, or null when it is not of the image form. */
        String image() {
            return isImage() ? images.get(0) : null;
        }

        /** True when the entry holds an image or audio part. */
        boolean isMedia() {
            return !images.isEmpty() || audio;
        }
    }

    private ScoreSides() {
    }

    /**
     * The entries of a side value: one entry, or a non-empty array of entries.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming {@code field} for any other shape
     */
    static List<Entry> side(Object raw, String field) {
        Entry single = entry(raw, field);
        if (single != null) {
            return List.of(single);
        }
        if (raw instanceof List<?> list && !list.isEmpty()) {
            List<Entry> out = new ArrayList<>();
            for (int i = 0; i < list.size(); i++) {
                Entry at = entry(list.get(i), field + "[" + i + "]");
                if (at == null) {
                    throw ServerJson.badRequest(field + "[" + i + "] must be a string or an object with a content"
                            + " array");
                }
                out.add(at);
            }
            return out;
        }
        throw ServerJson.badRequest(field + " must be a string, an object with a content array, or a non-empty"
                + " array of those");
    }

    /**
     * The entry of one value: a string, an object with a {@code content} array, or (the Cohere document form
     * {@code {"text": ...}}) an object with a string {@code text}; null when the value has no entry shape.
     */
    static Entry entry(Object raw, String field) {
        if (raw instanceof String s) {
            return new Entry(s, List.of(), false, null, field);
        }
        if (raw instanceof Map<?, ?> map) {
            if (map.containsKey("content")) {
                return contentEntry(map.get("content"), field);
            }
            if (map.get("text") instanceof String s) {
                return new Entry(s, List.of(), false, null, field);
            }
        }
        return null;
    }

    /**
     * The entry of one content part list: text parts join with newlines, {@code image_url} parts collect their
     * references, {@code input_audio} and {@code audio_url} parts set the audio flag. The parts stay raw for the
     * media prompt path.
     */
    private static Entry contentEntry(Object raw, String field) {
        if (!(raw instanceof List<?> parts) || parts.isEmpty()) {
            throw ServerJson.badRequest(field + ".content must be a non-empty array of text, image_url, input_audio"
                    + " and audio_url parts");
        }
        List<String> texts = new ArrayList<>();
        List<String> images = new ArrayList<>();
        boolean audio = false;
        for (int i = 0; i < parts.size(); i++) {
            String at = field + ".content[" + i + "]";
            if (!(parts.get(i) instanceof Map<?, ?> part)) {
                throw ServerJson.badRequest(at + " must be an object with a type");
            }
            Object type = part.get("type");
            if (ChatMessages.PART_IMAGE.equals(type)) {
                images.add(imageUrl(part.get(ChatMessages.PART_IMAGE), at));
            } else if ("text".equals(type)) {
                if (!(part.get("text") instanceof String t)) {
                    throw ServerJson.badRequest(at + " is a text part without a string 'text'");
                }
                texts.add(t);
            } else if (ChatMessages.PART_AUDIO.equals(type) || ChatMessages.PART_AUDIO_URL.equals(type)) {
                audio = true;
            } else {
                throw ServerJson.badRequest(at + " has unsupported type '" + type + "': score and rerank inputs"
                        + " take text, image_url, input_audio and audio_url parts");
            }
        }
        return new Entry(String.join("\n", texts), List.copyOf(images), audio, parts, field);
    }

    /**
     * The CLIP dual encoder's entry forms: text-only, or exactly one image. A mixed, multi-image or audio entry
     * refuses with vLLM's CLIP messages.
     */
    static void requireClipForm(Entry entry) {
        if (entry.images.size() > 1) {
            throw ServerJson.badRequest(entry.field + " holds " + entry.images.size() + " image_url parts; at most"
                    + " 1 image may be provided per input");
        }
        if (entry.audio) {
            throw ServerJson.badRequest(entry.field + " holds an audio part, but the CLIP dual encoder reads"
                    + " images only");
        }
        if (!entry.images.isEmpty()) {
            throw ServerJson.badRequest(entry.field + ": CLIP accepts text-only or image-only inputs, not both");
        }
    }

    /**
     * The URL of an {@code image_url} value: the URL string itself, or an object with a string {@code url}.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming {@code field} when neither holds a URL
     */
    static String imageUrl(Object value, String field) {
        if (value instanceof String s && !s.isBlank()) {
            return s;
        }
        if (value instanceof Map<?, ?> m && m.get("url") instanceof String s && !s.isBlank()) {
            return s;
        }
        throw ServerJson.badRequest(field + " requires image_url.url");
    }

    /** The complete score-side pair of {@code body}, or a 400 naming the accepted forms. */
    static String[] scorePair(JsonView body) {
        String[][] pairs = {{"queries", "documents"}, {"queries", "items"}, {"data_1", "data_2"},
                {"text_1", "text_2"}};
        for (String[] pair : pairs) {
            if (body.has(pair[0]) && body.has(pair[1])) {
                return pair;
            }
        }
        throw ServerJson.badRequest("scoring takes two sides: queries and documents, queries and items, data_1"
                + " and data_2, or text_1 and text_2");
    }
}
