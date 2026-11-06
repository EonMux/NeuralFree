package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.media.MediaLimits;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The multimodal features a render endpoint answers with and a generate endpoint reads (vLLM's
 * {@code MultiModalFeatures}): per modality the item hashes, their placeholder row spans, and the serialized
 * processor payload that lets another server reproduce the items.
 *
 * <ul>
 *   <li>{@code mm_hashes}: per modality, the items' key handles ({@link MediaKey#toString()});
 *       {@code image:content:<hex>}, {@code image:uuid:<uuid>} or {@code embeddings:rows:<hex>}.</li>
 *   <li>{@code mm_placeholders}: per modality, one {@code {"offset", "length"}} row span per item, parallel to
 *       the hashes; spans are globally ordered and do not overlap.</li>
 *   <li>{@code kwargs_data}: per modality, one entry parallel to the hashes: the base64 media source of a
 *       content- or uuid-keyed item, the base64 rows of an embedding item, or {@code null} for an item the
 *       receiving server already holds in its media cache.</li>
 * </ul>
 *
 * The generate endpoint rebuilds the prompt from these features: text tokens outside the spans, and each item
 * encoded through its key, so the encoded-media cache serves an item the render step already encoded on the same
 * server and the serialized payload re-encodes it elsewhere.
 */
final class ScaleOutFeatures {

    /** The token id a media row occupies in a rendered prompt's token ids. */
    static final int PLACEHOLDER = 0;

    /** The feature fields this server reads; anything else refuses by name. */
    private static final Set<String> FIELDS = Set.of("mm_hashes", "mm_placeholders", "kwargs_data");

    /** The modality whose item payload is precomputed embedding rows rather than media bytes. */
    private static final String EMBEDDINGS = "embeddings";

    private ScaleOutFeatures() {
    }

    /** The features of a built media prompt's placements, or null for a text prompt. */
    static Map<String, Object> of(List<ChatMedia.Placement> placements) {
        if (placements.isEmpty()) {
            return null;
        }
        Map<String, List<Object>> hashes = new LinkedHashMap<>();
        Map<String, List<Object>> placeholders = new LinkedHashMap<>();
        Map<String, List<Object>> kwargs = new LinkedHashMap<>();
        boolean payload = false;
        for (ChatMedia.Placement placement : placements) {
            String modality = placement.key().modality();
            hashes.computeIfAbsent(modality, key -> new ArrayList<>()).add(placement.key().toString());
            Map<String, Object> range = new LinkedHashMap<>();
            range.put("offset", placement.offset());
            range.put("length", placement.length());
            placeholders.computeIfAbsent(modality, key -> new ArrayList<>()).add(range);
            String data = placement.item() != null ? encode(placement.item().bytes())
                    : encodeRows(placement.rows());
            payload |= data != null;
            kwargs.computeIfAbsent(modality, key -> new ArrayList<>()).add(data);
        }
        Map<String, Object> features = new LinkedHashMap<>();
        features.put("mm_hashes", hashes);
        features.put("mm_placeholders", placeholders);
        if (payload) {
            features.put("kwargs_data", kwargs);
        }
        return features;
    }

    /** The {@code mm_placeholders} map of a built prompt's placements, per modality in prompt order. */
    static Map<String, Object> placeholdersOf(List<ChatMedia.Placement> placements) {
        Map<String, List<Object>> placeholders = new LinkedHashMap<>();
        for (ChatMedia.Placement placement : placements) {
            Map<String, Object> range = new LinkedHashMap<>();
            range.put("offset", placement.offset());
            range.put("length", placement.length());
            placeholders.computeIfAbsent(placement.key().modality(), key -> new ArrayList<>()).add(range);
        }
        return new LinkedHashMap<>(placeholders);
    }

    /**
     * The items to splice into an expanded token sequence, in prompt-row order.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming a malformed field
     */
    static List<ChatMedia.Placed> placed(EngineService service, JsonView features, int tokenCount) {
        refuseFields(features);
        JsonView hashes = requireMap(features, "mm_hashes");
        JsonView placeholders = requireMap(features, "mm_placeholders");
        JsonView kwargs = features.raw("kwargs_data") == null ? null : features.obj("kwargs_data");
        Set<String> modalities = hashes.keys();
        if (modalities.isEmpty()) {
            throw ServerJson.badRequest("features.mm_hashes must name at least one modality");
        }
        if (!modalities.equals(placeholders.keys())) {
            throw ServerJson.badRequest("features.mm_hashes and features.mm_placeholders must use the same"
                    + " modalities, got " + modalities + " and " + placeholders.keys());
        }
        if (kwargs != null && !modalities.equals(kwargs.keys())) {
            throw ServerJson.badRequest("features.kwargs_data must use the same modalities as features.mm_hashes,"
                    + " got " + kwargs.keys());
        }
        List<ChatMedia.Placed> out = new ArrayList<>();
        for (String modality : modalities) {
            List<?> hashList = requireList(hashes, modality, "features.mm_hashes");
            List<?> rangeList = requireList(placeholders, modality, "features.mm_placeholders");
            List<?> payloads = kwargs == null ? null : requireList(kwargs, modality, "features.kwargs_data");
            if (hashList.size() != rangeList.size()) {
                throw ServerJson.badRequest("features.mm_hashes." + modality + " holds " + hashList.size()
                        + " item(s) for " + rangeList.size() + " placeholder(s) in features.mm_placeholders."
                        + modality);
            }
            if (payloads != null && payloads.size() != hashList.size()) {
                throw ServerJson.badRequest("features.kwargs_data." + modality + " holds " + payloads.size()
                        + " item(s) for " + hashList.size() + " hash(es)");
            }
            for (int i = 0; i < hashList.size(); i++) {
                String where = "features.mm_hashes." + modality + "[" + i + "]";
                String handle = hashList.get(i) instanceof String s && !s.isBlank() ? s
                        : refuse(where + " must be a non-blank media key handle");
                MediaKey key;
                try {
                    key = MediaKey.parse(handle);
                } catch (IllegalArgumentException e) {
                    throw ServerJson.badRequest(where + ": " + e.getMessage());
                }
                if (!key.modality().equals(modality)) {
                    throw ServerJson.badRequest(where + " names the modality '" + key.modality() + "', not '"
                            + modality + "'");
                }
                JsonView range = JsonView.of(rangeList.get(i), "features.mm_placeholders." + modality + "[" + i + "]");
                int offset = range.i32("offset");
                int length = range.i32("length");
                if (offset < 0 || length <= 0 || (long) offset + length > tokenCount) {
                    throw ServerJson.badRequest("features.mm_placeholders." + modality + "[" + i + "] must be a"
                            + " positive span inside the " + tokenCount + "-token prompt, got offset " + offset
                            + " and length " + length);
                }
                Object payload = payloads == null ? null : payloads.get(i);
                out.add(item(key, payload, offset, length, where));
            }
        }
        out.sort(Comparator.comparingInt(ChatMedia.Placed::offset));
        int cursor = 0;
        for (ChatMedia.Placed placed : out) {
            if (placed.offset() < cursor) {
                throw ServerJson.badRequest("features.mm_placeholders must be globally ordered and not overlap:"
                        + " an item starts at row " + placed.offset() + " while the previous one ends at "
                        + cursor);
            }
            cursor = placed.offset() + placed.length();
        }
        return out;
    }

    /**
     * The media items of {@code content_parts}, in order: each is an {@code image_url}, {@code audio_url} or
     * {@code video_url} object with a {@code url} and optional {@code uuid}, or an inline
     * {@code input_audio}/{@code input_video} object.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming a malformed part
     */
    static List<ChatMedia.Item> contentItems(EngineService service, List<?> parts) {
        List<ChatMedia.Item> out = new ArrayList<>(parts.size());
        for (int i = 0; i < parts.size(); i++) {
            String where = "content_parts[" + i + "]";
            if (!(parts.get(i) instanceof Map<?, ?> part)) {
                throw ServerJson.badRequest(where + " must be an object");
            }
            try {
                out.add(service.media().contentPart(part, where));
            } catch (IllegalArgumentException e) {
                throw ServerJson.badRequest(e.getMessage());
            }
        }
        return out;
    }

    /** The expanded token ids of a media prompt: its text tokens and one placeholder per media row. */
    static int[] expandedIds(MediaPrompt prompt) {
        int[] out = new int[prompt.nTokens()];
        int at = 0;
        for (MediaChunk chunk : prompt.chunks()) {
            if (chunk.isText()) {
                int[] tokens = chunk.tokens();
                System.arraycopy(tokens, 0, out, at, tokens.length);
                at += tokens.length;
            } else {
                for (int i = 0; i < chunk.nTokens(); i++) {
                    out[at++] = PLACEHOLDER;
                }
            }
        }
        return out;
    }

    /** One parsed feature entry: its key, the media item or precomputed rows, and its row span. */
    private static ChatMedia.Placed item(MediaKey key, Object payload, int offset, int length, String where) {
        if (key.source() == MediaKey.Source.ROWS) {
            if (payload == null) {
                throw ServerJson.badRequest(where + " is an embedding item and needs its rows in"
                        + " features.kwargs_data");
            }
            return new ChatMedia.Placed(key, null, decodeRows(payload, where), offset, length);
        }
        byte[] data = payload == null ? null : decoded(payload, where);
        MediaLimits.Modality modality = modality(key, where);
        ChatMedia.Item item = new ChatMedia.Item(modality, null, data == null ? null : () -> data, where);
        return new ChatMedia.Placed(key, item, null, offset, length);
    }

    /** The item modality a key handle names. */
    private static MediaLimits.Modality modality(MediaKey key, String where) {
        for (MediaLimits.Modality modality : MediaLimits.Modality.values()) {
            if (modality.key().equals(key.modality())) {
                return modality;
            }
        }
        throw ServerJson.badRequest(where + " names the modality '" + key.modality() + "', which this server"
                + " does not take");
    }

    /** A base64 payload string, decoded. */
    private static byte[] decoded(Object payload, String where) {
        return decoded(requireBase64(payload, where), where);
    }

    /** The base64 string of a payload. */
    private static String requireBase64(Object payload, String where) {
        if (!(payload instanceof String text) || text.isEmpty()) {
            throw ServerJson.badRequest(where + " must be a non-empty base64 string");
        }
        return text;
    }

    /** A base64 string, decoded. */
    private static byte[] decoded(String text, String where) {
        try {
            return Base64.getDecoder().decode(text);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(where + " is not valid base64: " + e.getMessage());
        }
    }

    /** {@code bytes} as base64, or null. */
    private static String encode(byte[] bytes) {
        return bytes == null ? null : Base64.getEncoder().encodeToString(bytes);
    }

    /** Embedding rows as base64: a row count, a width, then little-endian float32 values. */
    static String encodeRows(float[][] rows) {
        int width = rows.length == 0 ? 0 : rows[0].length;
        ByteBuffer buffer = ByteBuffer.allocate(8 + rows.length * width * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(rows.length).putInt(width);
        for (float[] row : rows) {
            for (float value : row) {
                buffer.putFloat(value);
            }
        }
        return Base64.getEncoder().encodeToString(buffer.array());
    }

    /** The rows {@link #encodeRows} serialized from a feature payload. */
    static float[][] decodeRows(Object payload, String where) {
        return decodeRows(requireBase64(payload, where), where);
    }

    /** The rows {@link #encodeRows} serialized. */
    static float[][] decodeRows(String data, String where) {
        byte[] bytes;
        try {
            bytes = Base64.getDecoder().decode(data);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(where + " is not valid base64: " + e.getMessage());
        }
        if (bytes.length < 8) {
            throw ServerJson.badRequest(where + " is too short for an embedding payload");
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int count = buffer.getInt();
        int width = buffer.getInt();
        if (count <= 0 || width <= 0 || bytes.length != 8 + (long) count * width * Float.BYTES) {
            throw ServerJson.badRequest(where + " describes " + count + " x " + width + " embedding values in "
                    + bytes.length + " bytes");
        }
        float[][] rows = new float[count][width];
        for (int r = 0; r < count; r++) {
            for (int c = 0; c < width; c++) {
                rows[r][c] = buffer.getFloat();
            }
        }
        return rows;
    }

    /** Refuses a feature field this server does not read, naming it. */
    private static void refuseFields(JsonView features) {
        for (String key : features.keys()) {
            if (!FIELDS.contains(key)) {
                throw ServerJson.badRequest("features." + key + " is not supported by this server");
            }
        }
    }

    /** A required object member of {@code features}. */
    private static JsonView requireMap(JsonView features, String key) {
        Object raw = features.raw(key);
        if (!(raw instanceof Map<?, ?>)) {
            throw ServerJson.badRequest("features." + key + " is required and must be an object");
        }
        return features.obj(key);
    }

    /** A required array member of a features map. */
    private static List<?> requireList(JsonView map, String key, String where) {
        Object raw = map.raw(key);
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest(where + "." + key + " must be a non-empty array");
        }
        return list;
    }

    /** Throws the 400 {@code message}; the call sites use it as an expression. */
    private static String refuse(String message) {
        throw ServerJson.badRequest(message);
    }
}
