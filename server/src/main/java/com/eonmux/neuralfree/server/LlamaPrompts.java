package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * The prompts of a llama.cpp native completion ({@code tokenize_input_prompts}, {@code tools/server/server-common.cpp}):
 * a string; an array of token ids; an array mixing strings and token ids (one prompt, the strings tokenized in
 * place, BOS only before a leading string); an object with {@code prompt_string}; or, when an array holds no token id
 * at all, an array of any of these, one prompt each. On the generation routes a {@code prompt_string} object may
 * carry {@code multimodal_data}: base64 images and audio clips the projector encodes at the prompt's media markers
 * ({@link ChatMessages#MEDIA_MARKER}, the {@code media_marker} of {@code /props}, or mtmd's default
 * {@code <__media__>}), one marker per entry in order, the prompt starting with BOS when the vocabulary adds one
 * ({@code process_mtmd_prompt}). Token-only routes refuse {@code multimodal_data} by name.
 */
final class LlamaPrompts {

    /** mtmd's default media marker ({@code mtmd_default_marker}), accepted beside the server's own. */
    static final String MTMD_DEFAULT_MARKER = "<__media__>";

    /**
     * One generation prompt: its token ids, and the media prompt when it carries {@code multimodal_data} (the
     * tokens are then the media prompt's text tokens).
     */
    record Prompt(int[] tokens, MediaPrompt media) {

        /** The KV rows the prompt fills: its tokens, or the media prompt's tokens and media rows. */
        int rows() {
            return media == null ? tokens.length : media.nTokens();
        }
    }

    private LlamaPrompts() {
    }

    /** The generation prompts of {@code raw}: token prompts, or media prompts of {@code multimodal_data}. */
    static List<Prompt> generation(Object raw, EngineService service) {
        return generation(raw, service, () -> null);
    }

    /**
     * {@link #generation(Object, EngineService)} for a request naming adapters {@code mediaLoras}: its media
     * encodes in a LoRA session over the projector and its prompt chunks key by the adapters' identity.
     */
    static List<Prompt> generation(Object raw, EngineService service, Supplier<LoraSlotKey> mediaLoras) {
        if (raw == null) {
            throw ServerJson.badRequest("prompt is required");
        }
        List<Prompt> out = new ArrayList<>();
        if (raw instanceof List<?> list && !containsTokenId(list)) {
            if (list.isEmpty()) {
                throw ServerJson.badRequest("prompt must not be an empty array");
            }
            for (int i = 0; i < list.size(); i++) {
                out.add(generationPrompt(list.get(i), service, "prompt[" + i + "]", mediaLoras));
            }
        } else {
            out.add(generationPrompt(raw, service, "prompt", mediaLoras));
        }
        return out;
    }

    private static Prompt generationPrompt(Object raw, EngineService service, String key,
                                           Supplier<LoraSlotKey> mediaLoras) {
        if (raw instanceof Map<?, ?> map && map.get("prompt_string") instanceof String text
                && map.get("multimodal_data") != null) {
            MediaPrompt media = mediaPrompt(text, map.get("multimodal_data"), service, key, mediaLoras);
            return new Prompt(media.samplerTokens(), media);
        }
        return new Prompt(nonEmpty(subprompt(raw, service, key), key), null);
    }

    /**
     * The media prompt of a {@code prompt_string} with its {@code multimodal_data} entries: BOS when the model adds
     * it, then the text segments between media markers ({@link #MTMD_DEFAULT_MARKER} or
     * {@link ChatMessages#MEDIA_MARKER}) with each entry's media where its marker sits.
     *
     * @throws jakarta.ws.rs.WebApplicationException a 400 without a projector, for malformed entries, or markers
     *         that do not pair with the entries
     */
    static MediaPrompt mediaPrompt(String text, Object data, EngineService service, String key) {
        return mediaPromptRendered(text, data, service, key).prompt();
    }

    /**
     * {@link #mediaPrompt(String, Object, EngineService, String)} for a request naming adapters
     * {@code mediaLoras}: the entries encode under the adapters' identity and in a LoRA session over the projector.
     */
    static MediaPrompt mediaPrompt(String text, Object data, EngineService service, String key,
                                   Supplier<LoraSlotKey> mediaLoras) {
        return mediaPromptRendered(text, data, service, key, mediaLoras).prompt();
    }

    /**
     * {@link #mediaPrompt} with the row spans and keys of the spliced items, for a render endpoint that sends
     * the prompt to another server.
     *
     * @throws jakarta.ws.rs.WebApplicationException a 400 without a projector, for malformed entries, or markers
     *         that do not pair with the entries
     */
    static ChatMedia.Rendered mediaPromptRendered(String text, Object data, EngineService service, String key) {
        return mediaPromptRendered(text, data, service, key, () -> null);
    }

    /**
     * {@link #mediaPromptRendered(String, Object, EngineService, String)} for a request naming adapters
     * {@code mediaLoras}: the entries encode under the adapters' identity and in a LoRA session over the projector.
     */
    static ChatMedia.Rendered mediaPromptRendered(String text, Object data, EngineService service, String key,
                                                  Supplier<LoraSlotKey> mediaLoras) {
        String field = key + ".multimodal_data";
        if (!(data instanceof List<?> entries) || entries.isEmpty()) {
            throw ServerJson.badRequest(field + " must be a non-empty array of base64 strings");
        }
        if (!service.media().enabled()) {
            throw ServerJson.badRequest("Multimodal data provided, but model does not support multimodal requests."
                    + " Start the server with --mmproj");
        }
        try {
            List<ChatMedia.Item> items = service.media().items(entries, field);
            String rendered = text.replace(MTMD_DEFAULT_MARKER, ChatMessages.MEDIA_MARKER);
            return service.media().promptWithPlacements(rendered, items, List.of(), service.addedBos(),
                    segment -> service.encode(segment, false, true), mediaLoras);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
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
            if (map.get("multimodal_data") != null) {
                throw ServerJson.badRequest(key + ".multimodal_data is refused on this route: it takes token"
                        + " prompts only; send media to /completion or the chat routes");
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
