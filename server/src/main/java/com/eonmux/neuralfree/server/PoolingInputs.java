package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The inputs of a pooling request, one token array each. The completion forms of {@code input} are OpenAI's: a
 * string, strings, token ids or token-id arrays, text getting the vocabulary's special tokens unless
 * {@code add_special_tokens} is false. The chat forms are vLLM's chat-message embedding input: a {@code
 * messages} conversation, batched {@code messages}, or {@code input} holding one conversation or batched
 * conversations (a conversation is a non-empty array of objects with a string {@code role}), each rendered
 * through the model's chat template (or the request's {@code chat_template} on a server that trusts it) and
 * tokenized with special tokens parsed and none added, as every chat rendering here does. A conversation holding
 * an {@code image_url} part is an image input instead (vLLM's dual-encoder pooling): a CLIP dual encoder embeds it
 * through its vision tower, so the conversation holds exactly one image and no text, as vLLM's CLIP processor
 * requires. On the other models a conversation with image or audio parts ({@code image_url}, {@code
 * input_audio}, {@code audio_url}) is a media input when the server carries a projector
 * ({@code neuralfree.server.mmproj}): it renders through the template with media markers and pools as one media
 * prompt; without a projector it refuses by name.
 */
final class PoolingInputs {

    /** The template knobs a chat-form request carries beside its conversations ({@code OpenAiRequests.chat}). */
    private static final String[] CHAT_KNOBS = {"add_generation_prompt", "continue_final_message",
            "enable_thinking", "reasoning_effort", "chat_template", "chat_template_kwargs"};

    private PoolingInputs() {
    }

    /**
     * One pooling input: its tokens, the reference of its one image (the dual-encoder form), the conversation of
     * its media prompt (the projector form), and the request field it came from.
     */
    record Input(int[] tokens, String image, ChatRequest chat, String field) {

        /** True when the input is an image. */
        boolean isImage() {
            return image != null;
        }

        /** True when the input is a conversation whose media parts pool as one media prompt. */
        boolean isMedia() {
            return chat != null;
        }
    }

    /**
     * The token arrays of {@code body}'s inputs, for the tasks that read text only: an image or media input
     * refuses by name, naming {@code task}.
     */
    static List<int[]> tokens(JsonView body, EngineService service, EmbeddingOptions options, String task) {
        List<int[]> out = new ArrayList<>();
        for (Input input : of(body, service, options)) {
            if (input.isImage()) {
                throw ServerJson.badRequest(input.field() + " holds an image; " + task + " reads text inputs only");
            }
            if (input.isMedia()) {
                throw ServerJson.badRequest(input.field() + " holds an image or audio part; " + task
                        + " reads text inputs only");
            }
            out.add(input.tokens());
        }
        return out;
    }

    /** The inputs of {@code body} over {@code service}, text shaped by {@code options}. */
    static List<Input> of(JsonView body, EngineService service, EmbeddingOptions options) {
        Object messages = body.raw("messages");
        if (messages != null) {
            return conversations(messages, body, service, options, "messages");
        }
        Object input = body.raw("input");
        if (input == null) {
            throw ServerJson.badRequest("input is required: a string, an array of strings, an array of token ids"
                    + ", an array of token-id arrays or chat messages");
        }
        if (isConversation(input)) {
            return List.of(conversation((List<?>) input, body, service, options, "input"));
        }
        if (isBatch(input)) {
            List<?> batch = (List<?>) input;
            List<Input> out = new ArrayList<>();
            for (int i = 0; i < batch.size(); i++) {
                Object entry = batch.get(i);
                if (!isConversation(entry)) {
                    throw ServerJson.badRequest("input[" + i + "] must be an array of chat messages");
                }
                out.add(conversation((List<?>) entry, body, service, options, "input[" + i + "]"));
            }
            return out;
        }
        return completion(input, service, options);
    }

    /** The conversations of a {@code messages} value: one conversation or an array of them. */
    private static List<Input> conversations(Object messages, JsonView body, EngineService service,
                                             EmbeddingOptions options, String field) {
        if (isConversation(messages)) {
            return List.of(conversation((List<?>) messages, body, service, options, field));
        }
        if (isBatch(messages)) {
            List<?> batch = (List<?>) messages;
            List<Input> out = new ArrayList<>();
            for (int i = 0; i < batch.size(); i++) {
                out.add(conversation((List<?>) batch.get(i), body, service, options, field + "[" + i + "]"));
            }
            return out;
        }
        throw ServerJson.badRequest(field + " must be an array of chat messages, or an array of those arrays");
    }

    /**
     * One conversation: its image when it holds the one {@code image_url} part a dual encoder takes, its media
     * prompt when its image or audio parts pool through the projector, else its rendering through the model's chat
     * template, tokenized.
     */
    private static Input conversation(List<?> messages, JsonView body, EngineService service,
                                      EmbeddingOptions options, String field) {
        MediaParts parts = mediaParts(messages, field);
        if (parts.images.size() == 1 && !parts.audio && !parts.text) {
            if (!service.dualEncoder()) {
                throw ServerJson.badRequest(field + " holds an image_url part, but " + service.architecture()
                        + " embeds text only; image embeddings need a CLIP dual-encoder model (a two-tower clip"
                        + " GGUF) or a media projector (" + MediaOptions.MMPROJ_PROPERTY + ")");
            }
            return new Input(null, parts.images.get(0), null, field);
        }
        if (service.dualEncoder()) {
            if (parts.audio) {
                throw ServerJson.badRequest(field + " holds an audio part, but the CLIP dual encoder reads images"
                        + " only");
            }
            if (parts.images.size() > 1) {
                throw ServerJson.badRequest(field + " holds " + parts.images.size() + " image_url parts; at most 1"
                        + " image may be provided per input");
            }
            if (!parts.images.isEmpty()) {
                throw ServerJson.badRequest(field + ": CLIP accepts text-only or image-only inputs, not both");
            }
            throw ServerJson.badRequest(field + ": the CLIP dual encoder has no chat template to render text"
                    + " messages; send text through input, an image as a conversation of one image_url part");
        }
        ChatRequest request = chatRequest(messages, body, service, field);
        if (parts.hasMedia()) {
            if (!service.media().enabled()) {
                throw ServerJson.badRequest(field + " holds an image or audio part, but " + service.architecture()
                        + " embeds text only; media embeddings need a media projector (" + MediaOptions.MMPROJ_PROPERTY
                        + ") or a CLIP dual-encoder model");
            }
            if (options.truncate() != 0) {
                throw ServerJson.badRequest("truncate_prompt_tokens cannot be combined with media inputs:"
                        + " the prompt holds image and audio rows truncation cannot cut");
            }
            return new Input(null, null, request, field);
        }
        return new Input(nonEmpty(options.truncated(service.encode(service.formatChat(request), false, true))),
                null, null, field);
    }

    /** The chat request of one conversation: its messages and the template knobs the body carries. */
    private static ChatRequest chatRequest(List<?> messages, JsonView body, EngineService service, String field) {
        Map<String, Object> chat = new LinkedHashMap<>();
        chat.put("messages", new ArrayList<>(messages));
        for (String knob : CHAT_KNOBS) {
            if (body.has(knob)) {
                chat.put(knob, body.raw(knob));
            }
        }
        return OpenAiRequests.chat(JsonView.of(chat, field), service.options());
    }

    /** The image and audio parts of a conversation, and whether text parts or text content sit beside them. */
    private record MediaParts(List<String> images, boolean audio, boolean text) {

        boolean hasMedia() {
            return !images.isEmpty() || audio;
        }
    }

    /**
     * The media parts of one conversation: every {@code image_url} part's URL and whether any {@code
     * input_audio} or {@code audio_url} part or any text sits in it. A video part refuses by name: the pooling
     * inputs take image and audio parts.
     */
    private static MediaParts mediaParts(List<?> messages, String field) {
        List<String> images = new ArrayList<>();
        boolean audio = false;
        boolean text = false;
        for (int m = 0; m < messages.size(); m++) {
            Object content = ((Map<?, ?>) messages.get(m)).get(ChatMessages.CONTENT);
            if (content instanceof String s) {
                text |= !s.isEmpty();
            } else if (content instanceof List<?> parts) {
                for (int i = 0; i < parts.size(); i++) {
                    if (!(parts.get(i) instanceof Map<?, ?> part)) {
                        continue;
                    }
                    String at = field + "[" + m + "].content[" + i + "]";
                    Object type = part.get("type");
                    if (ChatMessages.PART_IMAGE.equals(type)) {
                        images.add(ScoreSides.imageUrl(part.get(ChatMessages.PART_IMAGE), at));
                    } else if (ChatMessages.PART_AUDIO.equals(type) || ChatMessages.PART_AUDIO_URL.equals(type)) {
                        audio = true;
                    } else if (ChatMessages.PART_VIDEO_URL.equals(type)
                            || ChatMessages.PART_INPUT_VIDEO.equals(type)) {
                        throw ServerJson.badRequest(at + " is a '" + type + "' part: the pooling inputs take"
                                + " image and audio parts, not video");
                    } else if ("text".equals(type) && part.get("text") instanceof String t) {
                        text |= !t.isEmpty();
                    }
                }
            }
        }
        return new MediaParts(images, audio, text);
    }

    /** OpenAI's completion forms of one {@code input} value (no chat shape). */
    private static List<Input> completion(Object raw, EngineService service, EmbeddingOptions options) {
        List<int[]> out = new ArrayList<>();
        if (raw instanceof String s) {
            out.add(text(s, service, options));
        } else if (raw instanceof List<?> list && !list.isEmpty()) {
            if (list.get(0) instanceof Number) {
                out.add(options.truncated(OpenAiRequests.tokens(list, service.vocabSize(), "input")));
            } else {
                for (int i = 0; i < list.size(); i++) {
                    Object item = list.get(i);
                    if (item instanceof String s) {
                        out.add(text(s, service, options));
                    } else if (item instanceof List<?> ids && !ids.isEmpty()) {
                        out.add(options.truncated(OpenAiRequests.tokens(ids, service.vocabSize(),
                                "input[" + i + "]")));
                    } else {
                        throw ServerJson.badRequest("input[" + i + "] must be a string or a non-empty array of"
                                + " token ids");
                    }
                }
            }
        } else {
            throw ServerJson.badRequest("input is required: a string, an array of strings, an array of token ids"
                    + " or an array of token-id arrays");
        }
        List<Input> inputs = new ArrayList<>(out.size());
        for (int i = 0; i < out.size(); i++) {
            inputs.add(new Input(nonEmpty(out.get(i)), null, null, out.size() == 1 ? "input" : "input[" + i + "]"));
        }
        return inputs;
    }

    private static int[] text(String s, EngineService service, EmbeddingOptions options) {
        return options.truncated(service.encode(s, options.addSpecial(), true));
    }

    private static int[] nonEmpty(int[] tokens) {
        if (tokens.length == 0) {
            throw ServerJson.badRequest("Input content cannot be empty");
        }
        return tokens;
    }

    /** True when {@code value} is a non-empty array of chat messages (objects with a string {@code role}). */
    private static boolean isConversation(Object value) {
        if (!(value instanceof List<?> list) || list.isEmpty()) {
            return false;
        }
        for (Object item : list) {
            if (!isMessage(item)) {
                return false;
            }
        }
        return true;
    }

    /** True when {@code value} is a non-empty array of conversations. */
    private static boolean isBatch(Object value) {
        if (!(value instanceof List<?> list) || list.isEmpty()) {
            return false;
        }
        for (Object item : list) {
            if (!isConversation(item)) {
                return false;
            }
        }
        return true;
    }

    private static boolean isMessage(Object value) {
        return value instanceof Map<?, ?> message && message.get("role") instanceof String;
    }
}
