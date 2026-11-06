package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * The decision prompt of the systemone template: the rendered state and questions, split on the
 * {@code <<clef:sep>>} marker, each piece tokenized on its own, and one decision-order code per token, as
 * llama.cpp's {@code server_decision_context::fill_task_joint} builds it (tools/server/server-decision.cpp).
 * The vllm-decision2 head reads one question per prompt (its {@code fill_task} delegates to the joint one),
 * the clef head reads all questions of a request as one joint prompt. The serverone request's text values are
 * sanitized (the {@code <<clef:} marker replaced) and every object's keys sorted before rendering, so no input
 * can forge a marker the head would read as structure.
 */
final class DecisionPrompt {

    /** The marker every sanitized input loses: llama.cpp's {@code CLEF_MARKER}. */
    static final String MARKER = "<<clef:";
    /** The piece separator the template emits between the parts of the joint prompt. */
    static final String SEP = "<<clef:sep>>";
    /** The piece prefix the template emits before one question's prompt. */
    static final String MARK_QUESTION = "<<clef:question>>";
    /** The piece prefix the template emits before one option's prompt. */
    static final String MARK_OPTION = "<<clef:option>>";

    /** Objects nest at most this deep: llama.cpp's {@code RECURSIVE_LIMIT}. */
    private static final int RECURSIVE_LIMIT = 128;

    /** One built prompt: the token ids, one decision-order code per token, and the scored option count. */
    record Built(int[] tokens, int[] order, int nScores) {
    }

    /** One built media prompt: the prompt, one decision-order code per prompt row, and the scored option count. */
    record MediaBuilt(MediaPrompt prompt, int[] order, int nScores) {
    }

    private DecisionPrompt() {
    }

    /**
     * Builds the prompt of {@code questions} over {@code state} through the model's systemone template. The
     * marker-carrying pieces must line up with the questions and their options, or the prompt is refused: the
     * template is part of the model, so a mismatched template is a model defect.
     *
     * @throws IllegalStateException when the rendered prompt does not line up with the questions
     * @throws IllegalArgumentException when a marked piece is empty (a request-side refusal, as llama.cpp's)
     */
    static Built build(EngineService service, List<DecisionQuestions.Question> questions, Object state) {
        return tokenize(service, render(service, questions, state, 0), questions);
    }

    /**
     * Builds the prompt of {@code questions} over {@code state} and the request's media: the template renders one
     * media marker per item, the text up to and including the first marker-containing piece runs through the
     * projector, and the pieces after it are tokenized on their own with their decision codes, as llama.cpp's
     * {@code fill_task_joint} builds a media prompt.
     *
     * @throws IllegalStateException when the rendered prompt does not line up with the questions
     * @throws IllegalArgumentException when a marked piece is empty
     */
    static MediaBuilt buildMedia(EngineService service, ChatMedia chat, List<DecisionQuestions.Question> questions,
                                 Object state, List<ChatMedia.Item> items) {
        String rendered = render(service, questions, state, items.size());
        List<String> pieces = split(rendered, SEP);
        StringBuilder head = new StringBuilder();
        int mediaPiece = -1;
        for (int i = 0; i < pieces.size() && mediaPiece < 0; i++) {
            head.append(pieces.get(i));
            if (pieces.get(i).contains(ChatMessages.MEDIA_MARKER)) {
                mediaPiece = i;
            }
        }
        if (mediaPiece < 0 || head.indexOf(MARKER) >= 0) {
            throw new IllegalStateException("unexpected layout of the decision prompt");
        }
        ChatMedia.TextEncoder encoder = text -> service.encode(text, false, true);
        MediaPrompt headPrompt = chat.prompt(head.toString(), items, List.of(), -1, encoder);
        List<MediaChunk> chunks = new ArrayList<>(headPrompt.chunks());
        List<Integer> order = new ArrayList<>(Collections.nCopies(headPrompt.nTokens(),
                ClassifierHead.DecisionOrder.NONE));
        int iQuestion = 0;
        int nScores = 0;
        for (int i = mediaPiece + 1; i < pieces.size(); i++) {
            String piece = pieces.get(i);
            int code = ClassifierHead.DecisionOrder.NONE;
            if (piece.startsWith(MARK_QUESTION)) {
                if (iQuestion >= questions.size()) {
                    throw new IllegalStateException("unexpected layout of the decision prompt");
                }
                piece = piece.substring(MARK_QUESTION.length());
                code = questions.get(iQuestion++).type() + ClassifierHead.DecisionOrder.QUESTION_NOUL;
            } else if (piece.startsWith(MARK_OPTION)) {
                piece = piece.substring(MARK_OPTION.length());
                code = ClassifierHead.DecisionOrder.OPTION;
                nScores++;
            }
            int[] pieceTokens = service.encode(piece, false, true);
            if (code != ClassifierHead.DecisionOrder.NONE && pieceTokens.length == 0) {
                throw new IllegalArgumentException("the instructions and the options of a question must not be"
                        + " empty");
            }
            if (pieceTokens.length > 0) {
                chunks.add(MediaChunk.text(pieceTokens));
            }
            for (int t = 0; t < pieceTokens.length; t++) {
                order.add(code);
            }
        }
        int nOptions = 0;
        for (DecisionQuestions.Question question : questions) {
            nOptions += question.options().size();
        }
        if (iQuestion != questions.size() || nScores != nOptions) {
            throw new IllegalStateException("unexpected layout of the decision prompt");
        }
        MediaPrompt.Builder prompt = MediaPrompt.builder();
        for (MediaChunk chunk : chunks) {
            prompt.add(chunk);
        }
        return new MediaBuilt(prompt.build(), toArray(order), nScores);
    }

    /**
     * Renders the systemone template for {@code questions} over {@code state} with one media marker per item: the
     * template input is the sorted, marker-sanitized request (llama.cpp's {@code fill_task_joint}), and a request
     * that attaches media loses the marker text from its own strings.
     */
    private static String render(EngineService service, List<DecisionQuestions.Question> questions, Object state,
                                 int nImages) {
        String template = service.tokenizer().systemoneTemplate();
        if (template == null) {
            throw new IllegalStateException(service.architecture()
                    + " carries no \"tokenizer.chat_template.systemone\" template; the decision prompt cannot be"
                    + " built");
        }
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("state", state);
        List<Map<String, Object>> inpQuestions = new ArrayList<>();
        for (DecisionQuestions.Question question : questions) {
            List<Map<String, Object>> options = new ArrayList<>();
            for (DecisionQuestions.Option option : question.options()) {
                Map<String, Object> opt = new LinkedHashMap<>();
                opt.put("key", option.key());
                opt.put("description", option.description());
                options.add(opt);
            }
            Map<String, Object> q = new LinkedHashMap<>();
            q.put("id", question.id());
            q.put("type", typeName(question.type()));
            q.put("instructions", question.instructions());
            q.put("options", options);
            inpQuestions.add(q);
        }
        inp.put("questions", inpQuestions);
        @SuppressWarnings("unchecked")
        Map<String, Object> clean = (Map<String, Object>) replaceMarker(sortKeys(inp, 0));
        inp = clean;
        List<String> images = new ArrayList<>();
        if (nImages > 0) {
            @SuppressWarnings("unchecked")
            Map<String, Object> sanitized = (Map<String, Object>) replaceText(inp, ChatMessages.MEDIA_MARKER, " ");
            inp = sanitized;
            for (int i = 0; i < nImages; i++) {
                images.add(ChatMessages.MEDIA_MARKER);
            }
        }
        inp.put("images", images);
        inp.put("sep", SEP);
        inp.put("mark_question", MARK_QUESTION);
        inp.put("mark_option", MARK_OPTION);

        return MiniJinja.render(template, inp);
    }

    /** The question type's name the template reads: llama.cpp's {@code decision_question_type_name}. */
    static String typeName(int type) {
        return switch (type) {
            case DecisionQuestions.NOUL -> "noul";
            case DecisionQuestions.CHOICE -> "choice";
            default -> "score";
        };
    }

    /**
     * Splits the rendered prompt on the separator, tokenizes each piece on its own (no special-token insertion,
     * special tokens parsed), and lays one order code per token: question pieces carry their question's order,
     * option pieces carry {@link ClassifierHead.DecisionOrder#OPTION}, everything else NONE.
     */
    private static Built tokenize(EngineService service, String prompt, List<DecisionQuestions.Question> questions) {
        List<String> pieces = split(prompt, SEP);
        List<Integer> tokens = new ArrayList<>();
        List<Integer> order = new ArrayList<>();
        int iQuestion = 0;
        int nScores = 0;
        for (String piece : pieces) {
            int pieceOrder = ClassifierHead.DecisionOrder.NONE;
            if (piece.startsWith(MARK_QUESTION)) {
                if (iQuestion >= questions.size()) {
                    throw new IllegalStateException("unexpected layout of the decision prompt");
                }
                piece = piece.substring(MARK_QUESTION.length());
                pieceOrder = questions.get(iQuestion++).type() + ClassifierHead.DecisionOrder.QUESTION_NOUL;
            } else if (piece.startsWith(MARK_OPTION)) {
                piece = piece.substring(MARK_OPTION.length());
                pieceOrder = ClassifierHead.DecisionOrder.OPTION;
                nScores++;
            }
            int[] pieceTokens = service.encode(piece, false, true);
            if (pieceOrder != ClassifierHead.DecisionOrder.NONE && pieceTokens.length == 0) {
                throw new IllegalArgumentException("the instructions and the options of a question must not be"
                        + " empty");
            }
            for (int token : pieceTokens) {
                tokens.add(token);
                order.add(pieceOrder);
            }
        }
        int nOptions = 0;
        for (DecisionQuestions.Question question : questions) {
            nOptions += question.options().size();
        }
        if (iQuestion != questions.size() || nScores != nOptions) {
            throw new IllegalStateException("unexpected layout of the decision prompt");
        }
        return new Built(toArray(tokens), toArray(order), nScores);
    }

    /** {@link String#split} without its trailing-empty-string drop: llama.cpp's {@code string_split}. */
    private static List<String> split(String text, String sep) {
        List<String> out = new ArrayList<>();
        int start = 0;
        int at;
        while ((at = text.indexOf(sep, start)) >= 0) {
            out.add(text.substring(start, at));
            start = at + sep.length();
        }
        out.add(text.substring(start));
        return out;
    }

    private static int[] toArray(List<Integer> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    /**
     * Replaces the decision marker in every string of a JSON value, recursing into arrays and object values
     * (object keys are kept): llama.cpp's {@code decision_replace_text}.
     *
     * @throws IllegalArgumentException when the value nests more than {@link #RECURSIVE_LIMIT} levels deep
     */
    private static Object replaceMarker(Object value) {
        return walk(value, 0, DecisionPrompt::stripMarker);
    }

    private static String stripMarker(String text) {
        return text.replace(MARKER, "<<clef ");
    }

    /**
     * Replaces {@code search} with {@code replace} in every string of a JSON value, recursing into arrays and
     * object values (keys are kept): llama.cpp's {@code decision_replace_text}.
     *
     * @throws IllegalArgumentException when the value nests more than {@link #RECURSIVE_LIMIT} levels deep
     */
    static Object replaceText(Object value, String search, String replace) {
        return walk(value, 0, text -> text.replace(search, replace));
    }

    /**
     * Maps every string of a JSON value through {@code escape}, object keys included: llama.cpp's
     * {@code decision_escape} (the lfm3 d1 input breaks the special-token spellings in keys too).
     *
     * @throws IllegalArgumentException when the value nests more than {@link #RECURSIVE_LIMIT} levels deep
     */
    static Object escapeText(Object value, java.util.function.UnaryOperator<String> escape) {
        return walkKeys(value, 0, escape);
    }

    /** Walks a JSON value, mapping its strings and object keys through {@code strings}, at the given depth. */
    private static Object walkKeys(Object value, int depth, java.util.function.UnaryOperator<String> strings) {
        if (depth >= RECURSIVE_LIMIT) {
            throw new IllegalArgumentException("JSON is nested more than " + RECURSIVE_LIMIT + " levels deep");
        }
        if (value instanceof List<?> list) {
            List<Object> out = new ArrayList<>(list.size());
            for (Object item : list) {
                out.add(walkKeys(item, depth + 1, strings));
            }
            return out;
        }
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> out = new TreeMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                out.put(strings.apply(String.valueOf(entry.getKey())), walkKeys(entry.getValue(), depth + 1,
                        strings));
            }
            return out;
        }
        if (value instanceof String text) {
            return strings.apply(text);
        }
        return value;
    }

    /** Sorts every object of a JSON value by key, recursing: llama.cpp's {@code decision_sort_keys}. */
    static Object sortKeys(Object value) {
        return sortKeys(value, 0);
    }

    private static Object sortKeys(Object value, int depth) {
        return walk(value, depth, DecisionPrompt::keep);
    }

    private static String keep(String text) {
        return text;
    }

    /** Walks a JSON value, mapping its strings through {@code strings}, at the given depth. */
    private static Object walk(Object value, int depth, java.util.function.UnaryOperator<String> strings) {
        if (depth >= RECURSIVE_LIMIT) {
            throw new IllegalArgumentException("JSON is nested more than " + RECURSIVE_LIMIT + " levels deep");
        }
        if (value instanceof List<?> list) {
            List<Object> out = new ArrayList<>(list.size());
            for (Object item : list) {
                out.add(walk(item, depth + 1, strings));
            }
            return out;
        }
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> out = new TreeMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                out.put(String.valueOf(entry.getKey()), walk(entry.getValue(), depth + 1, strings));
            }
            return out;
        }
        if (value instanceof String text) {
            return strings.apply(text);
        }
        return value;
    }
}
