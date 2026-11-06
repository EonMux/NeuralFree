package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The questions of a decision request, validated as llama.cpp's {@code server_decision_context::parse_questions}
 * validates the {@code /v1/systemone} body (tools/server/server-decision.cpp:245-352): a {@code state} (null only
 * for the families that allow it), a non-empty {@code questions} object, per question the instructions (optional
 * for vllm-d3, defaulting to "" with a choice default for h2o-lightning, refused when set for minicpm-ocr-jev)
 * and its type's criteria, and at most {@link Flags#nOptionsMax()} options. The head flags of llama.cpp's
 * {@code init} select the refusals and the option order: {@code strict_payload} (vllm-decision2),
 * {@code choice_sorted}/{@code noul_true_first}/{@code noul_criteria_order}, the "yes"/"no" noul aliases, the
 * one fixed question of the pooled read and the spoken-instruction refusal of the family without an audio input
 * (lfm3-d1). Immutable.
 */
final class DecisionQuestions {

    /** Question types in {@link com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead.DecisionOrder} codes minus one. */
    static final int NOUL = 0;
    static final int CHOICE = 1;
    static final int SCORE = 2;

    /** One option: its answer key and the description the prompt shows (a JSON value, or null). */
    record Option(String key, Object description) {
    }

    /**
     * One question: its prompt id, its answer name ({@code /v1/decisions} only, null on {@code /v1/systemone}),
     * type, instructions (text or a JSON payload) and options in prompt order; the {@code /v1/decisions} route
     * also carries the echoed choice values and score-level labels its answers repeat, and {@code orders}, the
     * request index of each prompt-order option (null when the prompt reads the request's own order).
     */
    record Question(String id, String name, int type, Object instructions, List<Option> options,
                    List<Object> choiceValues, List<String> levelLabels, int[] orders) {

        /** One {@code /v1/systemone} question: answers are keyed by id, nothing is echoed. */
        static Question systemOne(String id, int type, Object instructions, List<Option> options) {
            return new Question(id, null, type, instructions, options, null, null, null);
        }

        /** One {@code /v1/decisions} question: the answer echoes {@code name} and the raw choice values. */
        static Question vllm(String id, String name, int type, Object instructions, List<Option> options,
                             List<Object> choiceValues, List<String> levelLabels, int[] orders) {
            return new Question(id, name, type, instructions, options, choiceValues, levelLabels, orders);
        }

        /** The option probabilities of the prompt order placed into the request's option order. */
        double[] requestOrder(double[] probs) {
            if (orders == null) {
                return probs;
            }
            double[] out = new double[probs.length];
            for (int i = 0; i < probs.length; i++) {
                out[orders[i]] = probs[i];
            }
            return out;
        }
    }

    /**
     * The head flags that shape parsing: llama.cpp's {@code strict_payload}, {@code choice_sorted},
     * {@code noul_true_first}, {@code noul_criteria_order} and {@code n_options_max}, the family's allowances of
     * a null state and a question without instructions, the fixed-question, default-choice, noul-alias and
     * open-score-level rules of the served families, and the spoken-instruction refusal of the families without
     * an audio input (tools/server/server-decision.cpp init and parse_questions).
     */
    record Flags(boolean strictPayload, boolean choiceSorted, boolean noulTrueFirst, boolean noulCriteriaOrder,
                 int nOptionsMax, boolean nullState, boolean optionalInstructions, boolean fixedQuestion,
                 boolean defaultChoice, boolean noulAliases, boolean openScoreLevels, boolean refusesSpoken) {

        /** vllm-decision2: strict payloads, at most 255 options, noul keeps a true-first criteria order. */
        static final Flags VLLM_DECISION2 = new Flags(true, false, false, true, 255, false, false, false, false,
                false, false, false);
        /** clef: choice options sorted by key, noul options true first, at most 255 options. */
        static final Flags CLEF = new Flags(false, true, true, false, 255, false, false, false, false, false,
                false, false);
    }

    /**
     * True when {@code instructions} is a spoken question, an object with {@code "type": "input_audio"} (llama.cpp's
     * {@code decision_is_spoken}).
     */
    static boolean isSpoken(Object instructions) {
        return instructions instanceof Map<?, ?> map && "input_audio".equals(String.valueOf(map.get("type")));
    }

    private DecisionQuestions() {
    }

    /** True when {@code value} is a payload llama.cpp's {@code decision_is_payload} accepts: text, object or array. */
    static boolean isPayload(Object value) {
        return value instanceof String || value instanceof Map || value instanceof List;
    }

    /**
     * The questions of a {@code /v1/systemone} body: the {@code state} and every {@code questions} entry.
     *
     * @throws com.eonmux.neuralfree.server.ServerJson-reachable 400 naming the field llama.cpp's parser refuses
     */
    static List<Question> parseSystemOne(JsonView body, Flags flags) {
        if (!body.has("state") || (body.raw("state") == null && !flags.nullState())) {
            throw ServerJson.badRequest("\"state\" must be provided");
        }
        if (flags.strictPayload() && !isPayload(body.raw("state"))) {
            throw ServerJson.badRequest("\"state\" must be text, an object or an array");
        }
        Object raw = body.raw("questions");
        if (!(raw instanceof Map<?, ?> byId) || byId.isEmpty()) {
            throw ServerJson.badRequest("\"questions\" must be a non-empty object");
        }
        if (flags.fixedQuestion() && byId.size() != 1) {
            throw ServerJson.badRequest("this model answers exactly one question");
        }
        List<Question> out = new ArrayList<>();
        for (Map.Entry<?, ?> entry : byId.entrySet()) {
            out.add(parseQuestion(String.valueOf(entry.getKey()), entry.getValue(), flags));
        }
        return out;
    }

    /** One question object: its instructions, type and criteria (llama.cpp's per-question validations). */
    private static Question parseQuestion(String id, Object raw, Flags flags) {
        if (!(raw instanceof Map<?, ?> q)) {
            throw ServerJson.badRequest("questions." + id + ": must be an object");
        }
        // the vllm-d3 prompt has a default question; h2o-lightning's instructions default to "" and its
        // type to choice; minicpm-ocr-jev answers one fixed question of its template
        Object instructions = q.containsKey("instructions") ? q.get("instructions")
                : flags.defaultChoice() ? "" : null;
        if (instructions == null && !flags.optionalInstructions() && !flags.defaultChoice()
                && !flags.fixedQuestion()) {
            throw ServerJson.badRequest("questions." + id + ": \"instructions\" must be provided");
        }
        if (flags.strictPayload() && (!isPayload(instructions) || "".equals(instructions))) {
            throw ServerJson.badRequest("questions." + id + ": \"instructions\" must be non-empty text, an object"
                    + " or an array");
        }
        if (flags.refusesSpoken() && isSpoken(instructions)) {
            throw ServerJson.badRequest("questions." + id + ": spoken instructions need a model with audio input");
        }
        String typeName = q.get("type") instanceof String named ? named : flags.defaultChoice() ? "choice" : "";
        Object criteria = q.getOrDefault("criteria", null);
        List<Option> options;
        int type;
        switch (typeName) {
            case "choice" -> {
                type = CHOICE;
                options = parseChoice(id, criteria, flags);
            }
            case "score" -> {
                type = SCORE;
                options = parseScore(id, criteria, flags);
            }
            case "noul" -> {
                type = NOUL;
                options = parseNoul(id, criteria, flags);
            }
            default -> throw ServerJson.badRequest("questions." + id + ": \"type\" must be one of: choice, score,"
                    + " noul");
        }
        // the classes are fixed: "0" is inconsistent, "1" is consistent (llama.cpp's per-question post-checks)
        if (flags.fixedQuestion()) {
            options = checkFixedQuestion(id, type, instructions, criteria, options);
        }
        if (options.size() > flags.nOptionsMax()) {
            throw ServerJson.badRequest("questions." + id + ": too many options (" + options.size()
                    + "), this model supports at most " + flags.nOptionsMax());
        }
        return Question.systemOne(id, type, instructions, List.copyOf(options));
    }

    /**
     * The fixed question of minicpm-ocr-jev: no instructions (the task text is in the template), a choice whose
     * criteria are exactly {@code {"0": null, "1": null}}, a noul without criteria, no score question.
     */
    private static List<Option> checkFixedQuestion(String id, int type, Object instructions, Object criteria,
                                                   List<Option> options) {
        if (instructions != null) {
            throw ServerJson.badRequest("questions." + id + ": \"instructions\" is not used by this model, the"
                    + " task is part of its prompt");
        }
        if (type == CHOICE) {
            if (options.size() != 2 || !"0".equals(options.get(0).key()) || !"1".equals(options.get(1).key())
                    || options.get(0).description() != null || options.get(1).description() != null) {
                throw ServerJson.badRequest("questions." + id + ": \"criteria\" must be {\"0\": null, \"1\": null}");
            }
            return options;
        }
        if (type == NOUL) {
            if (criteria != null) {
                throw ServerJson.badRequest("questions." + id + ": \"criteria\" is not supported for a noul"
                        + " question");
            }
            return options;
        }
        throw ServerJson.badRequest("questions." + id + ": only choice and noul questions are supported");
    }

    /** Choice criteria: a non-empty object of {@code key: description}; sorted by key under {@code choice_sorted}. */
    private static List<Option> parseChoice(String id, Object criteria, Flags flags) {
        if (!(criteria instanceof Map<?, ?> byKey) || byKey.isEmpty()) {
            throw ServerJson.badRequest("questions." + id + ": \"criteria\" must be a non-empty object");
        }
        List<Option> options = new ArrayList<>();
        for (Map.Entry<?, ?> entry : byKey.entrySet()) {
            String key = String.valueOf(entry.getKey());
            Object description = entry.getValue();
            if (flags.strictPayload() && (key.isEmpty() || !(description == null || isPayload(description)))) {
                throw ServerJson.badRequest("questions." + id + ": option keys must be non-empty, descriptions must"
                        + " be null, text, an object or an array");
            }
            options.add(new Option(key, description));
        }
        if (flags.strictPayload() && options.size() < 2) {
            throw ServerJson.badRequest("questions." + id + ": \"criteria\" must have at least 2 options");
        }
        if (flags.choiceSorted()) {
            options.sort((a, b) -> a.key().compareTo(b.key()));
        }
        return options;
    }

    /**
     * Score criteria: an array of level descriptions, two to ten of them, or from one up to the option cap for
     * h2o-lightning ({@code open_score_levels}); the options' keys are the level indexes.
     */
    private static List<Option> parseScore(String id, Object criteria, Flags flags) {
        if (!(criteria instanceof List<?> levels) || levels.size() < 1
                || !flags.openScoreLevels() && (levels.size() < 2 || levels.size() > 10)) {
            throw ServerJson.badRequest(flags.openScoreLevels()
                    ? "questions." + id + ": \"criteria\" must be a non-empty array"
                    : "questions." + id + ": \"criteria\" must be an array of 2 to 10 levels");
        }
        List<Option> options = new ArrayList<>();
        for (int i = 0; i < levels.size(); i++) {
            Object description = levels.get(i);
            if (flags.strictPayload() && !isPayload(description)) {
                throw ServerJson.badRequest("questions." + id + ": level descriptions must be text, an object or"
                        + " an array");
            }
            options.add(new Option(String.valueOf(i), description));
        }
        return options;
    }

    /**
     * Noul criteria: an object (or null) describing only {@code "false"} and {@code "true"} under
     * {@code strict_payload}; the options are false and true, swapped to true first under {@code noul_true_first},
     * or under {@code noul_criteria_order} when the criteria names true first.
     */
    private static List<Option> parseNoul(String id, Object criteria, Flags flags) {
        if (criteria != null && !(criteria instanceof Map<?, ?> byKey)) {
            throw ServerJson.badRequest("questions." + id + ": \"criteria\" must be an object");
        }
        if (flags.strictPayload() && criteria != null) {
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) criteria).entrySet()) {
                String key = String.valueOf(entry.getKey());
                if (!"false".equals(key) && !"true".equals(key) || !isPayload(entry.getValue())) {
                    throw ServerJson.badRequest("questions." + id + ": \"criteria\" may only describe \"false\" and"
                            + " \"true\" with text, an object or an array");
                }
            }
        }
        List<Option> options = new ArrayList<>();
        options.add(new Option("false", noulDescription(criteria, "false", "no", flags)));
        options.add(new Option("true", noulDescription(criteria, "true", "yes", flags)));
        boolean trueFirst = flags.noulCriteriaOrder() && criteria instanceof Map<?, ?> byKey && byKey.size() == 2
                && "true".equals(String.valueOf(byKey.keySet().iterator().next()));
        if (flags.noulTrueFirst() || trueFirst) {
            java.util.Collections.swap(options, 0, 1);
        }
        return options;
    }

    /**
     * A noul option's description: the criteria's entry under {@code key}, or under the {@code alias} ("yes" or
     * "no") when the family reads aliases and the key is absent — h2o-lightning and lfm2-d1-omni.
     */
    private static Object noulDescription(Object criteria, String key, String alias, Flags flags) {
        if (!(criteria instanceof Map<?, ?> byKey)) {
            return null;
        }
        if (byKey.containsKey(key)) {
            return byKey.get(key);
        }
        return flags.noulAliases() && byKey.containsKey(alias) ? byKey.get(alias) : null;
    }

    // ------------------------------------------------------------------------------------------
    // /v1/decisions (vLLM): predicate, choice and score questions with echoed answers
    // ------------------------------------------------------------------------------------------

    /** The option cap of the {@code /v1/decisions} request: vLLM's {@code choices} model bounds. */
    static final int MAX_VLLM_OPTIONS = 255;
    /**
     * The option cap of a {@code /v1/decisions} read: every strategy that serves the label-reading heads holds
     * {@code max_options = len(LABELS)} (26 answer codes), and the read adapter caps at
     * {@code min(max_options, len(LABELS))}.
     */
    static final int MAX_READ_OPTIONS = 26;
    /** The question cap the {@code /v1/decisions} serving applies per read set (every strategy's limit). */
    static final int MAX_VLLM_QUESTIONS = 64;

    /**
     * The questions of a {@code /v1/decisions} body: predicate questions read false then true, choice questions
     * carry two to 255 distinct string or boolean values, score questions two to 10 labeled levels; unknown
     * fields refuse, as vLLM's {@code extra="forbid"} models do. The options are built with the head's own
     * flags, and {@code orders} maps each prompt-order option back to its request index.
     *
     * @throws com.eonmux.neuralfree.server.ServerJson-reachable 400 naming the field vLLM's models refuse
     */
    static List<Question> parseVllm(JsonView body, Flags flags, int maxOptions) {
        Object raw = body.raw("questions");
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest("\"questions\" must be a non-empty array");
        }
        if (list.size() > MAX_VLLM_QUESTIONS) {
            throw ServerJson.badRequest("questions: at most " + MAX_VLLM_QUESTIONS + " for this model");
        }
        List<Question> out = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            out.add(parseVllmQuestion(String.valueOf(i), list.get(i), flags, maxOptions));
        }
        return out;
    }

    /** One {@code /v1/decisions} question object: its type, instructions, name and its type's options. */
    private static Question parseVllmQuestion(String id, Object raw, Flags flags, int maxOptions) {
        if (!(raw instanceof Map<?, ?> q)) {
            throw ServerJson.badRequest("questions." + id + ": must be an object");
        }
        String typeName = q.get("type") instanceof String named ? named : "";
        int type = vllmType(id, typeName);
        for (Object key : q.keySet()) {
            if (!vllmAllowedFields(typeName).contains(String.valueOf(key))) {
                throw ServerJson.badRequest("questions." + id + ": unknown field \"" + key + "\"");
            }
        }
        Object instructions = q.get("instructions");
        if (!(instructions instanceof String named)) {
            throw ServerJson.badRequest("questions." + id + ": \"instructions\" must be provided as text");
        }
        String name = vllmName(id, q);
        List<Option> requestOptions;
        List<Object> choiceValues = null;
        List<String> levelLabels = null;
        switch (type) {
            case NOUL -> requestOptions = List.of(new Option("false", null), new Option("true", null));
            case CHOICE -> {
                VllmChoice choice = parseVllmChoice(id, q);
                requestOptions = choice.options();
                choiceValues = choice.values();
            }
            default -> {
                VllmScore score = parseVllmScore(id, q);
                requestOptions = score.options();
                levelLabels = score.labels();
            }
        }
        if (requestOptions.size() > maxOptions) {
            throw ServerJson.badRequest("This model supports at most " + maxOptions
                    + " choices per question");
        }
        List<Option> promptOptions = reorder(requestOptions, type, flags);
        return Question.vllm(id, name, type, instructions, promptOptions, choiceValues, levelLabels,
                orders(promptOptions, requestOptions));
    }

    /** The question type's code: vLLM's discriminator over predicate, choice and score. */
    private static int vllmType(String id, String typeName) {
        return switch (typeName) {
            case "predicate" -> NOUL;
            case "choice" -> CHOICE;
            case "score" -> SCORE;
            default -> throw ServerJson.badRequest("questions." + id + ": \"type\" must be one of: predicate,"
                    + " choice, score");
        };
    }

    /** The fields vLLM's {@code extra="forbid"} models accept for {@code typeName}. */
    private static List<String> vllmAllowedFields(String typeName) {
        return switch (typeName) {
            case "predicate" -> List.of("type", "instructions", "name");
            case "choice" -> List.of("type", "instructions", "name", "choices");
            default -> List.of("type", "instructions", "name", "levels");
        };
    }

    /** The question's optional {@code name}, a string when present (vLLM's {@code validate_name}). */
    private static String vllmName(String id, Map<?, ?> q) {
        Object name = q.get("name");
        if (name == null) {
            return null;
        }
        if (!(name instanceof String named)) {
            throw ServerJson.badRequest("questions." + id + ": \"name\" must be a string when provided");
        }
        return named;
    }

    /** The request-side options and echoed values of one {@code /v1/decisions} choice question. */
    private record VllmChoice(List<Option> options, List<Object> values) {
    }

    /** One {@code /v1/decisions} choice question's options: two to 255 distinct string or boolean values. */
    private static VllmChoice parseVllmChoice(String id, Map<?, ?> q) {
        Object rawChoices = q.get("choices");
        if (!(rawChoices instanceof List<?> choices) || choices.size() < 2 || choices.size() > MAX_VLLM_OPTIONS) {
            throw ServerJson.badRequest("questions." + id + ": \"choices\" must be an array of 2 to "
                    + MAX_VLLM_OPTIONS + " choices");
        }
        List<Option> options = new ArrayList<>();
        List<Object> values = new ArrayList<>();
        for (Object rawChoice : choices) {
            if (!(rawChoice instanceof Map<?, ?> choice)
                    || !(choice.get("value") instanceof String || choice.get("value") instanceof Boolean)) {
                throw ServerJson.badRequest("questions." + id + ": every choice needs a string or boolean"
                        + " \"value\"");
            }
            Object value = choice.get("value");
            Object description = choice.get("description") == null ? "" : choice.get("description");
            if (!(description instanceof String)) {
                throw ServerJson.badRequest("questions." + id + ": choice \"description\" must be text");
            }
            options.add(new Option(jsonKey(value), description));
            values.add(value);
        }
        if (distinct(values) != values.size()) {
            throw ServerJson.badRequest("questions." + id + ": choice values must be distinct");
        }
        return new VllmChoice(options, values);
    }

    /** The request-side options and echoed labels of one {@code /v1/decisions} score question. */
    private record VllmScore(List<Option> options, List<String> labels) {
    }

    /** One {@code /v1/decisions} score question's options: two to 10 labeled levels. */
    private static VllmScore parseVllmScore(String id, Map<?, ?> q) {
        Object rawLevels = q.get("levels");
        if (!(rawLevels instanceof List<?> levels) || levels.size() < 2 || levels.size() > 10) {
            throw ServerJson.badRequest("questions." + id + ": \"levels\" must be an array of 2 to 10"
                    + " levels");
        }
        List<Option> options = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        for (Object rawLevel : levels) {
            if (!(rawLevel instanceof Map<?, ?> level) || !(level.get("label") instanceof String label)) {
                throw ServerJson.badRequest("questions." + id + ": every level needs a \"label\"");
            }
            Object description = level.get("description") == null ? "" : level.get("description");
            if (!(description instanceof String)) {
                throw ServerJson.badRequest("questions." + id + ": level \"description\" must be text");
            }
            options.add(new Option(label, description));
            labels.add(label);
        }
        return new VllmScore(options, labels);
    }

    /** The choice value's option key: vLLM's {@code json.dumps(value, ensure_ascii=False)}. */
    private static String jsonKey(Object value) {
        return com.eonmux.neuralfree.utils.json.Json.toString(value,
                com.eonmux.neuralfree.utils.json.Json.Style.CANONICAL);
    }

    /** How many distinct values a list holds, comparing type and value as vLLM's validator does. */
    private static int distinct(List<Object> values) {
        Set<String> seen = new java.util.HashSet<>();
        for (Object value : values) {
            seen.add(value.getClass().getSimpleName() + ":" + value);
        }
        return seen.size();
    }

    /**
     * The options in the order the head reads them: choice options sorted by key under {@code choice_sorted},
     * noul options true first under {@code noul_true_first}, score levels kept by index.
     */
    static List<Option> reorder(List<Option> requestOptions, int type, Flags flags) {
        List<Option> out = new ArrayList<>(requestOptions);
        if (flags.choiceSorted()) {
            out.sort((a, b) -> a.key().compareTo(b.key()));
        }
        if (type == NOUL && flags.noulTrueFirst() && requestOptions.size() == 2) {
            java.util.Collections.swap(out, 0, 1);
        }
        return out;
    }

    /** The request index of each prompt-order option: identity when the prompt keeps the request's order. */
    static int[] orders(List<Option> promptOptions, List<Option> requestOptions) {
        int[] orders = new int[promptOptions.size()];
        boolean identity = true;
        for (int i = 0; i < orders.length; i++) {
            orders[i] = requestOptions.indexOf(promptOptions.get(i));
            identity &= orders[i] == i;
        }
        return identity ? null : orders;
    }
}
