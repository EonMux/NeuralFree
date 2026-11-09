package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.media.MediaLimits.Modality;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead;
import com.eonmux.neuralfree.gguf.ops.heads.KevScorer;
import com.eonmux.neuralfree.gguf.ops.heads.ModernBertDecisionHead;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One served decision model: its {@link DecisionFamily}, the option labels the family's readout names (letters,
 * codes or the {@code decision.labels} metadata), and the reads that answer a request — the ordered per-option head
 * (vllm-decision2), the joint clef head, the answer-code logits of the label families, the marker columns of a
 * modern-bert decision head, the kev dot products of the last token with every option's end token, the d1-omni
 * piece-budget read, and the pooled classifier of a fixed-question model.
 * The reads follow llama.cpp's {@code server_decision_context}
 * (tools/server/server-decision.cpp): the systemone template renders every question, the option order and the label
 * text are the ones the family passes to it, and the scores are softmaxed with the model's per-type temperatures and
 * score offsets.
 *
 * <p>A request that attaches media renders one media marker per item, splices the projector's rows where the markers
 * stand, and reads the prompt through the media path of the engine ({@link EnginePooling#nextLogits(LoraSelection,
 * MediaPrompt)} and {@link EnginePooling#decisionScores(LoraSelection, MediaPrompt, int[])}). Immutable.</p>
 */
final class DecisionModel {

    /** llama.cpp's bound on the media of one decision request ({@code DECISION_MAX_IMAGES}). */
    static final int MAX_MEDIA = 8;
    /** The levels of lev's rating scale (0 = certainly no, 8 = certainly yes). */
    private static final int LEV_RATINGS = 9;
    /** The per-option token budget of a marker read (llama.cpp's {@code max_option_tokens}). */
    private static final int MAX_OPTION_TOKENS = 48;
    /** The state of h2o-lightning keeps at most this many tokens ({@code DECISION_H2O_MAX_STATE_TOKENS}). */
    private static final int H2O_MAX_STATE_TOKENS = 32000;
    /** The prompt pieces of lfm2-d1-omni (llama.cpp's {@code D1OMNI_*}). */
    private static final String D1OMNI_SEP = "<<d1omni:sep>>";
    private static final String D1OMNI_MARK_STATE = "<<d1omni:state>>";
    private static final String D1OMNI_MARK_QUESTION = "<<d1omni:question>>";
    private static final String D1OMNI_MARK_OPTION = "<<d1omni:option>>";
    /** The max_length, image_text_length and audio_text_length of the d1-omni config, checked by the converter. */
    private static final int D1OMNI_MAX_TOKENS = 16384;
    private static final int D1OMNI_MAX_TOKENS_IMAGE = 896;
    private static final int D1OMNI_MAX_TOKENS_AUDIO = 15360;

    private final EngineService service;
    private final DecisionFamily family;
    private final String[] labelTexts;
    private final int[] labelIds;
    /** The special-token spellings this family quotes in the request's text, null when it reads none. */
    private final DecisionQuotes quotes;

    private DecisionModel(EngineService service, DecisionFamily family, String[] labelTexts, int[] labelIds,
                          DecisionQuotes quotes) {
        this.service = service;
        this.family = family;
        this.labelTexts = labelTexts;
        this.labelIds = labelIds;
        this.quotes = quotes;
    }

    /** One request's media items and the state left after their parts were taken out of it. */
    record Media(Object state, List<ChatMedia.Item> items) {

        static Media none(Object state) {
            return new Media(state, List.of());
        }
    }

    /** One evaluated request: each question's option probabilities in prompt order, and the input tokens read. */
    record Result(List<double[]> probs, long tokens) {
    }

    /**
     * The decision model of {@code service}.
     *
     * @throws jakarta.ws.rs.WebApplicationException 501 when the model is not a decision model, or its family is
     *                                               not served
     * @throws IllegalStateException                 when the checkpoint's label metadata is unusable (a model
     *                                               defect)
     */
    static DecisionModel of(EngineService service) {
        String type = service.pooling().decisionType();
        DecisionFamily family = DecisionFamily.of(type);
        if (family == null) {
            if (type == null && !service.pooling().decisionHead()) {
                throw ServerJson.error(501, "This model is not a decision model", "not_supported_error");
            }
            throw ServerJson.error(501, "decision model type '" + (type != null ? type
                    : service.architecture() + " decision body") + "' is not served: this endpoint serves the"
                    + " vllm-decision2, clef, openjev, lev, nimble, pplx-decider, vllm-d3, laya, lfm2-d1,"
                    + " h2o-lightning, openbmb-minicpm-ocr-jev, lfm2-d1-omni, lfm3-d1, lfm3-d1-omni,"
                    + " granite-switch and kev decision heads",
                    "not_supported_error");
        }
        if (family.read == DecisionFamily.Read.KEV) {
            // the option end token must be one token of the vocabulary (llama.cpp's init)
            kevMarker(service);
        }
        Resolved resolved = resolveLabels(service, family);
        return new DecisionModel(service, family, resolved.texts, resolved.ids, quotes(service, family));
    }

    /**
     * The id of the {@value KevScorer#MARKER_TEXT} token that ends every option of a kev prompt.
     *
     * @throws IllegalStateException when the vocabulary holds no single {@value KevScorer#MARKER_TEXT} token
     */
    private static int kevMarker(EngineService service) {
        int[] toks = service.encode(KevScorer.MARKER_TEXT, false, true);
        if (toks.length != 1) {
            throw new IllegalStateException("decision model has no " + KevScorer.MARKER_TEXT + " token");
        }
        return toks[0];
    }

    /** The spellings a family quotes in the request's text (llama.cpp's {@code init_quoted_spellings}). */
    private static DecisionQuotes quotes(EngineService service, DecisionFamily family) {
        if (family.type.equals("h2o-lightning")) {
            return DecisionQuotes.of(service.tokenizer().vocab(),
                    TokenizerVocab.ATTR_CONTROL | TokenizerVocab.ATTR_USER_DEFINED, 1);
        }
        if (family.type.startsWith("lfm3")) {
            return DecisionQuotes.of(service.tokenizer().vocab(), TokenizerVocab.ATTR_CONTROL, 2);
        }
        return null;
    }

    /** The family's option labels: their texts and their single-token ids. */
    private record Resolved(String[] texts, int[] ids) {
    }

    /**
     * The label texts of a family, each one distinct token as llama.cpp's {@code init} requires: openjev's letters
     * throw when one is not a single token, h2o-lightning's {@code decision.labels} when one is not a single
     * distinct token (and when there are none), and the code families keep the single-token codes of A-Z then AA-ZZ.
     */
    private static Resolved resolveLabels(EngineService service, DecisionFamily family) {
        List<String> texts = new ArrayList<>();
        List<Integer> ids = new ArrayList<>();
        switch (family.labels) {
            case LETTERS -> {
                String letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
                for (int i = 0; i < letters.length(); i++) {
                    String text = String.valueOf(letters.charAt(i));
                    int[] toks = service.encode(text, false, false);
                    if (toks.length != 1) {
                        throw new IllegalStateException("decision label '" + text + "' is not a single token");
                    }
                    texts.add(text);
                    ids.add(toks[0]);
                }
            }
            case CODES -> {
                List<String> codes = new ArrayList<>();
                for (char a = 'A'; a <= 'Z'; a++) {
                    codes.add(String.valueOf(a));
                }
                for (char a = 'A'; a <= 'Z'; a++) {
                    for (char b = 'A'; b <= 'Z'; b++) {
                        codes.add("" + a + b);
                    }
                }
                for (String code : codes) {
                    int[] toks = service.encode(code, false, false);
                    if (toks.length == 1 && texts.size() < 255) {
                        texts.add(code);
                        ids.add(toks[0]);
                    }
                }
            }
            case METADATA -> {
                List<String> labels = service.pooling().decisionLabels();
                if (labels.isEmpty()) {
                    throw new IllegalStateException("decision model has no labels");
                }
                for (String label : labels) {
                    int[] toks = service.encode(label, false, false);
                    if (toks.length != 1 || ids.contains(toks[0])) {
                        throw new IllegalStateException("decision label \"" + label
                                + "\" is not a single distinct token");
                    }
                    texts.add(label);
                    ids.add(toks[0]);
                }
            }
            default -> {
                return new Resolved(new String[0], new int[0]);
            }
        }
        int[] idArray = new int[ids.size()];
        for (int i = 0; i < idArray.length; i++) {
            idArray[i] = ids.get(i);
        }
        return new Resolved(texts.toArray(new String[0]), idArray);
    }

    /** The family this model serves. */
    DecisionFamily family() {
        return family;
    }

    /** The option cap of a {@code /v1/systemone} question (llama.cpp's {@code n_options_max}). */
    int nOptionsMax() {
        return switch (family.labels) {
            case LETTERS, CODES, METADATA -> labelTexts.length;
            default -> family.nOptionsMax;
        };
    }

    /** The option cap of a {@code /v1/decisions} question (the vLLM strategy's limit). */
    int vllmMaxOptions() {
        if (family.type.equals("vllm-d3")) {
            return labelTexts.length;
        }
        return family.vllmMaxOptions;
    }

    /** The parsing flags llama.cpp's {@code init} sets for the family, at this model's option cap. */
    DecisionQuestions.Flags flags() {
        return family.flags(nOptionsMax());
    }

    // ------------------------------------------------------------------ reads

    /**
     * Reads every question of {@code questions} over {@code state} and the request's media.
     *
     * @throws IOException when a forward fails
     * @throws IllegalStateException when the model's head cannot evaluate the request
     */
    Result read(List<DecisionQuestions.Question> questions, Object state, List<ChatMedia.Item> items,
                LoraSelection lora) throws IOException {
        if (family.read == DecisionFamily.Read.JOINT) {
            return readJoint(questions, state, items, lora);
        }
        if (family.read == DecisionFamily.Read.MARKERS) {
            return readMarkers(questions, state, items, lora);
        }
        if (family.read == DecisionFamily.Read.KEV) {
            return readKev(questions, state, items, lora);
        }
        if (family.read == DecisionFamily.Read.LABELS) {
            return readLabels(questions, state, items, lora);
        }
        if (family.read == DecisionFamily.Read.D1OMNI) {
            return readD1Omni(questions, state, items, lora);
        }
        if (family.read == DecisionFamily.Read.POOLED) {
            return readPooled(questions, state, items, lora);
        }
        return readOrder(questions, state, lora);
    }

    /** The vllm-decision2 head: one prompt per question, one score per option span. */
    private Result readOrder(List<DecisionQuestions.Question> questions, Object state, LoraSelection lora)
            throws IOException {
        long tokens = 0;
        List<double[]> probs = new ArrayList<>();
        for (DecisionQuestions.Question question : questions) {
            DecisionPrompt.Built prompt = DecisionPrompt.build(service, List.of(question), state);
            float[] scores = service.pooling().decisionScores(lora, prompt.tokens(), prompt.order());
            probs.add(probabilities(question, scores, 0, false));
            tokens += prompt.tokens().length;
        }
        return new Result(probs, tokens);
    }

    /** The clef head: every question of the request in one prompt, with the request's media when it has any. */
    private Result readJoint(List<DecisionQuestions.Question> questions, Object state, List<ChatMedia.Item> items,
                             LoraSelection lora) throws IOException {
        float[] scores;
        int nTokens;
        if (items.isEmpty()) {
            DecisionPrompt.Built prompt = DecisionPrompt.build(service, questions, state);
            scores = service.pooling().decisionScores(lora, prompt.tokens(), prompt.order());
            nTokens = prompt.tokens().length;
        } else {
            DecisionPrompt.MediaBuilt prompt = DecisionPrompt.buildMedia(service, service.media(), questions, state,
                    items);
            scores = service.pooling().decisionScores(lora, prompt.prompt(), prompt.order());
            nTokens = prompt.prompt().nTokens();
        }
        List<double[]> probs = new ArrayList<>();
        int at = 0;
        for (DecisionQuestions.Question question : questions) {
            probs.add(probabilities(question, scores, at, items.size() > 0));
            at += question.options().size();
        }
        if (at != scores.length) {
            throw new IllegalStateException("decision result does not match the number of options");
        }
        return new Result(probs, nTokens);
    }

    /**
     * The label families: one prompt per question, the logits of each option's label token at the last prompt token,
     * and the option's score its group's maximum. lev reads a choice's two option orders and averages them, and its
     * noul question the nine-point rating scale. h2o-lightning reads its metadata labels over the prepared state; a
     * spoken question of lfm3-d1-omni is heard where its text would stand, after the media of the state.
     */
    private Result readLabels(List<DecisionQuestions.Question> questions, Object state, List<ChatMedia.Item> items,
                              LoraSelection lora) throws IOException {
        Object readState = family.labels == DecisionFamily.Labels.METADATA ? prepareH2oState(state) : state;
        long tokens = 0;
        List<double[]> probs = new ArrayList<>();
        for (DecisionQuestions.Question question : questions) {
            int nOptions = question.options().size();
            boolean rating = family.ratingNoul && question.type() == DecisionQuestions.NOUL;
            int nOutputs = rating ? LEV_RATINGS : nOptions;
            int variants = family.buckets == DecisionFamily.Buckets.LEV
                    && question.type() == DecisionQuestions.CHOICE && nOptions > 1 ? 2 : 1;
            double[] questionProbs = new double[nOutputs];
            for (int variant = 0; variant < variants; variant++) {
                List<DecisionQuestions.Option> options = variant == 0 ? question.options()
                        : reversed(question.options());
                List<String> labels;
                int[][] groups;
                if (family.labels == DecisionFamily.Labels.D1) {
                    D1Labels d1 = d1Labels(question, options);
                    labels = d1.texts();
                    groups = d1.groups();
                } else {
                    labels = optionLabels(nOutputs);
                    groups = labelGroups(nOutputs);
                }
                String prompt = render(question, questions, options, labels, readState, items.size(), false);
                List<ChatMedia.Item> promptItems = spokenMedia(question, items);
                float[] logits;
                if (promptItems.isEmpty()) {
                    int[] ids = service.encode(prompt, false, true);
                    logits = service.pooling().nextLogits(lora, ids);
                    tokens += ids.length;
                } else {
                    MediaPrompt media = mediaPrompt(prompt, promptItems);
                    logits = service.pooling().nextLogits(lora, media);
                    tokens += media.nTokens();
                }
                float[] scores = new float[nOutputs];
                for (int i = 0; i < nOutputs; i++) {
                    float best = Float.NEGATIVE_INFINITY;
                    for (int id : groups[i]) {
                        best = Math.max(best, logits[id]);
                    }
                    scores[i] = best;
                }
                float temperature = temperature(question, nOptions, !promptItems.isEmpty());
                float[] bias = question.type() == DecisionQuestions.SCORE
                        ? service.pooling().decisionScoreBias().get(nOutputs) : null;
                double[] p = DecisionAnswers.probabilities(scores, question.type() == DecisionQuestions.SCORE, bias,
                        temperature);
                for (int i = 0; i < nOutputs; i++) {
                    questionProbs[variant == 0 ? i : nOutputs - 1 - i] += p[i] / variants;
                }
            }
            probs.add(questionProbs);
        }
        return new Result(probs, tokens);
    }

    /**
     * The modern-bert decision head (laya): the prompt tokenized with one mask token per option, the head's score of
     * the question's type at each marker. The question and options are cut to the checkpoint's
     * {@code decision.max_head_tokens} as the model was trained.
     */
    private Result readMarkers(List<DecisionQuestions.Question> questions, Object state, List<ChatMedia.Item> items,
                               LoraSelection lora) throws IOException {
        if (!items.isEmpty()) {
            throw ServerJson.badRequest("this decision model reads text prompts only");
        }
        int maxHeadTokens = service.pooling().decisionMaxHeadTokens();
        if (maxHeadTokens <= 0) {
            throw new IllegalStateException("decision model has no valid max_head_tokens");
        }
        long tokens = 0;
        List<double[]> probs = new ArrayList<>();
        for (DecisionQuestions.Question question : questions) {
            String prompt = render(question, questions, question.options(), List.of(), state, 0, false);
            LayaInput input = layaInput(service.encode(prompt, false, true), question.options().size(),
                    maxHeadTokens);
            float[] scores = service.pooling().decisionScores(lora, input.tokens(), null);
            int n = question.options().size();
            float[] questionScores = new float[n];
            for (int i = 0; i < n; i++) {
                questionScores[i] = scores[input.markers()[i] * ModernBertDecisionHead.N_DECISION_TYPES
                        + question.type()];
            }
            probs.add(probabilities(question, questionScores, 0, false));
            tokens += input.tokens().length;
        }
        return new Result(probs, tokens);
    }

    /**
     * The kev read: the prompt tokenized as text, one {@value KevScorer#MARKER_TEXT} marker per option, and the
     * decision scores the dot products of the last prompt token's hidden state with every marker's (llama.cpp's
     * {@code fill_task} and the pointer read of {@code send_decision}). The question and its options must fit one
     * prompt, and every option must end on its marker.
     */
    private Result readKev(List<DecisionQuestions.Question> questions, Object state, List<ChatMedia.Item> items,
                           LoraSelection lora) throws IOException {
        if (!items.isEmpty()) {
            throw ServerJson.badRequest("this decision model reads text prompts only");
        }
        int marker = kevMarker(service);
        long tokens = 0;
        List<double[]> probs = new ArrayList<>();
        for (DecisionQuestions.Question question : questions) {
            String prompt = render(question, questions, question.options(), List.of(), state, 0, false);
            int[] ids = service.encode(prompt, false, true);
            int nMarkers = 0;
            for (int id : ids) {
                nMarkers += id == marker ? 1 : 0;
            }
            if (nMarkers != question.options().size()) {
                throw new IllegalStateException("unexpected layout of the decision prompt");
            }
            probs.add(probabilities(question, service.pooling().decisionScores(lora, ids, null), 0, false));
            tokens += ids.length;
        }
        return new Result(probs, tokens);
    }

    /**
     * The pooled classifier of minicpm-ocr-jev: the fixed question of the template over exactly two images (the
     * original, then the rendered one), the pooled row's raw class scores — llama.cpp's {@code send_decision} reads
     * them through {@code llama_get_embeddings_seq} and format_answer softmaxes them like any choice.
     */
    private Result readPooled(List<DecisionQuestions.Question> questions, Object state, List<ChatMedia.Item> items,
                              LoraSelection lora) throws IOException {
        long tokens = 0;
        List<double[]> probs = new ArrayList<>();
        for (DecisionQuestions.Question question : questions) {
            String prompt = render(question, questions, question.options(), List.of(), state, items.size(), false);
            float[] scores;
            if (items.isEmpty()) {
                int[] ids = service.encode(prompt, false, true);
                scores = service.pooling().classifyOutput(lora, ids, false).scores();
                tokens += ids.length;
            } else {
                MediaPrompt media = mediaPrompt(prompt, items);
                scores = service.pooling().classifyOutputMedia(lora, media, false).scores();
                tokens += media.nTokens();
            }
            probs.add(probabilities(question, scores, 0, !items.isEmpty()));
        }
        return new Result(probs, tokens);
    }

    // ------------------------------------------------------------------ the lfm2-d1-omni read

    /** One prepared d1-omni read: the assembled text tokens and the position of each option's marker. */
    private record D1OmniInput(int[] tokens, int[] markers) {
    }

    /**
     * The d1-omni read of lfm2-d1-omni (llama.cpp's {@code fill_task_d1omni}): the media run through the projector
     * first and set the kind and the room of the read, and the rendered prompt's pieces are cut to the token budgets
     * the model was trained on, the state getting the room that is left. The decision head scores one question-type
     * column at each option's mask marker.
     *
     * @throws IllegalArgumentException when the media crowd out the text or the options do not fit the room left
     * @throws IllegalStateException    when the rendered prompt is not in the d1-omni layout (a model defect)
     */
    private Result readD1Omni(List<DecisionQuestions.Question> questions, Object state, List<ChatMedia.Item> items,
                              LoraSelection lora) throws IOException {
        int mask = service.tokenizer().specialTokens().mask();
        if (mask < 0) {
            throw new IllegalStateException("decision model has no mask token");
        }
        long tokens = 0;
        List<double[]> probs = new ArrayList<>();
        for (DecisionQuestions.Question question : questions) {
            String markerText = ChatMessages.MEDIA_MARKER.repeat(items.size());
            MediaPrompt media = null;
            int nMedia = 0;
            boolean isAudio = false;
            if (!items.isEmpty()) {
                media = mediaPrompt(markerText, items);
                nMedia = media.nTokens();
                for (ChatMedia.Item item : items) {
                    isAudio |= item.modality() == Modality.AUDIO;
                }
                if (isAudio && items.size() > 1) {
                    throw ServerJson.badRequest("a request has images or one audio clip, not both");
                }
            }
            int nMax = D1OMNI_MAX_TOKENS;
            if (!items.isEmpty()) {
                nMax = Math.min(isAudio ? D1OMNI_MAX_TOKENS_AUDIO : D1OMNI_MAX_TOKENS_IMAGE,
                        D1OMNI_MAX_TOKENS - Math.min(D1OMNI_MAX_TOKENS, nMedia));
                if (nMax < 64) {
                    throw ServerJson.badRequest("the media take " + nMedia + " of the " + D1OMNI_MAX_TOKENS
                            + " positions, send fewer images");
                }
            }
            String prompt = renderD1Omni(question, state, items.size(), isAudio);
            D1OmniInput input = d1OmniInput(prompt, markerText, nOptions(question), nMax);
            int[] markers = input.markers();
            int n = nOptions(question);
            float[] questionScores = new float[n];
            if (media == null) {
                float[] scores = service.pooling().decisionScores(lora, input.tokens(), null);
                for (int i = 0; i < n; i++) {
                    questionScores[i] = scores[markers[i] * ModernBertDecisionHead.N_DECISION_TYPES
                            + question.type()];
                }
                tokens += input.tokens().length;
            } else {
                MediaPrompt.Builder full = MediaPrompt.builder();
                for (MediaChunk chunk : media.chunks()) {
                    full.add(chunk);
                }
                full.addText(input.tokens());
                MediaPrompt prompt2 = full.build();
                float[] scores = service.pooling().decisionScores(lora, prompt2, null);
                for (int i = 0; i < n; i++) {
                    questionScores[i] = scores[markers[i] * ModernBertDecisionHead.N_DECISION_TYPES
                            + question.type()];
                }
                tokens += prompt2.nTokens();
            }
            probs.add(probabilities(question, questionScores, 0, media != null));
        }
        return new Result(probs, tokens);
    }

    /** The option count of a question. */
    private static int nOptions(DecisionQuestions.Question question) {
        return question.options().size();
    }

    /**
     * Assembles the d1-omni prompt's pieces: the state gets the room the head and the tail leave, the question and
     * the options are cut to their budgets, and the whole stays inside {@code nMax} (llama.cpp's
     * {@code fill_task_d1omni}). The mask markers are counted after the cut: there must be one per option.
     */
    private D1OmniInput d1OmniInput(String prompt, String markerText, int nOptions, int nMax) {
        List<String> pieces = splitPrompt(prompt, D1OMNI_SEP);
        if (pieces.isEmpty() || !pieces.get(0).equals(markerText)) {
            throw new IllegalStateException("unexpected layout of the decision prompt");
        }
        int nBudget = Math.max(96, Math.min(24 * nOptions + 32, nMax / 2));
        int nOptionMax = Math.max(2, (nBudget - 3 * nOptions) / nOptions);
        int nQuestionMax = Math.max(16, nBudget);

        List<Integer> head = new ArrayList<>();
        List<Integer> body = new ArrayList<>();
        List<Integer> tail = new ArrayList<>();
        boolean hasState = false;
        for (int i = 1; i < pieces.size(); i++) {
            String piece = pieces.get(i);
            int nPieceMax = -1;
            boolean isState = false;
            if (piece.startsWith(D1OMNI_MARK_STATE)) {
                piece = piece.substring(D1OMNI_MARK_STATE.length());
                isState = true;
            } else if (piece.startsWith(D1OMNI_MARK_QUESTION)) {
                piece = piece.substring(D1OMNI_MARK_QUESTION.length());
                nPieceMax = nQuestionMax;
            } else if (piece.startsWith(D1OMNI_MARK_OPTION)) {
                piece = piece.substring(D1OMNI_MARK_OPTION.length());
                nPieceMax = nOptionMax;
            }
            List<Integer> tokens = toInts(service.encode(piece, false, true));
            if (nPieceMax >= 0 && tokens.size() > nPieceMax) {
                tokens = tokens.subList(0, nPieceMax);
            }
            if (isState) {
                if (hasState) {
                    throw new IllegalStateException("unexpected layout of the decision prompt");
                }
                body = tokens;
                hasState = true;
            } else if (hasState) {
                tail.addAll(tokens);
            } else {
                head.addAll(tokens);
            }
        }
        if (!hasState) {
            throw new IllegalStateException("unexpected layout of the decision prompt");
        }
        int nRoom = nMax - Math.min(nMax, head.size() + tail.size());
        if (body.size() > nRoom) {
            body = body.subList(0, nRoom);
        }
        List<Integer> out = new ArrayList<>(head);
        out.addAll(body);
        out.addAll(tail);
        if (out.size() > nMax) {
            out = out.subList(0, nMax);
        }
        int mask = service.tokenizer().specialTokens().mask();
        List<Integer> markers = new ArrayList<>();
        for (int i = 0; i < out.size(); i++) {
            if (out.get(i) == mask) {
                markers.add(i);
            }
        }
        if (markers.size() != nOptions) {
            throw ServerJson.badRequest("the options do not fit in the context");
        }
        return new D1OmniInput(toArray(out), toArray(markers));
    }

    /** {@link String#split} without its trailing-empty-string drop (llama.cpp's {@code string_split}). */
    private static List<String> splitPrompt(String text, String sep) {
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

    private static List<Integer> toInts(int[] values) {
        List<Integer> out = new ArrayList<>(values.length);
        for (int value : values) {
            out.add(value);
        }
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
     * Renders the d1-omni template of one question: the input escaped for the tokenizer, the {@code <<d1omni:}
     * marker replaced, the kind of media, and the piece markers of the layout (llama.cpp's {@code render} for
     * LFM2_D1_OMNI).
     */
    private String renderD1Omni(DecisionQuestions.Question question, Object state, int nImages, boolean isAudio) {
        Map<String, Object> inp = d1OmniInp(question, state);
        boolean spoken = DecisionQuestions.isSpoken(question.instructions());
        if (spoken) {
            inp.put("instructions", "");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> escaped = (Map<String, Object>) DecisionPrompt
                .escapeText(inp, DecisionText::escapeSpecialTokens);
        @SuppressWarnings("unchecked")
        Map<String, Object> clean = (Map<String, Object>) DecisionPrompt.replaceText(escaped, "<<d1omni:",
                "<<d1omni ");
        inp = clean;
        inp.put("audio", isAudio);
        inp.put("sep", D1OMNI_SEP);
        inp.put("mark_state", D1OMNI_MARK_STATE);
        inp.put("mark_question", D1OMNI_MARK_QUESTION);
        inp.put("mark_option", D1OMNI_MARK_OPTION);
        inp.put("images", imagesInp(inp, nImages, spoken));
        if (spoken) {
            inp.put("instructions", ChatMessages.MEDIA_MARKER);
        }
        return MiniJinja.render(systemoneTemplate(), inp);
    }

    /** The d1-omni template input: the question and its options with their d1 labels. */
    private Map<String, Object> d1OmniInp(DecisionQuestions.Question question, Object state) {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("id", question.id());
        inp.put("type", DecisionPrompt.typeName(question.type()));
        inp.put("instructions", question.instructions());
        inp.put("state", state);
        D1Labels d1 = d1Labels(question, question.options());
        List<Map<String, Object>> options = new ArrayList<>();
        for (int i = 0; i < question.options().size(); i++) {
            Map<String, Object> option = new LinkedHashMap<>();
            option.put("key", question.options().get(i).key());
            option.put("description", question.options().get(i).description());
            option.put("label", d1.texts().get(i));
            options.add(option);
        }
        inp.put("options", options);
        return inp;
    }

    /** The systemone template of the checkpoint, refused by name when the model carries none. */
    private String systemoneTemplate() {
        String template = service.tokenizer().systemoneTemplate();
        if (template == null) {
            throw new IllegalStateException(service.architecture()
                    + " carries no \"tokenizer.chat_template.systemone\" template; the decision prompt cannot be"
                    + " built");
        }
        return template;
    }

    /**
     * The images input of a template: one media marker per item when the request attaches media or the question is
     * spoken, with the marker text removed from the input's own strings (llama.cpp's images block).
     */
    private static List<String> imagesInp(Map<String, Object> inp, int nImages, boolean spoken) {
        List<String> images = new ArrayList<>();
        if (nImages > 0 || spoken) {
            @SuppressWarnings("unchecked")
            Map<String, Object> sanitized = (Map<String, Object>) DecisionPrompt.replaceText(inp,
                    ChatMessages.MEDIA_MARKER, " ");
            inp.clear();
            inp.putAll(sanitized);
            for (int i = 0; i < nImages; i++) {
                images.add(ChatMessages.MEDIA_MARKER);
            }
        }
        return images;
    }

    // ------------------------------------------------------------------ the h2o-lightning render

    /**
     * The state of h2o-lightning, prepared once per request (llama.cpp's {@code prepare_state}): compact JSON of a
     * non-string state, the cut to {@link #H2O_MAX_STATE_TOKENS} tokens, and the special-token spellings quoted.
     */
    private String prepareH2oState(Object state) {
        String text = state instanceof String s ? s : DecisionText.dumps(state, -1, ",", ":", false);
        if (text.length() <= H2O_MAX_STATE_TOKENS) {
            return quotes.quote(text);
        }
        int[] tokens = service.encode(text, false, true);
        if (tokens.length <= H2O_MAX_STATE_TOKENS) {
            return quotes.quote(text);
        }
        int half = H2O_MAX_STATE_TOKENS / 2;
        String head = h2oStateOf(tokens, 0, half);
        String tail = h2oStateOf(tokens, tokens.length - half, tokens.length);
        return quotes.quote(head + "\n...\n" + tail);
    }

    /** The text of the token pieces of one half, a cut inside a character decoded to U+FFFD. */
    private String h2oStateOf(int[] tokens, int from, int end) {
        StringBuilder out = new StringBuilder();
        for (int i = from; i < end; i++) {
            out.append(service.tokenizer().getToken(tokens[i]));
        }
        byte[] bytes = out.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
        return new String(bytes, 0, bytes.length, java.nio.charset.StandardCharsets.UTF_8);
    }

    /**
     * The audio clip of a spoken question, heard after the media of the state (llama.cpp's fill_task appends the
     * spoken audio to the task's media).
     */
    private List<ChatMedia.Item> spokenMedia(DecisionQuestions.Question question, List<ChatMedia.Item> items) {
        if (!DecisionQuestions.isSpoken(question.instructions()) || family.read == DecisionFamily.Read.D1OMNI) {
            return items;
        }
        Map<?, ?> instructions = (Map<?, ?>) question.instructions();
        Object data = ((Map<?, ?>) instructions.get("input_audio")).get("data");
        if (!(data instanceof String encoded) || encoded.isEmpty()) {
            throw ServerJson.badRequest("the question's \"input_audio\" needs a \"data\" recording");
        }
        List<ChatMedia.Item> out = new ArrayList<>(items);
        out.add(service.media().decisionAudio(encoded, "questions"));
        return List.copyOf(out);
    }

    // ------------------------------------------------------------------ labels and rendering

    /** One option's answer label group and the text the template shows, for the d1 families. */
    private record D1Labels(List<String> texts, int[][] groups) {
    }

    /**
     * The d1 labels of a question (llama.cpp's {@code d1_labels}): a score level reads its digit, a noul option the
     * yes or no forms, and a choice the option key when it is one letter, else A to Z, else (lfm3) AA, AB, and
     * (lfm2) two digits, with the remaining single-token codes as a fallback pool.
     */
    private D1Labels d1Labels(DecisionQuestions.Question question, List<DecisionQuestions.Option> options) {
        List<String> texts = new ArrayList<>();
        List<int[]> groups = new ArrayList<>();
        if (question.type() != DecisionQuestions.CHOICE) {
            for (DecisionQuestions.Option option : options) {
                int[] group;
                if (question.type() == DecisionQuestions.SCORE) {
                    group = singleTokens(List.of(option.key()));
                } else if ("true".equals(option.key())) {
                    group = singleTokens(List.of("yes", "Yes", "YES"));
                } else {
                    group = singleTokens(List.of("no", "No", "NO"));
                }
                if (group.length == 0) {
                    throw new IllegalStateException("decision label is not a single token: " + option.key());
                }
                texts.add(option.key());
                groups.add(group);
            }
            return new D1Labels(texts, groups.toArray(new int[0][]));
        }
        boolean letters = true;
        for (DecisionQuestions.Option option : options) {
            letters &= option.key().length() == 1 && Character.isLetter(option.key().charAt(0));
        }
        List<String> codes = new ArrayList<>();
        for (int i = 0; i < options.size(); i++) {
            if (letters) {
                codes.add(options.get(i).key());
            } else if (options.size() <= 26) {
                codes.add(String.valueOf((char) ('A' + i)));
            } else if (family.type.startsWith("lfm3")) {
                codes.add(i < 26 ? String.valueOf((char) ('A' + i))
                        : "" + (char) ('A' + (i - 26) / 26) + (char) ('A' + (i - 26) % 26));
            } else {
                codes.add(String.format("%02d", i));
            }
        }
        List<String> pool = new ArrayList<>();
        for (char c = 'A'; c <= 'Z'; c++) {
            pool.add(String.valueOf(c));
        }
        for (int i = 0; i < 100; i++) {
            pool.add(String.format("%02d", i));
        }
        for (char c = 'a'; c <= 'z'; c++) {
            pool.add(String.valueOf(c));
        }
        for (int i = 0; i < 200; i++) {
            pool.add("#" + i);
        }
        for (char a = 'A'; a <= 'Z'; a++) {
            for (char b = 'A'; b <= 'Z'; b++) {
                pool.add("" + a + b);
            }
        }
        List<Integer> used = new ArrayList<>();
        for (String code : codes) {
            if (!take(code, used, texts, groups)) {
                boolean taken = false;
                for (int i = 0; !taken && i < pool.size(); i++) {
                    taken = take(pool.get(i), used, texts, groups);
                }
                if (!taken) {
                    throw new IllegalStateException("no single-token label left for " + options.size() + " options");
                }
            }
        }
        return new D1Labels(texts, groups.toArray(new int[0][]));
    }

    /** Takes {@code code} as an option label: one unused token, its group also the " code" form. */
    private boolean take(String code, List<Integer> used, List<String> texts, List<int[]> groups) {
        int[] toks = service.encode(code, false, false);
        if (toks.length != 1 || used.contains(toks[0])) {
            return false;
        }
        used.add(toks[0]);
        List<Integer> group = new ArrayList<>();
        group.add(toks[0]);
        for (int spaced : singleTokens(List.of(" " + code))) {
            if (spaced != toks[0]) {
                group.add(spaced);
            }
        }
        int[] ids = new int[group.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = group.get(i);
        }
        texts.add(code);
        groups.add(ids);
        return true;
    }

    /** The single tokens of {@code forms}, in order, each form once (llama.cpp's {@code get_single_tokens}). */
    private int[] singleTokens(List<String> forms) {
        List<Integer> out = new ArrayList<>();
        for (String form : forms) {
            int[] toks = service.encode(form, false, false);
            if (toks.length == 1 && !out.contains(toks[0])) {
                out.add(toks[0]);
            }
        }
        int[] ids = new int[out.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = out.get(i);
        }
        return ids;
    }

    /** The label texts of the first {@code n} labels. */
    private List<String> optionLabels(int n) {
        List<String> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(labelTexts[i]);
        }
        return out;
    }

    /** Each label's one-token group. */
    private int[][] labelGroups(int n) {
        int[][] groups = new int[n][];
        for (int i = 0; i < n; i++) {
            groups[i] = new int[]{labelIds[i]};
        }
        return groups;
    }

    /** The options in reverse order (lev's second choice variant). */
    private static List<DecisionQuestions.Option> reversed(List<DecisionQuestions.Option> options) {
        List<DecisionQuestions.Option> out = new ArrayList<>(options);
        java.util.Collections.reverse(out);
        return out;
    }

    /**
     * Renders the systemone template of one question: the question's id, type, instructions and options with their
     * labels, the state, and one media marker per item. h2o-lightning renders its own input, nimble also receives
     * every question of the request, lev sorts every object's keys as it was trained, minicpm-ocr-jev escapes the
     * OCR state, kev flattens the state, the instructions and the option texts, and the lfm3 d1 families break the
     * special-token spellings of the input.
     */
    private String render(DecisionQuestions.Question question, List<DecisionQuestions.Question> all,
                          List<DecisionQuestions.Option> options, List<String> labels, Object state, int nImages,
                          boolean isAudio) {
        Map<String, Object> inp = family.type.equals("h2o-lightning")
                ? h2oInp(state, question) : baseInp(question, all, options, labels, state);
        if (family.buckets == DecisionFamily.Buckets.LEV) {
            @SuppressWarnings("unchecked")
            Map<String, Object> sorted = (Map<String, Object>) DecisionPrompt.sortKeys(inp);
            inp = sorted;
        }
        if (family.read == DecisionFamily.Read.MARKERS) {
            // the input must not carry the marker of the options (llama.cpp's text_marker)
            int mask = service.tokenizer().specialTokens().mask();
            String marker = mask >= 0 ? service.tokenizer().getToken(mask) : "";
            if (!marker.isEmpty()) {
                @SuppressWarnings("unchecked")
                Map<String, Object> sanitized = (Map<String, Object>) DecisionPrompt.replaceText(inp, marker, " ");
                inp = sanitized;
            }
        }
        if (family.read == DecisionFamily.Read.POOLED) {
            // the OCR text must not be parsed as special tokens
            inp.put("state", DecisionText.escapeSpecialTokens((String) state));
        }
        if (family.read == DecisionFamily.Read.KEV) {
            // the kev template takes text: the state, the instructions and the option texts flatten,
            // their special-token spellings broken (llama.cpp's decision_kev_text)
            inp = kevInp(inp);
        }
        boolean spoken = DecisionQuestions.isSpoken(question.instructions());
        if (spoken) {
            // a spoken question is heard where its text would be
            inp.put("instructions", "");
        }
        if (quotes != null && family.type.startsWith("lfm3")) {
            @SuppressWarnings("unchecked")
            Map<String, Object> escaped = (Map<String, Object>) DecisionPrompt.escapeText(inp,
                    quotes::breakTokens);
            inp = escaped;
        }
        inp.put("images", imagesInp(inp, nImages, spoken));
        if (spoken) {
            inp.put("instructions", ChatMessages.MEDIA_MARKER);
        }
        return MiniJinja.render(systemoneTemplate(), inp);
    }

    /** The base template input of the generic families. */
    private Map<String, Object> baseInp(DecisionQuestions.Question question, List<DecisionQuestions.Question> all,
                                        List<DecisionQuestions.Option> options, List<String> labels, Object state) {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("id", question.id());
        inp.put("type", DecisionPrompt.typeName(question.type()));
        inp.put("instructions", question.instructions());
        inp.put("state", state);
        inp.put("options", renderOptions(options, labels));
        if (family.type.equals("nimble")) {
            List<Map<String, Object>> questions = new ArrayList<>();
            for (DecisionQuestions.Question other : all) {
                Map<String, Object> q = new LinkedHashMap<>();
                q.put("id", other.id());
                q.put("type", DecisionPrompt.typeName(other.type()));
                q.put("instructions", other.instructions());
                q.put("options", renderOptions(other.options(), optionLabels(other.options().size())));
                questions.add(q);
            }
            inp.put("questions", questions);
        }
        return inp;
    }

    /**
     * The h2o-lightning template input (llama.cpp's {@code render_h2o}): the instructions are JSON with an indent
     * of 1, the descriptions are default JSON with a score level that is shown as JSON null, the key and the text
     * that ends the user turn keep their Python-stripped forms.
     */
    private Map<String, Object> h2oInp(Object state, DecisionQuestions.Question question) {
        Object instructions = question.instructions();
        String instructionText = DecisionText.lstrip(instructions instanceof String s ? s
                : DecisionText.dumps(instructions, 1, ",", ": ", true));
        List<Map<String, Object>> options = new ArrayList<>();
        int nOptions = question.options().size();
        for (int i = 0; i < nOptions; i++) {
            DecisionQuestions.Option option = question.options().get(i);
            String description;
            if (question.type() == DecisionQuestions.SCORE && option.description() == null) {
                description = "null";
            } else if (option.description() != null) {
                Object raw = option.description();
                description = quotes.quote(raw instanceof String s ? s
                        : DecisionText.dumps(raw, -1, ", ", ": ", false));
            } else {
                description = null;
            }
            String key = quotes.quote(option.key());
            // the user turn is stripped like Python's str.strip(), the template only trims ASCII whitespace,
            // so the text that ends the turn is stripped here, a space keeps a description that is only whitespace
            if (i + 1 == nOptions) {
                boolean isKey = question.type() == DecisionQuestions.CHOICE && description == null;
                if (isKey || description != null) {
                    String text = isKey ? key : description;
                    String stripped = DecisionText.rstrip(text);
                    if (!stripped.equals(text)) {
                        description = stripped.isEmpty() ? " " : stripped;
                    }
                }
            }
            Map<String, Object> shown = new LinkedHashMap<>();
            shown.put("label", labelTexts[i]);
            shown.put("key", key);
            shown.put("description", description);
            options.add(shown);
        }
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("type", DecisionPrompt.typeName(question.type()));
        inp.put("state", state);
        inp.put("instructions", quotes.quote(instructionText));
        inp.put("options", options);
        return inp;
    }

    /** The template's options: their key, description and (when the family labels them) label. */
    private static List<Map<String, Object>> renderOptions(List<DecisionQuestions.Option> options,
                                                           List<String> labels) {
        List<Map<String, Object>> out = new ArrayList<>(options.size());
        for (int i = 0; i < options.size(); i++) {
            Map<String, Object> option = new LinkedHashMap<>();
            option.put("key", options.get(i).key());
            option.put("description", options.get(i).description());
            if (labels != null && !labels.isEmpty()) {
                option.put("label", labels.get(i));
            }
            out.add(option);
        }
        return out;
    }

    /**
     * The kev template input: the state, the instructions and every option's key and description flatten to text,
     * their special-token spellings broken, while the id and the type stay as they are (llama.cpp's
     * {@code decision_kev_text} of exactly those fields).
     */
    private static Map<String, Object> kevInp(Map<String, Object> inp) {
        Map<String, Object> out = new LinkedHashMap<>(inp);
        out.put("state", DecisionText.kevText(inp.get("state")));
        out.put("instructions", DecisionText.kevText(inp.get("instructions")));
        if (inp.get("options") instanceof List<?> options) {
            List<Map<String, Object>> flat = new ArrayList<>(options.size());
            for (Object raw : options) {
                Map<?, ?> option = (Map<?, ?>) raw;
                Map<String, Object> shown = new LinkedHashMap<>();
                shown.put("key", DecisionText.kevText(option.get("key")));
                shown.put("description", option.get("description") == null ? null
                        : DecisionText.kevText(option.get("description")));
                flat.add(shown);
            }
            out.put("options", flat);
        }
        return out;
    }

    /** The media prompt of a rendered text with one media marker per item. */
    private MediaPrompt mediaPrompt(String rendered, List<ChatMedia.Item> items) {
        ChatMedia.TextEncoder encoder = text -> service.encode(text, false, true);
        return service.media().prompt(rendered, items, List.of(), -1, encoder);
    }

    // ------------------------------------------------------------------ scoring

    /** One question's option probabilities from the head's scores: calibrated and softmaxed. */
    private double[] probabilities(DecisionQuestions.Question question, float[] scores, int at, boolean media) {
        int n = question.options().size();
        if (at + n > scores.length) {
            throw new IllegalStateException("decision result does not match the number of options");
        }
        float[] questionScores = new float[n];
        System.arraycopy(scores, at, questionScores, 0, n);
        float temperature = temperature(question, n, media);
        float[] bias = question.type() == DecisionQuestions.SCORE
                ? service.pooling().decisionScoreBias().get(n) : null;
        return DecisionAnswers.probabilities(questionScores, question.type() == DecisionQuestions.SCORE, bias,
                temperature);
    }

    /**
     * The model's temperature for a question, with the request's media temperature when it defines one — and
     * temperature one for a media read of lfm2-d1-omni, whose buckets are for text only (llama.cpp's
     * format_answer reads the uncalibrated logits at temperature one).
     */
    private float temperature(DecisionQuestions.Question question, int nOptions, boolean media) {
        Map<String, Float> temperatures = service.pooling().decisionTemperatures();
        if (media && temperatures.containsKey("image")) {
            return temperatures.get("image");
        }
        if (media && family.uncalibratedMedia && !temperatures.containsKey("image")) {
            return 1.0f;
        }
        return DecisionAnswers.temperature(temperatures, DecisionPrompt.typeName(question.type()),
                family.bucket(nOptions));
    }

    // ------------------------------------------------------------------ the laya prompt cut

    /** One prepared marker read: the truncated tokens and the position of each option's marker. */
    private record LayaInput(int[] tokens, int[] markers) {
    }

    /**
     * The laya prompt prepared as the model was trained (llama.cpp's {@code fill_task_laya}): the question and its
     * options cut to {@code max_head_tokens}, the options evenly shrunk when they do not fit, the state kept whole
     * when it fits.
     *
     * @throws IllegalStateException when the prompt is not [cls] question [sep] (marker option)* [sep] state [sep]
     */
    private LayaInput layaInput(int[] tokens, int nOptions, int maxHeadTokens) {
        int mask = service.tokenizer().specialTokens().mask();
        int sep = service.tokenizer().specialTokens().sep();
        if (mask < 0 || sep < 0) {
            throw new IllegalStateException("decision model has no mask or sep token");
        }
        List<Integer> markers = new ArrayList<>();
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] == mask) {
                markers.add(i);
            }
        }
        if (markers.size() != nOptions || markers.get(0) < 2 || tokens[markers.get(0) - 1] != sep
                || tokens[tokens.length - 1] != sep) {
            throw new IllegalStateException("unexpected layout of the decision prompt");
        }
        int headEnd = markers.get(0) - 1;
        int optsEnd = markers.get(markers.size() - 1);
        while (optsEnd < tokens.length && tokens[optsEnd] != sep) {
            optsEnd++;
        }
        if (optsEnd + 1 >= tokens.length) {
            throw new IllegalStateException("unexpected layout of the decision prompt");
        }
        List<int[]> options = new ArrayList<>();
        for (int i = 0; i < nOptions; i++) {
            int end = i + 1 < nOptions ? markers.get(i + 1) : optsEnd;
            options.add(java.util.Arrays.copyOfRange(tokens, markers.get(i), end));
        }
        int nOptionsTokens = setMax(options, MAX_OPTION_TOKENS + 1);
        if (nOptionsTokens + 16 > maxHeadTokens) {
            setMax(options, Math.max(4, (maxHeadTokens - Math.min(maxHeadTokens, 16)) / nOptions));
        }
        int questionMax = Math.max(8, maxHeadTokens - Math.min(maxHeadTokens, nOptionsTokens));

        List<Integer> out = new ArrayList<>();
        out.add(tokens[0]);
        for (int i = 1; i < Math.min(headEnd, 1 + questionMax); i++) {
            out.add(tokens[i]);
        }
        out.add(sep);
        List<Integer> markerPositions = new ArrayList<>();
        for (int[] option : options) {
            markerPositions.add(out.size());
            for (int token : option) {
                out.add(token);
            }
        }
        for (int i = optsEnd; i < tokens.length; i++) {
            out.add(tokens[i]);
        }
        int[] prepared = new int[out.size()];
        for (int i = 0; i < prepared.length; i++) {
            prepared[i] = out.get(i);
        }
        int[] markerArray = new int[markerPositions.size()];
        for (int i = 0; i < markerArray.length; i++) {
            markerArray[i] = markerPositions.get(i);
        }
        return new LayaInput(prepared, markerArray);
    }

    /** Truncates every option to at most {@code nMax} tokens and answers their total. */
    private static int setMax(List<int[]> options, int nMax) {
        int total = 0;
        for (int i = 0; i < options.size(); i++) {
            int[] option = options.get(i);
            if (option.length > nMax) {
                option = java.util.Arrays.copyOf(option, nMax);
                options.set(i, option);
            }
            total += option.length;
        }
        return total;
    }
}
