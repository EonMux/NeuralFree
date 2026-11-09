package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.ops.heads.ModernBertDecisionHead;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The omni d1 families over the real service. lfm2-d1-omni scores the question-type column of every option's mask
 * marker through the media, over a prompt whose pieces are cut to the token budgets the model was trained on, with
 * the state taking the room that is left and an audio clip shrinking the text budget; a spoken question keeps its
 * marker in the text (llama.cpp's d1-omni quirk). lfm3-d1-omni reads the answer codes from its own output logits,
 * its score answer is the median level and its confidence the answer's probability, and the special-token
 * spellings of the input are broken with a zero-width space. Every read is compared against the same computation
 * over the model's own decision scores or output logits on the prompt the family renders.
 */
class DecisionD1OmniTest {

    /** The piece markers of the d1-omni layout (server-decision.cpp:942-945). */
    private static final String SEP = "<<d1omni:sep>>";
    private static final String MARK_STATE = "<<d1omni:state>>";
    private static final String MARK_QUESTION = "<<d1omni:question>>";
    private static final String MARK_OPTION = "<<d1omni:option>>";
    private static final int D1OMNI_MAX_TOKENS = 16384;
    private static final int D1OMNI_MAX_IMAGE = 896;
    private static final int D1OMNI_MAX_AUDIO = 15360;

    private static EngineService omni;
    private static EngineService omniMedia;
    private static EngineService lfm3;
    private static DecisionResource omniResource;
    private static DecisionResource omniMediaResource;
    private static DecisionResource lfm3Resource;
    private static String wav64;
    private static String audioUrl;
    private static String image;

    @BeforeAll
    static void open() throws IOException {
        EngineConfig config = ServerTestSupport.config().build();
        Path omniModel = SyntheticModels.lfm2D1Omni().writeTempFile();
        // the d1-omni read answers up to 16384 tokens in one memory-less step
        omni = new EngineService(omniModel.toString(), ServerTestSupport.config()
                .memory(MemoryConfig.builder().nCtx(16384).ubatch(16384)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build()).build());
        omniResource = new DecisionResource(omni);
        int width = SyntheticModels.Config.lfm2().nEmbd();
        omniMedia = new EngineService(omniModel.toString(), ServerTestSupport.config()
                        .memory(MemoryConfig.builder().nCtx(40960).ubatch(4096).parallel(2)
                                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build(),
                ServerOptions.defaults(),
                MediaOptions.of(MmprojTestFiles.qwen25Omni(width).toString()));
        omniMediaResource = new DecisionResource(omniMedia);
        lfm3 = new EngineService(SyntheticModels.lfm3D1Omni(context()).writeTempFile().toString(), config);
        lfm3Resource = new DecisionResource(lfm3);
        wav64 = Base64.getEncoder().encodeToString(wav(1));
        audioUrl = "data:audio/wav;base64," + wav64;
        image = "data:image/png;base64," + Base64.getEncoder().encodeToString(png());
    }

    @AfterAll
    static void close() {
        for (EngineService s : List.of(omni, omniMedia, lfm3)) {
            if (s != null) {
                s.close();
            }
        }
    }

    private static SyntheticModels.Config context() {
        // the non-causal trunk runs a centred short convolution, which needs an odd kernel
        return SyntheticModels.Config.lfm3().toBuilder().nCtxTrain(1024).ssmConv(3).build();
    }

    /** A 12x8 RGB PNG the projector takes. */
    private static byte[] png() throws IOException {
        java.awt.image.BufferedImage image = new java.awt.image.BufferedImage(12, 8,
                java.awt.image.BufferedImage.TYPE_INT_RGB);
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        javax.imageio.ImageIO.write(image, "png", out);
        return out.toByteArray();
    }

    /** A 16 kHz mono PCM16 sine WAV of {@code seconds}. */
    private static byte[] wav(int seconds) {
        int rate = 16000;
        int frames = seconds * rate;
        int dataBytes = frames * 2;
        ByteBuffer bb = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        bb.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt(36 + dataBytes);
        bb.put("WAVE".getBytes(StandardCharsets.US_ASCII));
        bb.put("fmt ".getBytes(StandardCharsets.US_ASCII)).putInt(16);
        bb.putShort((short) 1).putShort((short) 1).putInt(rate).putInt(rate * 2);
        bb.putShort((short) 2).putShort((short) 16);
        bb.put("data".getBytes(StandardCharsets.US_ASCII)).putInt(dataBytes);
        for (int i = 0; i < frames; i++) {
            bb.putShort((short) (Math.sin(2.0 * Math.PI * 440 * i / rate) * 12000));
        }
        return bb.array();
    }

    private static final String CHOICE = """
            "questions": {"q": {"type": "choice", "instructions": "Pick one",
                "criteria": {"red": "warm", "blue": null}}}""";

    // ------------------------------------------------------------------ lfm2-d1-omni

    @Test
    void theMarkerReadScoresTheChoiceColumnThroughTheBudgets() throws IOException {
        String body = "{\"state\": \"x\", " + CHOICE + "}";
        JsonView answer = view(omniResource.systemone(body).getEntity()).obj("answers").obj("q");
        double[] expected = d1omniProbabilities(omni, "x", "Pick one", false, List.of(),
                new String[] {"red", "blue"}, new Object[] {"warm", null});
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    @Test
    void longStateAndPiecesAreCutToTheirBudgets() throws IOException {
        StringBuilder state = new StringBuilder();
        while (state.length() < 30000) {
            state.append("alpha beta ");
        }
        String body = "{\"state\": \"" + state + "\", " + CHOICE + "}";
        JsonView out = view(omniResource.systemone(body).getEntity());
        JsonView answer = out.obj("answers").obj("q");
        double[] expected = d1omniProbabilities(omni, state.toString(), "Pick one", false, List.of(),
                new String[] {"red", "blue"}, new Object[] {"warm", null});
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    @Test
    void aSpokenQuestionKeepsItsMarkerInTheText() throws IOException {
        // d1-omni never appends the spoken clip to its media: the marker stays a text token in the question piece
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "choice",
                     "instructions": {"type": "input_audio", "input_audio": {"data": "%s", "format": "wav"}},
                     "criteria": {"red": "warm", "blue": null}}}}""".formatted(wav64);
        JsonView answer = view(omniResource.systemone(body).getEntity()).obj("answers").obj("q");
        Map<String, Object> spoken = new LinkedHashMap<>();
        spoken.put("type", "input_audio");
        Map<String, Object> audio = new LinkedHashMap<>();
        audio.put("data", wav64);
        audio.put("format", "wav");
        spoken.put("input_audio", audio);
        double[] expected = d1omniProbabilities(omni, "x", spoken, false, List.of(),
                new String[] {"red", "blue"}, new Object[] {"warm", null});
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    @Test
    void anImageShrinksTheTextBudgetAndTheReadRunsThroughTheProjector() throws IOException {
        String body = "{\"state\": \"the picture\", \"files\": [\"" + image + "\"], " + CHOICE + "}";
        JsonView answer = view(omniMediaResource.systemone(body).getEntity()).obj("answers").obj("q");
        List<ChatMedia.Item> items = omniMedia.media().decisionItems(List.of(image), "files");
        double[] expected = d1omniProbabilities(omniMedia, "the picture", "Pick one", false, items,
                new String[] {"red", "blue"}, new Object[] {"warm", null});
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    @Test
    void audioAndImagesDoNotMix() {
        String body = "{\"state\": \"x\", \"files\": [\"" + audioUrl + "\", \"" + image + "\"], " + CHOICE + "}";
        assertEquals(400, statusOf(() -> omniMediaResource.systemone(body)));
        assertEquals("a request has images or one audio clip, not both",
                errorOf(() -> omniMediaResource.systemone(body)));
    }

    @Test
    void anAudioClipUsesTheAudioTextBudget() throws IOException {
        String body = "{\"state\": \"x\", \"files\": [\"" + audioUrl + "\"], " + CHOICE + "}";
        JsonView answer = view(omniMediaResource.systemone(body).getEntity()).obj("answers").obj("q");
        List<ChatMedia.Item> items = omniMedia.media().decisionItems(List.of(audioUrl), "files");
        assertEquals(com.eonmux.neuralfree.gguf.media.MediaLimits.Modality.AUDIO, items.get(0).modality());
        double[] expected = d1omniProbabilities(omniMedia, "x", "Pick one", true, items,
                new String[] {"red", "blue"}, new Object[] {"warm", null});
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    // ------------------------------------------------------------------ lfm3-d1-omni

    @Test
    void lfm3ReadsTheAnswerCodesAtTheLastPromptToken() throws IOException {
        String body = "{\"state\": \"x\", " + CHOICE + "}";
        JsonView answer = view(lfm3Resource.systemone(body).getEntity()).obj("answers").obj("q");
        String prompt = lfm3Prompt("x", "Pick one", lfm3Option("A", "red", "warm"),
                lfm3Option("B", "blue", null));
        double[] expected = lfm3Probabilities(prompt, 0.7f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
        assertEquals(expected[0] > expected[1] ? "red" : "blue", answer.str("choice"));
        // the confidence of the lfm3 d1 read is the answer's own probability
        assertEquals(Math.max(expected[0], expected[1]), answer.f64("confidence"), 1e-9);
    }

    @Test
    void lfm3NoulReadsTheYesAndNoTokenGroups() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "noul", "instructions": "Does it hold?",
                     "criteria": {"true": "holds", "false": "not"}}}}""";
        JsonView answer = view(lfm3Resource.systemone(body).getEntity()).obj("answers").obj("q");
        // true first; the label of each side is its key, the read takes the best of the yes or no forms
        String prompt = lfm3Prompt("x", "Does it hold?", lfm3Option("true", "true", "holds"),
                lfm3Option("false", "false", "not"));
        int[] tokens = lfm3.encode(prompt, false, true);
        float[] logits = lfm3.pooling().nextLogits(null, tokens);
        float trueScore = groupMax(lfm3, logits, "yes", "Yes", "YES");
        float falseScore = groupMax(lfm3, logits, "no", "No", "NO");
        double[] probs = softmax(new float[] {trueScore, falseScore}, 0.5f);
        assertEquals(probs[0], answer.f64("noul"), 1e-9);
    }

    @Test
    void lfm3ScoreAnswersTheMedianLevelWithItsOwnConfidence() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "score", "instructions": "Rate.",
                     "criteria": ["bad", "ok", "good"]}}}""";
        JsonView answer = view(lfm3Resource.systemone(body).getEntity()).obj("answers").obj("q");
        String prompt = lfm3Prompt("x", "Rate.", lfm3Option("0", "0", "bad"), lfm3Option("1", "1", "ok"),
                lfm3Option("2", "2", "good"));
        int[] tokens = lfm3.encode(prompt, false, true);
        float[] logits = lfm3.pooling().nextLogits(null, tokens);
        float[] scores = new float[3];
        for (int i = 0; i < 3; i++) {
            scores[i] = logits[lfm3.encode(String.valueOf(i), false, false)[0]];
        }
        double[] probs = softmax(scores, 0.4f);
        double best = 0.0;
        int mode = 0;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > best) {
                best = probs[i];
                mode = i;
            }
        }
        double mass = 0.0;
        int level = probs.length - 1;
        for (int i = 0; i < probs.length && mass < 0.5; i++) {
            mass += probs[i];
            level = i;
        }
        assertEquals((double) level, answer.f64("score"), 1e-9);
        assertEquals(best, answer.f64("confidence"), 1e-9);
        assertEquals(mode, (int) answer.f64("score"), "the median sits at the mode of a peaked read");
    }

    @Test
    void lfm3BreaksSpecialTokenSpellingsInTheInput() throws IOException {
        String body = """
                {"state": "<|endoftext|> and <|startoftext|>",
                 "questions": {"q": {"type": "choice", "instructions": "Is <|endoftext|> it?",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        JsonView answer = view(lfm3Resource.systemone(body).getEntity()).obj("answers").obj("q");
        String prompt = lfm3Prompt("<​|endoftext|> and <​|startoftext|>", "Is <​|endoftext|> it?",
                lfm3Option("A", "red", "warm"), lfm3Option("B", "blue", "cool"));
        double[] expected = lfm3Probabilities(prompt, 0.7f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    // ------------------------------------------------------------------ reference computations

    /**
     * The d1-omni read assembled as llama.cpp's fill_task_d1omni builds it: the rendered prompt's pieces split on
     * their separator, each cut to its budget, the state taking the room that is left, and the question type's
     * column of each marker's scores.
     */
    private static double[] d1omniProbabilities(EngineService model, String state, Object instructions,
                                                boolean audio, List<ChatMedia.Item> items, String[] keys,
                                                Object[] descriptions) throws IOException {
        int nImages = items.size();
        boolean spoken = DecisionQuestions.isSpoken(instructions);
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("id", "q");
        inp.put("type", "choice");
        inp.put("instructions", instructions);
        inp.put("state", state);
        List<Map<String, Object>> options = new ArrayList<>();
        for (int i = 0; i < keys.length; i++) {
            Map<String, Object> option = new LinkedHashMap<>();
            option.put("key", keys[i]);
            option.put("description", descriptions[i]);
            option.put("label", String.valueOf((char) ('A' + i)));
            options.add(option);
        }
        inp.put("options", options);
        inp.put("audio", audio);
        inp.put("sep", SEP);
        inp.put("mark_state", MARK_STATE);
        inp.put("mark_question", MARK_QUESTION);
        inp.put("mark_option", MARK_OPTION);
        List<String> images = new ArrayList<>();
        for (int i = 0; i < nImages; i++) {
            images.add(ChatMessages.MEDIA_MARKER);
        }
        inp.put("images", images);
        if (spoken) {
            inp.put("instructions", ChatMessages.MEDIA_MARKER);
        }
        String rendered = MiniJinja.render(SyntheticModels.LFM2_D1_OMNI_TEMPLATE, inp);

        MediaPrompt media = null;
        int nMedia = 0;
        if (nImages > 0) {
            media = model.media().prompt(ChatMessages.MEDIA_MARKER.repeat(nImages), items, List.of(), -1,
                    text -> model.encode(text, false, true));
            nMedia = media.nTokens();
        }
        int nMax = D1OMNI_MAX_TOKENS;
        if (nImages > 0) {
            nMax = Math.min(audio ? D1OMNI_MAX_AUDIO : D1OMNI_MAX_IMAGE,
                    D1OMNI_MAX_TOKENS - Math.min(D1OMNI_MAX_TOKENS, nMedia));
        }
        int nOptions = keys.length;
        int nBudget = Math.max(96, Math.min(24 * nOptions + 32, nMax / 2));
        int nOptionMax = Math.max(2, (nBudget - 3 * nOptions) / nOptions);
        int nQuestionMax = Math.max(16, nBudget);

        List<String> pieces = new ArrayList<>();
        int start = 0;
        int at;
        while ((at = rendered.indexOf(SEP, start)) >= 0) {
            pieces.add(rendered.substring(start, at));
            start = at + SEP.length();
        }
        pieces.add(rendered.substring(start));
        assertEquals(ChatMessages.MEDIA_MARKER.repeat(nImages), pieces.get(0), "the media markers stand first");

        List<Integer> head = new ArrayList<>();
        List<Integer> bodyTokens = new ArrayList<>();
        List<Integer> tail = new ArrayList<>();
        boolean hasState = false;
        for (int i = 1; i < pieces.size(); i++) {
            String piece = pieces.get(i);
            int nPieceMax = -1;
            boolean isState = false;
            if (piece.startsWith(MARK_STATE)) {
                piece = piece.substring(MARK_STATE.length());
                isState = true;
            } else if (piece.startsWith(MARK_QUESTION)) {
                piece = piece.substring(MARK_QUESTION.length());
                nPieceMax = nQuestionMax;
            } else if (piece.startsWith(MARK_OPTION)) {
                piece = piece.substring(MARK_OPTION.length());
                nPieceMax = nOptionMax;
            }
            List<Integer> tokens = new ArrayList<>();
            for (int token : model.encode(piece, false, true)) {
                tokens.add(token);
            }
            if (nPieceMax >= 0 && tokens.size() > nPieceMax) {
                tokens = tokens.subList(0, nPieceMax);
            }
            if (isState) {
                bodyTokens = tokens;
                hasState = true;
            } else if (hasState) {
                tail.addAll(tokens);
            } else {
                head.addAll(tokens);
            }
        }
        assertTrue(hasState, "the state piece stands between the head and the tail");
        int nRoom = nMax - Math.min(nMax, head.size() + tail.size());
        if (bodyTokens.size() > nRoom) {
            bodyTokens = bodyTokens.subList(0, nRoom);
        }
        List<Integer> out = new ArrayList<>(head);
        out.addAll(bodyTokens);
        out.addAll(tail);
        if (out.size() > nMax) {
            out = out.subList(0, nMax);
        }
        int[] tokens = new int[out.size()];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = out.get(i);
        }
        int mask = model.tokenizer().specialTokens().mask();
        List<Integer> markers = new ArrayList<>();
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] == mask) {
                markers.add(i);
            }
        }
        assertEquals(nOptions, markers.size(), "one marker per option survives the cuts");
        float[] column = new float[nOptions];
        if (media == null) {
            float[] scores = model.pooling().decisionScores(null, tokens, null);
            for (int i = 0; i < nOptions; i++) {
                column[i] = scores[markers.get(i) * ModernBertDecisionHead.N_DECISION_TYPES
                        + DecisionQuestions.CHOICE];
            }
        } else {
            MediaPrompt.Builder full = MediaPrompt.builder();
            for (MediaChunk chunk : media.chunks()) {
                full.add(chunk);
            }
            full.addText(tokens);
            float[] scores = model.pooling().decisionScores(null, full.build(), null);
            for (int i = 0; i < nOptions; i++) {
                column[i] = scores[markers.get(i) * ModernBertDecisionHead.N_DECISION_TYPES
                        + DecisionQuestions.CHOICE];
            }
        }
        // the media read of lfm2-d1-omni answers the uncalibrated logits (temperature one); the text read
        // takes the checkpoint's choice bucket
        return softmax(column, media == null ? 0.9f : 1.0f);
    }

    /** The lfm3 d1 prompt: the d1 template with the think marker over the rendered input. */
    private static String lfm3Prompt(Object state, Object instructions,
                                     Map<String, Object>... options) {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("id", "q");
        inp.put("type", "choice");
        inp.put("instructions", instructions);
        inp.put("state", state);
        inp.put("options", List.of(options));
        inp.put("images", List.of());
        return MiniJinja.render(SyntheticModels.LFM2_D1_TEMPLATE + "<think></think>", inp);
    }

    /** One rendered option of the d1 template: its label, key and description. */
    private static Map<String, Object> lfm3Option(String label, String key, Object description) {
        Map<String, Object> option = new LinkedHashMap<>();
        option.put("label", label);
        option.put("key", key);
        option.put("description", description);
        return option;
    }

    /** The label read over the model's own output logits at the two answer codes. */
    private static double[] lfm3Probabilities(String prompt, float temperature) throws IOException {
        int[] tokens = lfm3.encode(prompt, false, true);
        float[] logits = lfm3.pooling().nextLogits(null, tokens);
        float[] scores = new float[2];
        String[] labels = {"A", "B"};
        for (int i = 0; i < 2; i++) {
            scores[i] = logits[lfm3.encode(labels[i], false, false)[0]];
        }
        return softmax(scores, temperature);
    }

    /** The best logit of one answer's token forms. */
    private static float groupMax(EngineService model, float[] logits, String... forms) throws IOException {
        float best = Float.NEGATIVE_INFINITY;
        for (String form : forms) {
            int[] ids = model.encode(form, false, false);
            assertEquals(1, ids.length, form + " must be a single token");
            best = Math.max(best, logits[ids[0]]);
        }
        return best;
    }

    /** The softmax the answers apply: the max-shifted exponential over the temperature. */
    private static double[] softmax(float[] scores, float temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (float score : scores) {
            max = Math.max(max, score);
        }
        double[] out = new double[scores.length];
        double sum = 0.0;
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.exp((scores[i] - max) / temperature);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) {
            out[i] /= sum;
        }
        return out;
    }
}
