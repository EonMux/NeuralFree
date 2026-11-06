package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.WavWriter;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.speech.Lfm3aBackboneFixture;
import com.eonmux.neuralfree.gguf.speech.Lfm3aCodec;
import com.eonmux.neuralfree.gguf.speech.Lfm3aGenClip;
import com.eonmux.neuralfree.gguf.speech.Lfm3aGenClipFixture;
import com.eonmux.neuralfree.gguf.speech.Lfm3aSpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.SpeechOptions;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Chat completions that answer with speech ({@code modalities}/{@code output_modalities} naming {@code audio},
 * OpenAI's audio output) over the real service with the synthetic lfm3a omni backbone beside its generator file:
 * an audio-only request runs a fresh AUDIO turn that speaks and stops at the end-of-audio frame, a text+audio
 * request a MIXED turn that interleaves the fixture's filler text with audio blocks, and the answer carries the
 * OpenAI audio object under {@code message.audio} (base64 data, the turn's text as {@code transcript}). The
 * waveform equals a direct omni turn over the same prompt and options, and every refusal names its piece.
 */
class OmniChatAudioResourceTest {

    /** {@code max_tokens} caps the sampled steps, as llama.cpp's {@code n_predict} does. */
    private static final int AUDIO_STEPS = 4;
    /** Six text slots open a MIXED turn, then the audio block. */
    private static final int MIXED_STEPS = Lfm3aSpeechGenerator.N_TEXT_BLOCK + 2;
    private static final int CHUNK_FIRST = Lfm3aCodec.DEFAULT_FRAME_STACK * Lfm3aCodec.DEFAULT_HOP
            - (Lfm3aCodec.DEFAULT_N_FFT - Lfm3aCodec.DEFAULT_HOP) / 2;
    private static final int CHUNK_REST = Lfm3aCodec.DEFAULT_FRAME_STACK * Lfm3aCodec.DEFAULT_HOP;
    /** The flush drains {@code n_fft - hop} and drops its own tail padding (mtmd.cpp:2371-2373). */
    private static final int FLUSH = Lfm3aCodec.DEFAULT_N_FFT - Lfm3aCodec.DEFAULT_HOP
            - (Lfm3aCodec.DEFAULT_N_FFT - Lfm3aCodec.DEFAULT_HOP) / 2;
    private static final int AUDIO_SAMPLES = CHUNK_FIRST + (AUDIO_STEPS - 1) * CHUNK_REST + FLUSH;
    private static final int MIXED_SAMPLES = CHUNK_FIRST + CHUNK_REST + FLUSH;
    /** The fixture's filler piece, what the interleaved text slots sample when the fillers win. */
    private static final String FILLER = "<|filler_0|>";

    private static Path backbone;
    private static Path fillerBackbone;
    private static Path generator;
    private static Path bareBackbone;
    private static Path llamaFile;
    private static int audioPromptTokens;
    private static EngineService omni;
    private static EngineService interleaved;
    private static EngineService noGenerator;
    private static EngineService llama;

    @BeforeAll
    static void open() throws IOException {
        backbone = new Lfm3aBackboneFixture(false, true).write();
        fillerBackbone = new Lfm3aBackboneFixture(false, true, true).write();
        generator = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false).write();
        bareBackbone = new Lfm3aBackboneFixture(false, true).write();
        llamaFile = SyntheticModels.llama().writeTempFile();
        audioPromptTokens = promptTokens(backbone, "Hi");
        omni = service(backbone, MediaOptions.of(generator.toString()));
        interleaved = service(fillerBackbone, MediaOptions.of(generator.toString()));
        noGenerator = service(bareBackbone, MediaOptions.none());
        llama = new EngineService(llamaFile.toString(), ServerTestSupport.pagedConfig(64, 1));
    }

    /** The tokens of the conversation's render plus the fresh turn's audio start token. */
    private static int promptTokens(Path model, String text) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), ServerTestSupport.config()
                .memory(MemoryConfig.builder().nCtx(128).ubatch(32)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build()).build())) {
            String rendered = m.tokenizer().formatSpeechChat(chat(text), "", "");
            return m.tokenizer().encode(rendered, false, true).length + 1;
        }
    }

    private static ChatRequest chat(String text) {
        return ChatRequest.builder().addMessage(Map.of("role", "user", "content", text)).build();
    }

    /** A service over {@code model} with the given media options, no warmup decode. */
    private static EngineService service(Path model, MediaOptions media) throws IOException {
        EngineConfig config = EngineConfig.builder().backend(ServerTestSupport.backend())
                .sourceOptions(ModelSourceOptions.builder().warmup(false).build())
                .memory(MemoryConfig.builder().nCtx(128).ubatch(32).parallel(1)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        return new EngineService(model.toString(), config, ServerOptions.defaults(), media);
    }

    @AfterAll
    static void close() throws IOException {
        omni.close();
        interleaved.close();
        noGenerator.close();
        llama.close();
        Files.deleteIfExists(backbone);
        Files.deleteIfExists(fillerBackbone);
        Files.deleteIfExists(generator);
        Files.deleteIfExists(bareBackbone);
        Files.deleteIfExists(llamaFile);
    }

    private static JsonView view(Object entity) {
        return JsonView.of(Json.parse(String.valueOf(entity), Json.Limits.UNTRUSTED), "body");
    }

    /** The first choice of a chat-completions body or stream chunk. */
    private static JsonView firstChoice(Object entity) {
        List<?> choices = (List<?>) view(entity).raw("choices");
        return JsonView.of(choices.get(0), "choices[0]");
    }

    /** The spoken-answer request body: one user message, the modalities and the step cap. */
    private static String body(String modalities, int steps, String extra) {
        return "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"modalities\":" + modalities
                + ",\"seed\":0,\"max_tokens\":" + steps + (extra.isEmpty() ? "" : ", " + extra) + "}";
    }

    /** The {@code data:} payloads of an SSE stream, the trailing {@code [DONE]} stripped. */
    private static List<String> dataFrames(String sse) {
        List<String> frames = new ArrayList<>();
        for (String line : sse.split("\n\n")) {
            if (line.startsWith("data: ") && !line.equals("data: [DONE]")) {
                frames.add(line.substring("data: ".length()));
            }
        }
        return frames;
    }

    @Test
    void spokenAnswerCarriesTheOpenAiAudioObject() throws IOException {
        Response response = new OpenAiResource(omni).chatCompletions(body("[\"audio\"]", AUDIO_STEPS, ""));
        assertEquals(200, response.getStatus());
        JsonView answer = view(response.getEntity());
        assertEquals("chat.completion", answer.optStr("object", ""));
        assertTrue(answer.optStr("id", "").startsWith("chatcmpl-"));
        long created = answer.optI64("created", 0);
        JsonView message = firstChoice(response.getEntity()).obj("message");
        assertEquals("assistant", message.optStr("role", ""));
        assertNull(message.raw("content"), "the audio-only modality asks for no text");
        JsonView audio = message.obj("audio");
        assertTrue(audio.optStr("id", "").startsWith("audio-"));
        assertEquals("wav", audio.optStr("format", ""));
        assertEquals(created + 3600, audio.optI64("expires_at", 0));
        assertEquals("", audio.optStr("transcript", ""), "a fresh speech turn emits no text");
        byte[] wav = Base64.getDecoder().decode(audio.optStr("data", ""));
        assertArrayEquals("RIFF".getBytes(StandardCharsets.UTF_8), Arrays.copyOf(wav, 4));
        assertEquals(AUDIO_SAMPLES, WavWriter.dataSamples(wav));
        AudioClip direct = directTurn(backbone, chat("Hi"), false, AUDIO_STEPS);
        assertArrayEquals(WavWriter.bytes(direct.samples(), direct.sampleRate()), wav,
                "the waveform equals a direct omni turn over the same prompt and options");
        assertEquals("stop", firstChoice(response.getEntity()).optStr("finish_reason", ""));
        JsonView usage = JsonView.of(answer.raw("usage"), "usage");
        assertEquals(audioPromptTokens, usage.optI64("prompt_tokens", 0));
        assertEquals(AUDIO_STEPS, usage.optI64("completion_tokens", 0));
        assertEquals(audioPromptTokens + AUDIO_STEPS, usage.optI64("total_tokens", 0));
    }

    @Test
    void textAndAudioInterleaveThroughTheConversation() throws IOException {
        Response response = new OpenAiResource(interleaved).chatCompletions(
                body("[\"text\",\"audio\"]", MIXED_STEPS, ""));
        assertEquals(200, response.getStatus());
        JsonView message = firstChoice(response.getEntity()).obj("message");
        String text = FILLER.repeat(Lfm3aSpeechGenerator.N_TEXT_BLOCK);
        assertEquals(text, message.optStr("content", ""),
                "the MIXED turn's first block is text, then the audio block");
        JsonView audio = message.obj("audio");
        assertEquals(text, audio.optStr("transcript", ""));
        byte[] wav = Base64.getDecoder().decode(audio.optStr("data", ""));
        assertEquals(MIXED_SAMPLES, WavWriter.dataSamples(wav));
        AudioClip direct = directTurn(fillerBackbone, chat("Hi"), true, MIXED_STEPS);
        assertArrayEquals(WavWriter.bytes(direct.samples(), direct.sampleRate()), wav);
        JsonView usage = JsonView.of(view(response.getEntity()).raw("usage"), "usage");
        assertEquals(MIXED_STEPS, usage.optI64("completion_tokens", 0));
    }

    @Test
    void outputModalitiesIsTheSameRequest() {
        Response a = new OpenAiResource(omni).chatCompletions(body("[\"audio\"]", AUDIO_STEPS, ""));
        Response b = new OpenAiResource(omni).chatCompletions(
                body("[\"audio\"]", AUDIO_STEPS, "").replace("modalities", "output_modalities"));
        assertEquals(200, a.getStatus());
        assertEquals(200, b.getStatus());
        assertEquals(view(a.getEntity()).optObj("usage").optI64("completion_tokens", 0),
                view(b.getEntity()).optObj("usage").optI64("completion_tokens", 0));
    }

    @Test
    void theAudioObjectTakesTheOpenAiFormatAndVoice() {
        Response wav = new OpenAiResource(omni).chatCompletions(body("[\"audio\"]", AUDIO_STEPS,
                "\"audio\": {\"voice\": \"female\", \"format\": \"wav\"}"));
        assertEquals(200, wav.getStatus());
        JsonView audio = firstChoice(wav.getEntity()).obj("message").obj("audio");
        assertEquals("wav", audio.optStr("format", ""));

        Response pcm = new OpenAiResource(omni).chatCompletions(body("[\"audio\"]", AUDIO_STEPS,
                "\"audio\": {\"format\": \"pcm16\"}"));
        assertEquals(200, pcm.getStatus());
        JsonView pcmAudio = firstChoice(pcm.getEntity()).obj("message").obj("audio");
        assertEquals("pcm16", pcmAudio.optStr("format", ""));
        byte[] raw = Base64.getDecoder().decode(pcmAudio.optStr("data", ""));
        assertEquals(2 * AUDIO_SAMPLES, raw.length, "raw little-endian int16");
    }

    @Test
    void streamedAnswerCarriesDeltaTextAndAudio() throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        Response response = new OpenAiResource(interleaved).chatCompletions(
                body("[\"text\",\"audio\"]", MIXED_STEPS,
                        "\"stream\": true, \"stream_options\": {\"include_usage\": true}"));
        assertEquals(200, response.getStatus());
        ((StreamingOutput) response.getEntity()).write(out);
        String sse = out.toString(StandardCharsets.UTF_8);
        assertTrue(sse.endsWith("data: [DONE]\n\n"), sse);
        List<String> frames = dataFrames(sse);
        assertEquals(Lfm3aSpeechGenerator.N_TEXT_BLOCK + 3, frames.size(),
                "role, six text deltas, audio, stop");
        assertEquals("assistant", firstChoice(frames.get(0)).obj("delta").optStr("role", ""));
        StringBuilder text = new StringBuilder();
        for (int i = 1; i <= Lfm3aSpeechGenerator.N_TEXT_BLOCK; i++) {
            text.append(firstChoice(frames.get(i)).obj("delta").optStr("content", ""));
        }
        assertEquals(FILLER.repeat(Lfm3aSpeechGenerator.N_TEXT_BLOCK), text.toString());
        JsonView audio = firstChoice(frames.get(frames.size() - 2)).obj("delta").obj("audio");
        assertTrue(audio.optStr("id", "").startsWith("audio-"));
        assertEquals(MIXED_SAMPLES, WavWriter.dataSamples(Base64.getDecoder().decode(audio.optStr("data", ""))));
        JsonView stop = firstChoice(frames.get(frames.size() - 1));
        assertEquals("stop", stop.optStr("finish_reason", ""));
        assertEquals(MIXED_STEPS, view(frames.get(frames.size() - 1)).optObj("usage").optI64("completion_tokens", 0),
                "the stop chunk carries the usage at the frame's top level");
    }

    @Test
    void refusalsNameTheirField() {
        OpenAiResource resource = new OpenAiResource(omni);
        assertEquals(400, statusOf(() -> resource.chatCompletions(
                "{\"messages\":[],\"modalities\":[\"image\"]}")));
        assertTrue(errorOf(() -> resource.chatCompletions("{\"messages\":[],\"modalities\":[\"image\"]}"))
                .contains("image"));
        assertEquals(400, statusOf(() -> resource.chatCompletions(
                "{\"messages\":[],\"modalities\":[\"audio\"],\"output_modalities\":[\"audio\"]}")));
        assertTrue(errorOf(() -> resource.chatCompletions(
                "{\"messages\":[],\"modalities\":[\"audio\"],\"output_modalities\":[\"audio\"]}"))
                .contains("output_modalities"));
        assertEquals(400, statusOf(() -> resource.chatCompletions(
                "{\"messages\":[],\"modalities\":[\"audio\"],\"n\":2}")));
        assertTrue(errorOf(() -> resource.chatCompletions(
                "{\"messages\":[],\"modalities\":[\"audio\"],\"n\":2}")).contains("one choice"));
        assertEquals(400, statusOf(() -> resource.chatCompletions(body("[\"audio\"]", AUDIO_STEPS,
                "\"top_p\": 1.5"))));
        assertTrue(errorOf(() -> resource.chatCompletions(body("[\"audio\"]", AUDIO_STEPS, "\"top_p\": 1.5")))
                .contains("topP"), "the sampler's refusal names the parameter");
        assertEquals(400, statusOf(() -> resource.chatCompletions(body("[\"audio\"]", AUDIO_STEPS,
                "\"audio\": {\"format\": \"mp3\"}"))));
        assertTrue(errorOf(() -> resource.chatCompletions(body("[\"audio\"]", AUDIO_STEPS,
                "\"audio\": {\"format\": \"mp3\"}"))).contains("mp3"));
        assertEquals(400, statusOf(() -> resource.chatCompletions(body("[\"audio\"]", AUDIO_STEPS,
                "\"audio\": {\"voice\": \"female\"}, \"voice\": \"male\""))));
        assertTrue(errorOf(() -> resource.chatCompletions(body("[\"audio\"]", AUDIO_STEPS,
                "\"audio\": {\"voice\": \"female\"}, \"voice\": \"male\""))).contains("voice"));
    }

    @Test
    void modelsWithoutTheOmniPipelineRefuseByName() {
        String audio = body("[\"audio\"]", AUDIO_STEPS, "");
        OpenAiResource silent = new OpenAiResource(llama);
        assertEquals(400, statusOf(() -> silent.chatCompletions(audio)));
        assertTrue(errorOf(() -> silent.chatCompletions(audio)).contains("no speech-generator backbone"));
        OpenAiResource bare = new OpenAiResource(noGenerator);
        assertEquals(400, statusOf(() -> bare.chatCompletions(audio)));
        assertTrue(errorOf(() -> bare.chatCompletions(audio)).contains("generator"));
    }

    /**
     * The same turn the service speaks, run straight over the fixtures: the conversation render plus the fresh
     * turn's audio start token, or the continued turn's mixed start token, over the same sampler.
     */
    private static AudioClip directTurn(Path modelPath, ChatRequest chat, boolean mixed, int steps)
            throws IOException {
        try (GGUFModel model = new GGUFModel(modelPath.toString(), ServerTestSupport.config()
                .memory(MemoryConfig.builder().nCtx(128).ubatch(32)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build()).build());
             Lfm3aGenClip clip = Lfm3aGenClip.open(generator.toString());
             Lfm3aSpeechGenerator generator = Lfm3aSpeechGenerator.open(model, clip)) {
            JsonView request = view(body(mixed ? "[\"text\",\"audio\"]" : "[\"audio\"]", steps, ""));
            SpeechOptions options = SpeechOptions.defaults().withSeed(0)
                    .withSampler(SamplingParams.of(request)).withMaxFrames(steps);
            String rendered = model.tokenizer().formatSpeechChat(chat, "", "");
            int[] prompt = model.tokenizer().encode(rendered, false, true);
            Lfm3aSpeechGenerator.Mode mode = mixed ? Lfm3aSpeechGenerator.Mode.MIXED
                    : Lfm3aSpeechGenerator.Mode.AUDIO;
            if (!mixed) {
                int audioStart = model.tokenizer().getTokenId(Lfm3aSpeechGenerator.AUDIO_START_PIECE);
                prompt = Arrays.copyOf(prompt, prompt.length + 1);
                prompt[prompt.length - 1] = audioStart;
            }
            return generator.generate(prompt, mode, mixed, options, null).audio();
        }
    }
}
