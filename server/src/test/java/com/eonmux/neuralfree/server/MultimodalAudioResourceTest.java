package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Audio input over the real service with a whisper / qwen2a projector beside the text model: the chat
 * {@code input_audio} part, the Responses {@code input_audio} part and {@code /completion} {@code multimodal_data}
 * WAV entries each add the clip's 750 encoder rows, the token count equals the usage, {@code /props} reports audio
 * and not vision, and an image part is refused with llama.cpp's hint.
 */
class MultimodalAudioResourceTest {

    /** The rows of one 30 s whisper window after the conv front end and the qwen2a average pool. */
    private static final int AUDIO_ROWS = 750;

    private static EngineService service;
    private static OpenAiResource openai;
    private static String wav64;

    @BeforeAll
    static void open() throws IOException {
        java.nio.file.Path model = SyntheticModels.llama().writeTempFile();
        int width = SyntheticModels.Config.llama().nEmbd();
        wav64 = Base64.getEncoder().encodeToString(wav(1));
        EngineConfig config = ServerTestSupport.config()
                .memory(MemoryConfig.builder()
                        .nCtx(2048).ubatch(64).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(MmprojTestFiles.qwen2aAudio(width).toString()));
        openai = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
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

    private static String chat(String part) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"hear\"}, "
                + part + "]}], \"max_tokens\": 2, \"temperature\": 0}";
    }

    /** The response of a call that must succeed; a refusal fails the test with its message. */
    private static Response ok(java.util.function.Supplier<Response> call) {
        try {
            Response response = call.get();
            assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
            return response;
        } catch (jakarta.ws.rs.WebApplicationException e) {
            throw new AssertionError("refused: " + e.getResponse().getEntity(), e);
        }
    }

    @Test
    void anInputAudioPartAddsItsEncoderRowsAndTheCountMatchesTheUsage() {
        String body = chat("{\"type\": \"input_audio\", \"input_audio\": {\"data\": \"" + wav64
                + "\", \"format\": \"wav\"}}");
        Response response = ok(() -> openai.chatCompletions(body));
        long prompt = JsonView.of(view(response.getEntity()).raw("usage"), "usage").i64("prompt_tokens");
        assertTrue(prompt > AUDIO_ROWS, prompt + " prompt rows");
        assertEquals(prompt, view(openai.chatInputTokens(body).getEntity()).i64("input_tokens"));

        ResponsesResource responses = new ResponsesResource(service);
        Response viaResponses = ok(() -> responses.create("{\"input\": [{\"role\": \"user\", \"content\": [{\"type\":"
                + " \"input_text\", \"text\": \"hear\"}, {\"type\": \"input_audio\", \"input_audio\": {\"data\": \""
                + wav64 + "\", \"format\": \"wav\"}}]}], \"max_output_tokens\": 2, \"temperature\": 0}"));
        assertEquals(prompt, JsonView.of(view(viaResponses.getEntity()).raw("usage"), "usage").i64("input_tokens"));
    }

    @Test
    void anAudioUuidLetsALaterPartOmitItsData() {
        long named = promptTokens(chat("{\"type\": \"input_audio\", \"input_audio\": {\"data\": \"" + wav64
                + "\", \"format\": \"wav\"}, \"uuid\": \"clip-1\"}"));
        assertEquals(named, promptTokens(chat("{\"type\": \"input_audio\", \"uuid\": \"clip-1\"}")));
        assertEquals(named, promptTokens(chat("{\"type\": \"input_audio\", \"input_audio\": {\"format\": \"wav\"},"
                + " \"uuid\": \"clip-1\"}")));
        assertTrue(errorOf(() -> openai.chatCompletions(chat("{\"type\": \"input_audio\", \"uuid\": \"clip-2\"}")))
                .contains("uuid 'clip-2'"));
        assertTrue(errorOf(() -> openai.chatCompletions(chat("{\"type\": \"input_audio\", \"input_audio\": {}}")))
                .contains("requires 'data'"));
    }

    private static long promptTokens(String body) {
        Response response = ok(() -> openai.chatCompletions(body));
        return JsonView.of(view(response.getEntity()).raw("usage"), "usage").i64("prompt_tokens");
    }

    @Test
    void completionMultimodalDataSniffsAudio() {
        Response response = ok(() -> new LlamaCompletionResource(service).completion("{\"prompt\": {\"prompt_string\":"
                + " \"<__media__>\", \"multimodal_data\": [\"" + wav64 + "\"]}, \"n_predict\": 1}"));
        assertTrue(view(response.getEntity()).i64("tokens_evaluated") > AUDIO_ROWS);
    }

    @Test
    void imagesAreRefusedAndPropsReportAudioOnly() {
        assertTrue(errorOf(() -> openai.chatCompletions(chat("{\"type\": \"image_url\", \"image_url\":"
                + " {\"url\": \"data:image/png;base64,AAAA\"}}"))).contains("image input is not supported - hint"));
        JsonView modalities = view(new LlamaCppResource(service).props(null, null).getEntity()).obj("modalities");
        assertTrue(modalities.bool("audio"));
        assertFalse(modalities.bool("vision"));
    }
}
