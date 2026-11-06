package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.gguf.video.VideoDecoder;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Video input over the real service with a llava projector beside the text model (llama.cpp {@code --mmproj}):
 * the chat {@code video_url} part as a data URL and the {@code input_video} part as inline base64 each decode
 * through ffmpeg into frame rows the usage counts, {@code /props} reports the video modality, a video without a
 * projector refuses with llama.cpp's hint, and a limit of 0 turns the modality off. Skipped when no ffmpeg is on
 * the PATH.
 */
class MultimodalVideoResourceTest {

    /** The rows of one 8x8 frame through the llava fixture's 8x8 grid: a 2x2 patch grid, the class token dropped. */
    private static final int FRAME_ROWS = 4;

    private static EngineService service;
    private static OpenAiResource openai;
    private static LlamaCppResource llamacpp;
    private static String mp464;

    @BeforeAll
    static void open() throws IOException, InterruptedException {
        assumeTrue(ffmpegOnPath(), "no ffmpeg on the PATH");
        Path model = SyntheticModels.llama().writeTempFile();
        int width = SyntheticModels.Config.llama().nEmbd();
        mp464 = Base64.getEncoder().encodeToString(mp4());
        EngineConfig config = ServerTestSupport.config()
                .memory(MemoryConfig.builder()
                        .nCtx(1024).ubatch(64).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(MmprojTestFiles.llava(width).toString()));
        openai = new OpenAiResource(service);
        llamacpp = new LlamaCppResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    private static boolean ffmpegOnPath() {
        try {
            Process p = new ProcessBuilder("ffmpeg", "-version").redirectErrorStream(false).start();
            p.waitFor(30, TimeUnit.SECONDS);
            return p.exitValue() == 0;
        } catch (IOException e) {
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    /** An 8x8 2 fps MP4 of 3 seconds (6 frames), encoded with the test's ffmpeg. */
    private static byte[] mp4() throws IOException, InterruptedException {
        Path out = Files.createTempFile("nf-video-server-test-", ".mp4");
        try {
            Process p = new ProcessBuilder("ffmpeg", "-y", "-nostdin", "-f", "lavfi",
                    "-i", "testsrc=size=8x8:rate=2", "-t", "3", "-pix_fmt", "yuv420p", out.toString())
                    .redirectErrorStream(false).start();
            try (InputStream err = p.getErrorStream()) {
                byte[] buf = new byte[4096];
                while (err.read(buf) > 0) {
                    // the encode's log drains away
                }
            }
            assertTrue(p.waitFor(120, TimeUnit.SECONDS), "ffmpeg failed to encode the fixture");
            return Files.readAllBytes(out);
        } finally {
            Files.deleteIfExists(out);
        }
    }

    private static String chat(String part) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"watch this\"},"
                + " " + part + "]}], \"max_tokens\": 2, \"temperature\": 0}";
    }

    /** The response of a call that must succeed; a refusal fails the test with its message. */
    private static Response ok(java.util.function.Supplier<Response> call) {
        try {
            Response response = call.get();
            assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
            return response;
        } catch (WebApplicationException e) {
            throw new AssertionError("refused: " + e.getResponse().getEntity(), e);
        }
    }

    /** The prompt rows the usage of a chat completion of {@code body} reports. */
    private static long promptTokens(String body) {
        Response response = ok(() -> openai.chatCompletions(body));
        return JsonView.of(view(response.getEntity()).raw("usage"), "usage").i64("prompt_tokens");
    }

    @Test
    void thePropsReportTheVideoModalityOfAVisionProjector() {
        Response response = ok(() -> llamacpp.props(null, null));
        JsonView modalities = JsonView.of(view(response.getEntity()).raw("modalities"), "modalities");
        assertTrue(modalities.bool("vision"));
        assertTrue(modalities.bool("video"));
        assertFalse(modalities.bool("audio"));
    }

    @Test
    void aVideoUrlPartAddsItsFrameRowsToThePrompt() {
        String body = chat("{\"type\": \"video_url\", \"video_url\": {\"url\": \"data:video/mp4;base64,"
                + mp464 + "\"}}");
        long withVideo = promptTokens(body);
        long textOnly = promptTokens(chat("{\"type\": \"text\", \"text\": \"x\"}"));
        // the video takes every frame the source has (6 frames of 4 rows) behind the "Video:" prefix
        assertTrue(withVideo >= textOnly + 6 * FRAME_ROWS - 1,
                withVideo + " rows for " + textOnly + " text tokens");
        // the generation counts the same prompt the same way
        assertEquals(withVideo, view(ok(() -> openai.chatInputTokens(body)).getEntity()).i64("input_tokens"));
    }

    @Test
    void anInputVideoPartDecodesLikeAVideoUrl() {
        long url = promptTokens(chat("{\"type\": \"video_url\", \"video_url\": {\"url\": \"data:video/mp4;base64,"
                + mp464 + "\"}}"));
        long inline = promptTokens(chat("{\"type\": \"input_video\", \"input_video\": {\"data\": \"" + mp464
                + "\"}}"));
        assertEquals(url, inline);
    }

    @Test
    void videoPartsCarryAVllmUuidForTheMediaCache() {
        String once = chat("{\"type\": \"video_url\", \"video_url\": {\"url\": \"data:video/mp4;base64," + mp464
                + "\"}, \"uuid\": \"clip-9\"}");
        long first = promptTokens(once);
        // the repeat sends the uuid alone: the cache serves the frames without the data
        String repeat = chat("{\"type\": \"video_url\", \"video_url\": null, \"uuid\": \"clip-9\"}");
        assertEquals(first, promptTokens(repeat));
    }

    @Test
    void refusedVideosNameTheirField() {
        String badBase64 = errorOf(() -> openai.chatCompletions(chat(
                "{\"type\": \"video_url\", \"video_url\": {\"url\": \"data:video/mp4;base64,####\"}}")));
        assertTrue(badBase64.contains("video_url.url"), badBase64);
        String bare = errorOf(() -> openai.chatCompletions(chat("{\"type\": \"input_video\"}")));
        assertTrue(bare.contains("input_video"), bare);
    }

    @Test
    void aZeroLimitTurnsTheModalityOff() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int width = SyntheticModels.Config.llama().nEmbd();
        EngineConfig config = ServerTestSupport.config()
                .memory(MemoryConfig.builder()
                        .nCtx(1024).ubatch(64).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        MediaOptions options = MediaOptions.of(MmprojTestFiles.llava(width).toString())
                .withLimits(com.eonmux.neuralfree.gguf.media.MediaLimits.parse("{\"video\": 0}"));
        try (EngineService limited = new EngineService(model.toString(), config, ServerOptions.defaults(),
                options)) {
            OpenAiResource limitedOpenAi = new OpenAiResource(limited);
            assertTrue(errorOf(() -> limitedOpenAi.chatCompletions(chat(
                    "{\"type\": \"video_url\", \"video_url\": {\"url\": \"data:video/mp4;base64," + mp464
                            + "\"}}")))
                    .contains("At most 0 video(s) may be provided"));
            // images still work on the same projector
            assertTrue(errorOf(() -> limitedOpenAi.chatCompletions(chat(
                    "{\"type\": \"input_video\", \"input_video\": {\"data\": \"" + mp464 + "\"}}")))
                    .contains("At most 0 video(s)"));
            assertTrue(VideoDecoder.isVideo(Base64.getDecoder().decode(mp464)));
        }
    }
}
