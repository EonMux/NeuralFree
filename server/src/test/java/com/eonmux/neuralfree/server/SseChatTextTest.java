package com.eonmux.neuralfree.server;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** The chat-stream to speech-delta parser: deltas, the finish reason, split writes and error frames. */
class SseChatTextTest {

    @Test
    void joinsContentDeltasAndKeepsTheFinishReason() throws IOException {
        List<String> deltas = new ArrayList<>();
        SseChatText parser = new SseChatText(deltas::add);
        write(parser,
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
                        + "\"finish_reason\":null}]}\n\n"
                        + "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n"
                        + "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n"
                        + "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n"
                        + "data: {\"choices\":[],\"usage\":{\"total_tokens\":3}}\n\n"
                        + "data: [DONE]\n\n");
        parser.finish();
        assertEquals(List.of("Hel", "lo"), deltas);
        assertEquals("Hello", parser.text());
        assertEquals("length", parser.finishReason());
    }

    @Test
    void processesDeltasSplitAcrossWrites() throws IOException {
        List<String> deltas = new ArrayList<>();
        SseChatText parser = new SseChatText(deltas::add);
        byte[] frame = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n"
                .getBytes(StandardCharsets.UTF_8);
        for (byte b : frame) {
            parser.write(b);
        }
        parser.finish();
        assertEquals(List.of("ok"), deltas);
        assertEquals("ok", parser.text());
    }

    @Test
    void anErrorFrameOrABrokenConsumerFailsTheStream() throws IOException {
        SseChatText error = new SseChatText(delta -> { });
        write(error, "data: {\"error\":{\"message\":\"the engine failed\",\"type\":\"server_error\"}}\n\n");
        IOException failure = assertThrows(IOException.class, error::finish);
        assertTrue(String.valueOf(failure.getMessage()).contains("the engine failed"));

        SseChatText broken = new SseChatText(delta -> {
            throw new UncheckedIOException(new IOException("the client went away"));
        });
        IOException gone = assertThrows(IOException.class,
                () -> write(broken, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n"));
        assertEquals("the client went away", gone.getMessage());
    }

    private static void write(SseChatText parser, String frames) throws IOException {
        byte[] bytes = frames.getBytes(StandardCharsets.UTF_8);
        parser.write(bytes, 0, bytes.length);
    }
}
