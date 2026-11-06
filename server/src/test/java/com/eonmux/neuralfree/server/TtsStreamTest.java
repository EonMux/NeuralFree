package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.WavWriter;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The streamed speech response: byte layout, server-sent events and disconnect cancellation. */
class TtsStreamTest {

    private static final int RATE = 8000;

    @Test
    void aCancelledClientStopsAfterTheChunkItInterrupted() throws IOException {
        AtomicBoolean cancelled = new AtomicBoolean();
        ByteArrayOutputStream out = new ByteArrayOutputStream() {
            @Override
            public synchronized void write(byte[] bytes, int off, int length) {
                super.write(bytes, off, length);
                if (size() > WavWriter.HEADER_BYTES) {
                    cancelled.set(true);
                }
            }
        };
        TtsStream stream = new TtsStream(sink -> {
            sink.accept(clip(0.5f));
            sink.accept(clip(0.25f));
            sink.accept(clip(0.125f));
        }, cancelled::get, RATE, true, false);
        stream.write(out);
        byte[] body = out.toByteArray();
        assertEquals(WavWriter.HEADER_BYTES + 4, body.length,
                "the header and the first chunk, nothing after the disconnect");
        byte[] first = WavWriter.pcmChunk(new float[]{0.5f, 0.5f}, 0, 2);
        assertEquals(4, first.length);
        assertEquals(-1, WavWriter.dataSamples(body));
    }

    @Test
    void rawPcmStreamsWithoutAHeader() throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        TtsStream stream = new TtsStream(sink -> sink.accept(clip(0.5f)), () -> false, RATE, false, false);
        stream.write(out);
        assertEquals(4, out.size());
    }

    @Test
    void sseStreamsDeltasAndADoneEvent() throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        TtsStream stream = new TtsStream(sink -> sink.accept(clip(0.5f)), () -> false, RATE, true, true);
        stream.write(out);
        String text = new String(out.toByteArray(), StandardCharsets.UTF_8);
        assertTrue(text.contains("event: speech.audio.delta"));
        assertTrue(text.contains("\"type\":\"speech.audio.delta\""));
        assertTrue(text.contains("event: speech.audio.done"));
        assertTrue(text.contains("\"type\":\"speech.audio.done\""));
        assertTrue(text.contains("\"usage\""));
        assertFalse(text.contains("[DONE]"));
    }

    @Test
    void awaitFirstRethrowsTheSourcesRefusal() {
        TtsStream stream = new TtsStream(sink -> {
            throw new UnsupportedOperationException("no voice preset");
        }, () -> false, RATE, true, false);
        UnsupportedOperationException thrown = org.junit.jupiter.api.Assertions.assertThrows(
                UnsupportedOperationException.class, stream::awaitFirst);
        assertEquals("no voice preset", thrown.getMessage());
    }

    @Test
    void aFailureAfterTheStreamStartedAbortsIt() {
        TtsStream stream = new TtsStream(sink -> {
            throw new IllegalStateException("the decoder broke");
        }, () -> false, RATE, true, false);
        IOException thrown = org.junit.jupiter.api.Assertions.assertThrows(IOException.class,
                () -> stream.write(new ByteArrayOutputStream()));
        assertTrue(thrown.getMessage().contains("the decoder broke"));
    }

    private static AudioClip clip(float value) {
        return new AudioClip(new float[]{value, value}, RATE);
    }
}
