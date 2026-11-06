package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.WavWriter;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/**
 * One streamed speech response: the waveform chunks a {@link Source} produces are written as they arrive,
 * either as raw audio bytes (a WAV header with the unknown-length sentinel, then 16-bit PCM runs; or raw PCM
 * when {@code wav} is false), or as OpenAI's speech server-sent events {@code speech.audio.delta} and
 * {@code speech.audio.done}.
 *
 * <p>The synthesis runs on its own daemon thread and hands each clip to an unbounded queue, so a slow client
 * never stalls the engine thread that produces it. A client that goes away sets {@code cancelled}: the writer
 * stops, and the sink passed to the source throws, which aborts the generation at its next decoded chunk.
 * Not thread-safe beyond its own fields; one instance serves one response.
 */
final class TtsStream {

    /** Produces the waveform chunks of one request; {@code sink} accepts them in order. */
    interface Source {
        void generate(Consumer<AudioClip> sink) throws IOException;
    }

    /** End of the source's output. */
    private static final Object DONE = new Object();

    /** A source failure, carried to the writer. */
    private static final class Failure {
        final Throwable error;

        Failure(Throwable error) {
            this.error = error;
        }
    }

    /** Raised into the source's sink so a cancellation unwinds the generation. */
    private static final class Cancelled extends RuntimeException {
        Cancelled() {
            super("the speech stream was cancelled");
        }
    }

    private final Source source;
    private final BooleanSupplier cancelled;
    private final int sampleRate;
    private final boolean wav;
    private final boolean sse;
    private final BlockingQueue<Object> queue = new LinkedBlockingQueue<>();
    /** Set by the writer once it stops, so the producer does not keep generating behind a gone client. */
    private final AtomicBoolean stopped = new AtomicBoolean();
    private final AtomicBoolean started = new AtomicBoolean();
    /** The event {@link #awaitFirst()} read and left for {@link #write}; null before it runs. */
    private Object first;

    TtsStream(Source source, BooleanSupplier cancelled, int sampleRate, boolean wav, boolean sse) {
        this.source = source;
        this.cancelled = cancelled;
        this.sampleRate = sampleRate;
        this.wav = wav;
        this.sse = sse;
    }

    /** Starts the synthesis on its daemon thread. */
    void start() {
        if (started.compareAndSet(false, true)) {
            Thread producer = new Thread(this::produce, "neuralfree-tts-stream");
            producer.setDaemon(true);
            producer.start();
        }
    }

    /**
     * Waits until the source produces its first event, so a request the generator refuses is answered with the
     * refusal's status instead of a started stream: a failure before any clip is rethrown here (an
     * {@link UnsupportedOperationException} or {@link IllegalArgumentException} input refusal, or an
     * {@link IOException} generation failure). The first clip, or the end, is kept for {@link #write}. A cancelled
     * request stops waiting, and the response then carries no audio.
     */
    void awaitFirst() throws IOException, InterruptedException {
        start();
        while (first == null) {
            if (cancelled.getAsBoolean()) {
                first = DONE;
                return;
            }
            Object event = queue.poll(50, TimeUnit.MILLISECONDS);
            if (event == null) {
                continue;
            }
            if (event instanceof Failure failure) {
                rethrow(failure.error);
            }
            first = event;
        }
    }

    /**
     * Starts the synthesis (unless {@link #start()} did) and writes the response through {@code out} until the
     * source finishes, the client goes away, or the source fails. A failure after the response started is an
     * {@code error} event on an SSE stream and a closed connection on a raw stream.
     */
    void write(OutputStream out) throws IOException {
        start();
        try {
            if (wav) {
                emit(out, WavWriter.streamingHeader(sampleRate));
            }
            while (true) {
                if (cancelled.getAsBoolean()) {
                    stopped.set(true);
                    return;
                }
                Object event = next();
                if (event == null) {
                    continue;
                }
                if (event == DONE) {
                    break;
                }
                if (event instanceof Failure failure) {
                    fail(out, failure.error);
                    return;
                }

                AudioClip clip = (AudioClip) event;
                emit(out, WavWriter.pcmChunk(clip.samples(), 0, clip.length()));
            }
            if (sse) {
                done(out);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            stopped.set(true);
            throw new IOException("the speech stream was interrupted", e);
        } finally {
            stopped.set(true);
        }
    }

    /** The next event: the one {@link #awaitFirst()} kept, else the next queued one, or null on a poll timeout. */
    private Object next() throws InterruptedException {
        if (first != null) {
            Object event = first;
            first = null;
            return event;
        }
        return queue.poll(50, TimeUnit.MILLISECONDS);
    }

    /** Rethrows a source failure as it happened: an IOException, an input refusal, or a wrapped IOException. */
    private static void rethrow(Throwable error) throws IOException {
        if (error instanceof IOException io) {
            throw io;
        }
        if (error instanceof RuntimeException runtime) {
            throw runtime;
        }
        throw new IOException(error.getMessage() == null ? error.toString() : error.getMessage(), error);
    }

    /** Runs the source on the generation thread, routing its clips and its end through the queue. */
    private void produce() {
        try {
            source.generate(clip -> {
                if (cancelled.getAsBoolean() || stopped.get()) {
                    throw new Cancelled();
                }
                queue.add(clip);
            });
            queue.add(DONE);
        } catch (Cancelled e) {
            queue.add(DONE);
        } catch (Throwable e) {
            if (!cancelled.getAsBoolean() && !stopped.get()) {
                queue.add(new Failure(e));
            }
        }
    }

    /** One audio run: raw bytes, or a {@code speech.audio.delta} event carrying them base64-encoded. */
    private void emit(OutputStream out, byte[] bytes) throws IOException {
        if (bytes.length == 0) {
            return;
        }
        if (!sse) {
            out.write(bytes);
            return;
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("type", "speech.audio.delta");
        body.put("audio", Base64.getEncoder().encodeToString(bytes));
        ServerJson.event(out, "speech.audio.delta", body);
    }

    /** The ending {@code speech.audio.done} event, with the usage shape the OpenAI speech stream carries. */
    private static void done(OutputStream out) throws IOException {
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("type", "tokens");
        usage.put("input_tokens", 0L);
        usage.put("output_tokens", 0L);
        usage.put("total_tokens", 0L);
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("type", "speech.audio.done");
        body.put("usage", usage);
        ServerJson.event(out, "speech.audio.done", body);
    }

    private void fail(OutputStream out, Throwable error) throws IOException {
        if (error instanceof IOException io) {
            // the client is gone, or the response cannot be written: the stream ends
            throw io;
        }
        String message = error.getMessage() == null ? error.toString() : error.getMessage();
        if (sse) {
            ServerJson.event(out, "error", ServerJson.errorBody(message, "server_error"));
            return;
        }
        throw new IOException("speech generation failed: " + message, error);
    }
}
