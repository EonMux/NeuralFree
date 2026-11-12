package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.TokenScore;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;

/**
 * The per-token cost of a streamed OpenAI chat response on the server: one {@code chat.completion.chunk} built by
 * {@link OpenAiBodies} (its head, the content delta and, under {@code top_logprobs}, the token's chat logprobs
 * entry) and written as one server-sent event by {@link ServerJson#event} into a reused buffer, without the flush a
 * drained run of events shares. The logprob tokens render as {@code token_id:<id>}
 * ({@code return_tokens_as_token_ids}), which needs no tokenizer. A streaming server pays this once per token per
 * stream.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(1)
public class SseChunkEncodeBenchmark {

    private static final String ID = "chatcmpl-6f1c2a9e4b7d4c0f8a35e2d1b9c07a64";
    private static final String MODEL = "qwen3-0.6b-q8_0.gguf";
    private static final long CREATED = 1_794_000_000L;
    private static final String[] WORDS = {
        "the", "model", "token", "attention", "decode", "batch", "kernel", "memory", "über", "café", "日本語",
        "привет", "{\"k\": 1}", "\n", "\"quoted\"", "x86_64", "42", ",", ".", "throughput"
    };
    private static final int PIECES = 1024;

    /** Alternatives per token: 0 streams no logprobs, 5 is a typical {@code top_logprobs}. */
    @Param({"0", "5"})
    public int topLogprobs;

    private OpenAiBodies bodies;
    private String[] pieces;
    private TokenScore[][] scores;
    private ByteArrayOutputStream out;
    private int next;

    @Setup
    public void setUp() throws IOException {
        bodies = new OpenAiBodies(null, topLogprobs > 0 ? topLogprobs : -1, MODEL, true);
        SplittableRandom random = new SplittableRandom(17);
        pieces = new String[PIECES];
        scores = new TokenScore[PIECES][];
        for (int i = 0; i < PIECES; i++) {
            pieces[i] = " " + WORDS[random.nextInt(WORDS.length)];
            TokenScore[] step = new TokenScore[topLogprobs + 1];
            for (int j = 0; j < step.length; j++) {
                step[j] = new TokenScore(random.nextInt(151_936), (float) (-0.1 - j - random.nextDouble()));
            }
            scores[i] = step;
        }
        out = new ByteArrayOutputStream(4096);
        if (encode() == 0) {
            throw new IllegalStateException("the chunk encodes to nothing");
        }
    }

    @Benchmark
    public int encode() throws IOException {
        int i = next;
        next = (next + 1) % PIECES;
        Map<String, Object> delta = new LinkedHashMap<>();
        delta.put("content", pieces[i]);
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("delta", delta);
        if (topLogprobs > 0) {
            choice.put("logprobs", bodies.chatLogprobs(scores[i]));
        }
        choice.put("finish_reason", null);
        Map<String, Object> chunk = bodies.head(ID, "chat.completion.chunk", CREATED);
        chunk.put("choices", List.of(choice));
        out.reset();
        ServerJson.event(out, null, chunk, false);
        return out.size();
    }
}
