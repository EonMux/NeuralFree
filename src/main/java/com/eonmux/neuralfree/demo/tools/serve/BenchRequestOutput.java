package com.eonmux.neuralfree.demo.tools.serve;

import java.util.ArrayList;
import java.util.List;

/**
 * The outcome of one benchmark request, the fields the serving metrics aggregate: the time to first
 * token, the inter-token latencies of the streamed tokens, the end-to-end latency (last token time
 * minus send time), the generated text and the token counts (the stream's {@code usage} chunk when
 * the server reports one). One object per request; failed requests carry the error and nothing else.
 */
public final class BenchRequestOutput {

    /** Whether a first token arrived and the usage closed the stream. */
    public boolean success;

    /** Seconds from send to the first content chunk; set on every successful request. */
    public double ttft;

    /** Seconds between consecutive content chunks, first-token exclusive. */
    public final List<Double> itls = new ArrayList<>();

    /** Seconds from send to the last content chunk. */
    public double latency;

    /** The generated text, every content chunk concatenated. */
    public final StringBuilder generatedText = new StringBuilder();

    /** The usage chunk's {@code completion_tokens}; null when the server sent no usage. */
    public Integer outputTokens;

    /** The prompt's token count: the sample's, or the usage chunk's {@code prompt_tokens}. */
    public int promptLen;

    /** The send time, in seconds relative to the run's start. */
    public double startTime;

    /** Seconds between the scheduled arrival and the actual send, the concurrency queue's wait. */
    public double clientQueueTime;

    /** The failure: the HTTP status line for a non-200 reply, the I/O error's message otherwise. */
    public String error;

    /** The audio the request transcribed, in seconds; zero for a request without audio. */
    public double inputAudioDuration;

    /** The number of input sequences the request carried (the documents of a rerank, else one). */
    public int numInputSequences = 1;

    /** The output length the metrics count: the usage's, else one per content chunk. */
    public int outputLenOrChunkCount() {
        if (outputTokens != null) {
            return outputTokens;
        }
        return itls.size() + 1;
    }

    @Override
    public String toString() {
        if (!success) {
            return "BenchRequestOutput{failed, error=" + error + "}";
        }
        return "BenchRequestOutput{ttft=" + ttft + "s, tokens=" + outputLenOrChunkCount()
                + ", latency=" + latency + "s}";
    }
}
