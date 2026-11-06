package com.eonmux.neuralfree.demo.tools.serve;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * One request of a benchmark workload: its prompt, the prompt's token count, the output length the
 * request asks for and the request id it names ({@code X-Request-Id}). A prompt is one of four
 * things, and the endpoint the run targets follows from it: text or token ids for the generation
 * endpoints, a list of chat content parts for {@code /v1/chat/completions}, a query with its
 * documents for {@code /rerank}, or an audio file for {@code /v1/audio/transcriptions}. A timed
 * trace's sample also carries its arrival timestamp. Immutable.
 */
public final class BenchSample {

    /** What a sample's payload is. */
    public enum Kind {
        /** Text sent to a generation endpoint. */
        TEXT,
        /** Token ids sent to {@code /v1/completions}. */
        TOKENS,
        /** Chat content parts sent as one user message. */
        CHAT,
        /** A rerank query and its documents. */
        RERANK,
        /** An audio file transcribed by {@code /v1/audio/transcriptions}. */
        AUDIO
    }

    private final Kind kind;
    private final String text;
    private final int[] tokens;
    private final List<Object> chatContent;
    private final List<String> rerankTexts;
    private final String audioPath;
    private final double audioDurationSec;
    private final int promptLen;
    private final int outputLen;
    private final String requestId;
    private final Double timestamp;
    private final Map<String, Object> requestOverrides;

    /** One sample with a text prompt of {@code promptLen} tokens (the server counts the truth). */
    public static BenchSample ofText(String text, int promptLen, int outputLen, String requestId) {
        return new BenchSample(Kind.TEXT, text, null, null, null, null, 0.0, promptLen, outputLen,
                requestId, null, null);
    }

    /** One sample whose prompt is the token ids themselves, so {@code promptLen} is exact. */
    public static BenchSample ofTokens(int[] tokens, int outputLen, String requestId) {
        return new BenchSample(Kind.TOKENS, null, tokens.clone(), null, null, null, 0.0, tokens.length,
                outputLen, requestId, null, null);
    }

    /**
     * One sample whose prompt is a list of chat content parts (text and media dictionaries), sent as
     * the user message of a chat request.
     */
    public static BenchSample ofChat(List<Object> content, int promptLen, int outputLen, String requestId) {
        if (content == null || content.isEmpty()) {
            throw new IllegalArgumentException("a chat sample needs at least one content part");
        }
        return new BenchSample(Kind.CHAT, null, null, Collections.unmodifiableList(new ArrayList<>(content)),
                null, null, 0.0, promptLen, outputLen, requestId, null, null);
    }

    /** One rerank sample: the first text is the query, the rest are the documents. */
    public static BenchSample ofRerank(List<String> texts, int promptLen, String requestId) {
        if (texts == null || texts.size() < 2) {
            throw new IllegalArgumentException("a rerank sample needs a query and at least one document");
        }
        return new BenchSample(Kind.RERANK, null, null, null,
                Collections.unmodifiableList(new ArrayList<>(texts)), null, 0.0, promptLen, 0, requestId,
                null, null);
    }

    /** One audio sample: the file the transcription uploads and its duration in seconds. */
    public static BenchSample ofAudio(String audioPath, double durationSec, int promptLen, int outputLen,
                                      String requestId) {
        if (audioPath == null || audioPath.isEmpty()) {
            throw new IllegalArgumentException("an audio sample needs its file path");
        }
        if (!(durationSec >= 0.0)) {
            throw new IllegalArgumentException("an audio sample's duration cannot be negative, got "
                    + durationSec);
        }
        return new BenchSample(Kind.AUDIO, null, null, null, null, audioPath, durationSec, promptLen,
                outputLen, requestId, null, null);
    }

    BenchSample(Kind kind, String text, int[] tokens, List<Object> chatContent, List<String> rerankTexts,
                String audioPath, double audioDurationSec, int promptLen, int outputLen, String requestId,
                Double timestamp, Map<String, Object> requestOverrides) {
        if (kind != Kind.AUDIO && kind != Kind.RERANK && text == null && tokens == null
                && chatContent == null) {
            throw new IllegalArgumentException("a bench sample needs a text, token or chat prompt");
        }
        if (promptLen < 0) {
            throw new IllegalArgumentException("a bench sample's prompt length cannot be negative, got "
                    + promptLen);
        }
        this.kind = kind;
        this.text = text;
        this.tokens = tokens;
        this.chatContent = chatContent;
        this.rerankTexts = rerankTexts;
        this.audioPath = audioPath;
        this.audioDurationSec = audioDurationSec;
        this.promptLen = promptLen;
        this.outputLen = outputLen;
        this.requestId = requestId;
        this.timestamp = timestamp;
        this.requestOverrides = requestOverrides == null ? Collections.emptyMap()
                : Collections.unmodifiableMap(new LinkedHashMap<>(requestOverrides));
    }

    /** What the payload is. */
    public Kind kind() {
        return kind;
    }

    /** The prompt text, null unless the kind is {@link Kind#TEXT} or {@link Kind#AUDIO}. */
    public String text() {
        return text;
    }

    /** The prompt token ids, null unless the kind is {@link Kind#TOKENS}. */
    public int[] tokens() {
        return tokens == null ? null : tokens.clone();
    }

    /** Whether the prompt is token ids. */
    public boolean hasTokens() {
        return tokens != null;
    }

    /** The chat content parts, null unless the kind is {@link Kind#CHAT}. */
    public List<Object> chatContent() {
        return chatContent == null ? null : Collections.unmodifiableList(new ArrayList<>(chatContent));
    }

    /** The rerank query and documents, null unless the kind is {@link Kind#RERANK}. */
    public List<String> rerankTexts() {
        return rerankTexts == null ? null : Collections.unmodifiableList(new ArrayList<>(rerankTexts));
    }

    /** The audio file, null unless the kind is {@link Kind#AUDIO}. */
    public String audioPath() {
        return audioPath;
    }

    /** The audio duration in seconds, zero unless the kind is {@link Kind#AUDIO}. */
    public double audioDurationSec() {
        return audioDurationSec;
    }

    /** The prompt's token count. */
    public int promptLen() {
        return promptLen;
    }

    /** The output length the request asks for. */
    public int outputLen() {
        return outputLen;
    }

    /** The request id sent as {@code X-Request-Id}. */
    public String requestId() {
        return requestId;
    }

    /** The trace arrival time in seconds, null unless a timed trace supplied it. */
    public Double timestamp() {
        return timestamp;
    }

    /** Per-request body fields, empty when the sample carries none. */
    public Map<String, Object> requestOverrides() {
        return requestOverrides;
    }

    /** The same sample under a new request id. */
    public BenchSample withRequestId(String newRequestId) {
        return new BenchSample(kind, text, tokens, chatContent, rerankTexts, audioPath, audioDurationSec,
                promptLen, outputLen, newRequestId, timestamp, requestOverrides);
    }

    /** The same sample with its trace arrival time. */
    public BenchSample withTimestamp(double arrivalSeconds) {
        return new BenchSample(kind, text, tokens, chatContent, rerankTexts, audioPath, audioDurationSec,
                promptLen, outputLen, requestId, arrivalSeconds, requestOverrides);
    }

    @Override
    public String toString() {
        String prompt = kind == Kind.TOKENS ? "tokens(" + tokens.length + ")"
                : kind == Kind.CHAT ? "chat(" + chatContent.size() + " parts)"
                : kind == Kind.RERANK ? "rerank(" + (rerankTexts.size() - 1) + " documents)"
                : kind == Kind.AUDIO ? "audio(" + audioPath + ")"
                : "text(" + promptLen + " tokens)";
        return "BenchSample{" + prompt + ", outputLen=" + outputLen + ", id=" + requestId + "}";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof BenchSample)) {
            return false;
        }
        BenchSample other = (BenchSample) o;
        return kind == other.kind && promptLen == other.promptLen && outputLen == other.outputLen
                && Double.compare(audioDurationSec, other.audioDurationSec) == 0
                && Arrays.equals(tokens, other.tokens)
                && Objects.equals(text, other.text)
                && Objects.equals(chatContent, other.chatContent)
                && Objects.equals(rerankTexts, other.rerankTexts)
                && Objects.equals(audioPath, other.audioPath)
                && Objects.equals(requestId, other.requestId)
                && Objects.equals(timestamp, other.timestamp)
                && Objects.equals(requestOverrides, other.requestOverrides);
    }

    @Override
    public int hashCode() {
        int result = Objects.hashCode(kind);
        result = 31 * result + Objects.hashCode(text);
        result = 31 * result + Arrays.hashCode(tokens);
        result = 31 * result + Objects.hashCode(chatContent);
        result = 31 * result + Objects.hashCode(rerankTexts);
        result = 31 * result + Objects.hashCode(audioPath);
        result = 31 * result + Double.hashCode(audioDurationSec);
        result = 31 * result + promptLen;
        result = 31 * result + outputLen;
        result = 31 * result + Objects.hashCode(requestId);
        result = 31 * result + Objects.hashCode(timestamp);
        result = 31 * result + Objects.hashCode(requestOverrides);
        return result;
    }
}
