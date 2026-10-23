package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.TokenCallback;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.BooleanSupplier;

/**
 * One autoregressive generation: how many tokens, how to sample them and when to stop. Immutable (the cancellation
 * hook is a caller-owned supplier that the generation polls once per token).
 */
public final class GenerationRequest {

    /** {@link #maxNewTokens()} value meaning "until end of generation or a full context" (llama.cpp {@code -n -1}). */
    public static final int UNLIMITED = -1;
    /** Default new-token budget. */
    public static final int DEFAULT_MAX_NEW_TOKENS = 512;

    private static final BooleanSupplier NEVER = new BooleanSupplier() {
        @Override
        public boolean getAsBoolean() {
            return false;
        }

        @Override
        public String toString() {
            return "never";
        }
    };

    private final int maxNewTokens;
    private final SamplerConfig sampler;
    private final List<String> stopStrings;
    private final boolean ignoreEos;
    private final BooleanSupplier cancellation;

    private GenerationRequest(Builder b) {
        this.maxNewTokens = b.maxNewTokens;
        this.sampler = b.sampler;
        this.stopStrings = Collections.unmodifiableList(new ArrayList<>(b.stopStrings));
        this.ignoreEos = b.ignoreEos;
        this.cancellation = b.cancellation;
    }

    /** {@link #DEFAULT_MAX_NEW_TOKENS} new tokens with {@link SamplerConfig#defaults()}. */
    public static GenerationRequest defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** New tokens to generate at most, or {@link #UNLIMITED}. */
    public int maxNewTokens() { return maxNewTokens; }
    public boolean unlimited() { return maxNewTokens == UNLIMITED; }
    public SamplerConfig sampler() { return sampler; }
    /** Strings that end generation when they appear in the decoded output; excluded from the result text. */
    public List<String> stopStrings() { return stopStrings; }
    /** Whether end-of-generation tokens are sampled through instead of stopping (llama.cpp {@code ignore_eos}). */
    public boolean ignoreEos() { return ignoreEos; }
    /** Polled once per generated token; true stops generation with {@link StopReason#CANCELLED}. */
    public BooleanSupplier cancellation() { return cancellation; }
    /** Whether a cancellation hook was supplied. */
    public boolean cancellable() { return cancellation != NEVER; }

    /**
     * A streaming filter that withholds text which may begin a stop string and stops forwarding at the first
     * complete one; see {@link StopStringFilter}.
     *
     * @param downstream receives the released text, or null to only track the stop
     */
    public StopStringFilter newStopFilter(TokenCallback downstream) {
        return new StopStringFilter(stopStrings, downstream);
    }

    public Builder toBuilder() {
        return builder().maxNewTokens(maxNewTokens).sampler(sampler).stopStrings(stopStrings).ignoreEos(ignoreEos)
                .cancellation(cancellation);
    }

    @Override
    public String toString() {
        return "GenerationRequest{maxNew=" + (unlimited() ? "unlimited" : String.valueOf(maxNewTokens)) + ", "
                + sampler + ", stops=" + stopStrings.size() + ", ignoreEos=" + ignoreEos + ", cancellable="
                + cancellable() + '}';
    }

    public static final class Builder {
        private int maxNewTokens = DEFAULT_MAX_NEW_TOKENS;
        private SamplerConfig sampler = SamplerConfig.defaults();
        private final List<String> stopStrings = new ArrayList<>();
        private boolean ignoreEos;
        private BooleanSupplier cancellation = NEVER;

        private Builder() {
        }

        public Builder maxNewTokens(int v) { this.maxNewTokens = v; return this; }
        public Builder sampler(SamplerConfig v) { this.sampler = Objects.requireNonNull(v, "sampler"); return this; }

        /** Replaces the stop strings. */
        public Builder stopStrings(List<String> v) {
            Objects.requireNonNull(v, "stopStrings");
            this.stopStrings.clear();
            for (String s : v) {
                stopString(s);
            }
            return this;
        }

        /** Replaces the stop strings. */
        public Builder stopStrings(String... v) {
            return stopStrings(Arrays.asList(Objects.requireNonNull(v, "stopStrings")));
        }

        /** Adds one stop string. */
        public Builder stopString(String v) {
            Objects.requireNonNull(v, "stop string");
            if (v.isEmpty()) {
                throw new IllegalArgumentException("stop strings must not be empty");
            }
            this.stopStrings.add(v);
            return this;
        }

        public Builder ignoreEos(boolean v) { this.ignoreEos = v; return this; }

        /** Cancellation hook; null removes it. */
        public Builder cancellation(BooleanSupplier v) { this.cancellation = v == null ? NEVER : v; return this; }

        /** @throws IllegalArgumentException when {@code maxNewTokens} is neither positive nor {@link #UNLIMITED} */
        public GenerationRequest build() {
            if (maxNewTokens <= 0 && maxNewTokens != UNLIMITED) {
                throw new IllegalArgumentException(
                        "maxNewTokens must be positive, or -1 for unlimited, got " + maxNewTokens);
            }
            return new GenerationRequest(this);
        }
    }

    /**
     * Streaming stop-string matcher (the hold-back rule of llama.cpp server {@code find_stopping_strings}): text is
     * released as soon as it can no longer be the start of a stop string; the earliest complete stop string (first
     * in list order on ties) ends the stream, and neither it nor anything after it is released. Not thread-safe;
     * one instance per generation.
     */
    public static final class StopStringFilter implements TokenCallback {

        private final List<String> stops;
        private final TokenCallback downstream;
        private final StringBuilder pending = new StringBuilder();
        private final StringBuilder released = new StringBuilder();
        private String matched;

        StopStringFilter(List<String> stops, TokenCallback downstream) {
            this.stops = stops;
            this.downstream = downstream;
        }

        /** Feeds the text of one token; ignored once a stop string matched. */
        @Override
        public void onToken(int tokenId, String text) {
            if (matched != null || text == null || text.isEmpty()) {
                return;
            }
            pending.append(text);
            int hit = -1;
            String hitStop = null;
            for (String s : stops) {
                int at = pending.indexOf(s);
                if (at >= 0 && (hit < 0 || at < hit)) {
                    hit = at;
                    hitStop = s;
                }
            }
            if (hitStop != null) {
                matched = hitStop;
                release(tokenId, hit);
                pending.setLength(0);
                return;
            }
            release(tokenId, pending.length() - heldBack());
        }

        /** Releases the withheld tail when the stream ended without a stop string. */
        public void finish() {
            if (matched == null) {
                release(-1, pending.length());
            }
        }

        /** Whether a stop string matched. */
        public boolean stopped() {
            return matched != null;
        }

        /** The stop string that matched, or null. */
        public String matched() {
            return matched;
        }

        /** All text released so far. */
        public String releasedText() {
            return released.toString();
        }

        /** Length of the longest suffix of the pending text that is a proper prefix of a stop string. */
        private int heldBack() {
            int best = 0;
            int len = pending.length();
            for (String s : stops) {
                int max = Math.min(len, s.length() - 1);
                for (int k = max; k > best; k--) {
                    if (pendingEndsWith(s, k)) {
                        best = k;
                        break;
                    }
                }
            }
            return best;
        }

        /** Whether the pending text ends with the first {@code k} chars of {@code s}. */
        private boolean pendingEndsWith(String s, int k) {
            int base = pending.length() - k;
            for (int i = 0; i < k; i++) {
                if (pending.charAt(base + i) != s.charAt(i)) {
                    return false;
                }
            }
            return true;
        }

        private void release(int tokenId, int count) {
            if (count <= 0) {
                return;
            }
            String out = pending.substring(0, count);
            pending.delete(0, count);
            released.append(out);
            if (downstream != null) {
                downstream.onToken(tokenId, out);
            }
        }
    }
}
