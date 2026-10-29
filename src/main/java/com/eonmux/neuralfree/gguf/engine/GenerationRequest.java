package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.TokenCallback;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BooleanSupplier;

/**
 * One autoregressive generation: how many tokens, how to sample them and when to stop. Immutable (the cancellation
 * hook is a caller-owned supplier that the generation polls once per token).
 *
 * <p>The token budget is explicit when {@link Builder#maxNewTokens} set it; otherwise it is a fallback
 * ({@link #DEFAULT_MAX_NEW_TOKENS}, or {@link Builder#fallbackMaxNewTokens}) that a model's
 * {@code max_new_tokens} default replaces ({@link #withModelDefaults}).
 */
public final class GenerationRequest {

    /** {@link #maxNewTokens()} value meaning "until end of generation or a full context" (llama.cpp {@code -n -1}). */
    public static final int UNLIMITED = -1;
    /** Default new-token budget when neither the request nor the model's sampling defaults give one. */
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
    private final boolean maxNewTokensExplicit;
    private final SamplerConfig sampler;
    private final List<String> stopStrings;
    private final List<String> badWords;
    private final boolean ignoreEos;
    private final BooleanSupplier cancellation;
    private final int logprobs;
    private final int minTokens;
    private final Map<Integer, Float> logitBias;

    private GenerationRequest(Builder b) {
        this.maxNewTokens = b.maxNewTokens;
        this.maxNewTokensExplicit = b.maxNewTokensExplicit;
        this.sampler = b.sampler;
        this.stopStrings = Collections.unmodifiableList(new ArrayList<>(b.stopStrings));
        this.badWords = Collections.unmodifiableList(new ArrayList<>(b.badWords));
        this.ignoreEos = b.ignoreEos;
        this.cancellation = b.cancellation;
        this.logprobs = b.logprobs;
        this.minTokens = b.minTokens;
        this.logitBias = Collections.unmodifiableMap(new LinkedHashMap<>(b.logitBias));
    }

    /**
     * {@link #DEFAULT_MAX_NEW_TOKENS} new tokens with {@link SamplerConfig#defaults()}, all fallbacks a model's
     * sampling defaults replace.
     */
    public static GenerationRequest defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** New tokens to generate at most, or {@link #UNLIMITED}. */
    public int maxNewTokens() { return maxNewTokens; }
    public boolean unlimited() { return maxNewTokens == UNLIMITED; }
    /** Whether {@link Builder#maxNewTokens} set the token budget, so a model's default does not replace it. */
    public boolean maxNewTokensExplicit() { return maxNewTokensExplicit; }
    public SamplerConfig sampler() { return sampler; }
    /** Strings that end generation when they appear in the decoded output; excluded from the result text. */
    public List<String> stopStrings() { return stopStrings; }
    /** Whether end-of-generation tokens are sampled through instead of stopping (llama.cpp {@code ignore_eos}). */
    public boolean ignoreEos() { return ignoreEos; }
    /** Polled once per generated token; true stops generation with {@link StopReason#CANCELLED}. */
    public BooleanSupplier cancellation() { return cancellation; }

    /** Alternatives to report per generated token (the sampled token included), or 0 when off. */
    public int logprobs() { return logprobs; }

    /** Phrases the output must not spell ({@code bad_words}); empty when none. */
    public List<String> badWords() { return badWords; }

    /** The generated-token minimum before end-of-generation tokens may be sampled (0 = off). */
    public int minTokens() { return minTokens; }

    /**
     * The request's {@code logit_bias} map: token id to the additive bias applied at the sampler's first stage
     * (llama.cpp {@code llama_sampler_init_logit_bias}); empty when none. A bias of {@code -100} or below means
     * ban to OpenAI clients — the caller translates that into the banned ids before the sampler sees it.
     */
    public Map<Integer, Float> logitBias() { return logitBias; }

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

    /** A builder holding these values; the token budget keeps its explicit or fallback state. */
    public Builder toBuilder() {
        Builder b = builder().sampler(sampler).stopStrings(stopStrings).badWords(badWords).ignoreEos(ignoreEos)
                .cancellation(cancellation).logprobs(logprobs).minTokens(minTokens).logitBias(logitBias);
        return maxNewTokensExplicit ? b.maxNewTokens(maxNewTokens) : b.fallbackMaxNewTokens(maxNewTokens);
    }

    /**
     * The request a generation runs for a model with {@code defaults}: the sampler takes the model's defaults for
     * every setting it leaves unset ({@link SamplerConfig#withModelDefaults}), and a token budget that is not
     * explicit becomes the model's {@code max_new_tokens} when it has one. Returns this request when
     * {@code defaults} is empty.
     *
     * @throws IllegalArgumentException when a taken sampler value is out of range
     */
    public GenerationRequest withModelDefaults(SamplingDefaults defaults) {
        Objects.requireNonNull(defaults, "defaults");
        if (defaults.isEmpty()) {
            return this;
        }
        Builder b = toBuilder().sampler(sampler.withModelDefaults(defaults));
        if (!maxNewTokensExplicit && defaults.maxNewTokens() != null) {
            b.fallbackMaxNewTokens(defaults.maxNewTokens());
        }
        return b.build();
    }

    @Override
    public String toString() {
        return "GenerationRequest{maxNew=" + (unlimited() ? "unlimited" : String.valueOf(maxNewTokens)) + ", "
                + sampler + ", stops=" + stopStrings.size() + ", ignoreEos=" + ignoreEos + ", cancellable="
                + cancellable() + '}';
    }

    public static final class Builder {
        private int maxNewTokens = DEFAULT_MAX_NEW_TOKENS;
        private boolean maxNewTokensExplicit;
        private SamplerConfig sampler = SamplerConfig.defaults();
        private final List<String> stopStrings = new ArrayList<>();
        private boolean ignoreEos;
        private int logprobs;
        private int minTokens;
        private List<String> badWords = new ArrayList<>();
        private Map<Integer, Float> logitBias = new LinkedHashMap<>();

        /** Copies {@code logitBias}; keys must be non-negative ids, values finite. */
        public Builder logitBias(Map<Integer, Float> v) {
            Objects.requireNonNull(v, "logitBias");
            for (Map.Entry<Integer, Float> e : v.entrySet()) {
                logitBias(e.getKey(), e.getValue());
            }
            return this;
        }

        /** Adds one {@code logit_bias} entry; the key must be a non-negative id, the value finite. */
        public Builder logitBias(int tokenId, float bias) {
            Objects.requireNonNull(bias, "bias");
            if (tokenId < 0) {
                throw new IllegalArgumentException("logit_bias token id " + tokenId + " is negative");
            }
            if (!Float.isFinite(bias)) {
                throw new IllegalArgumentException("the bias of token " + tokenId + " is not finite");
            }
            this.logitBias.put(tokenId, bias);
            return this;
        }
        private BooleanSupplier cancellation = NEVER;

        private Builder() {
        }

        /** Sets the token budget explicitly: a model's {@code max_new_tokens} default does not replace it. */
        public Builder maxNewTokens(int v) {
            this.maxNewTokens = v;
            this.maxNewTokensExplicit = true;
            return this;
        }

        /** Sets the token budget as a fallback, which a model's {@code max_new_tokens} default replaces. */
        public Builder fallbackMaxNewTokens(int v) {
            this.maxNewTokens = v;
            this.maxNewTokensExplicit = false;
            return this;
        }

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

        /**
         * Reports the post-chain distribution per generated token: the sampled token plus its {@code n} top
         * alternatives as log probabilities (llama.cpp's {@code n_probs}; 0 = off, the default).
         */
        /**
         * Bans end-of-generation tokens until {@code n} tokens were generated (vLLM's {@code min_tokens};
         * 0 = off, the default).
         */
        public Builder minTokens(int n) {
            if (n < 0) {
                throw new IllegalArgumentException("min tokens needs a non-negative count, got " + n);
            }
            this.minTokens = n;
            return this;
        }

        /** Phrases the output must not spell (vLLM's {@code bad_words}); tokenized into sampling bans. */
        public Builder badWords(List<String> v) {
            this.badWords = new ArrayList<>(Objects.requireNonNull(v, "badWords"));
            return this;
        }

        public Builder badWord(String v) {
            this.badWords.add(Objects.requireNonNull(v, "badWord"));
            return this;
        }

        public Builder logprobs(int n) {
            if (n < 0) {
                throw new IllegalArgumentException("logprobs needs a non-negative alternative count, got " + n);
            }
            this.logprobs = n;
            return this;
        }

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
