package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Masked-diffusion generation (dream, llada, llada-moe, rnd1), following llama.cpp examples/diffusion/diffusion.cpp
 * {@code diffusion_generate} and the parameter resolution of diffusion-cli.cpp.
 *
 * <p>The canvas is the prompt followed by mask tokens up to the resolved max length (default: the ubatch, as
 * llama.cpp uses {@code n_ubatch}). The mask token is {@code tokenizer.ggml.mask_token_id}
 * ({@link SpecialTokens#mask()}) unless the configuration names one. Each step:
 * <ol>
 *   <li>collects the masked positions: every masked canvas position for the timestep schedule (including mask
 *       tokens placed in the prompt), the masked positions of the current block for the block schedule; with none
 *       left the block ends;</li>
 *   <li>runs one non-causal {@link ForwardBackend#forwardAll} over the canvas, reading only the logits rows the
 *       masked positions use: row {@code pos - 1} ({@code 0} for position 0) with {@code shift_logits}, row
 *       {@code pos} without. With classifier-free guidance a second pass runs with the prompt positions masked
 *       and the rows combine as {@code u + (cfg + 1) * (c - u)};</li>
 *   <li>adds Gumbel noise {@code exp(l) / (-ln u)^T} to those rows when enabled and the temperature is
 *       positive;</li>
 *   <li>ORIGIN: unmasks each masked position with probability {@code transfer / masked}, drawing its token
 *       through the sampler chain. Otherwise every masked position is sampled through the chain
 *       ({@link Sampler#diffusionChain}: top-k, top-p, temperature, draw) and scored; the {@code transfer}
 *       positions of highest confidence are unmasked (ties: lower position first), or, with a positive
 *       {@code alg_temp}, {@code transfer} positions are drawn from {@code softmax(confidence / alg_temp)}
 *       without replacement.</li>
 * </ol>
 * Confidence: CONFIDENCE_BASED the probability of the sampled token; ENTROPY_BASED the negative entropy
 * {@code sum p ln(p + 1e-10)} of the final candidate distribution (less uncertain rows first); MARGIN_BASED the
 * gap between the two most probable candidates; RANDOM a uniform draw.
 *
 * <p>Transfer counts: TIMESTEP {@code floor(masked * (1 - s/t))} with {@code t = 1 - step/steps * (1 - eps)},
 * {@code s = 1 - (step+1)/steps * (1 - eps)} and everything at the last step; BLOCK the block's masked count split
 * evenly over its steps, earlier steps taking the remainder. Blocks start after the prompt and are clamped to the
 * canvas, as in llama.cpp.
 *
 * <p>Three RNG streams share the seed, as in llama.cpp: the sampler chain's draw, the confidence draw
 * ({@code alg_temp}), and the stream for Gumbel noise, ORIGIN transfers and RANDOM confidence.
 *
 * <p>Immutable; one generation runs at a time, on the backend's thread.
 */
public final class DiffusionGenerator {

    /** Progress observer, called before each step (llama.cpp {@code diffusion_step_callback_t}). */
    @FunctionalInterface
    public interface StepListener {
        /**
         * @param step       global step index, from 0
         * @param totalSteps the configured number of steps
         * @param canvas     copy of the canvas (prompt and generated tokens, masks where still undecided)
         * @return false to stop generation; the canvas is returned as it stands
         */
        boolean onStep(int step, int totalSteps, int[] canvas);
    }

    private static final float ENTROPY_EPS = 1e-10f;

    private final ForwardBackend backend;
    private final GGUFTokenizer tokenizer;
    private final int ubatch;
    private final boolean modelShiftLogits;

    /**
     * @param ubatch           largest canvas one forward pass evaluates; the default max length
     * @param modelShiftLogits the model's {@code diffusion.shift_logits} (true when the key is absent)
     * @throws IllegalArgumentException when {@code ubatch < 2}
     */
    public DiffusionGenerator(ForwardBackend backend, GGUFTokenizer tokenizer, int ubatch, boolean modelShiftLogits) {
        this.backend = Objects.requireNonNull(backend, "backend");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        if (ubatch < 2) {
            throw new IllegalArgumentException("diffusion needs a ubatch of at least 2 tokens, got " + ubatch);
        }
        this.ubatch = ubatch;
        this.modelShiftLogits = modelShiftLogits;
    }

    /**
     * The generator of an engine model: its backend and tokenizer, {@code config}'s ubatch and the model's
     * {@code diffusion.shift_logits}.
     *
     * @throws UnsupportedOperationException when the architecture is not a diffusion model
     */
    public static DiffusionGenerator forModel(EngineModel model, EngineConfig config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        if (!model.spec().arch().isDiffusion()) {
            throw new UnsupportedOperationException(model.architecture() + " is not a masked-diffusion model");
        }
        return new DiffusionGenerator(model.forwardBackend(), model.tokenizer(), config.memory().ubatch(),
                model.spec().hparams().diffusionShiftLogits());
    }

    /** {@link #generate(int[], DiffusionConfig, TokenCallback, StepListener)} without a step listener. */
    public int[] generate(int[] prompt, DiffusionConfig config, TokenCallback callback) {
        return generate(prompt, config, callback, null);
    }

    /**
     * Generates the canvas after {@code prompt} and streams its decoded text to {@code callback} at the end.
     *
     * @param callback receives the generated text; may be null
     * @param listener progress observer; may be null
     * @return the canvas after the prompt ({@code maxLength - prompt.length} tokens; mask tokens remain where
     *         generation stopped early)
     * @throws IllegalArgumentException when the prompt is empty or has negative ids, the canvas leaves no room
     *                                  after the prompt, exceeds the ubatch, or the schedule does not divide it
     * @throws IllegalStateException    when no mask token is known
     */
    public int[] generate(int[] prompt, DiffusionConfig config, TokenCallback callback, StepListener listener) {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(config, "config");
        if (prompt.length == 0) {
            throw new IllegalArgumentException("diffusion prompt must not be empty");
        }
        for (int i = 0; i < prompt.length; i++) {
            if (prompt[i] < 0) {
                throw new IllegalArgumentException("prompt token " + i + " has negative id " + prompt[i]);
            }
        }
        int mask = resolveMask(config);
        int maxLength = config.resolveMaxLength(prompt.length, ubatch);
        if (maxLength > ubatch) {
            throw new IllegalArgumentException("diffusion canvas of " + maxLength + " tokens exceeds the ubatch of "
                    + ubatch + " that one non-causal pass evaluates; raise -Dneuralfree.ubatch");
        }
        int stepsPerBlock = config.stepsPerBlock(maxLength);
        boolean shift = config.resolveShiftLogits(modelShiftLogits);
        long seed = config.seed() == SamplerConfig.RANDOM_SEED ? Sampler.randomSeed() : config.seed();

        int[] canvas = Arrays.copyOf(prompt, maxLength);
        Arrays.fill(canvas, prompt.length, maxLength, mask);
        new Run(config, canvas, prompt.length, mask, shift, stepsPerBlock, seed).execute(listener);

        int[] generated = Arrays.copyOfRange(canvas, prompt.length, maxLength);
        if (callback != null) {
            stream(generated, callback);
        }
        return generated;
    }

    private int resolveMask(DiffusionConfig config) {
        int mask = config.maskTokenId() != DiffusionConfig.MASK_FROM_TOKENIZER
                ? config.maskTokenId() : tokenizer.specialTokens().mask();
        if (mask == SpecialTokens.NULL) {
            throw new IllegalStateException("diffusion generation needs a mask token, but the model has no"
                    + " tokenizer.ggml.mask_token_id and the configuration names none");
        }
        if (mask >= tokenizer.vocabSize()) {
            throw new IllegalArgumentException("mask token id " + mask + " is outside the vocabulary of "
                    + tokenizer.vocabSize() + " tokens");
        }
        return mask;
    }

    private void stream(int[] generated, TokenCallback callback) {
        GGUFTokenizer.StreamingDecoder decoder = tokenizer.createStreamingDecoder();
        for (int token : generated) {
            String text = decoder.decodeToken(token);
            if (!text.isEmpty()) {
                callback.onToken(token, text);
            }
        }
        String tail = decoder.flush();
        if (!tail.isEmpty()) {
            callback.onToken(-1, tail);
        }
    }

    // ------------------------------------------------------------------ schedule arithmetic

    /**
     * Rows to unmask this step (diffusion.cpp {@code calculate_transfer_count}).
     *
     * @param blockTransfer per-step counts of the block schedule; null for the timestep schedule
     */
    static int transferCount(int step, int totalSteps, int remainingMasked, float eps, int[] blockTransfer) {
        if (blockTransfer == null) {
            float t = 1.0f - (float) step / totalSteps * (1.0f - eps);
            float s = 1.0f - (float) (step + 1) / totalSteps * (1.0f - eps);
            float pTransfer = step < totalSteps - 1 ? 1.0f - s / t : 1.0f;
            return (int) (remainingMasked * pTransfer);
        }
        if (step < blockTransfer.length) {
            return blockTransfer[step];
        }
        return remainingMasked / (totalSteps - step);
    }

    /** A block's masked count split over its steps (diffusion.cpp {@code get_num_transfer_tokens}). */
    static int[] blockTransferCounts(int maskCount, int steps) {
        int[] out = new int[steps];
        int base = maskCount / steps;
        int remainder = maskCount % steps;
        for (int i = 0; i < steps; i++) {
            out[i] = base + (i < remainder ? 1 : 0);
        }
        return out;
    }

    /** Logits row read for canvas position {@code pos}. */
    static int logitsRow(int pos, boolean shift) {
        return shift ? (pos == 0 ? 0 : pos - 1) : pos;
    }

    /** Confidence of a sampled candidate set (diffusion.cpp {@code calculate_confidence}). */
    static float confidence(Sampler.Candidates c, DiffusionConfig.Algorithm algorithm, SplittableRandom rng) {
        switch (algorithm) {
            case ENTROPY_BASED: {
                float negEntropy = 0f;
                for (int i = 0; i < c.size(); i++) {
                    float p = c.prob(i);
                    negEntropy += p * (float) Math.log(p + ENTROPY_EPS);
                }
                return negEntropy;
            }
            case MARGIN_BASED: {
                float first = Float.NEGATIVE_INFINITY;
                float second = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < c.size(); i++) {
                    float p = c.prob(i);
                    if (p > first) {
                        second = first;
                        first = p;
                    } else if (p > second) {
                        second = p;
                    }
                }
                return c.size() > 1 ? first - second : first;
            }
            case RANDOM:
                return nextFloat(rng);
            case ORIGIN:
            case CONFIDENCE_BASED:
            default:
                return c.prob(c.selected());
        }
    }

    /** Uniform float in [0, 1) with 24 random bits. */
    static float nextFloat(SplittableRandom rng) {
        return (rng.nextInt() >>> 8) * 0x1.0p-24f;
    }

    /**
     * Draws {@code take} distinct indices from {@code softmax(scores)} one at a time, each draw renormalized over
     * the indices not yet drawn (the dist sampler's single-pass draw per pick).
     */
    static int[] drawWithoutReplacement(float[] scores, int take, SplittableRandom rng) {
        int n = scores.length;
        boolean[] drawn = new boolean[n];
        int[] out = new int[take];
        for (int t = 0; t < take; t++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < n; i++) {
                if (!drawn[i] && scores[i] > max) {
                    max = scores[i];
                }
            }
            double u = rng.nextDouble();
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                if (!drawn[i]) {
                    sum += (float) Math.exp(scores[i] - max);
                }
            }
            double target = sum * u;
            double run = 0.0;
            int pick = -1;
            int lastOpen = -1;
            for (int i = 0; i < n && pick < 0; i++) {
                if (drawn[i]) {
                    continue;
                }
                lastOpen = i;
                float p = (float) Math.exp(scores[i] - max);
                run += p;
                if (p > 0f && run >= target) {
                    pick = i;
                }
            }
            if (pick < 0) {
                pick = lastOpen;
            }
            drawn[pick] = true;
            out[t] = pick;
        }
        return out;
    }

    // ------------------------------------------------------------------ one generation

    /** Mutable state of one generation. */
    private final class Run {

        private final DiffusionConfig config;
        private final int[] canvas;
        private final int nInput;
        private final int mask;
        private final boolean shift;
        private final int stepsPerBlock;
        private final Sampler sampler;
        private final SplittableRandom rng;
        private final SplittableRandom confRng;
        private final int[] rowSlot;

        Run(DiffusionConfig config, int[] canvas, int nInput, int mask, boolean shift, int stepsPerBlock, long seed) {
            this.config = config;
            this.canvas = canvas;
            this.nInput = nInput;
            this.mask = mask;
            this.shift = shift;
            this.stepsPerBlock = stepsPerBlock;
            this.sampler = Sampler.diffusionChain(config.topK(), config.topP(), config.temperature(), seed);
            this.rng = new SplittableRandom(seed);
            this.confRng = new SplittableRandom(seed);
            this.rowSlot = new int[canvas.length];
        }

        void execute(StepListener listener) {
            boolean block = config.schedule() == DiffusionConfig.Schedule.BLOCK;
            int len = canvas.length;
            int numBlocks = block ? len / config.blockLength() : 1;
            for (int b = 0; b < numBlocks; b++) {
                int blockStart = block ? nInput + b * config.blockLength() : 0;
                int blockEnd = block ? Math.min(nInput + (b + 1) * config.blockLength(), len) : len;
                int[] transfer = block ? blockTransferCounts(countMasks(blockStart, blockEnd), stepsPerBlock) : null;
                for (int step = 0; step < stepsPerBlock; step++) {
                    int globalStep = b * stepsPerBlock + step;
                    if (listener != null && !listener.onStep(globalStep, config.steps(), canvas.clone())) {
                        return;
                    }
                    int[] masked = maskedPositions(block, blockStart, blockEnd);
                    if (masked.length == 0) {
                        break;
                    }
                    float[][] rows = logits(masked);
                    if (config.addGumbelNoise() && config.temperature() > 0f) {
                        for (float[] row : rows) {
                            addGumbelNoise(row);
                        }
                    }
                    int count = transferCount(step, stepsPerBlock, masked.length, config.eps(), transfer);
                    if (config.algorithm() == DiffusionConfig.Algorithm.ORIGIN) {
                        unmaskOrigin(masked, rows, count);
                    } else {
                        unmaskByConfidence(masked, rows, count);
                    }
                }
            }
        }

        private int countMasks(int from, int to) {
            int n = 0;
            for (int i = from; i < to; i++) {
                if (canvas[i] == mask) {
                    n++;
                }
            }
            return n;
        }

        private int[] maskedPositions(boolean block, int blockStart, int blockEnd) {
            int from = block ? blockStart : 0;
            int to = block ? blockEnd : canvas.length;
            int[] out = new int[Math.max(0, to - from)];
            int n = 0;
            for (int i = from; i < to; i++) {
                if (canvas[i] == mask) {
                    out[n++] = i;
                }
            }
            return Arrays.copyOf(out, n);
        }

        /**
         * Logits of the distinct rows the masked positions read, with CFG applied; {@link #rowSlot} maps each
         * masked position to its row in the result.
         */
        private float[][] logits(int[] masked) {
            boolean[] needed = new boolean[canvas.length];
            int distinct = 0;
            for (int pos : masked) {
                int r = logitsRow(pos, shift);
                if (!needed[r]) {
                    needed[r] = true;
                    distinct++;
                }
            }
            int[] rows = new int[distinct];
            int[] slotOfRow = new int[canvas.length];
            for (int r = 0, k = 0; r < canvas.length; r++) {
                if (needed[r]) {
                    slotOfRow[r] = k;
                    rows[k++] = r;
                }
            }
            for (int pos : masked) {
                rowSlot[pos] = slotOfRow[logitsRow(pos, shift)];
            }
            float[][] cond = checkRows(backend.forwardAll(canvas, rows), rows.length, "conditional");
            if (!config.cfgEnabled()) {
                return cond;
            }
            float[][] c = new float[cond.length][];
            for (int k = 0; k < cond.length; k++) {
                c[k] = cond[k].clone();
            }
            int[] uncondCanvas = canvas.clone();
            Arrays.fill(uncondCanvas, 0, nInput, mask);
            float[][] u = checkRows(backend.forwardAll(uncondCanvas, rows), rows.length, "unconditional");
            float scale = config.cfgScale() + 1.0f;
            for (int k = 0; k < c.length; k++) {
                if (u[k].length != c[k].length) {
                    throw new IllegalStateException("unconditional logits row " + k + " has " + u[k].length
                            + " values, the conditional row " + c[k].length);
                }
                for (int v = 0; v < c[k].length; v++) {
                    c[k][v] = u[k][v] + scale * (c[k][v] - u[k][v]);
                }
            }
            return c;
        }

        private float[][] checkRows(float[][] rows, int expected, String pass) {
            if (rows == null || rows.length != expected) {
                throw new IllegalStateException("the " + pass + " diffusion pass returned "
                        + (rows == null ? "no" : String.valueOf(rows.length)) + " logits rows, expected " + expected);
            }
            for (int k = 0; k < rows.length; k++) {
                if (rows[k] == null || rows[k].length == 0) {
                    throw new IllegalStateException("the " + pass + " diffusion pass returned an empty logits row "
                            + k);
                }
            }
            return rows;
        }

        /** diffusion.cpp {@code add_gumbel_noise}: {@code l <- exp(l) / (-ln max(u, 1e-20))^T}. */
        private void addGumbelNoise(float[] row) {
            double temp = config.temperature();
            for (int v = 0; v < row.length; v++) {
                double noise = Math.max(rng.nextDouble(), 1e-20);
                double gumbel = Math.pow(-Math.log(noise), temp);
                row[v] = (float) ((float) Math.exp(row[v]) / gumbel);
            }
        }

        private void unmaskOrigin(int[] masked, float[][] rows, int count) {
            float pTransfer = (float) count / masked.length;
            for (int pos : masked) {
                if (nextFloat(rng) < pTransfer) {
                    canvas[pos] = sampler.sample(rows[rowSlot[pos]]);
                }
            }
        }

        private void unmaskByConfidence(int[] masked, float[][] rows, int count) {
            int n = masked.length;
            float[] conf = new float[n];
            int[] sampled = new int[n];
            for (int i = 0; i < n; i++) {
                Sampler.Candidates c = sampler.apply(rows[rowSlot[masked[i]]]);
                sampled[i] = c.selectedId();
                conf[i] = confidence(c, config.algorithm(), rng);
            }
            if (count <= 0) {
                return;
            }
            int take = Math.min(count, n);
            if (config.algTemp() == 0f) {
                long[] order = new long[n];
                for (int i = 0; i < n; i++) {
                    order[i] = Sampler.descendingKey(Float.isNaN(conf[i]) ? Float.NEGATIVE_INFINITY : conf[i], i);
                }
                Arrays.sort(order);
                for (int j = 0; j < take; j++) {
                    int i = (int) order[j];
                    canvas[masked[i]] = sampled[i];
                }
                return;
            }
            float[] scores = new float[n];
            for (int i = 0; i < n; i++) {
                scores[i] = conf[i] / config.algTemp();
            }
            for (int i : drawWithoutReplacement(scores, take, confRng)) {
                canvas[masked[i]] = sampled[i];
            }
        }
    }
}
