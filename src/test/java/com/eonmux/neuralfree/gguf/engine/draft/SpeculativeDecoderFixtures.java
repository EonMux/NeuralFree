package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Fixtures that run a {@link com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder} over a 4-token vocabulary: a
 * fake target whose rows are plain scripted logits (with temperature 1 and no truncation stages the sampler chain
 * leaves the probabilities intact, so the script is the processed target distribution itself), its KV or recurrent
 * fake memory, and scripted drafters with and without proposal probabilities.
 */
final class SpeculativeDecoderFixtures {

    static final int N_CTX = 32;
    private static final int UBATCH = 8;
    static final long SEED = 42;

    private SpeculativeDecoderFixtures() {
    }

    /** Logits of probabilities: temperature 1 with no truncation stages gives them back. */
    static float[] logits(double... probs) {
        float[] row = new float[probs.length];
        for (int i = 0; i < probs.length; i++) {
            row[i] = probs[i] <= 0.0 ? Float.NEGATIVE_INFINITY : (float) Math.log(probs[i]);
        }
        return row;
    }

    /** A gpt2 tokenizer over the tokens {@code a b c d}. */
    static GGUFTokenizer tokenizer() {
        List<Object> tokens = new ArrayList<>(Arrays.<Object>asList("a", "b", "c", "d"));
        List<Object> types = new ArrayList<>(Arrays.<Object>asList(1, 1, 1, 1));
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "gpt2");
        md.put(TokenizerVocab.KEY_PRE, "gpt-2");
        md.put(TokenizerVocab.KEY_TOKENS, tokens);
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
        md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>());
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    /** A gpt2 tokenizer over {@code a b c d} and the end-of-sequence control token {@code </s>} (id 4). */
    static GGUFTokenizer tokenizerWithEos() {
        List<Object> tokens = new ArrayList<>(Arrays.<Object>asList("a", "b", "c", "d", "</s>"));
        List<Object> types = new ArrayList<>(Arrays.<Object>asList(1, 1, 1, 1, 3));
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "gpt2");
        md.put(TokenizerVocab.KEY_PRE, "gpt-2");
        md.put(TokenizerVocab.KEY_TOKENS, tokens);
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
        md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>());
        md.put("tokenizer.ggml.eos_token_id", 4);
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    /** Temperature 1 without truncation, seeded with {@link #SEED}. */
    static SamplerConfig sampling() {
        return SamplerConfig.builder().temperature(1f).topK(0).topP(1f).minP(0f).seed(SEED).build();
    }

    /** A request for {@code maxNewTokens} tokens under {@code config}. */
    static GenerationRequest request(int maxNewTokens, SamplerConfig config) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(config).build();
    }

    /** KV-only fake memory with a plain sequence record. */
    static final class FakeMemory implements ModelMemory {
        private final MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, N_CTX, UBATCH)
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.NONE).build())
                .build();
        final SequenceState sequence = new SequenceState(N_CTX);
        int snapshots;
        int restores;
        boolean recurrent;
        private int saved = -1;

        @Override public MemoryPlan plan() { return plan; }
        @Override public SequenceState sequence() { return sequence; }
        @Override public void reset() {
            sequence.clear();
            saved = -1;
        }
        @Override public boolean canRollback(int nTokens) { return !recurrent && nTokens <= sequence.nPast(); }
        @Override public void rollback(int nTokens) { sequence.truncate(nTokens); }
        @Override public boolean hasRecurrentState() { return recurrent; }
        @Override public boolean canSnapshot() { return recurrent; }
        @Override public boolean ringsKeepWindow(int nWrite, int nDrop) { return true; }
        @Override public void snapshot() {
            saved = sequence.nPast();
            snapshots++;
        }
        @Override public void restore() {
            if (saved < 0) {
                throw new IllegalStateException("no snapshot");
            }
            sequence.truncate(sequence.nPast() - saved);
            restores++;
        }
        @Override public void close() { }
    }

    /** Target backend scripted per forward call, in call order; each entry services one call's rows. */
    static final class FakeTarget implements ForwardBackend {
        final FakeMemory memory = new FakeMemory();
        private final Deque<float[][]> scripts = new ArrayDeque<>();
        final List<int[]> verifyCalls = new ArrayList<>();
        final List<Integer> verifyPositions = new ArrayList<>();
        final List<Integer> rollbacks = new ArrayList<>();
        final List<int[]> prefills = new ArrayList<>();
        final List<Integer> prefillPositions = new ArrayList<>();
        final List<Integer> decodes = new ArrayList<>();
        final List<int[]> shifts = new ArrayList<>();
        /** Whether the memory shifts its context ({@link #shiftContext}); unsupported otherwise. */
        boolean shiftable;
        int snapshots;
        int restores;
        private float[] fallback;

        FakeTarget(boolean recurrent) {
            memory.recurrent = recurrent;
        }

        FakeTarget add(float[]... rows) {
            scripts.add(rows);
            return this;
        }

        /** Every row of the calls after the scripted ones is {@code row}. */
        FakeTarget otherwise(float[] row) {
            this.fallback = row.clone();
            return this;
        }

        private float[][] nextRow(int rowsWanted) {
            float[][] script = scripts.poll();
            if (script == null && fallback != null) {
                float[][] rows = new float[rowsWanted][];
                Arrays.fill(rows, fallback);
                return rows;
            }
            if (script == null) {
                throw new IllegalStateException("no scripted rows left for a forward call");
            }
            if (script.length != rowsWanted) {
                throw new IllegalStateException("scripted " + script.length + " rows, wanted " + rowsWanted);
            }
            return script;
        }

        @Override public float[] prefill(int[] tokens, int pos0) {
            if (pos0 > memory.sequence.nPast()) {
                throw new IllegalStateException("prefill at " + pos0 + " after " + memory.sequence.nPast());
            }
            if (pos0 < memory.sequence.nPast()) {
                memory.rollback(memory.sequence.nPast() - pos0);
            }
            memory.sequence.append(tokens, 0, tokens.length);
            prefills.add(tokens.clone());
            prefillPositions.add(pos0);
            return nextRow(1)[0];
        }

        @Override public float[] decode(int token, int pos) {
            memory.sequence.append(new int[]{token}, 0, 1);
            decodes.add(token);
            return nextRow(1)[0];
        }

        @Override public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
            if (pos0 != memory.sequence.nPast()) {
                throw new IllegalStateException("verify at " + pos0 + " after " + memory.sequence.nPast());
            }
            memory.sequence.append(tokens, 0, tokens.length);
            verifyCalls.add(tokens.clone());
            verifyPositions.add(pos0);
            return nextRow(tokens.length);
        }

        @Override public float[][] forwardAll(int[] tokens, int[] outputRows) {
            throw new UnsupportedOperationException();
        }

        @Override public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
            throw new UnsupportedOperationException();
        }

        @Override public void reset() {
            memory.reset();
        }

        @Override public boolean canRollback(int nTokens) {
            return memory.canRollback(nTokens);
        }

        @Override public void rollback(int nTokens) {
            memory.rollback(nTokens);
            rollbacks.add(nTokens);
        }

        @Override public void shiftContext(int nKeep, int nDiscard) {
            if (!shiftable) {
                throw new UnsupportedOperationException("the fake memory does not shift");
            }
            memory.sequence.shift(nKeep, nDiscard);
            shifts.add(new int[]{nKeep, nDiscard});
        }

        @Override public ModelMemory memory() {
            FakeMemory memory = this.memory;
            return new ModelMemory() {
                @Override public MemoryPlan plan() { return memory.plan(); }
                @Override public SequenceState sequence() { return memory.sequence(); }
                @Override public void reset() { memory.reset(); }
                @Override public boolean canRollback(int nTokens) { return memory.canRollback(nTokens); }
                @Override public void rollback(int nTokens) { memory.rollback(nTokens); }
                @Override public boolean hasRecurrentState() { return memory.hasRecurrentState(); }
                @Override public boolean canSnapshot() { return memory.canSnapshot(); }
                @Override public boolean ringsKeepWindow(int nWrite, int nDrop) {
                    return memory.ringsKeepWindow(nWrite, nDrop);
                }
                @Override public void snapshot() { snapshots++; memory.snapshot(); }
                @Override public void restore() { restores++; memory.restore(); }
                @Override public void close() { }
            };
        }

        @Override public void setHiddenTap(HiddenTap tap) {
        }

        @Override public void close() {
        }
    }

    /** A scripted drafter with per-position proposal distributions. */
    static final class ProbDrafter implements DraftModel, DraftProbs {
        final List<int[]> proposals = new ArrayList<>();
        final List<double[][]> qRows = new ArrayList<>();
        int call;
        double faultyAt;
        boolean faultyZero;
        boolean faultyNull;
        boolean faultyShort;
        boolean faultyNaN;

        ProbDrafter(int[]... tokens) {
            for (int[] t : tokens) {
                proposals.add(t);
            }
        }

        static ProbDrafter with(int[] tokens, double[]... q) {
            ProbDrafter d = new ProbDrafter(tokens);
            d.qRows.add(q);
            return d;
        }

        @Override public void bindTarget(EngineModel target) {
        }

        @Override public int[] draft(int lastToken, int n) {
            if (call >= proposals.size()) {
                return new int[0];
            }
            return proposals.get(call++);
        }

        @Override public boolean supportsWatermarking() {
            return true;
        }

        @Override public float[] draftProbs(int position) {
            double[][] rows = qRows.get(call - 1);
            if (faultyNull && position == 0) {
                return null;
            }
            if (faultyShort) {
                // a row shorter than the drafted token id
                return new float[position + 1];
            }
            if (faultyZero) {
                // a full row whose entry for the drafted token is 0
                return new float[]{0.1f, 0.2f, 0.0f, 0.4f};
            }
            if (faultyNaN) {
                return new float[]{0.1f, 0.2f, Float.NaN, 0.4f};
            }
            if (rows == null || position >= rows.length) {
                throw new IllegalStateException("no proposal probabilities at position " + position);
            }
            double[] row = rows[position];
            float[] out = new float[row.length];
            for (int i = 0; i < row.length; i++) {
                out[i] = (float) row[i];
            }
            return out;
        }

        @Override public void accept(int nAccepted) {
        }

        @Override public void reset() {
            call = 0;
        }
    }

    /** A plain drafter without probabilities (one-hot {@code q}); repeats its last proposal. */
    static final class PlainDrafter implements DraftModel {
        final List<int[]> proposals = new ArrayList<>();
        int call;
        int resets;
        /** Whether the drafter reads the target's hidden rows, as the hidden-fed drafters do. */
        boolean readsHidden;

        PlainDrafter(int[]... tokens) {
            for (int[] t : tokens) {
                proposals.add(t);
            }
        }

        @Override public void bindTarget(EngineModel target) {
        }

        @Override public int[] draft(int lastToken, int n) {
            if (call >= proposals.size()) {
                return new int[0];
            }
            return proposals.get(call++);
        }

        @Override public void accept(int nAccepted) {
        }

        @Override public boolean readsTargetHidden() {
            return readsHidden;
        }

        @Override public void reset() {
            call = 0;
            resets++;
        }
    }
}
