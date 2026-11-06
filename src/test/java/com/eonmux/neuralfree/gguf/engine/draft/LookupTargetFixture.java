package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;

/**
 * Fake targets of the lookup drafters: an {@link EngineModel} whose memory is a plain {@link SequenceState},
 * prefilled with a test's token stream, exactly the view a bound drafter has of its target's history.
 */
final class LookupTargetFixture {

    private LookupTargetFixture() {
    }

    static final class FakeMem implements ModelMemory {
        final SequenceState sequence = new SequenceState(64);

        void prefill(int[] tokens) {
            if (sequence.nPast() != 0) {
                throw new IllegalStateException();
            }
            sequence.append(tokens, 0, tokens.length);
        }

        int[] append(int[] tokens) {
            sequence.append(tokens, 0, tokens.length);
            return tokens;
        }

        @Override public MemoryPlan plan() {
            return null;
        }

        @Override public SequenceState sequence() {
            return sequence;
        }

        @Override public void reset() {
            sequence.clear();
        }

        @Override public boolean canRollback(int nTokens) {
            return nTokens <= sequence.nPast();
        }

        @Override public void rollback(int nTokens) {
            sequence.truncate(nTokens);
        }

        @Override public boolean hasRecurrentState() {
            return false;
        }

        @Override public boolean canSnapshot() {
            return false;
        }

        @Override public boolean ringsKeepWindow(int nWrite, int nDrop) {
            return true;
        }

        @Override public void snapshot() {
        }

        @Override public void restore() {
        }

        @Override public void close() {
        }
    }

    static final class FakeBackend implements ForwardBackend {
        final FakeMem memory = new FakeMem();

        @Override public float[] prefill(int[] tokens, int pos0) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public float[] decode(int token, int pos) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public float[][] forwardAll(int[] tokens, int[] outputRows) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public void reset() {
            memory.reset();
        }

        @Override public boolean canRollback(int nTokens) {
            return memory.canRollback(nTokens);
        }

        @Override public void rollback(int nTokens) {
            memory.rollback(nTokens);
        }

        @Override public void shiftContext(int nKeep, int nDiscard) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public ModelMemory memory() {
            return memory;
        }

        @Override public void setHiddenTap(HiddenTap tap) {
        }

        @Override public void close() {
        }
    }

    static final class FakeModel implements EngineModel {
        final FakeBackend backend = new FakeBackend();

        FakeModel prefill(int[] tokens) {
            backend.memory.prefill(tokens);
            return this;
        }

        @Override public com.eonmux.neuralfree.gguf.GGUFTokenizer tokenizer() {
            return null;
        }

        @Override public void close() {
        }

        @Override public ForwardBackend forwardBackend() {
            return backend;
        }

        @Override public com.eonmux.neuralfree.gguf.spec.ModelSpec spec() {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public String name() {
            return "fake";
        }

        @Override public String architecture() {
            return "fake-llama";
        }

        @Override public int vocabSize() {
            return 16;
        }

        @Override public int contextLength() {
            return 64;
        }

        @Override public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP,
                                        TokenCallback callback) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP,
                                                  TokenCallback callback) {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public void resetKVCache() {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public void printInfo() {
            throw new UnsupportedOperationException("not used by the drafter tests");
        }

        @Override public int kvCacheContext() {
            return 64;
        }

        @Override public int embeddingLength() {
            return 0;
        }

        @Override public int layerCount() {
            return 0;
        }

        @Override public int headCount() {
            return 0;
        }

        @Override public int headCountKv() {
            return 0;
        }

        @Override public int feedForwardLength() {
            return 0;
        }

        @Override public float rmsNormEps() {
            return 0.0f;
        }

        @Override public float ropeFreqBase() {
            return 0.0f;
        }

        @Override public int headDim() {
            return 0;
        }

        @Override public int kvDim() {
            return 0;
        }
    }
}
