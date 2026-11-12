package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The generation loop's per-step selection over a device readback, beyond the plain greedy argmax: a greedy chain
 * whose only edits are the fixed bias stage's bans and non-positive {@code logit_bias} entries reads back the
 * largest raw logits (the raw argmax plus one candidate per edited token) and selects over them exactly the token it
 * selects over the whole row. A positive bias raises a token from anywhere in the row, so it keeps the whole-row
 * path, and a positive bias or a sparse chain of more than the readback's width must change no token either way.
 * The backend reports the device readout and counts its calls, so the test sees which path ran; it computes the
 * readout on the host over the same rows, which is the device's ranking ({@code GpuDeviceVerifyTest} checks that
 * equality on the GPU).
 */
public class DeviceSamplingSparseTest {

    private static final int[] PROMPT = {4, 17, 30, 43, 56, 69};
    private static final int TOKENS = 5;

    private Path model;

    @After
    public void deleteModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
        return EngineConfig.builder().memory(mc).build();
    }

    /** A greedy request with {@code topK} and the given {@code logit_bias} entries. */
    private static GenerationRequest request(int topK, Map<Integer, Float> bias) {
        GenerationRequest.Builder b = GenerationRequest.builder()
                .maxNewTokens(TOKENS)
                .sampler(SamplerConfig.builder().greedy(true).topK(topK).penaltyLastN(0).build());
        for (Map.Entry<Integer, Float> e : bias.entrySet()) {
            b.logitBias(e.getKey(), e.getValue());
        }
        return b.build();
    }

    private static int[] generate(GGUFModel one, ForwardBackend backend, GenerationRequest request,
                                  boolean onDevice) {
        return GenerationLoop.builder(backend, one.tokenizer())
                .samplingOnDevice(onDevice)
                .build()
                .generate(PROMPT, request, null)
                .tokens();
    }

    private void openModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    /**
     * The whole-row tokens of a request banning the raw argmax, then the tokens of a loop whose backend reports the
     * device readout: equal tokens, the ban applied on both, and every step read the ban/bias prefix back.
     */
    @Test
    public void aGreedyChainWithBansReadsItsPrefixBack() throws IOException {
        openModel();
        try (GGUFModel one = new GGUFModel(model.toString(), config())) {
            int rawFirst = generate(one, wrap(one.forwardBackend()),
                    request(0, new LinkedHashMap<Integer, Float>()), true)[0];
            Map<Integer, Float> bias = new LinkedHashMap<>();
            bias.put(rawFirst, -100f);   // an OpenAI-style ban of the raw argmax
            bias.put(9, -3f);
            GenerationRequest req = request(0, bias);
            int[] host = generate(one, wrap(one.forwardBackend()), req, false);
            TopKBackend device = wrap(one.forwardBackend());
            int[] tokens = generate(one, device, req, true);
            assertArrayEquals("the prefix readback selects the whole row's tokens", host, tokens);
            assertTrue("the banned token leaves the selection", host[0] != rawFirst);
            assertTrue("the ban chain read a prefix back: " + device.decodeTopKCalls, device.decodeTopKCalls > 0);
        }
    }

    /**
     * A positive logit bias raises a token from anywhere in the row: the whole-row path selects it on both runs, and
     * the bias changes the argmax (so a device argmax that ignored the bias would differ).
     */
    @Test
    public void aGreedyPositiveLogitBiasSelectsItsTokenOnBothPaths() throws IOException {
        openModel();
        try (GGUFModel one = new GGUFModel(model.toString(), config())) {
            int rawFirst = generate(one, wrap(one.forwardBackend()),
                    request(0, new LinkedHashMap<Integer, Float>()), true)[0];
            int biased = rawFirst == 7 ? 8 : 7;
            Map<Integer, Float> bias = new LinkedHashMap<>();
            bias.put(biased, 1e4f);
            GenerationRequest req = request(0, bias);
            int[] host = generate(one, wrap(one.forwardBackend()), req, false);
            TopKBackend device = wrap(one.forwardBackend());
            int[] tokens = generate(one, device, req, true);
            assertTrue("the run generated " + host.length + " tokens", host.length >= 3);
            assertArrayEquals("the biased token is the chain's selection on both paths", host, tokens);
            assertEquals("the biased token wins the first step", biased, tokens[0]);
            assertEquals("the raw-argmax path never runs over a biased chain", 0, device.greedyCalls);
            assertEquals("a positive bias reads the row back", 0, device.decodeTopKCalls);
        }
    }

    /** A plain greedy chain keeps the on-device argmax fast path (no readback, no whole row). */
    @Test
    public void aPlainGreedyChainSelectsTheDeviceArgmax() throws IOException {
        openModel();
        GenerationRequest req = request(40, new LinkedHashMap<Integer, Float>());
        try (GGUFModel one = new GGUFModel(model.toString(), config())) {
            TopKBackend device = wrap(one.forwardBackend());
            int[] tokens = generate(one, device, req, true);
            assertTrue("tokens were generated", tokens.length > 0);
            assertTrue("the argmax path ran: " + device.greedyCalls, device.greedyCalls > 0);
            assertEquals("no candidate readback", 0, device.decodeTopKCalls);
        }
    }

    /**
     * A greedy chain banning a hundred tokens selects among the largest 101 raw logits: wider than a few dozen
     * candidates, inside the device readout's radix-select width, so it reads those candidates back, not the row.
     */
    @Test
    public void aBanChainOfAHundredTokensReadsItsCandidatesBack() throws IOException {
        openModel();
        Map<Integer, Float> bias = new LinkedHashMap<>();
        for (int id = 1; id <= 100; id++) {
            bias.put(id, -100f);
        }
        GenerationRequest req = request(1, bias);
        try (GGUFModel one = new GGUFModel(model.toString(), config())) {
            int[] host = generate(one, wrap(one.forwardBackend()), req, false);
            TopKBackend device = wrap(one.forwardBackend());
            int[] tokens = generate(one, device, req, true);
            assertArrayEquals(host, tokens);
            assertTrue("a 101-candidate cone reads the candidates back: " + device.decodeTopKCalls,
                    device.decodeTopKCalls > 0);
            assertEquals("every candidate readback is the cone's width", 101, device.widestK);
        }
    }

    /** A chain wider than the device readback keeps the whole-row path and the same tokens. */
    @Test
    public void aChainWiderThanTheReadbackReadsTheRow() throws IOException {
        openModel();
        GenerationRequest req = GenerationRequest.builder()
                .maxNewTokens(TOKENS)
                .sampler(SamplerConfig.builder().topK(RowTopK.MAX_DEVICE_CANDIDATES).temperature(0.8f).seed(7L)
                        .penaltyLastN(0).build())
                .logitBias(9, -3f)
                .build();
        try (GGUFModel one = new GGUFModel(model.toString(), config())) {
            int[] host = generate(one, wrap(one.forwardBackend()), req, false);
            TopKBackend device = wrap(one.forwardBackend());
            int[] tokens = generate(one, device, req, true);
            assertArrayEquals(host, tokens);
            assertEquals("cone " + (RowTopK.MAX_DEVICE_CANDIDATES + 1) + " exceeds the readback", 0,
                    device.decodeTopKCalls);
        }
    }

    private static TopKBackend wrap(ForwardBackend backend) {
        return new TopKBackend(backend);
    }

    /** The CPU backend reporting the device top-k readout: every call runs the real CPU forward path. */
    private static final class TopKBackend implements ForwardBackend {

        private final ForwardBackend delegate;
        int decodeTopKCalls;
        int widestK;
        int greedyCalls;

        TopKBackend(ForwardBackend delegate) {
            this.delegate = delegate;
        }

        @Override
        public boolean topKOnDevice() {
            return true;
        }

        @Override
        public RowTopK decodeTopKOnDevice(int token, int pos, int k) {
            decodeTopKCalls++;
            widestK = Math.max(widestK, k);
            return RowTopK.of(new float[][]{delegate.decode(token, pos)}, k);
        }

        @Override
        public int decodeGreedyOnDevice(int token, int pos) {
            greedyCalls++;
            return ForwardBackend.super.decodeGreedyOnDevice(token, pos);
        }

        @Override
        public float[] prefill(int[] tokens, int pos0) {
            return delegate.prefill(tokens, pos0);
        }

        @Override
        public float[] decode(int token, int pos) {
            return delegate.decode(token, pos);
        }

        @Override
        public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
            return delegate.forwardRows(tokens, pos0, outputRows);
        }

        @Override
        public float[][] forwardAll(int[] tokens, int[] outputRows) {
            return delegate.forwardAll(tokens, outputRows);
        }

        @Override
        public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
            return delegate.hiddenRows(tokens, tokenTypes);
        }

        @Override
        public void reset() {
            delegate.reset();
        }

        @Override
        public boolean canRollback(int nTokens) {
            return delegate.canRollback(nTokens);
        }

        @Override
        public void rollback(int nTokens) {
            delegate.rollback(nTokens);
        }

        @Override
        public void shiftContext(int nKeep, int nDiscard) {
            delegate.shiftContext(nKeep, nDiscard);
        }

        @Override
        public com.eonmux.neuralfree.gguf.memory.ModelMemory memory() {
            return delegate.memory();
        }

        @Override
        public void setHiddenTap(HiddenTap tap) {
            delegate.setHiddenTap(tap);
        }

        @Override
        public void close() {
            delegate.close();
        }
    }
}
