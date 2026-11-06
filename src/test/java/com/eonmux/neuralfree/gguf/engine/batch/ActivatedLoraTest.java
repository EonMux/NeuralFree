package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraHandle;
import com.eonmux.neuralfree.gguf.engine.lora.LoraPlan;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.testutil.LoraAdapterFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Activated LoRA (aLoRA, vLLM's {@code alora_invocation_tokens}, llama.cpp's {@code adapter.alora.invocation_tokens}):
 * the adapter applies per row from the last occurrence of its invocation sequence in the prompt on, the rows before
 * it run without it, and a prompt without the sequence runs without it throughout. Checked against direct backend
 * passes — the rows before the invocation prefilled in the bare slot, the rest and the greedy decode in the adapter's
 * slot 0 — through the batch engine (beside a base request in shared steps) and the single-sequence loop, for a
 * per-request and a startup aLoRA, prompt log-probabilities included; an aLoRA beside another adapter refuses by
 * name.
 */
public class ActivatedLoraTest {

    private static final int RANK = 4;
    private static final float ALPHA = 8.0f;
    private static final float SCALE = 0.9f;
    private static final int N_NEW = 6;
    private static final int EOS = 2;
    private static final TokenCallback NONE = null;

    private static EngineConfig.Builder config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .prefixReuse(true);
        if (parallel > 1) {
            mc = mc.parallel(parallel);
        }
        return EngineConfig.builder().memory(mc.build())
                .loraConfig(LoraConfig.builder().maxAdapters(2).maxRank(64).cacheEntries(2).build());
    }

    private static GenerationRequest request(LoraSelection lora) {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding())
                .lora(lora).build();
    }

    /** A 14-token prompt holding the invocation {@code {77, 91}} at positions 3 and 9 (the last one counts). */
    private static int[] invokedPrompt() {
        return new int[]{11, 12, 13, 77, 91, 14, 15, 16, 17, 77, 91, 18, 19, 20};
    }

    private static final int[] INVOCATION = {77, 91};
    private static final int LAST_START = 9;

    private static int[] plainPrompt() {
        return new int[]{11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    }

    /**
     * The greedy tokens of a model loaded with {@code adapter} at {@code SCALE} as its startup adapter, the rows
     * before {@code start} prefilled in the bare slot and every later row, decode included, in slot 0; a negative
     * {@code start} runs every row bare.
     */
    private static int[] reference(Path model, Path adapter, int[] prompt, int start) throws IOException {
        EngineConfig cfg = config(1).addLora(adapter.toString(), SCALE).build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            LoraRegistry registry = one.loraRegistry();
            ForwardBackend backend = one.forwardBackend();
            backend.reset();
            int bareRows = start < 0 ? prompt.length : start;
            float[] logits = null;
            if (bareRows > 0) {
                registry.useSlot(registry.bareSlot());
                logits = backend.prefill(Arrays.copyOfRange(prompt, 0, bareRows), 0);
            }
            if (start >= 0) {
                registry.useSlot(0);
            }
            if (start >= 0 && start < prompt.length) {
                logits = backend.prefill(Arrays.copyOfRange(prompt, start, prompt.length), start);
            }
            int[] out = new int[N_NEW];
            int n = 0;
            int pos = prompt.length;
            while (n < N_NEW) {
                int token = ForwardBackend.argmax(logits);
                out[n++] = token;
                if (token == EOS || n == N_NEW) {
                    break;
                }
                logits = backend.decode(token, pos++);
            }
            return Arrays.copyOf(out, n);
        }
    }

    /**
     * The prompt log-probabilities ({@code prompt_logprobs: 1}) of {@code prompt} on a model loaded with
     * {@code adapter} at {@code SCALE} as its startup adapter: the rows before {@code start} read back in the bare
     * slot, the rest in slot 0 ({@link PromptLogprobs#capture} over the direct passes).
     */
    private static List<TokenScore[]> referencePromptLogprobs(Path model, Path adapter, int[] prompt, int start)
            throws IOException {
        EngineConfig cfg = config(1).addLora(adapter.toString(), SCALE).build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            LoraRegistry registry = one.loraRegistry();
            ForwardBackend backend = one.forwardBackend();
            backend.reset();
            float[][] rows = new float[prompt.length][];
            if (start > 0) {
                registry.useSlot(registry.bareSlot());
                float[][] pre = backend.forwardRows(Arrays.copyOfRange(prompt, 0, start), 0, null);
                System.arraycopy(pre, 0, rows, 0, start);
            }
            registry.useSlot(0);
            float[][] post = backend.forwardRows(Arrays.copyOfRange(prompt, start, prompt.length), start, null);
            System.arraycopy(post, 0, rows, start, post.length);
            return PromptLogprobs.capture(rows, prompt, 1);
        }
    }

    /** Asserts that {@code got} scores every prompt token as {@code reference} does, within float tolerance. */
    private static void assertPromptLogprobs(List<TokenScore[]> reference, List<TokenScore[]> got) {
        assertEquals("one entry per prompt token", reference.size(), got.size());
        for (int r = 1; r < reference.size(); r++) {
            assertEquals("row " + r + " scores the prompt token", reference.get(r)[0].tokenId(),
                    got.get(r)[0].tokenId());
            assertEquals("row " + r, reference.get(r)[0].logprob(), got.get(r)[0].logprob(), 1e-4f);
        }
    }

    private static void assertMatches(int[] reference, GenerationResult result) {
        int[] got = result.tokens();
        assertTrue("generated something", got.length > 0);
        assertArrayEquals("the generated tokens follow the reference",
                Arrays.copyOf(reference, Math.min(reference.length, got.length)),
                Arrays.copyOf(got, Math.min(reference.length, got.length)));
    }

    private static void assertMatches(int[] reference, BatchSequence sequence) {
        assertEquals("expected DONE but was " + sequence.state()
                + (sequence.error() == null ? "" : ": " + sequence.error()), BatchSequence.State.DONE,
                sequence.state());
        assertMatches(reference, sequence.result());
    }

    @Test
    public void theInvocationStartIsTheLastOccurrence() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path alora = LoraAdapterFiles.activated(3000, RANK, ALPHA, INVOCATION);
        EngineConfig cfg = config(2).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            LoraRegistry registry = m.loraRegistry();
            LoraHandle h = registry.load("al", alora.toString(), SCALE);
            assertTrue(h.adapter().isActivated());
            assertArrayEquals(INVOCATION, h.adapter().invocationTokens());
            assertEquals(LAST_START, h.adapter().invocationStart(invokedPrompt()));
            assertEquals(-1, h.adapter().invocationStart(plainPrompt()));
            assertEquals(-1, h.adapter().invocationStart(new int[]{91}));

            LoraPlan plan = LoraPlan.resolve(registry, LoraSelection.of("al"), invokedPrompt());
            assertEquals(LAST_START, plan.from());
            assertTrue("the rows before run the bare model", plan.preKey() == null || plan.preKey().isBare());
            assertTrue(plan.identity(), plan.identity().endsWith("@" + LAST_START));
            LoraPlan absent = LoraPlan.resolve(registry, LoraSelection.of("al"), plainPrompt());
            assertTrue("without the sequence the aLoRA is off", absent.isDefault());
            UnsupportedOperationException media = assertThrows(UnsupportedOperationException.class,
                    () -> LoraPlan.resolve(registry, LoraSelection.of("al"), (int[]) null));
            assertTrue(media.getMessage(), media.getMessage().contains("activated LoRA"));
            UnsupportedOperationException pooled = assertThrows(UnsupportedOperationException.class,
                    () -> LoraPlan.uniform(registry, LoraSelection.of("al"), "a pooling forward runs one slot"));
            assertTrue(pooled.getMessage(), pooled.getMessage().endsWith("a pooling forward runs one slot"));
        }
    }

    @Test
    public void aPerRequestAloraAppliesFromItsInvocationInTheBatchEngineAndTheLoop() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path alora = LoraAdapterFiles.activated(3000, RANK, ALPHA, INVOCATION);
        int[] invoked = invokedPrompt();
        int[] refInvoked = reference(model, alora, invoked, LAST_START);
        int[] refAlways = reference(model, alora, invoked, 0);
        int[] refBase = reference(model, alora, plainPrompt(), -1);
        assertFalse("the invocation start must change the text", Arrays.equals(refInvoked, refAlways));

        EngineConfig cfg = config(2).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            m.loraRegistry().load("al", alora.toString(), SCALE);
            BatchSequence sAl = engine.submit(invoked, request(LoraSelection.of("al")), NONE);
            BatchSequence sBase = engine.submit(plainPrompt(), request(LoraSelection.of("al")), NONE);
            engine.drain();
            assertMatches(refInvoked, sAl);
            assertMatches(refBase, sBase);

            BatchSequence again = engine.submit(invoked, request(LoraSelection.of("al")), NONE);
            engine.drain();
            assertTrue("rows of the same invocation are reused", again.reusedPrefix() > 0);
            assertMatches(refInvoked, again);
            BatchSequence base = engine.submit(invoked, request(null), NONE);
            engine.drain();
            assertEquals("the base model reuses no row an aLoRA computed", 0, base.reusedPrefix());
        }

        EngineConfig single = config(1).build();
        try (GGUFModel one = new GGUFModel(model.toString(), single)) {
            one.loraRegistry().load("al", alora.toString(), SCALE);
            GenerationLoop loop = GenerationLoop.forModel(one, single);
            assertMatches(refInvoked, loop.generate(invoked, request(LoraSelection.of("al")), NONE));
            assertMatches(refInvoked, loop.generate(invoked, request(LoraSelection.of("al")), NONE));
            assertMatches(refBase, loop.generate(plainPrompt(), request(LoraSelection.of("al")), NONE));
        }
    }

    @Test
    public void promptLogprobsUnderAnAloraReadEachRowInItsSlot() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path alora = LoraAdapterFiles.activated(3000, RANK, ALPHA, INVOCATION);
        int[] invoked = invokedPrompt();
        List<TokenScore[]> reference = referencePromptLogprobs(model, alora, invoked, LAST_START);
        List<TokenScore[]> always = referencePromptLogprobs(model, alora, invoked, 0);
        assertFalse("the rows before the invocation must score differently without the adapter",
                Math.abs(reference.get(LAST_START - 1)[0].logprob() - always.get(LAST_START - 1)[0].logprob())
                        < 1e-4f);
        GenerationRequest logprobs = request(LoraSelection.of("al")).toBuilder().promptLogprobs(1).build();

        EngineConfig cfg = config(2).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            m.loraRegistry().load("al", alora.toString(), SCALE);
            BatchSequence seq = engine.submit(invoked, logprobs, NONE);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, seq.state());
            assertPromptLogprobs(reference, seq.result().promptLogprobs());
        }
        EngineConfig single = config(1).build();
        try (GGUFModel one = new GGUFModel(model.toString(), single)) {
            one.loraRegistry().load("al", alora.toString(), SCALE);
            GenerationLoop loop = GenerationLoop.forModel(one, single);
            loop.generate(invoked, request(LoraSelection.of("al")), NONE);
            GenerationResult result = loop.generate(invoked, logprobs, NONE);
            assertPromptLogprobs(reference, result.promptLogprobs());
            assertMatches(reference(model, alora, invoked, LAST_START), result);
        }
    }

    @Test
    public void aStartupAloraPlansItsInvocationForRequestsThatSelectNone() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path alora = LoraAdapterFiles.activated(3000, RANK, ALPHA, INVOCATION);
        int[] invoked = invokedPrompt();
        int[] refInvoked = reference(model, alora, invoked, LAST_START);
        int[] refBare = reference(model, alora, invoked, -1);

        EngineConfig cfg = config(2).addLora(alora.toString(), SCALE).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            String name = m.loraRegistry().list().get(0).name();
            BatchSequence sDefault = engine.submit(invoked, request(null), NONE);
            BatchSequence sOff = engine.submit(invoked,
                    request(LoraSelection.builder().add(name, 0f).build()), NONE);
            engine.drain();
            assertMatches(refInvoked, sDefault);
            assertMatches(refBare, sOff);
        }
        EngineConfig single = config(1).addLora(alora.toString(), SCALE).build();
        try (GGUFModel one = new GGUFModel(model.toString(), single)) {
            assertMatches(refInvoked, GenerationLoop.forModel(one, single).generate(invoked, request(null), NONE));
        }
    }

    @Test
    public void anAloraBesideAnotherAdapterRefusesByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path alora = LoraAdapterFiles.activated(3000, RANK, ALPHA, INVOCATION);
        Path plain = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        EngineConfig cfg = config(2).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            m.loraRegistry().load("al", alora.toString(), SCALE);
            m.loraRegistry().load("p", plain.toString(), SCALE);
            IllegalArgumentException both = assertThrows(IllegalArgumentException.class,
                    () -> engine.submit(invokedPrompt(),
                            request(LoraSelection.builder().add("al").add("p").build()), NONE));
            assertTrue(both.getMessage(), both.getMessage().contains("al is an activated LoRA"));
            assertEquals(0, engine.queued());
        }
        EngineConfig startup = config(1).addLora(plain.toString(), SCALE).addLora(alora.toString(), SCALE).build();
        IllegalArgumentException mixed = assertThrows(IllegalArgumentException.class,
                () -> new GGUFModel(model.toString(), startup).close());
        assertTrue(mixed.getMessage(), mixed.getMessage().contains("runs alone"));
    }

    /** The media prompt of the row-mapping tests: 3 text rows, a 16-row image (4 positions) and the tail. */
    private static MediaPrompt mediaPrompt(int[] tail) {
        return MediaPrompt.builder()
                .addText(new int[]{11, 12, 13})
                .addMedia(MediaRopePositions.image2d(3, 4, 4), imageRows(16, 0.3f), true)
                .addText(tail)
                .build();
    }

    private static float[][] imageRows(int n, float v) {
        float[][] r = new float[n][64];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < 64; d++) {
                r[i][d] = v * (1 + 0.02f * ((i + d) % 5));
            }
        }
        return r;
    }

    @Test
    public void theInvocationStartOfAMediaPromptMapsToItsRow() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path alora = LoraAdapterFiles.activated(3000, RANK, ALPHA, INVOCATION);
        EngineConfig cfg = config(2).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            LoraRegistry registry = m.loraRegistry();
            registry.load("al", alora.toString(), SCALE);
            MediaPrompt prompt = mediaPrompt(new int[]{77, 91, 18});
            assertArrayEquals(new int[]{0, 1, 2, 19, 20, 21}, prompt.samplerTokenRows());
            assertArrayEquals(new int[]{11, 12, 13, 77, 91, 18}, prompt.samplerTokens());
            LoraPlan plan = LoraPlan.resolve(registry, LoraSelection.of("al"), prompt);
            assertEquals(19, plan.from());
            assertTrue("the rows before run the bare model", plan.preKey() == null || plan.preKey().isBare());
            assertTrue(plan.identity(), plan.identity().endsWith("@19"));
            LoraPlan absent = LoraPlan.resolve(registry, LoraSelection.of("al"),
                    mediaPrompt(new int[]{18, 19, 20}));
            assertTrue("without the sequence the aLoRA is off", absent.isDefault());
            assertEquals(22, prompt.nTokens());
            assertEquals(10, prompt.nPositions());
        }
    }

    /**
     * An aLoRA over a prompt with media parts applies from its invocation row on: the rows before it (the image
     * rows included) run without the adapter, the rows from it on — under M-RoPE, whose positions differ from the
     * rows — in the adapter's slot, checked against direct per-slot passes of the same model.
     */
    @Test
    public void anAloraOverAMediaPromptAppliesFromItsInvocationRow() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        Path alora = LoraAdapterFiles.activatedQwen2vl(3100, RANK, ALPHA, INVOCATION);
        MediaPrompt invoked = mediaPrompt(new int[]{77, 91, 18});
        MediaPrompt absent = mediaPrompt(new int[]{18, 19, 20});
        int start = 19;
        int[] refInvoked = mediaReference(model, alora, invoked, start);
        int[] refBare = mediaReference(model, alora, absent, -1);
        assertFalse("the invocation start must change the text", Arrays.equals(refInvoked, refBare));

        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                        .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).prefixReuse(true).build())
                .loraConfig(LoraConfig.builder().maxAdapters(2).maxRank(64).cacheEntries(2).build()).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            m.loraRegistry().load("al", alora.toString(), SCALE);
            BatchSequence sInvoked = engine.submit(invoked, request(LoraSelection.of("al")), NONE);
            engine.drain();
            assertMatches(refInvoked, sInvoked);
            BatchSequence sAbsent = engine.submit(absent, request(LoraSelection.of("al")), NONE);
            engine.drain();
            assertMatches(refBare, sAbsent);
            BatchSequence again = engine.submit(invoked, request(LoraSelection.of("al")), NONE);
            engine.drain();
            assertTrue("rows of the same invocation are reused", again.reusedPrefix() > 0);
            assertMatches(refInvoked, again);
        }

        // a startup aLoRA serves the same media prompt for a request that selects nothing
        EngineConfig startup = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                        .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build())
                .addLora(alora.toString(), SCALE).build();
        try (GGUFModel m = new GGUFModel(model.toString(), startup);
             BatchEngine engine = BatchEngine.forModel(m, startup)) {
            BatchSequence sDefault = engine.submit(invoked, request(null), NONE);
            engine.drain();
            assertMatches(refInvoked, sDefault);
        }
    }

    /**
     * The greedy tokens of a qwen2vl model loaded with the aLoRA {@code adapter} at {@code SCALE}: the rows before
     * {@code start} prefill in the bare slot (the image rows included), the rest and the decode in slot 0; a
     * negative {@code start} runs every row bare. Each chunk prefills as the batch engine schedules it, the image
     * chunk as its own embedding batch at its M-RoPE positions.
     */
    private static int[] mediaReference(Path model, Path adapter, MediaPrompt prompt, int start) throws IOException {
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                        .typeK(KvDType.F32).typeV(KvDType.F32).parallel(1).kvPaged(true).build())
                .addLora(adapter.toString(), SCALE).build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            LoraRegistry registry = one.loraRegistry();
            ForwardBackend backend = one.forwardBackend();
            backend.reset();
            float[] logits = null;
            int row = 0;
            int pos = 0;
            for (MediaChunk c : prompt.chunks()) {
                boolean adapterRows = start >= 0 && row + c.nTokens() > start;
                registry.useSlot(adapterRows ? 0 : registry.bareSlot());
                ForwardBatch.Builder b = ForwardBatch.builder(c.nTokens());
                if (c.isText()) {
                    int[] tokens = c.tokens();
                    for (int i = 0; i < c.nTokens(); i++) {
                        b.add(tokens[i], pos + i, 0, i == c.nTokens() - 1);
                    }
                } else {
                    MediaRopePositions rp = c.positions();
                    float[][] rows = c.embeddings();
                    for (int i = 0; i < c.nTokens(); i++) {
                        b.addEmbedding(rows[i], rp.position(0, i), 0, i == c.nTokens() - 1);
                    }
                    b.ropePositions(rp.sections());
                    b.nonCausal();
                }
                float[][] out = backend.forwardBatch(b.build());
                logits = out[out.length - 1];
                row += c.nTokens();
                pos += c.nPositions();
            }
            registry.useSlot(start >= 0 && start < prompt.nTokens() ? 0 : registry.bareSlot());
            int[] out = new int[N_NEW];
            int n = 0;
            while (n < N_NEW) {
                int token = ForwardBackend.argmax(logits);
                out[n++] = token;
                if (token == EOS || n == N_NEW) {
                    break;
                }
                ForwardBatch.Builder d = ForwardBatch.builder(1);
                d.add(token, pos, 0, true);
                logits = backend.forwardBatch(d.build())[0];
                pos++;
            }
            return Arrays.copyOf(out, n);
        }
    }
}
