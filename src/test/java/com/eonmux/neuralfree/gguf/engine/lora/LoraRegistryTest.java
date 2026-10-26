package com.eonmux.neuralfree.gguf.engine.lora;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.heads.LoraFixtures;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link LoraRegistry} and its engine surface (LR-1, J3 and J7): a load makes the handle resident and its events
 * reach a {@link LoraAttachment} synchronously, refusals name the adapter or tensor, displacement keeps the host
 * cache under {@code neuralfree.lora.cache} and the residents under {@code neuralfree.lora.max}, an unload detaches
 * the deltas and reload reattaches them, {@code setScale} of a held adapter rescales its pairs to
 * {@code newScale * B} and creates a new epoch (also returning to an earlier value), an unheld adapter's scale is
 * the default of its later requests, and the content digest covers the adapter's content (§5.5.1 N-2).
 */
public class LoraRegistryTest {

    private static final float EPS = 1e-5f;
    private static final int N_EMBD = 6;
    private static final int N_VOCAB = 11;
    private static final String Q = TensorNames.weight(TensorNames.ATTN_Q);
    private static final String OUTPUT = TensorNames.weight(TensorNames.OUTPUT);
    private static final String EMBD = TensorNames.weight(TensorNames.TOKEN_EMBD);

    /** The events the registry delivers, in order. */
    private static final class Events implements LoraListener {
        final List<String> lines = new ArrayList<>();

        @Override
        public void loaded(LoraHandle handle) {
            lines.add("loaded " + handle.name() + " scale=" + handle.scale());
        }

        @Override
        public void unloaded(LoraHandle handle) {
            lines.add("unloaded " + handle.name());
        }

        @Override
        public void scaleChanged(LoraHandle handle, float previous) {
            lines.add("scale " + handle.name() + " " + previous + " -> " + handle.scale());
        }
    }

    private static LayerSpec layer(int il) {
        return LayerSpec.builder(il, MixerKind.NONE, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .build();
    }

    private static CpuTensorRef f32(long rows, long cols) {
        ByteBuffer buf = ByteBuffer.allocate((int) (rows * cols * Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
        return CpuTensorRef.of(GGMLType.F32, buf, rows, cols);
    }

    /** Deltas of a one-layer model serving atto_q, the output head and the token table. */
    private static CpuProjectionDeltas servedModel() {
        CpuProjectionDeltas d = new CpuProjectionDeltas();
        LayerSpec l = layer(0);
        d.serve(l, blkQ0(l), f32(N_EMBD, N_EMBD), CpuMatVec.PLAIN);
        d.serveModel(OUTPUT, N_EMBD, N_VOCAB, null);
        d.serveEmbedding(EMBD, N_VOCAB, N_EMBD, null);
        return d;
    }

    private static String blkQ0(LayerSpec l) {
        return l.tensorName(TensorNames.weight(TensorNames.ATTN_Q));
    }

    /** The in-memory adapter source: one path always builds equal content (equal digests, fresh objects). */
    private static final class Adapters implements LoraRegistry.Loader {
        private final Map<String, LoraAdapter> byPath = new HashMap<>();
        private final Map<String, LoraAdapter> rebuilt = new HashMap<>();

        void put(String path, LoraAdapter adapter) {
            byPath.put(path, adapter);
        }

        /** The factory-style path used where objects must be fresh per load. */
        void putRebuilt(String path, LoraAdapter adapter) {
            rebuilt.put(path, adapter);
        }

        @Override
        public LoraAdapter open(String path) throws IOException {
            if (rebuilt.containsKey(path)) {
                return rebuilt.get(path);
            }
            LoraAdapter adapter = byPath.get(path);
            if (adapter == null) {
                throw new IOException("no in-memory adapter at " + path);
            }
            return adapter;
        }
    }

    private static LoraConfig config(int max, int maxRank, int cache) {
        return LoraConfig.builder().maxAdapters(max).maxRank(maxRank).cacheEntries(cache).build();
    }

    // ------------------------------------------------------------------ registry operations

    @Test
    public void aLoadMakesTheHandleResidentAndDeliversItsEventsSynchronously() throws IOException {
        Adapters adapters = new Adapters();
        LoraAdapter a = LoraFixtures.adapter("a.gguf", 0.0f,
                LoraFixtures.projection(TensorNames.weight(TensorNames.ATTN_Q), N_EMBD, N_EMBD, 2, 1));
        adapters.put("a.gguf", a);
        LoraRegistry registry = LoraRegistry.of(adapters, config(4, 64, 8));
        Events events = new Events();
        registry.addListener(events);

        LoraHandle handle = registry.load("work", "a.gguf", 0.5f);
        assertNotNull(handle);
        assertEquals("work", handle.name());
        assertEquals("a.gguf", handle.path());
        assertSame(a, handle.adapter());
        assertEquals(0.5f, handle.scale(), 0f);
        assertEquals(0L, handle.revision());
        assertEquals(0, handle.holds());
        assertEquals(a.digest(), handle.digest());
        assertEquals(Arrays.asList("loaded work scale=0.5"), events.lines);
        assertEquals(Collections.singletonList(handle), registry.list());
        assertSame(handle, registry.handle("work"));

        LoraHandle second = registry.load("again", "a.gguf", 1.0f);
        assertSame(a, second.adapter());
        assertEquals(handle.digest(), second.digest());
        assertEquals(Arrays.asList("loaded work scale=0.5", "loaded again scale=1.0"), events.lines);
        assertEquals(2, registry.list().size());
        assertTrue(second.digest().matches("[0-9a-f]{64}"));
    }

    @Test
    public void aResidentNameAndInvalidInputsAreRefused() throws IOException {
        Adapters adapters = new Adapters();
        adapters.put("a.gguf", buildQ("a.gguf"));
        LoraRegistry registry = LoraRegistry.of(adapters, config(4, 64, 8));
        registry.load("the-adapter", "a.gguf");
        try {
            registry.load("the-adapter", "a.gguf");
            fail("a resident name must be refused");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("the-adapter"));
        }
        try {
            registry.load(" ", "a.gguf");
            fail("a blank name must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("name"));
        }
        try {
            registry.load("nan", "a.gguf", Float.NaN);
            fail("a non-finite scale must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("nan"));
        }
        try {
            registry.setScale("unknown", 1.0f);
            fail("a foreign name must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("unknown"));
        }
        try {
            registry.unload("unknown");
            fail("unloading a foreign name must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("unknown"));
        }
    }

    @Test
    public void aPairAboveTheMaxRankIsRefusedNamingItsTensorAndRank() throws IOException {
        Adapters adapters = new Adapters();
        adapters.put("wide.gguf", LoraFixtures.adapter("wide.gguf", 0.0f,
                LoraFixtures.projection(TensorNames.weight(TensorNames.ATTN_Q), N_EMBD, N_EMBD, 65, 1)));
        try {
            LoraRegistry.of(adapters, config(4, 64, 8)).load("w", "wide.gguf");
            fail("a rank 65 pair must be refused at max rank 64");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(TensorNames.weight(TensorNames.ATTN_Q)));
            assertTrue(expected.getMessage(), expected.getMessage().contains("65"));
        }
    }

    @Test
    public void unloadCachesTheHandleAndHostEvictsItsOldestEntry() throws IOException {
        Adapters adapters = new Adapters();
        adapters.put("a.gguf", buildQ("a0.gguf"));
        adapters.put("b.gguf", buildQ("b0.gguf"));
        LoraRegistry registry = LoraRegistry.of(adapters, config(4, 64, 1));
        LoraHandle first = registry.load("first", "a.gguf");
        LoraHandle second = registry.load("second", "b.gguf");
        registry.unload("first");
        assertEquals(Collections.singletonList(first), registry.cached());
        assertFalse(registry.list().contains(first));
        assertNotNull(registry.handle("first"));

        // The cache holds one entry: unloading a second one evicts the oldest cached handle.
        registry.unload("second");
        assertEquals(Collections.singletonList(second), registry.cached());
        // Unloading the already cached name again is refused as a state misuse.
        try {
            registry.unload("second");
            fail("a cached-only name is not resident");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("second"));
        }
    }

    @Test
    public void theResidencyLimitEvictsTheOldestUnheldResidentAndRefusesWhenEveryResidentIsHeld() throws IOException {
        Adapters adapters = new Adapters();
        adapters.put("a.gguf", buildQ("a0.gguf"));
        adapters.put("b0.gguf", buildQ("b0.gguf"));
        adapters.put("c0.gguf", buildQ("c0.gguf"));

        // Without an attachment every resident is unheld: the LRU evicts the oldest one to the host cache.
        LoraRegistry unheld = LoraRegistry.of(adapters, config(2, 64, 2));
        LoraHandle first = unheld.load("first", "a0.gguf");
        LoraHandle second = unheld.load("second", "b0.gguf");
        LoraHandle third = unheld.load("third", "c0.gguf");
        assertNull(unheld.handle("first"));
        assertEquals(Arrays.asList(second, third), unheld.list());
        assertEquals(Collections.singletonList(first), unheld.cached());

        // With an attachment holding both residents, the load is refused and leaves the residents.
        LoraRegistry held = LoraRegistry.of(adapters, config(2, 64, 2));
        CpuProjectionDeltas deltas = servedModel();
        LoraAttachment att = LoraAttachment.cpu(Collections.<LoraAttachment.Binding>emptyList(), deltas);
        att.follow(held);
        LoraHandle h1 = held.load("h1", "a0.gguf");
        LoraHandle h2 = held.load("h2", "b0.gguf");
        assertEquals(1, h1.holds());
        assertEquals(1, h2.holds());
        try {
            held.load("overflow", "c0.gguf");
            fail("loading past " + LoraConfig.PROP_MAX + " held residents must be refused");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("neuralfree.lora.max"));
        }
        assertEquals(Arrays.asList(h1, h2), held.list());
        assertTrue(held.cached().isEmpty());
        att.close();
    }

    @Test
    public void unloadDetachesTheDeltasAndAHotReloadReattachesThem() throws IOException {
        Adapters adapters = new Adapters();
        adapters.putRebuilt("a.gguf", LoraFixtures.adapter("a.gguf", 4.0f,
                LoraFixtures.projection(TensorNames.weight(TensorNames.ATTN_Q), N_EMBD, N_EMBD, 2, 1),
                LoraFixtures.embedding(EMBD, N_VOCAB, N_EMBD, 2, 7)));
        CpuProjectionDeltas deltas = servedModel();
        LoraRegistry registry = LoraRegistry.of(adapters, config(4, 64, 8));
        LoraAttachment att = LoraAttachment.cpu(Collections.<LoraAttachment.Binding>emptyList(), deltas);
        att.follow(registry);

        LoraHandle handle = registry.load("hot", "a.gguf");
        assertEquals(2, att.deltaCount());
        assertTrue(att.bindings().get(0).adapter().targets().contains(TensorNames.weight(TensorNames.ATTN_Q)));

        registry.unload("hot");
        assertEquals(0, att.deltaCount());
        assertTrue(deltas.served(TensorNames.weight(TensorNames.ATTN_Q)).isEmpty());
        assertTrue(deltas.served(EMBD).isEmpty());
        assertSame(handle, registry.handle("hot"));
        assertEquals(Collections.singletonList(handle), registry.cached());

        LoraHandle reloaded = registry.load("hot", "a.gguf", 0.25f);
        assertEquals(2, att.deltaCount());
        assertEquals(handle.digest(), reloaded.digest());
        assertEquals(handle.contentLength(), reloaded.contentLength());
        assertEquals(0.25f, att.bindings().get(0).scale(), 0f);
        assertTrue(registry.cached().isEmpty());
    }

    @Test
    public void aHotLoadOfAnUnrunnablePairFailsWholeAndRegistersNothing() throws IOException {
        Adapters adapters = new Adapters();
        adapters.putRebuilt("bad.gguf", LoraFixtures.adapter("bad.gguf", 0.0f,
                LoraFixtures.projection(TensorNames.weight(TensorNames.ATTN_Q), N_EMBD, N_EMBD, 2, 1),
                LoraFixtures.projection("blk.0.attn_k.weight", N_EMBD, N_EMBD, 2, 3)));
        CpuProjectionDeltas deltas = servedModel();
        LoraRegistry registry = LoraRegistry.of(adapters, config(4, 64, 8));
        LoraAttachment att = LoraAttachment.cpu(Collections.<LoraAttachment.Binding>emptyList(), deltas);
        att.follow(registry);
        try {
            registry.load("bad", "bad.gguf");
            fail("a pair whose base tensor no bound block serves must fail the load");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("bad.gguf"));
        }
        assertEquals(0, att.deltaCount());
        assertTrue(deltas.served(TensorNames.weight(TensorNames.ATTN_Q)).isEmpty());
        assertTrue(registry.list().isEmpty());
        assertTrue(registry.cached().isEmpty());
        att.close();
    }

    // ------------------------------------------------------------------ runtime scales

    @Test
    public void setScaleRescalesTheHeldDeltasToNewScaleTimesB() throws IOException {
        Adapters adapters = new Adapters();
        LoraAdapter a = LoraFixtures.adapter("c.gguf", 6.0f,
                LoraFixtures.projection(TensorNames.weight(TensorNames.OUTPUT), N_EMBD, N_VOCAB, 3, 5));
        adapters.put("c.gguf", a);
        LoraRegistry registry = LoraRegistry.of(adapters, config(4, 64, 8));
        CpuProjectionDeltas deltas = servedModel();
        LoraAttachment att = LoraAttachment.cpu(Collections.<LoraAttachment.Binding>emptyList(), deltas);
        att.follow(registry);
        Events events = new Events();
        registry.addListener(events);

        LoraHandle handle = registry.load("work", "c.gguf", 0.5f);
        float[] x = LoraFixtures.random(11, 2 * N_EMBD);
        float[] expectedHalf = new float[2 * N_VOCAB];
        for (int t = 0; t < 2; t++) {
            LoraFixtures.project(a.pair(TensorNames.weight(TensorNames.OUTPUT)), a.scale(
                    TensorNames.weight(TensorNames.OUTPUT), 0.5f), 0, x, t * N_EMBD, expectedHalf, t * N_VOCAB);
        }
        float[] atHalf = new float[2 * N_VOCAB];
        applyOutput(deltas, 2, x, atHalf);
        assertArrayEquals(expectedHalf, atHalf, EPS);

        registry.setScale("work", 0.25f);
        assertEquals(1L, handle.revision());
        assertEquals(Arrays.asList("loaded work scale=0.5", "scale work 0.5 -> 0.25"), events.lines);
        float[] expectedQuarter = new float[2 * N_VOCAB];
        for (int t = 0; t < 2; t++) {
            LoraFixtures.project(a.pair(TensorNames.weight(TensorNames.OUTPUT)), a.scale(
                    TensorNames.weight(TensorNames.OUTPUT), 0.25f), 0, x, t * N_EMBD, expectedQuarter, t * N_VOCAB);
        }
        float[] atQuarter = new float[2 * N_VOCAB];
        applyOutput(deltas, 2, x, atQuarter);
        assertArrayEquals(expectedQuarter, atQuarter, EPS);

        // Returning to an earlier user scale is a new epoch of the held state (§5.5.1 N-2), never the first one.
        registry.setScale("work", 0.5f);
        assertEquals(2L, handle.revision());
        float[] backAtHalf = new float[2 * N_VOCAB];
        applyOutput(deltas, 2, x, backAtHalf);
        assertArrayEquals(atHalf, backAtHalf, EPS);
        assertEquals(2, handle.holds());
        assertEquals(1, att.deltaCount());
        att.close();
    }

    @Test
    public void theScaleOfAnUnheldAdapterIsTheDefaultOfItsLaterRequestsAndRewritesNothing() throws IOException {
        Adapters adapters = new Adapters();
        adapters.put("p.gguf", buildQ("p.gguf"));
        Events events = new Events();
        LoraRegistry registry = LoraRegistry.of(adapters, config(4, 64, 8));
        registry.addListener(events);
        LoraHandle handle = registry.load("queue", "p.gguf", 1.0f);

        registry.setScale("queue", 3.0f);
        assertSame(handle, registry.handle("queue"));
        assertEquals(3.0f, handle.scale(), 0f);
        assertEquals(0L, handle.revision());
        assertEquals(0, handle.holds());
        assertEquals(Arrays.asList("loaded queue scale=1.0", "scale queue 1.0 -> 3.0"), events.lines);
    }

    // ------------------------------------------------------------------ the content digest

    @Test
    public void theDigestCoversTheContentAndEqualContentHoldsItsValue() {
        LoraAdapter a = LoraFixtures.adapter("one.gguf", 2.0f,
                LoraFixtures.projection(TensorNames.weight(TensorNames.ATTN_Q), 4, 3, 2, 1));
        LoraAdapter sameContent = LoraFixtures.adapter("same-content.gguf", 2.0f,
                LoraFixtures.projection(TensorNames.weight(TensorNames.ATTN_Q), 4, 3, 2, 1));
        LoraAdapter otherAlpha = LoraFixtures.adapter("alpha.gguf", 4.0f,
                LoraFixtures.projection(TensorNames.weight(TensorNames.ATTN_Q), 4, 3, 2, 1));
        LoraAdapter otherTensor = LoraFixtures.adapter("content.gguf", 2.0f,
                LoraFixtures.projection(TensorNames.weight(TensorNames.ATTN_Q), 4, 3, 2, 2));
        LoraAdapter otherRank = LoraFixtures.adapter("rank.gguf", 2.0f,
                LoraFixtures.projection(TensorNames.weight(TensorNames.ATTN_Q), 4, 3, 3, 1));
        LoraAdapter extraTarget = LoraFixtures.adapter("targets.gguf", 2.0f,
                LoraFixtures.projection(TensorNames.weight(TensorNames.ATTN_Q), 4, 3, 2, 1),
                LoraFixtures.projection(TensorNames.weight(TensorNames.OUTPUT), 4, 5, 2, 9));

        assertEquals(a.digest(), sameContent.digest());
        assertEquals(a.contentLength(), sameContent.contentLength());
        assertTrue(a.contentLength() > 0);
        assertTrue(a.digest().matches("[0-9a-f]{64}"));
        assertNotEquals(a.digest(), otherAlpha.digest());
        assertNotEquals(a.digest(), otherTensor.digest());
        assertNotEquals(a.digest(), otherRank.digest());
        assertNotEquals(a.digest(), extraTarget.digest());
    }

    // ------------------------------------------------------------------ properties

    @Test
    public void thePropertiesAndTheBuilderFillTheConfig() {
        assertEquals(4, LoraConfig.defaults().maxAdapters());
        assertEquals(64, LoraConfig.defaults().maxRank());
        assertEquals(8, LoraConfig.defaults().cacheEntries());

        Properties p = new Properties();
        p.setProperty(LoraConfig.PROP_MAX, "2");
        p.setProperty(LoraConfig.PROP_MAX_RANK, "16");
        p.setProperty(LoraConfig.PROP_CACHE, "0");
        LoraConfig fromProps = LoraConfig.fromProperties(p);
        assertEquals(2, fromProps.maxAdapters());
        assertEquals(16, fromProps.maxRank());
        assertEquals(0, fromProps.cacheEntries());
        assertEquals(fromProps, fromProps.toBuilder().build());

        LoraConfig built = LoraConfig.builder().maxAdapters(3).maxRank(8).cacheEntries(5).build();
        assertEquals(3, built.maxAdapters());
        assertEquals(8, built.maxRank());
        assertEquals(5, built.cacheEntries());

        try {
            LoraConfig.builder().maxAdapters(0).build();
            fail(LoraConfig.PROP_MAX + " below 1 must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(LoraConfig.PROP_MAX));
        }
        try {
            LoraConfig.builder().maxRank(-1).build();
            fail(LoraConfig.PROP_MAX_RANK + " below 1 must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(LoraConfig.PROP_MAX_RANK));
        }
        try {
            LoraConfig.builder().cacheEntries(-1).build();
            fail(LoraConfig.PROP_CACHE + " below 0 must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(LoraConfig.PROP_CACHE));
        }
        try {
            Properties bad = new Properties();
            bad.setProperty(LoraConfig.PROP_MAX, "many");
            LoraConfig.fromProperties(bad);
            fail("a non-integer property must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(LoraConfig.PROP_MAX));
        }
    }

    // ------------------------------------------------------------------ helpers

    private static LoraAdapter buildQ(String path) {
        return LoraFixtures.adapter(path, 0.0f,
                LoraFixtures.projection(TensorNames.weight(TensorNames.ATTN_Q), N_EMBD, N_EMBD, 2, 1));
    }

    private static void applyOutput(CpuProjectionDeltas deltas, int n, float[] x, float[] y) {
        ((CpuProjectionDeltas.Site) deltas.served(TensorNames.weight(TensorNames.OUTPUT)))
                .applyRows(n, x, 0, y, 0);
    }
}
