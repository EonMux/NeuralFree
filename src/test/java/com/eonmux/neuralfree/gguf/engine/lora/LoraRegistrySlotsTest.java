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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The per-request slot table of {@link LoraRegistry} (vLLM's {@code max_loras} slots, llama.cpp server's
 * per-request {@code lora} list) and a {@link LoraAttachment} serving it: a selection resolves to a slot key at
 * frozen scales, equal keys share a slot, a full table makes a request wait, idle slots stay open and the least
 * recently used one is displaced, adapters a running request holds are neither unloaded nor evicted, and the
 * attachment registers a slot's pairs at the slot so only that slot's rows see them.
 */
public final class LoraRegistrySlotsTest {

    private static final float EPS = 1e-5f;
    private static final int N_EMBD = 6;
    private static final int N_VOCAB = 11;
    private static final String OUTPUT = TensorNames.weight(TensorNames.OUTPUT);

    /** The slot events the registry delivers, in order. */
    private static final class SlotEvents implements LoraListener {
        final List<String> lines = new ArrayList<>();

        @Override
        public void loaded(LoraHandle handle) {
        }

        @Override
        public void unloaded(LoraHandle handle) {
        }

        @Override
        public void scaleChanged(LoraHandle handle, float previous) {
        }

        @Override
        public void slotOpened(int slot, LoraSlotKey key) {
            lines.add("open " + slot + " " + key.parts());
        }

        @Override
        public void slotClosed(int slot, LoraSlotKey key) {
            lines.add("close " + slot + " " + key.parts());
        }
    }

    private static LoraAdapter outputAdapter(String path, long seed) {
        return LoraFixtures.adapter(path, 0.0f, LoraFixtures.projection(OUTPUT, N_EMBD, N_VOCAB, 2, seed));
    }

    /** A registry over in-memory adapters {@code a}, {@code b} and {@code c} of the output head. */
    private static LoraRegistry registry(int maxAdapters) throws IOException {
        Map<String, LoraAdapter> byPath = new HashMap<>();
        for (String name : new String[]{"a", "b", "c"}) {
            byPath.put(name + ".gguf", outputAdapter(name + ".gguf", name.charAt(0)));
        }
        LoraRegistry registry = LoraRegistry.of(path -> {
            LoraAdapter a = byPath.get(path);
            if (a == null) {
                throw new IOException("no adapter at " + path);
            }
            return a;
        }, LoraConfig.builder().maxAdapters(maxAdapters).maxRank(64).cacheEntries(4).build());
        return registry;
    }

    private static LoraSlotKey key(LoraRegistry registry, String name) {
        return registry.resolve(LoraSelection.of(name));
    }

    // ------------------------------------------------------------------ selections and keys

    @Test
    public void aSelectionResolvesToResidentAdaptersAtFrozenScales() throws IOException {
        LoraRegistry registry = registry(4);
        LoraHandle a = registry.load("a", "a.gguf", 0.5f);
        LoraHandle b = registry.load("b", "b.gguf", 2.0f);

        LoraSlotKey key = registry.resolve(LoraSelection.builder().add("b").add("a", 0.25f).build());
        assertEquals(2, key.parts().size());
        assertSame(a, key.parts().get(0).handle());
        assertEquals(0.25f, key.parts().get(0).scale(), 0f);
        assertSame(b, key.parts().get(1).handle());
        assertEquals("the default scale freezes at resolution", 2.0f, key.parts().get(1).scale(), 0f);
        assertTrue(key.contains(a) && key.contains(b));

        registry.setScale("b", 3.0f);
        assertEquals(2.0f, key.parts().get(1).scale(), 0f);
        assertNotEquals(key, registry.resolve(LoraSelection.builder().add("a", 0.25f).add("b").build()));
        assertEquals("order-insensitive at equal scales",
                key, registry.resolve(LoraSelection.builder().add("a", 0.25f).add("b", 2.0f).build()));

        assertTrue(key.identity().contains(a.digest()));
        assertTrue(key.identity().contains(b.digest()));
        assertNull("scale 0 leaves an adapter out (llama.cpp)",
                registry.resolve(LoraSelection.builder().add("a", 0f).build()));
    }

    @Test
    public void aSelectionOfAnUnknownOrModelWideAdapterIsRefusedByName() throws IOException {
        LoraRegistry registry = registry(4);
        IllegalArgumentException unknown = assertThrows(IllegalArgumentException.class,
                () -> registry.resolve(LoraSelection.of("ghost")));
        assertTrue(unknown.getMessage(), unknown.getMessage().contains("ghost"));

        LoraRegistry followed = registry(4);
        LoraAttachment att = LoraAttachment.cpu(Collections.<LoraAttachment.Binding>emptyList(), servedModel());
        att.follow(followed);
        followed.load("wide", "a.gguf");
        IllegalArgumentException wide = assertThrows(IllegalArgumentException.class,
                () -> followed.resolve(LoraSelection.of("wide")));
        assertTrue(wide.getMessage(), wide.getMessage().contains("wide"));
        att.close();

        assertThrows(IllegalArgumentException.class, () -> LoraSelection.builder().add("x").add("x"));
        assertThrows(IllegalArgumentException.class, () -> LoraSelection.builder().add("x", Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> LoraSelection.of(" "));
        assertThrows(IllegalStateException.class, () -> LoraSelection.builder().build());
    }

    // ------------------------------------------------------------------ the slot table

    @Test
    public void equalKeysShareASlotAndAFullTableMakesOtherKeysWait() throws IOException {
        LoraRegistry registry = registry(2);
        SlotEvents events = new SlotEvents();
        registry.addListener(events);
        registry.load("a", "a.gguf");
        registry.load("b", "b.gguf");
        LoraSlotKey a = key(registry, "a");
        LoraSlotKey b = key(registry, "b");
        LoraSlotKey aHalf = registry.resolve(LoraSelection.builder().add("a", 0.5f).build());

        int sa = registry.acquire(a);
        assertEquals(sa, registry.acquire(key(registry, "a")));
        assertEquals(2, registry.slotRequests(sa));
        int sb = registry.acquire(b);
        assertNotEquals(sa, sb);
        assertEquals(2, registry.slotCount());
        assertFalse(registry.canAcquire(aHalf));
        assertEquals(-1, registry.acquire(aHalf));
        assertEquals(Arrays.asList("open " + sa + " [a:1.0]", "open " + sb + " [b:1.0]"), events.lines);

        registry.release(sa);
        assertFalse("one request still holds slot " + sa, registry.canAcquire(aHalf));
        registry.release(sa);
        assertTrue(registry.canAcquire(aHalf));
        assertSame("an idle slot stays open", a, registry.slotKey(sa));
        assertEquals(sa, registry.acquire(aHalf));
        assertEquals(Arrays.asList("open " + sa + " [a:1.0]", "open " + sb + " [b:1.0]",
                "close " + sa + " [a:1.0]", "open " + sa + " [a:0.5]"), events.lines);
        assertThrows(IllegalArgumentException.class, () -> registry.release(sb + 5));
    }

    @Test
    public void theLeastRecentlyUsedIdleSlotIsDisplacedAndAWarmSlotIsReused() throws IOException {
        LoraRegistry registry = registry(2);
        SlotEvents events = new SlotEvents();
        registry.addListener(events);
        registry.load("a", "a.gguf");
        registry.load("b", "b.gguf");
        LoraSlotKey a = key(registry, "a");
        LoraSlotKey b = key(registry, "b");
        LoraSlotKey bHalf = registry.resolve(LoraSelection.builder().add("b", 0.5f).build());

        int sa = registry.acquire(a);
        int sb = registry.acquire(b);
        registry.release(sb);
        registry.release(sa);
        // slot a was released last: slot b is the least recently used idle slot
        events.lines.clear();
        assertEquals(sa, registry.acquire(a));
        assertTrue("a warm slot opens nothing", events.lines.isEmpty());
        assertEquals(sb, registry.acquire(bHalf));
        assertEquals(Arrays.asList("close " + sb + " [b:1.0]", "open " + sb + " [b:0.5]"), events.lines);
    }

    @Test
    public void anAdapterARunningRequestHoldsIsNeitherUnloadedNorEvicted() throws IOException {
        LoraRegistry registry = registry(2);
        SlotEvents events = new SlotEvents();
        registry.addListener(events);
        registry.load("a", "a.gguf");
        registry.load("b", "b.gguf");
        int sa = registry.acquire(key(registry, "a"));
        LoraSlotKey b = key(registry, "b");
        int sb = registry.acquire(b);

        IllegalStateException busy = assertThrows(IllegalStateException.class, () -> registry.unload("a"));
        assertTrue(busy.getMessage(), busy.getMessage().contains("a") && busy.getMessage().contains("1"));
        IllegalStateException full = assertThrows(IllegalStateException.class, () -> registry.load("c", "c.gguf"));
        assertTrue(full.getMessage(), full.getMessage().contains("running request"));

        registry.release(sb);
        events.lines.clear();
        registry.load("c", "c.gguf");
        assertNull("the idle adapter b was evicted", registry.list().stream()
                .filter(h -> h.name().equals("b")).findFirst().orElse(null));
        assertEquals(Collections.singletonList("close " + sb + " [b:1.0]"), events.lines);
        IllegalStateException gone = assertThrows(IllegalStateException.class, () -> registry.acquire(b));
        assertTrue(gone.getMessage(), gone.getMessage().contains("b"));

        registry.release(sa);
        registry.unload("a");
        assertNull(registry.slotKey(sa));
        assertEquals(0, registry.requestsOf(registry.handle("a")));
    }

    // ------------------------------------------------------------------ the attachment's per-request mode

    @Test
    public void aServingAttachmentRegistersAnOpenedSlotsPairsAtTheSlotOnly() throws IOException {
        LoraRegistry registry = registry(2);
        CpuProjectionDeltas deltas = servedModel();
        LoraAttachment att = LoraAttachment.cpu(Collections.<LoraAttachment.Binding>emptyList(), deltas);
        att.serve(registry);
        LoraHandle a = registry.load("a", "a.gguf", 0.5f);
        assertEquals("a per-request adapter attaches to no row", 0, att.deltaCount());
        assertEquals(0, a.holds());
        assertTrue(deltas.served(OUTPUT).isEmpty());
        assertThrows(IllegalStateException.class, () -> att.follow(registry));

        int slot = registry.acquire(key(registry, "a"));
        assertTrue(deltas.holdsSlot(slot));
        assertEquals(1, att.deltaCount());

        float[] x = LoraFixtures.random(3, 2 * N_EMBD);
        float[] expected = new float[2 * N_VOCAB];
        LoraAdapter adapter = a.adapter();
        LoraFixtures.project(adapter.pair(OUTPUT), adapter.scale(OUTPUT, 0.5f), 0, x, N_EMBD, expected, N_VOCAB);
        deltas.rows().set(new int[]{0, slot});
        float[] y = new float[2 * N_VOCAB];
        ((CpuProjectionDeltas.Site) deltas.served(OUTPUT)).applyRows(2, x, 0, y, 0);
        assertArrayEquals("row 0 runs the base model, row 1 the adapter", expected, y, EPS);
        deltas.rows().clear();

        registry.release(slot);
        registry.unload("a");
        assertFalse(deltas.holdsSlot(slot));
        assertEquals(0, att.deltaCount());
        att.close();
    }

    @Test
    public void aSlotWhoseAdapterCannotRunStaysFree() throws IOException {
        LoraRegistry registry = LoraRegistry.of(path -> LoraFixtures.adapter(path, 0.0f,
                LoraFixtures.projection("blk.0.attn_k.weight", N_EMBD, N_EMBD, 2, 3)),
                LoraConfig.builder().maxAdapters(2).maxRank(64).cacheEntries(0).build());
        LoraAttachment att = LoraAttachment.cpu(Collections.<LoraAttachment.Binding>emptyList(), servedModel());
        att.serve(registry);
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> registry.load("k", "k.gguf"));
        assertTrue(e.getMessage(), e.getMessage().contains("k.gguf"));
        assertTrue(registry.list().isEmpty());
        att.close();
    }

    // ------------------------------------------------------------------ helpers

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

    /** Deltas of a one-layer model serving blk.0.attn_q and the output head. */
    private static CpuProjectionDeltas servedModel() {
        CpuProjectionDeltas d = new CpuProjectionDeltas();
        d.serve(layer(0), TensorNames.weight(TensorNames.ATTN_Q), f32(N_EMBD, N_EMBD), CpuMatVec.PLAIN);
        d.serveModel(OUTPUT, N_EMBD, N_VOCAB, null);
        return d;
    }
}
