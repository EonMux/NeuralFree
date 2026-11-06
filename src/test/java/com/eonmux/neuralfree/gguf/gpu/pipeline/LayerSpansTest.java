package com.eonmux.neuralfree.gguf.gpu.pipeline;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Layer-span assignment of a split: equal and weighted boundaries (llama.cpp {@code tensor-split}), at least one
 * layer per device, KV-sharing layers kept with their owner, and the refused shapes.
 */
public class LayerSpansTest {

    private static int[] starts(LayerSpans spans) {
        int[] out = new int[spans.devices() + 1];
        for (int d = 0; d < spans.devices(); d++) {
            out[d] = spans.start(d);
        }
        out[spans.devices()] = spans.start(spans.devices() - 1) + spans.spanLayers(spans.devices() - 1);
        return out;
    }

    private static void assertSpans(LayerSpans spans, int... expectedStarts) {
        assertTrue(Arrays.toString(starts(spans)), Arrays.equals(expectedStarts, starts(spans)));
    }

    // ------------------------------------------------------------------ boundaries

    @Test
    public void equalSharesSplitLayersEvenly() {
        LayerSpans spans = LayerSpans.of(SplitSpecs.trunk(8, com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE),
                2, new double[0]);
        assertSpans(spans, 0, 4, 8);
    }

    @Test
    public void weightedSharesFollowTheirShareOfTheLayers() {
        LayerSpans spans = LayerSpans.of(SplitSpecs.trunk(8, com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE),
                2, new double[]{3, 1});
        assertSpans(spans, 0, 6, 8);
    }

    @Test
    public void threeDevicesTakeTheirShares() {
        LayerSpans spans = LayerSpans.of(SplitSpecs.trunk(12, com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE),
                3, new double[]{1, 1, 2});
        assertSpans(spans, 0, 3, 6, 12);
    }

    @Test
    public void everyDeviceKeepsAtLeastOneLayer() {
        LayerSpans spans = LayerSpans.of(SplitSpecs.trunk(4, com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE),
                3, new double[]{1, 1, 10});
        for (int d = 0; d < spans.devices(); d++) {
            assertTrue("device " + d + " holds no layers", spans.spanLayers(d) >= 1);
        }
        assertEquals(4, spans.start(spans.devices() - 1) + spans.spanLayers(spans.devices() - 1));
    }

    @Test
    public void singleDeviceRunsTheWholeTrunk() {
        LayerSpans spans = LayerSpans.of(SplitSpecs.trunk(5, com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE),
                1, new double[0]);
        assertSpans(spans, 0, 5);
    }

    @Test
    public void explicitCountsBecomeTheSpans() {
        LayerSpans spans = LayerSpans.fromCounts(
                SplitSpecs.trunk(8, com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE), new int[]{5, 3}, 0);
        assertSpans(spans, 0, 5, 8);
        LayerSpans hosted = LayerSpans.fromCounts(
                SplitSpecs.trunk(8, com.eonmux.neuralfree.gguf.arch.MemoryKind.KV), new int[]{2, 3}, 3);
        assertSpans(hosted, 3, 5, 8);
    }

    @Test
    public void countsThatDoNotCoverTheDeviceLayersAreRefused() {
        IllegalArgumentException shortAll = assertThrows(IllegalArgumentException.class,
                () -> LayerSpans.fromCounts(SplitSpecs.trunk(8,
                        com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE), new int[]{3, 3}, 0));
        assertTrue(shortAll.getMessage(), shortAll.getMessage().contains("neuralfree.split.mode"));
        IllegalArgumentException idle = assertThrows(IllegalArgumentException.class,
                () -> LayerSpans.fromCounts(SplitSpecs.trunk(8,
                        com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE), new int[]{8, 0}, 0));
        assertTrue(idle.getMessage(), idle.getMessage().contains("without a layer"));
    }

    @Test
    public void weightLayersRouteToTheirSpan() {
        LayerSpans spans = LayerSpans.of(SplitSpecs.trunk(6, com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE),
                2, new double[0]);
        assertEquals(0, spans.deviceOfWeightLayer(0));
        assertEquals(0, spans.deviceOfWeightLayer(2));
        assertEquals(1, spans.deviceOfWeightLayer(3));
        assertEquals(1, spans.deviceOfWeightLayer(5));
        assertEquals(1, spans.deviceOfWeightLayer(6));
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void hostLayersLeaveTheSpansToTheDeviceLayers() {
        LayerSpans spans = LayerSpans.of(SplitSpecs.trunk(8, com.eonmux.neuralfree.gguf.arch.MemoryKind.KV), 2,
                new double[0], 2);
        assertSpans(spans, 2, 5, 8);
        assertEquals(1, spans.deviceOfWeightLayer(5));
        LayerSpans weighted = LayerSpans.of(SplitSpecs.trunk(8, com.eonmux.neuralfree.gguf.arch.MemoryKind.KV), 2,
                new double[]{1, 2}, 5);
        assertSpans(weighted, 5, 6, 8);
    }

    @Test
    public void hostLayersThatLeaveTooFewDeviceLayersAreRefusedByName() {
        IllegalArgumentException few = assertThrows(IllegalArgumentException.class,
                () -> LayerSpans.of(SplitSpecs.trunk(4, com.eonmux.neuralfree.gguf.arch.MemoryKind.KV), 2,
                        new double[0], 3));
        assertTrue(few.getMessage(), few.getMessage().contains("neuralfree.n.gpu.layers"));
        IllegalArgumentException none = assertThrows(IllegalArgumentException.class,
                () -> LayerSpans.of(SplitSpecs.trunk(4, com.eonmux.neuralfree.gguf.arch.MemoryKind.KV), 2,
                        new double[0], 4));
        assertTrue(none.getMessage(), none.getMessage().contains("on the GPUs"));
    }

    @Test
    public void aKvReaderAcrossTheHostBoundaryIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> LayerSpans.of(SplitSpecs.sharedKv(6, 3, 1), 2, new double[0], 2));
        assertTrue(e.getMessage(), e.getMessage().contains("host boundary"));
        LayerSpans together = LayerSpans.of(SplitSpecs.sharedKv(6, 3, 2), 2, new double[0], 2);
        assertEquals(together.deviceOfWeightLayer(2), together.deviceOfWeightLayer(3));
        assertEquals(2, together.start(0));
    }

    @Test
    public void moreDevicesThanLayersIsRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> LayerSpans.of(SplitSpecs.trunk(2, com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE), 3,
                        new double[0]));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.split.mode"));
    }

    @Test
    public void moreRatiosThanDevicesIsRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> LayerSpans.of(SplitSpecs.trunk(4, com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE), 2,
                        new double[]{1, 1, 1}));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.tensor.split"));
    }

    @Test
    public void fewerRatiosThanDevicesIsRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> LayerSpans.of(SplitSpecs.trunk(4, com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE), 3,
                        new double[]{1, 1}));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.tensor.split"));
    }

    @Test
    public void nonPositiveShareIsRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> LayerSpans.of(SplitSpecs.trunk(4, com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE), 2,
                        new double[]{1, 0}));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.tensor.split"));
    }

    // ------------------------------------------------------------------ shared KV

    @Test
    public void aSharedKvLayerStaysWithItsOwner() {
        int reader = 2;
        int owner = 1;
        LayerSpans spans = LayerSpans.of(SplitSpecs.sharedKv(4, reader, owner), 2, new double[0]);
        assertEquals(spans.deviceOfWeightLayer(owner), spans.deviceOfWeightLayer(reader));
        assertEquals(2, spans.devices());
        for (int d = 0; d < spans.devices(); d++) {
            assertTrue(spans.spanLayers(d) >= 1);
        }
    }

    @Test
    public void aSharedKvLayerMidStackMovesItsBoundary() {
        int reader = 5;
        int owner = 1;
        LayerSpans spans = LayerSpans.of(SplitSpecs.sharedKv(8, reader, owner), 2, new double[]{1, 3});
        assertEquals(spans.deviceOfWeightLayer(owner), spans.deviceOfWeightLayer(reader));
        assertEquals(spans.deviceOfWeightLayer(owner), spans.deviceOfWeightLayer(4));
        assertNotEquals(spans.deviceOfWeightLayer(owner), spans.deviceOfWeightLayer(6));
    }

    @Test
    public void anUnsplittableKvChainIsRefused() {
        // every layer from 1 on reads its predecessor: no boundary can separate them
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> LayerSpans.of(SplitSpecs.sharedKvChain(4), 2, new double[0]));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.tensor.split"));
    }

    @Test
    public void loopedSchedulesAreRefused() {
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> LayerSpans.of(SplitSpecs.looped(4, 2), 2, new double[0]));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.split.mode"));
    }

    @Test
    public void encoderDecoderModelsAreRefused() {
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> LayerSpans.of(SplitSpecs.encDec(2), 2, new double[0]));
        assertTrue(e.getMessage(), e.getMessage().contains("encoder-decoder"));
    }

    @Test
    public void nonPositiveDeviceCountIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> LayerSpans.of(
                SplitSpecs.trunk(4, com.eonmux.neuralfree.gguf.arch.MemoryKind.NONE), 0, new double[0]));
    }
}
