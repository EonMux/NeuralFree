package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.engine.GreedyStep;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import org.junit.After;
import org.junit.Test;

import java.lang.ref.WeakReference;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Greedy steps chained on the device over the real forward backend on the capture-capable fake queue, whose argmax
 * kernel selects {@link #select} of the step's input id as it sits in the token-id slot: a chained step copies the
 * previous selection into that slot on the device instead of writing a host id, so the chain selects the serial ids
 * only when the copy delivered them; the stored sequence ends with every input id once the steps completed, also for
 * a step chained on one whose id was read already; a device failure reaches every wait as the device's own
 * exception; a completed step keeps no earlier step reachable; a decode step writes no scratch position slot, which
 * no cached layer reads; and the property, a scratch read of the step's ids or a head with rows past the token table
 * keeps steps serial, while a malformed property fails before any block binds.
 */
public class GpuGreedyChainHostTest {

    private static final int VOCAB = 32;
    private static final int FIRST = 5;
    private static final int STEPS = 12;

    private final GpuDecodeReplayTest rig = new GpuDecodeReplayTest();

    @After
    public void tearDown() throws Exception {
        System.clearProperty(GpuForwardBackend.GREEDY_CHAIN_PROPERTY);
        if (rig.backend != null) {
            rig.backend.failWaits(null);
        }
        rig.tearDown();
    }

    /** The id the emulated argmax selects after input {@code token}. */
    private static int select(int token) {
        return (token * 7 + 3) % VOCAB;
    }

    private GpuForwardBackend open() {
        rig.argmaxEmulator = d -> {
            ScratchSlot ids = rig.scratch().slot(ScratchSlots.TOKEN_IDS);
            int input = ((FakeGpuBuffer) ids.buffer()).getInt(ids.offset());
            d.bufferArgs().get(1).putInt(0, select(input));
        };
        return rig.open(true);
    }

    @Test
    public void chainedStepsSelectTheSerialIdsThroughADeviceCopy() {
        GpuForwardBackend fb = open();
        assertFalse("an empty memory has no step to roll back", fb.chainsGreedyOnDevice());
        int[] expected = new int[STEPS];
        int[] inputs = new int[STEPS];
        int token = FIRST;
        for (int i = 0; i < STEPS; i++) {
            inputs[i] = token;
            token = select(token);
            expected[i] = token;
        }

        int[] ids = new int[STEPS];
        GreedyStep step = fb.startGreedy(FIRST, 0);
        assertTrue("a stored step can be rolled back", fb.chainsGreedyOnDevice());
        for (int i = 0; i < STEPS; i++) {
            GreedyStep next = i + 1 < STEPS ? fb.chainGreedy(step, i + 1) : null;
            ids[i] = step.id();
            step = next;
        }

        assertArrayEquals("the chain selects the serial ids", expected, ids);
        assertEquals("every later step read its id on the device", STEPS - 1, fb.greedyDeviceChained());
        assertArrayEquals("the sequence stores every input id", inputs, fb.memory().sequence().tokens());
        ScratchSlot tokenIds = rig.scratch().slot(ScratchSlots.TOKEN_IDS);
        assertEquals("only the first step writes its id from the host", 1, writesTo(tokenIds));
        assertEquals("each chained step copies the previous selection into the token-id slot", STEPS - 1,
                copiesInto(tokenIds.buffer()));
    }

    @Test
    public void aCallBetweenChainedStepsSeesTheirTokensStored() {
        GpuForwardBackend fb = open();
        GreedyStep first = fb.startGreedy(FIRST, 0);
        GreedyStep second = fb.chainGreedy(first, 1);
        assertEquals(2, fb.memory().sequence().nPast());
        float[] logits = fb.decode(select(select(FIRST)), 2);
        assertEquals(VOCAB, logits.length);
        assertArrayEquals("the decode settled both chained steps first",
                new int[]{FIRST, select(FIRST), select(select(FIRST))}, fb.memory().sequence().tokens());
        assertEquals(select(FIRST), first.id());
        assertEquals(select(select(FIRST)), second.id());
        GreedyStep third = fb.chainGreedy(second, 3);
        assertEquals("a step after another call waits for its input instead of chaining", 1,
                fb.greedyDeviceChained());
        assertEquals(select(select(select(FIRST))), third.id());
    }

    @Test
    public void aDroppedChainedStepRollsBackLikeADecode() {
        GpuForwardBackend fb = open();
        GreedyStep first = fb.startGreedy(FIRST, 0);
        GreedyStep dropped = fb.chainGreedy(first, 1);
        assertEquals(select(FIRST), first.id());
        assertEquals(select(select(FIRST)), dropped.id());
        assertTrue(fb.canRollback(1));
        fb.rollback(1);
        assertArrayEquals(new int[]{FIRST}, fb.memory().sequence().tokens());
        assertEquals("the decode after the drop selects what the dropped step selected", dropped.id(),
                fb.decodeGreedyOnDevice(select(FIRST), 1));
    }

    @Test
    public void aStepChainedOnACompletedStepStoresItsInputIdAtOnce() {
        GpuForwardBackend fb = open();
        GreedyStep first = fb.startGreedy(FIRST, 0);
        assertEquals(select(FIRST), first.id());
        GreedyStep second = fb.chainGreedy(first, 1);
        assertEquals("the step still reads its id on the device", 1, fb.greedyDeviceChained());
        assertArrayEquals("the entry holds the completed step's id when the call returns",
                new int[]{FIRST, select(FIRST)}, fb.memory().sequence().tokens());
        assertEquals(select(select(FIRST)), second.id());
        fb.decode(select(select(FIRST)), 2);
        assertArrayEquals("a later call keeps every input id",
                new int[]{FIRST, select(FIRST), select(select(FIRST))}, fb.memory().sequence().tokens());
    }

    @Test
    public void aDeviceFailureReachesEveryWaitAsTheDevicesOwnException() {
        GpuForwardBackend fb = open();
        GreedyStep first = fb.startGreedy(FIRST, 0);
        GreedyStep second = fb.chainGreedy(first, 1);
        rig.backend.failWaits(() -> new GpuBackendException("fake", -999, "device lost"));
        GpuBackendException failure = assertThrows("the call that completes the chain reports the failed wait",
                GpuBackendException.class, () -> fb.decode(3, 2));
        assertEquals("the step chained on the failed one adds no copy of the failure", 0,
                failure.getSuppressed().length);
        assertSame("the failed step reports its failure", failure, assertThrows(GpuBackendException.class,
                first::id));
        assertSame("the step chained on it fails with the same exception", failure,
                assertThrows(GpuBackendException.class, second::id));
        rig.backend.failWaits(null);
    }

    @Test
    public void aCompletedStepKeepsNoEarlierStepReachable() throws InterruptedException {
        GpuForwardBackend fb = open();
        GreedyStep step = fb.startGreedy(FIRST, 0);
        WeakReference<GreedyStep> first = new WeakReference<>(step);
        int expected = FIRST;
        for (int i = 1; i <= STEPS; i++) {
            GreedyStep next = fb.chainGreedy(step, i);
            expected = select(expected);
            assertEquals(expected, step.id());
            step = next;
        }
        assertEquals(select(expected), step.id());
        assertTrue("the first of " + (STEPS + 1) + " chained steps is unreachable from the last", collected(first));
        assertEquals(STEPS, fb.greedyDeviceChained());
    }

    /** Whether {@code ref} clears within a few collections. */
    private static boolean collected(WeakReference<?> ref) throws InterruptedException {
        for (int i = 0; i < 50 && ref.get() != null; i++) {
            System.gc();
            Thread.sleep(10);
        }
        return ref.get() == null;
    }

    @Test
    public void aHeadWithRowsPastTheTokenTableKeepsStepsSerial() {
        rig.outputRows = VOCAB + 8;
        GpuForwardBackend fb = open();
        GreedyStep first = fb.startGreedy(FIRST, 0);
        assertFalse("a selected id may have no token row", fb.chainsGreedyOnDevice());
        GreedyStep second = fb.chainGreedy(first, 1);
        assertEquals(select(select(FIRST)), second.id());
        assertEquals(0, fb.greedyDeviceChained());
    }

    @Test
    public void aMalformedPropertyFailsBeforeAnyBlockBinds() {
        System.setProperty(GpuForwardBackend.GREEDY_CHAIN_PROPERTY, "sometimes");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> rig.open(true));
        assertTrue(e.getMessage(), e.getMessage().contains(GpuForwardBackend.GREEDY_CHAIN_PROPERTY));
        assertEquals("no block, head or driver prepared a dispatch to leave behind", rig.preparedBeforeBackend,
                rig.backend.preparedKernels());
    }

    @Test
    public void decodeStepsWriteNoPositionSlotThatNoLayerReads() {
        rig.mixerReadsPositionSlot = false;
        GpuForwardBackend fb = open();
        assertFalse(fb.decodeWritesPositions());
        assertEquals("every decode step skips the position slot", 1, positionWrites(fb));
    }

    @Test
    public void decodeStepsWriteThePositionSlotALayerReads() {
        GpuForwardBackend fb = open();
        assertTrue(fb.decodeWritesPositions());
        assertEquals("every step writes the position slot", 5, positionWrites(fb));
    }

    /** The position-slot writes of a prefill and four decode steps of every kind. */
    private long positionWrites(GpuForwardBackend fb) {
        ScratchSlot pos = rig.scratch().slot(ScratchSlots.POS);
        fb.prefill(new int[]{1, 2, 3}, 0);
        assertEquals("a prefill step writes the position slot", 1, writesTo(pos));
        fb.decode(4, 3);
        fb.decodeGreedyOnDevice(5, 4);
        GreedyStep step = fb.startGreedy(6, 5);
        fb.chainGreedy(step, 6).id();
        return writesTo(pos);
    }

    @Test
    public void thePropertyKeepsStepsSerial() {
        System.setProperty(GpuForwardBackend.GREEDY_CHAIN_PROPERTY, "false");
        GpuForwardBackend fb = open();
        assertFalse(fb.chainsGreedyOnDevice());
        GreedyStep first = fb.startGreedy(FIRST, 0);
        GreedyStep second = fb.chainGreedy(first, 1);
        assertEquals(select(select(FIRST)), second.id());
        assertEquals(0, fb.greedyDeviceChained());
        assertEquals("every step writes its id from the host", 2,
                writesTo(rig.scratch().slot(ScratchSlots.TOKEN_IDS)));
    }

    @Test
    public void aMalformedPropertyIsRejected() {
        System.setProperty(GpuForwardBackend.GREEDY_CHAIN_PROPERTY, "sometimes");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                GpuForwardBackend::greedyChainConfigured);
        assertTrue(e.getMessage(), e.getMessage().contains(GpuForwardBackend.GREEDY_CHAIN_PROPERTY));
    }

    @Test
    public void theStepIdsOfAChainedStepAreNotOnTheHost() {
        GpuForwardBackend fb = open();
        GreedyStep first = fb.startGreedy(FIRST, 0);
        GreedyStep second = fb.chainGreedy(first, 1);
        second.id();
        GpuScratch scratch = rig.scratch();
        long reads = scratch.stepTokenReads();
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> scratch.readStepTokens(new int[1], 0, 1));
        assertTrue(e.getMessage(), e.getMessage().contains("only on the device"));
        assertEquals("the failed read counts, so a trunk that reads them is told apart", reads + 1,
                scratch.stepTokenReads());
        scratch.recordStepTokens(new int[]{7}, 1);
        int[] ids = new int[1];
        scratch.readStepTokens(ids, 0, 1);
        assertArrayEquals("a host step records its ids again", new int[]{7}, ids);
    }

    /** The writes into {@code slot}'s first element. */
    private long writesTo(ScratchSlot slot) {
        long offset = (long) slot.offset() * Float.BYTES;
        return rig.backend.ops().stream().filter(op -> op.kind() == FakeOpKind.WRITE
                && op.buffer() == slot.buffer() && op.offset() == offset).count();
    }

    /** The device copies into {@code target} at element 0. */
    private long copiesInto(GpuBuffer target) {
        return rig.backend.ops().stream().filter(op -> op.kind() == FakeOpKind.COPY
                && op.buffer() == target && op.offset() == 0 && op.source() != target).count();
    }
}
