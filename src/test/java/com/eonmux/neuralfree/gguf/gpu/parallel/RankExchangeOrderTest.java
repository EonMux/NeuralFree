package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeOp;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link RankExchange#replayAll} over three devices without peer copies: the source is read and finished once for
 * every peer, and each peer receives one write that its later work follows in queue order, without a finish of its
 * own, so the peers' parts can be enqueued before rank 0's.
 */
public class RankExchangeOrderTest {

    private static final int FLOATS = 6;

    private FakeComputeBackend root;
    private FakeComputeBackend peerA;
    private FakeComputeBackend peerB;

    @Before
    public void open() {
        root = FakeComputeBackend.create();
        peerA = FakeComputeBackend.create();
        peerB = FakeComputeBackend.create();
    }

    @After
    public void close() {
        for (FakeComputeBackend b : Arrays.asList(root, peerA, peerB)) {
            if (b != null) {
                b.close();
            }
        }
    }

    @Test
    public void everyPeerGetsOneWriteFromOneReadOfTheSource() {
        float[] input = {1f, -2f, 3.5f, 4f, 5f, -6f};
        GpuBuffer src = root.allocateBuffer((FLOATS + 2) * Float.BYTES);
        GpuBuffer dstA = peerA.allocateBuffer(FLOATS * Float.BYTES);
        GpuBuffer dstB = peerB.allocateBuffer(FLOATS * Float.BYTES);
        ComputeQueue rq = root.computeQueue();
        try (GpuEvent w = rq.enqueueWriteBuffer(src, 2L * Float.BYTES, (long) FLOATS * Float.BYTES, input)) {
            rq.finish();
        }
        root.clearLog();

        RankExchange.replayAll(rq, src, 2L, Arrays.asList(peerA.computeQueue(), peerB.computeQueue()),
                Arrays.asList(dstA, dstB), FLOATS);

        assertEquals(Arrays.asList(FakeOpKind.READ, FakeOpKind.FINISH, FakeOpKind.FINISH), kinds(root));
        assertEquals(Arrays.asList(FakeOpKind.WRITE), kinds(peerA));
        assertEquals(Arrays.asList(FakeOpKind.WRITE), kinds(peerB));
        assertArrayEquals(input, read(peerA, dstA), 0f);
        assertArrayEquals(input, read(peerB, dstB), 0f);
    }

    @Test
    public void aDestinationOnTheSourceQueueIsADeviceCopy() {
        float[] input = {7f, 8f, 9f, 10f, 11f, 12f};
        GpuBuffer src = root.allocateBuffer(FLOATS * Float.BYTES);
        GpuBuffer local = root.allocateBuffer(FLOATS * Float.BYTES);
        ComputeQueue rq = root.computeQueue();
        try (GpuEvent w = rq.enqueueWriteBuffer(src, 0L, (long) FLOATS * Float.BYTES, input)) {
            rq.finish();
        }
        root.clearLog();

        RankExchange.replayAll(rq, src, 0L, List.of(rq), List.of(local), FLOATS);

        assertEquals(Arrays.asList(FakeOpKind.COPY, FakeOpKind.FINISH), kinds(root));
        assertArrayEquals(input, read(root, local), 0f);
    }

    @Test
    public void destinationQueuesAndBuffersMustPair() {
        GpuBuffer src = root.allocateBuffer(FLOATS * Float.BYTES);
        assertThrows(IllegalArgumentException.class, () -> RankExchange.replayAll(root.computeQueue(), src, 0L,
                List.of(peerA.computeQueue()), List.of(), FLOATS));
    }

    private static List<FakeOpKind> kinds(FakeComputeBackend backend) {
        List<FakeOpKind> out = new ArrayList<>();
        for (FakeOp op : backend.ops()) {
            out.add(op.kind());
        }
        return out;
    }

    private static float[] read(FakeComputeBackend backend, GpuBuffer buffer) {
        float[] out = new float[FLOATS];
        ComputeQueue q = backend.computeQueue();
        try (GpuEvent r = q.enqueueReadBuffer(buffer, 0L, (long) FLOATS * Float.BYTES, out)) {
            q.finish();
        }
        return out;
    }
}
