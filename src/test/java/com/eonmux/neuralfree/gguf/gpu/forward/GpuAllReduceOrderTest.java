package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeOp;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The queue work of {@link GpuAllReduce#reduceToRoot} against {@link GpuAllReduce#sum} on two devices without peer
 * copies: the reduction finishes each peer once per chunk it reads, never finishes or reads rank 0's queue, and never
 * writes a peer; the all-reduce finishes every queue around the operation and broadcasts the sum.
 */
public class GpuAllReduceOrderTest {

    private static final int COUNT = 8;

    private FakeComputeBackend root;
    private FakeComputeBackend peer;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

    @Before
    public void open() {
        root = FakeComputeBackend.create();
        peer = FakeComputeBackend.create();
        scratch = GpuScratch.builder(root, 1).slot(ScratchSlots.X, COUNT).build();
        placeholders = GpuPlaceholders.allocate(root, root.computeQueue());
        ops = new GpuOps(root.computeQueue(), LlmKernels.open(root), scratch, placeholders);
    }

    @After
    public void close() {
        if (ops != null) ops.close();
        if (placeholders != null) placeholders.close();
        if (scratch != null) scratch.close();
        if (peer != null) peer.close();
        if (root != null) root.close();
    }

    @Test
    public void aReductionToTheRootWaitsOnlyForThePeer() {
        List<GpuAllReduce.Participant> ranks = participants();
        GpuBuffer temporary = root.allocateBuffer((long) COUNT * Float.BYTES);
        root.clearLog();
        peer.clearLog();

        GpuAllReduce.reduceToRoot(ops, temporary, ranks, COUNT);

        List<FakeOpKind> rootOps = kinds(root);
        assertFalse("rank 0's queue keeps running: " + rootOps, rootOps.contains(FakeOpKind.FINISH));
        assertFalse("nothing is read back from rank 0: " + rootOps, rootOps.contains(FakeOpKind.READ));
        assertEquals(FakeOpKind.WRITE, rootOps.get(0));
        assertTrue("the add follows the staged partial: " + rootOps, rootOps.contains(FakeOpKind.DISPATCH));
        assertEquals(Arrays.asList(FakeOpKind.READ, FakeOpKind.FINISH), kinds(peer));
    }

    @Test
    public void anAllReduceFinishesEveryQueueAndBroadcasts() {
        List<GpuAllReduce.Participant> ranks = participants();
        GpuBuffer temporary = root.allocateBuffer((long) COUNT * Float.BYTES);
        root.clearLog();
        peer.clearLog();

        GpuAllReduce.sum(ops, temporary, ranks, COUNT);

        List<FakeOpKind> rootOps = kinds(root);
        assertEquals(FakeOpKind.FINISH, rootOps.get(0));
        assertEquals(FakeOpKind.FINISH, rootOps.get(rootOps.size() - 1));
        assertTrue("the sum is read back for the broadcast: " + rootOps, rootOps.contains(FakeOpKind.READ));
        List<FakeOpKind> peerOps = kinds(peer);
        assertTrue("the peer receives the sum: " + peerOps, peerOps.contains(FakeOpKind.WRITE));
        assertEquals(FakeOpKind.FINISH, peerOps.get(peerOps.size() - 1));
    }

    private List<GpuAllReduce.Participant> participants() {
        ComputeQueue rq = root.computeQueue();
        GpuBuffer mine = root.allocateBuffer((long) COUNT * Float.BYTES);
        GpuBuffer theirs = peer.allocateBuffer((long) COUNT * Float.BYTES);
        return Arrays.asList(new GpuAllReduce.Participant(rq, mine, 0),
                new GpuAllReduce.Participant(peer.computeQueue(), theirs, 0));
    }

    private static List<FakeOpKind> kinds(FakeComputeBackend backend) {
        List<FakeOpKind> out = new ArrayList<>();
        for (FakeOp op : backend.ops()) {
            if (op.kind() != FakeOpKind.FLUSH) {
                out.add(op.kind());
            }
        }
        return out;
    }
}
