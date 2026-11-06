package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.engine.DraftTree;
import com.eonmux.neuralfree.gguf.engine.TreeAccept;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * The single-pass tree verify's acceptance walk readout ({@link GpuLogitsHead#TREE_ACCEPT},
 * {@code kernels/llm/sampling.cl}): after the head enqueued the verify rows' logits and their greedy ids, the
 * kernel follows the accepted path through the proposal's parent array over the rows' ids and only the accepted
 * depth, the path and the two emissions the round needs besides the path's node tokens are read back
 * ({@code 3 + nNodes} ints, against {@code (width + 1) * vocab} floats for the host walk).
 *
 * <p>{@link #stage} enqueues the tree upload, the kernel and the result read into this call's queue; the host
 * values are assembled by {@link #result()} after the queue finished, and the read's event is closed by
 * {@link #closeEvents()} (the discipline of {@link GpuCallReads}). One instance serves one batched call. The tree
 * and output buffers are scratch temporaries reused across calls. Not thread-safe.</p>
 */
final class GpuTreeAcceptReadout {

    /** The proposal's node count is the frontier bit set's width in the kernel (two 32-bit words). */
    static final int MAX_NODES = 63;

    private static final String TREE_TEMP = "tree.accept.tree";
    private static final String OUT_TEMP = "tree.accept.out";

    private final GpuOps ops;
    private final ComputeQueue queue;
    private final GpuScratch scratch;
    /** {@code [token, parent, depth] x nNodes} in kernel order, staged to the device per call. */
    private final int[] tree;
    private final ByteBuffer treeBytes;
    private final ByteBuffer outBytes;
    /** The read's destination, filled once the queue drained; {@link #result} copies it into {@code outBytes}. */
    private final byte[] outData;
    private final int nNodes;
    private final int local;
    private GpuEvent readEvent;

    private GpuTreeAcceptReadout(GpuOps ops, ComputeQueue queue, GpuScratch scratch, DraftTree proposal) {
        this.ops = ops;
        this.queue = queue;
        this.scratch = scratch;
        this.nNodes = proposal.size();
        this.tree = new int[3 * nNodes];
        for (int i = 0; i < nNodes; i++) {
            DraftTree.Node n = proposal.nodes().get(i);
            tree[3 * i] = n.token();
            tree[3 * i + 1] = n.parent();
            tree[3 * i + 2] = n.depth();
        }
        this.treeBytes = ByteBuffer.allocate(tree.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        this.outData = new byte[(3 + nNodes) * Integer.BYTES];
        this.outBytes = ByteBuffer.allocate(outData.length).order(ByteOrder.LITTLE_ENDIAN);
        this.local = Math.min(ops.local(), LlmKernelConstants.LLM_DIMX);
    }

    /**
     * Binds the readout of one tree verify round over {@code proposal}.
     *
     * @throws UnsupportedOperationException when the backend provides no {@value GpuLogitsHead#TREE_ACCEPT}
     *                                       kernel at the launch's work-group size
     */
    static GpuTreeAcceptReadout bind(GpuOps ops, ComputeQueue queue, GpuScratch scratch, DraftTree proposal) {
        Objects.requireNonNull(proposal, "proposal");
        if (proposal.size() < 1) {
            throw new IllegalArgumentException("a tree accept readout needs at least one proposal node, got "
                    + proposal.size());
        }
        if (proposal.size() > MAX_NODES) {
            throw new UnsupportedOperationException("the device accept walk holds the frontier as a " + MAX_NODES
                    + "-bit set, the proposal has " + proposal.size() + " nodes: raise neuralfree.draft.tree.nodes"
                    + " no further or run the host walk");
        }
        if (!ops.prewarm(GpuLogitsHead.TREE_ACCEPT, new long[]{Math.min(ops.local(), LlmKernelConstants.LLM_DIMX)})) {
            throw new UnsupportedOperationException("the backend provides no " + GpuLogitsHead.TREE_ACCEPT
                    + " kernel at the launch's work-group size");
        }
        return new GpuTreeAcceptReadout(ops, queue, scratch, proposal);
    }

    /**
     * After the head enqueued {@code nRows} rows' logits and greedy ids, enqueues the tree upload, the walk kernel
     * over them and the result read. The values are assembled by {@link #result()} once the queue finished.
     *
     * @param ids the head's argmax temporary holding {@code nRows} ids from element 0
     */
    void stage(ScratchSlot ids) {
        ScratchSlot treeSlot = scratch.temp(TREE_TEMP, tree.length);
        if (treeSlot.offset() != 0) {
            throw new IllegalStateException("scratch temporary " + TREE_TEMP + " starts at element " + treeSlot
                    .offset() + ", the accept kernel takes its buffers at element 0");
        }
        treeBytes.clear();
        for (int t : tree) {
            treeBytes.putInt(t);
        }
        GpuEvent up = queue.enqueueWriteBuffer(treeSlot.buffer(), 0, (long) tree.length * Integer.BYTES,
                treeBytes.array());
        if (up != null) {
            up.close();
        }
        ScratchSlot outSlot = scratch.temp(OUT_TEMP, 3L + nNodes);
        if (outSlot.offset() != 0) {
            throw new IllegalStateException("scratch temporary " + OUT_TEMP + " starts at element " + outSlot
                    .offset() + ", the accept kernel takes its buffers at element 0");
        }
        ops.enqueue(GpuLogitsHead.TREE_ACCEPT, new long[]{local}, new long[]{local}, nNodes, treeSlot.buffer(),
                ids.buffer(), outSlot.buffer());
        readEvent = queue.enqueueReadBuffer(outSlot.buffer(), 0, outData.length, outData);
    }

    /**
     * The device's acceptance walk, once the queue finished and {@link #stage} ran.
     *
     * @throws IllegalStateException when called before {@link #stage}
     */
    TreeAccept result() {
        if (readEvent == null) {
            throw new IllegalStateException("the accept readout assembled before its read was staged");
        }
        outBytes.clear();
        outBytes.put(outData);
        outBytes.flip();
        int acceptedDepth = outBytes.getInt();
        int firstId = outBytes.getInt();
        int finalDrawId = outBytes.getInt();
        int[] path = new int[acceptedDepth];
        for (int i = 0; i < acceptedDepth; i++) {
            path[i] = outBytes.getInt();
        }
        return new TreeAccept(acceptedDepth, path, firstId, finalDrawId);
    }

    /** Closes the result read's event, after the queue finished. Idempotent. */
    void closeEvents() {
        if (readEvent != null) {
            readEvent.close();
            readEvent = null;
        }
    }
}
