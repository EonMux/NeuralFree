package com.eonmux.neuralfree.nn.graph;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuReduceOps.SegmentOp;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Base of the message-passing graph layers. {@link #propagate} runs, as {@link TensorOps} compositions on the node
 * features' backend:
 * <ol>
 *   <li>{@link #transform}: an optional per-node transform (identity by default);</li>
 *   <li>a clamped row gather of the transformed features at each edge's source (and target, when
 *       {@link #needsTargetFeatures()}): an index outside {@code [0, rows)} reads the nearest valid row;</li>
 *   <li>{@link #message}, one row per edge;</li>
 *   <li>{@link #aggregate}, one row per node, through the graph's cached CSR ({@link GraphData#csrOffsets()}):
 *       edges whose target is outside {@code [0, numNodes)} contribute nothing, and the messages of a node are
 *       combined in ascending edge order;</li>
 *   <li>{@link #update} of the aggregate and the input features.</li>
 * </ol>
 * A graph without edges skips the gather and message steps: every node aggregates to zeros of the transformed
 * features' width, which is then the width {@link #message} must produce.
 *
 * <p><b>Autograd.</b> Every step is a recorded {@link TensorOps} operation or {@link Tensor} view, so a forward in
 * grad mode builds the graph from the output back to the parameters and to {@code x} when it requires grad: the
 * gathers record {@code IndexSelectRowsGrad} (a clamped index routes its row's gradient to the row it read) and the
 * aggregations {@code SegmentReduceGrad}, whose backward needs each message row at most once in the CSR; the
 * {@link GraphData} CSR lists every edge id at most once. The graph structure is a constant. The
 * zero aggregate of a graph without edges is a constant too: the transform then runs under
 * {@link ComputationGraph#noGrad()}, because only its width is read, and the parameters it reads get no gradient
 * through the aggregate. A recorded output is a node output: it is valid until the loss computed from it is
 * back-propagated, which releases it.</p>
 */
public abstract class MessagePassing extends Module {

    protected MessagePassing() {
    }

    /**
     * Per-node transform applied before the gather. Returning {@code x} itself means no transform; a new tensor is
     * closed by {@link #propagate} once the messages are aggregated.
     */
    protected Tensor transform(Tensor x) {
        return x;
    }

    /**
     * Checks that the layer supports the edge index of a graph with at least one edge; {@link #propagate} calls it
     * before any computation, so a rejected graph records nothing. The default accepts every graph: out-of-range
     * sources are clamped and out-of-range targets ignored.
     *
     * @throws IndexOutOfBoundsException when the layer cannot handle an index of the graph's edge index
     */
    protected void requireSupportedEdges(GraphData graph) {
    }

    /** Whether {@link #message} reads the target features; when false it receives null for them. */
    protected boolean needsTargetFeatures() {
        return false;
    }

    /**
     * The messages {@code [numEdges, width]}, one row per edge. The result may be {@code srcFeatures} itself; any
     * other tensor is closed by {@link #propagate} after aggregation.
     *
     * @param srcFeatures transformed features of each edge's (clamped) source node
     * @param dstFeatures transformed features of each edge's (clamped) target node, or null when
     *                    {@link #needsTargetFeatures()} is false
     */
    protected abstract Tensor message(Tensor srcFeatures, Tensor dstFeatures, GraphData graph);

    /** The per-node aggregate {@code [numNodes, width]} of the messages; must be a new tensor. */
    protected abstract Tensor aggregate(Tensor messages, GraphData graph);

    /**
     * The layer output from the aggregate and the input node features. The result may be {@code aggregated} itself;
     * otherwise {@link #propagate} closes the aggregate.
     */
    protected abstract Tensor update(Tensor aggregated, Tensor nodeFeatures);

    /**
     * Runs the layer on {@code x}, the node features (one row per node, on the graph's backend).
     *
     * @throws IllegalArgumentException  when {@code x} is not {@code [numNodes, features]} on the graph's backend
     * @throws IndexOutOfBoundsException as {@link #requireSupportedEdges}
     */
    public Tensor propagate(GraphData graph, Tensor x) {
        Objects.requireNonNull(graph, "graph cannot be null");
        Objects.requireNonNull(x, "x cannot be null");
        graph.requireNodeFeatures(x, getClass().getSimpleName());

        Tensor aggregated;
        if (graph.numEdges() == 0) {
            aggregated = zeroAggregate(graph, x);
        } else {
            requireSupportedEdges(graph);
            aggregated = transformedAggregate(graph, x);
        }
        try {
            Tensor output = update(aggregated, x);
            if (output != aggregated) {
                aggregated.close();
            }
            return output;
        } catch (RuntimeException e) {
            aggregated.close();
            throw e;
        }
    }

    /** The aggregate of the messages built from {@link #transform}{@code (x)}. */
    private Tensor transformedAggregate(GraphData graph, Tensor x) {
        Tensor transformed = transform(x);
        try {
            return messagesAggregated(graph, transformed);
        } finally {
            if (transformed != x) {
                transformed.close();
            }
        }
    }

    /**
     * Zeros {@code [numNodes, width]}, the aggregate of a graph without edges, with the width of
     * {@link #transform}{@code (x)}. The transform runs under {@link ComputationGraph#noGrad()}: only its width is
     * read, so it must leave no node behind.
     */
    private Tensor zeroAggregate(GraphData graph, Tensor x) {
        int width;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor transformed = transform(x);
            try {
                width = transformed.shape().dim(1);
            } finally {
                if (transformed != x) {
                    transformed.close();
                }
            }
        }
        return Tensor.zeros(x.backend(), Shape.matrix((int) graph.numNodes(), width));
    }

    private Tensor messagesAggregated(GraphData graph, Tensor features) {
        try (Tensor src = gatherNodes(features, graph.sourceIndex());
             Tensor dst = needsTargetFeatures() ? gatherNodes(features, graph.targetIndex()) : null) {
            Tensor messages = message(src, dst, graph);
            try {
                return aggregate(messages, graph);
            } finally {
                if (messages != src) {
                    messages.close();
                }
            }
        }
    }

    /**
     * Rows of {@code nodeFeatures} at {@code nodeIndices} (INT32 or INT64), each index clamped to
     * {@code [0, rows - 1]}.
     */
    protected Tensor gatherNodes(Tensor nodeFeatures, Tensor nodeIndices) {
        return TensorOps.indexSelectRows(nodeFeatures, nodeIndices, true);
    }

    /** Sum of each node's incoming messages in ascending edge order; 0 for a node without one. */
    protected Tensor scatterAdd(Tensor messages, GraphData graph) {
        return segment(messages, graph, SegmentOp.SUM);
    }

    /** {@link #scatterAdd}, then one division by the node's incoming edge count; 0 for a node without one. */
    protected Tensor scatterMean(Tensor messages, GraphData graph) {
        return segment(messages, graph, SegmentOp.MEAN);
    }

    /**
     * Java {@code Math.max} fold of each node's incoming messages from {@code -inf} (NaN propagates); a result of
     * {@code -inf}, which includes every node without an incoming edge, becomes 0.
     */
    protected Tensor scatterMax(Tensor messages, GraphData graph) {
        return segment(messages, graph, SegmentOp.MAX);
    }

    /** FLOAT32 {@code [numNodes]}: the number of edges into each node. The caller closes the result. */
    protected Tensor computeDegree(GraphData graph) {
        return graph.inDegree().to(DType.FLOAT32);
    }

    private static Tensor segment(Tensor messages, GraphData graph, SegmentOp op) {
        Objects.requireNonNull(messages, "messages cannot be null");
        if (messages.shape().rank() < 1 || messages.shape().dim(0) != graph.numEdges()) {
            throw new IllegalArgumentException("messages need one row per edge (" + graph.numEdges()
                + "), got shape " + messages.shape());
        }
        return TensorOps.segmentReduce(messages, graph.csrOffsets(), graph.csrEdges(), (int) graph.numNodes(), op,
            0.0f);
    }

    // ------------------------------------------------------------------ parameter construction for the graph modules

    /**
     * A parameter of {@code std * N(0, 1)} values drawn from {@code Random(seed)} on the host and
     * scaled on {@code backend}. Initialization runs under {@link ComputationGraph#noGrad()} and records nothing.
     */
    static Parameter normalParameter(ComputeBackend backend, Shape shape, float std, long seed, String name) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor noise = Tensor.randn(backend, shape, seed)) {
            return ownedParameter(TensorOps.mul(noise, std), name);
        }
    }

    /** A parameter of zeros. */
    static Parameter zeroParameter(ComputeBackend backend, Shape shape, String name) {
        return ownedParameter(Tensor.zeros(backend, shape), name);
    }

    /** A parameter that takes over {@code values}: the caller must not use or close {@code values} afterwards. */
    static Parameter ownedParameter(Tensor values, String name) {
        return new Parameter(values, name);
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException(
            "MessagePassing layers require GraphData input. Use propagate(graph, x) instead.");
    }

    public Tensor forward(GraphData graph) {
        Objects.requireNonNull(graph, "graph cannot be null");
        return propagate(graph, graph.nodeFeatures());
    }

    public Tensor forward(GraphData graph, Tensor x) {
        return propagate(graph, x);
    }
}
