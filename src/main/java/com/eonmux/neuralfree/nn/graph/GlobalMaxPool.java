package com.eonmux.neuralfree.nn.graph;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.GpuReduceOps.SegmentOp;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Global maximum pooling over the node features of each graph, through the graph's cached node grouping
 * ({@link GraphData#graphOffsets()}) on the node features' backend.
 */
public final class GlobalMaxPool extends Module {
    
    public GlobalMaxPool() {
    }
    
    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException(
            "GlobalMaxPool requires GraphData input. Use forward(graph, x) instead.");
    }
    
    public Tensor forward(GraphData graph) {
        Objects.requireNonNull(graph, "graph cannot be null");
        return forward(graph, graph.nodeFeatures());
    }
    
    /**
     * The elementwise maximum of each graph's node features: {@code [numGraphs, features]} for a batched graph,
     * {@code [1, features]} otherwise. A Java {@code Math.max} fold from {@code -inf} (NaN propagates); a result of
     * {@code -inf}, which includes every graph without nodes, becomes 0. Recorded: each graph's gradient goes to the
     * first of its nodes (ascending id) equal to the maximum, none for a NaN maximum or an all {@code -inf} graph.
     *
     * @throws IllegalArgumentException when {@code x} is not {@code [numNodes, features]} on the graph's backend
     */
    public Tensor forward(GraphData graph, Tensor x) {
        Objects.requireNonNull(graph, "graph cannot be null");
        Objects.requireNonNull(x, "x cannot be null");
        graph.requireNodeFeatures(x, "GlobalMaxPool");
        return TensorOps.segmentReduce(x, graph.graphOffsets(), graph.graphNodes(), graph.numGraphs(),
            SegmentOp.MAX, 0.0f);
    }
    
    @Override
    public String toString() {
        return "GlobalMaxPool{}";
    }
}
