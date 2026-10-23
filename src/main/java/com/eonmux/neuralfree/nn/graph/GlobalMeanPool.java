package com.eonmux.neuralfree.nn.graph;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.GpuReduceOps.SegmentOp;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Global mean pooling over the node features of each graph, through the graph's cached node grouping
 * ({@link GraphData#graphOffsets()}) on the node features' backend.
 */
public final class GlobalMeanPool extends Module {
    
    public GlobalMeanPool() {
    }
    
    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException(
            "GlobalMeanPool requires GraphData input. Use forward(graph, x) instead.");
    }
    
    public Tensor forward(GraphData graph) {
        Objects.requireNonNull(graph, "graph cannot be null");
        return forward(graph, graph.nodeFeatures());
    }
    
    /**
     * The mean of each graph's node features: {@code [numGraphs, features]} for a batched graph, {@code [1, features]}
     * otherwise. Each graph sums its nodes in ascending node order and divides once by its node count; a graph of a
     * batch without nodes gives 0, an unbatched graph without nodes gives NaN ({@code 0 / 0}). Recorded: each node
     * receives its graph's gradient divided by the graph's node count.
     *
     * @throws IllegalArgumentException when {@code x} is not {@code [numNodes, features]} on the graph's backend
     */
    public Tensor forward(GraphData graph, Tensor x) {
        Objects.requireNonNull(graph, "graph cannot be null");
        Objects.requireNonNull(x, "x cannot be null");
        graph.requireNodeFeatures(x, "GlobalMeanPool");
        return TensorOps.segmentReduce(x, graph.graphOffsets(), graph.graphNodes(), graph.numGraphs(),
            SegmentOp.MEAN, graph.isBatched() ? 0.0f : Float.NaN);
    }
    
    @Override
    public String toString() {
        return "GlobalMeanPool{}";
    }
}
