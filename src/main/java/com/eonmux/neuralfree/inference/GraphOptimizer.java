package com.eonmux.neuralfree.inference;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.norm.BatchNorm1d;
import com.eonmux.neuralfree.nn.norm.BatchNorm2d;
import com.eonmux.neuralfree.nn.activations.ReLU;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class GraphOptimizer {
    private final GraphOptimizerConfig config;
    private final List<OptimizationPass> passes;
    
    private long memorySavedBytes = 0;
    private int operationsFused = 0;
    private int constantsFolded = 0;
    private int deadNodesRemoved = 0;
    
    public GraphOptimizer(GraphOptimizerConfig config) {
        Objects.requireNonNull(config, "Config cannot be null");
        this.config = config;
        this.passes = new ArrayList<>();
        
        if (config.enableFusion()) {
            passes.add(new FusionPass());
        }
        if (config.enableConstantFolding()) {
            passes.add(new ConstantFoldingPass());
        }
        if (config.enableDeadCodeElimination()) {
            passes.add(new DeadCodeEliminationPass());
        }
        if (config.enableMemoryPlanning()) {
            passes.add(new MemoryPlanningPass());
        }
    }
    
    public GraphOptimizer() {
        this(GraphOptimizerConfig.defaults());
    }
    
    public OptimizedGraph optimize(Module model) {
        ensureOpen();
        Objects.requireNonNull(model, "Model cannot be null");
        
        ComputationGraphRepresentation graph = buildGraph(model);
        
        for (OptimizationPass pass : passes) {
            graph = pass.apply(graph);
        }
        
        return new OptimizedGraph(graph, memorySavedBytes, operationsFused, constantsFolded, deadNodesRemoved);
    }
    
    private ComputationGraphRepresentation buildGraph(Module model) {
        ComputationGraphRepresentation graph = new ComputationGraphRepresentation();
        
        if (model instanceof Sequential) {
            buildSequentialGraph((Sequential) model, graph);
        } else {
            buildModuleGraph(model, graph);
        }
        
        return graph;
    }
    
    private void buildSequentialGraph(Sequential sequential, ComputationGraphRepresentation graph) {
        List<Module> layers = sequential.layers();
        GraphNode previousNode = null;
        
        for (int i = 0; i < layers.size(); i++) {
            Module layer = layers.get(i);
            GraphNode node = new GraphNode(
                "layer_" + i,
                layer.getClass().getSimpleName(),
                layer,
                previousNode != null ? List.of(previousNode) : List.of()
            );
            graph.addNode(node);
            previousNode = node;
        }
    }
    
    private void buildModuleGraph(Module module, ComputationGraphRepresentation graph) {
        GraphNode node = new GraphNode(
            "module_" + module.moduleName(),
            module.moduleName(),
            module,
            List.of()
        );
        graph.addNode(node);
    }
    
    public long memorySavedBytes() {
        return memorySavedBytes;
    }
    
    public int operationsFused() {
        return operationsFused;
    }
    
    public int constantsFolded() {
        return constantsFolded;
    }
    
    public int deadNodesRemoved() {
        return deadNodesRemoved;
    }
    
    public GraphOptimizerConfig config() {
        return config;
    }
    
    private void ensureOpen() {
    }
    
    private interface OptimizationPass {
        ComputationGraphRepresentation apply(ComputationGraphRepresentation graph);
    }
    
    private class FusionPass implements OptimizationPass {
        @Override
        public ComputationGraphRepresentation apply(ComputationGraphRepresentation graph) {
            ComputationGraphRepresentation result = new ComputationGraphRepresentation();
            List<GraphNode> nodes = new ArrayList<>(graph.nodes());
            
            int i = 0;
            while (i < nodes.size()) {
                GraphNode current = nodes.get(i);
                
                if (i + 1 < nodes.size()) {
                    GraphNode next = nodes.get(i + 1);
                    
                    if (canFuse(current, next)) {
                        GraphNode fused = fuseNodes(current, next);
                        result.addNode(fused);
                        operationsFused++;
                        memorySavedBytes += estimateMemorySavings(current, next);
                        i += 2;
                        continue;
                    }
                }
                
                result.addNode(current);
                i++;
            }
            
            return result;
        }
        
        private boolean canFuse(GraphNode a, GraphNode b) {
            String aType = a.operationType();
            String bType = b.operationType();
            
            if (aType.equals("Conv2d") && bType.equals("BatchNorm2d")) {
                return true;
            }
            if (aType.equals("Conv2d") && bType.equals("ReLU")) {
                return true;
            }
            if (aType.equals("Linear") && bType.equals("ReLU")) {
                return true;
            }
            if (aType.equals("BatchNorm1d") && bType.equals("ReLU")) {
                return true;
            }
            if (aType.equals("BatchNorm2d") && bType.equals("ReLU")) {
                return true;
            }
            
            return false;
        }
        
        private GraphNode fuseNodes(GraphNode a, GraphNode b) {
            String fusedName = a.name() + "_" + b.name();
            String fusedType = "Fused" + a.operationType() + b.operationType();
            
            return new GraphNode(fusedName, fusedType, null, a.inputs());
        }
        
        private long estimateMemorySavings(GraphNode a, GraphNode b) {
            return 1024L;
        }
    }
    
    private class ConstantFoldingPass implements OptimizationPass {
        @Override
        public ComputationGraphRepresentation apply(ComputationGraphRepresentation graph) {
            ComputationGraphRepresentation result = new ComputationGraphRepresentation();
            
            for (GraphNode node : graph.nodes()) {
                if (isConstant(node)) {
                    constantsFolded++;
                    memorySavedBytes += 512L;
                } else {
                    result.addNode(node);
                }
            }
            
            return result;
        }
        
        private boolean isConstant(GraphNode node) {
            return false;
        }
    }
    
    private class DeadCodeEliminationPass implements OptimizationPass {
        @Override
        public ComputationGraphRepresentation apply(ComputationGraphRepresentation graph) {
            ComputationGraphRepresentation result = new ComputationGraphRepresentation();
            
            Set<String> reachable = new HashSet<>();
            findReachableNodes(graph, reachable);
            
            for (GraphNode node : graph.nodes()) {
                if (reachable.contains(node.name())) {
                    result.addNode(node);
                } else {
                    deadNodesRemoved++;
                }
            }
            
            return result;
        }
        
        private void findReachableNodes(ComputationGraphRepresentation graph, Set<String> reachable) {
            for (GraphNode node : graph.nodes()) {
                reachable.add(node.name());
            }
        }
    }
    
    private class MemoryPlanningPass implements OptimizationPass {
        @Override
        public ComputationGraphRepresentation apply(ComputationGraphRepresentation graph) {
            analyzeLifetime(graph);
            return graph;
        }
        
        private void analyzeLifetime(ComputationGraphRepresentation graph) {
            Map<String, int[]> lifetimes = new HashMap<>();
            int idx = 0;
            for (GraphNode node : graph.nodes()) {
                lifetimes.put(node.name(), new int[]{idx, idx});
                idx++;
            }
        }
    }
    
    private static class ComputationGraphRepresentation {
        private final List<GraphNode> nodes = new ArrayList<>();
        
        void addNode(GraphNode node) {
            nodes.add(node);
        }
        
        List<GraphNode> nodes() {
            return new ArrayList<>(nodes);
        }
    }
    
    public static final class GraphNode {
        private final String name;
        private final String operationType;
        private final Module module;
        private final List<GraphNode> inputs;
        
        GraphNode(String name, String operationType, Module module, List<GraphNode> inputs) {
            this.name = name;
            this.operationType = operationType;
            this.module = module;
            this.inputs = inputs != null ? new ArrayList<>(inputs) : new ArrayList<>();
        }
        
        public String name() { return name; }
        public String operationType() { return operationType; }
        public Module module() { return module; }
        public List<GraphNode> inputs() { return new ArrayList<>(inputs); }
    }
    
    public static final class OptimizedGraph {
        private final ComputationGraphRepresentation graph;
        private final long memorySavedBytes;
        private final int operationsFused;
        private final int constantsFolded;
        private final int deadNodesRemoved;
        
        OptimizedGraph(ComputationGraphRepresentation graph, long memorySavedBytes,
                      int operationsFused, int constantsFolded, int deadNodesRemoved) {
            this.graph = graph;
            this.memorySavedBytes = memorySavedBytes;
            this.operationsFused = operationsFused;
            this.constantsFolded = constantsFolded;
            this.deadNodesRemoved = deadNodesRemoved;
        }
        
        public int nodeCount() { return graph.nodes().size(); }
        public long memorySavedBytes() { return memorySavedBytes; }
        public int operationsFused() { return operationsFused; }
        public int constantsFolded() { return constantsFolded; }
        public int deadNodesRemoved() { return deadNodesRemoved; }
        
        @Override
        public String toString() {
            return "OptimizedGraph{" +
                "nodes=" + nodeCount() +
                ", memorySavedMB=" + (memorySavedBytes / (1024 * 1024)) +
                ", operationsFused=" + operationsFused +
                ", constantsFolded=" + constantsFolded +
                ", deadNodesRemoved=" + deadNodesRemoved +
                '}';
        }
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static final class Builder {
        private GraphOptimizerConfig config = GraphOptimizerConfig.defaults();
        
        public Builder config(GraphOptimizerConfig config) {
            this.config = config;
            return this;
        }
        
        public Builder enableFusion(boolean enable) {
            this.config = new GraphOptimizerConfig(
                enable, config.enableConstantFolding(), 
                config.enableDeadCodeElimination(), config.enableMemoryPlanning()
            );
            return this;
        }
        
        public Builder enableConstantFolding(boolean enable) {
            this.config = new GraphOptimizerConfig(
                config.enableFusion(), enable,
                config.enableDeadCodeElimination(), config.enableMemoryPlanning()
            );
            return this;
        }
        
        public GraphOptimizer build() {
            return new GraphOptimizer(config);
        }
    }
}
