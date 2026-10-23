package com.eonmux.neuralfree.meta;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class NASSearchSpace {
    private final int numNodes;
    private final List<Edge> edges;
    private final List<Operation> availableOperations;
    private final String name;

    public NASSearchSpace(int numNodes, List<Operation> availableOperations) {
        this(numNodes, availableOperations, "NASCell");
    }

    public NASSearchSpace(int numNodes, List<Operation> availableOperations, String name) {
        if (numNodes < 2) {
            throw new IllegalArgumentException("Number of nodes must be at least 2");
        }
        Objects.requireNonNull(availableOperations, "Available operations cannot be null");
        if (availableOperations.isEmpty()) {
            throw new IllegalArgumentException("Available operations cannot be empty");
        }

        this.numNodes = numNodes;
        this.availableOperations = new ArrayList<>(availableOperations);
        this.name = name;
        this.edges = new ArrayList<>();
        
        initializeEdges();
    }

    private void initializeEdges() {
        edges.clear();
        
        for (int i = 0; i < numNodes; i++) {
            for (int j = 0; j < i; j++) {
                edges.add(new Edge(j, i, availableOperations));
            }
        }
    }

    public NASSearchSpace copy() {
        NASSearchSpace copy = new NASSearchSpace(numNodes, availableOperations, name);
        copy.setArchitecture(getArchitecture());
        return copy;
    }

    public void setArchitecture(List<Integer> operationIndices) {
        Objects.requireNonNull(operationIndices, "Operation indices cannot be null");
        
        if (operationIndices.size() != edges.size()) {
            throw new IllegalArgumentException("Expected " + edges.size() + 
                " operation indices, got " + operationIndices.size());
        }

        for (int i = 0; i < edges.size(); i++) {
            int opIdx = operationIndices.get(i);
            edges.get(i).setSelectedOperation(opIdx);
        }
    }

    public List<Integer> getArchitecture() {
        List<Integer> arch = new ArrayList<>();
        for (Edge edge : edges) {
            arch.add(edge.selectedOperationIndex());
        }
        return Collections.unmodifiableList(arch);
    }

    public void setRandomArchitecture(long seed) {
        java.util.Random random = new java.util.Random(seed);
        for (Edge edge : edges) {
            int randomIdx = random.nextInt(availableOperations.size());
            edge.setSelectedOperation(randomIdx);
        }
    }

    public List<Edge> edges() {
        return Collections.unmodifiableList(edges);
    }

    public List<Edge> edgesToNode(int targetNode) {
        List<Edge> result = new ArrayList<>();
        for (Edge edge : edges) {
            if (edge.targetNode() == targetNode) {
                result.add(edge);
            }
        }
        return result;
    }

    public List<Edge> edgesFromNode(int sourceNode) {
        List<Edge> result = new ArrayList<>();
        for (Edge edge : edges) {
            if (edge.sourceNode() == sourceNode) {
                result.add(edge);
            }
        }
        return result;
    }

    public int numNodes() {
        return numNodes;
    }

    public int numEdges() {
        return edges.size();
    }

    public List<Operation> availableOperations() {
        return Collections.unmodifiableList(availableOperations);
    }

    public String name() {
        return name;
    }

    public int totalSearchSpaceSize() {
        return (int) Math.pow(availableOperations.size(), edges.size());
    }

    public ArchitectureSpec getArchitectureSpec() {
        List<OperationChoice> choices = new ArrayList<>();
        for (Edge edge : edges) {
            choices.add(new OperationChoice(
                edge.sourceNode(),
                edge.targetNode(),
                edge.selectedOperation(),
                edge.selectedOperationIndex()
            ));
        }
        return new ArchitectureSpec(name, numNodes, choices);
    }

    @Override
    public String toString() {
        return "NASSearchSpace{" +
            "name='" + name + '\'' +
            ", numNodes=" + numNodes +
            ", numEdges=" + edges.size() +
            ", availableOps=" + availableOperations.size() +
            '}';
    }

    public static final class Edge {
        private final int sourceNode;
        private final int targetNode;
        private final List<Operation> operations;
        private int selectedOperationIndex;

        Edge(int sourceNode, int targetNode, List<Operation> operations) {
            this.sourceNode = sourceNode;
            this.targetNode = targetNode;
            this.operations = new ArrayList<>(operations);
            this.selectedOperationIndex = 0;
        }

        public void setSelectedOperation(int index) {
            if (index < 0 || index >= operations.size()) {
                throw new IndexOutOfBoundsException("Operation index out of bounds: " + index);
            }
            this.selectedOperationIndex = index;
        }

        public int sourceNode() {
            return sourceNode;
        }

        public int targetNode() {
            return targetNode;
        }

        public Operation selectedOperation() {
            return operations.get(selectedOperationIndex);
        }

        public int selectedOperationIndex() {
            return selectedOperationIndex;
        }

        public List<Operation> availableOperations() {
            return Collections.unmodifiableList(operations);
        }

        public int numOperations() {
            return operations.size();
        }

        @Override
        public String toString() {
            return "Edge{" + sourceNode + "->" + targetNode + 
                ", op=" + selectedOperation().name() + "}";
        }
    }

    public static final class Operation {
        private final String name;
        private final OperationType type;
        private final Map<String, Object> params;

        public Operation(String name, OperationType type) {
            this(name, type, new HashMap<>());
        }

        public Operation(String name, OperationType type, Map<String, Object> params) {
            Objects.requireNonNull(name, "Operation name cannot be null");
            Objects.requireNonNull(type, "Operation type cannot be null");

            this.name = name;
            this.type = type;
            this.params = new HashMap<>(params);
        }

        public String name() {
            return name;
        }

        public OperationType type() {
            return type;
        }

        public Map<String, Object> params() {
            return Collections.unmodifiableMap(params);
        }

        @SuppressWarnings("unchecked")
        public <T> T param(String key, T defaultValue) {
            Object value = params.get(key);
            if (value == null) {
                return defaultValue;
            }
            return (T) value;
        }

        @Override
        public String toString() {
            return name;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Operation operation = (Operation) o;
            return name.equals(operation.name);
        }

        @Override
        public int hashCode() {
            return name.hashCode();
        }
    }

    public enum OperationType {
        NONE,
        SKIP_CONNECT,
        CONV_3X3,
        CONV_5X5,
        DILATED_CONV_3X3,
        DILATED_CONV_5X5,
        AVG_POOL_3X3,
        AVG_POOL_5X5,
        MAX_POOL_3X3,
        MAX_POOL_5X5,
        SEPARABLE_CONV_3X3,
        SEPARABLE_CONV_5X5,
        IDENTITY,
        LINEAR,
        RELU,
        TANH,
        SIGMOID
    }

    public static final class OperationChoice {
        private final int sourceNode;
        private final int targetNode;
        private final Operation operation;
        private final int operationIndex;

        public OperationChoice(int sourceNode, int targetNode, 
                               Operation operation, int operationIndex) {
            this.sourceNode = sourceNode;
            this.targetNode = targetNode;
            this.operation = operation;
            this.operationIndex = operationIndex;
        }

        public int sourceNode() {
            return sourceNode;
        }

        public int targetNode() {
            return targetNode;
        }

        public Operation operation() {
            return operation;
        }

        public int operationIndex() {
            return operationIndex;
        }

        @Override
        public String toString() {
            return sourceNode + "->" + targetNode + ":" + operation.name();
        }
    }

    public static final class ArchitectureSpec {
        private final String name;
        private final int numNodes;
        private final List<OperationChoice> choices;

        public ArchitectureSpec(String name, int numNodes, List<OperationChoice> choices) {
            this.name = name;
            this.numNodes = numNodes;
            this.choices = new ArrayList<>(choices);
        }

        public String name() {
            return name;
        }

        public int numNodes() {
            return numNodes;
        }

        public List<OperationChoice> choices() {
            return Collections.unmodifiableList(choices);
        }

        public String toCompactString() {
            StringBuilder sb = new StringBuilder();
            sb.append(name).append("[");
            for (int i = 0; i < choices.size(); i++) {
                if (i > 0) sb.append(",");
                OperationChoice choice = choices.get(i);
                sb.append(choice.sourceNode())
                  .append("->")
                  .append(choice.targetNode())
                  .append(":")
                  .append(choice.operationIndex());
            }
            sb.append("]");
            return sb.toString();
        }

        @Override
        public String toString() {
            return "ArchitectureSpec{" + toCompactString() + "}";
        }
    }

    public static List<Operation> standardOperations() {
        return Arrays.asList(
            new Operation("none", OperationType.NONE),
            new Operation("skip_connect", OperationType.SKIP_CONNECT),
            new Operation("conv_3x3", OperationType.CONV_3X3),
            new Operation("conv_5x5", OperationType.CONV_5X5),
            new Operation("avg_pool_3x3", OperationType.AVG_POOL_3X3),
            new Operation("max_pool_3x3", OperationType.MAX_POOL_3X3),
            new Operation("sep_conv_3x3", OperationType.SEPARABLE_CONV_3X3),
            new Operation("sep_conv_5x5", OperationType.SEPARABLE_CONV_5X5)
        );
    }

    public static List<Operation> dartsOperations() {
        return Arrays.asList(
            new Operation("none", OperationType.NONE),
            new Operation("skip_connect", OperationType.SKIP_CONNECT),
            new Operation("max_pool_3x3", OperationType.MAX_POOL_3X3),
            new Operation("avg_pool_3x3", OperationType.AVG_POOL_3X3),
            new Operation("sep_conv_3x3", OperationType.SEPARABLE_CONV_3X3),
            new Operation("sep_conv_5x5", OperationType.SEPARABLE_CONV_5X5),
            new Operation("dil_conv_3x3", OperationType.DILATED_CONV_3X3),
            new Operation("dil_conv_5x5", OperationType.DILATED_CONV_5X5)
        );
    }
}
