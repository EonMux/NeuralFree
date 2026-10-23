package com.eonmux.neuralfree.acceleration.pipeline;

import com.eonmux.neuralfree.acceleration.pipeline.ExecutionNode.Operation;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;

/**
 * The steps of one pass of a dense network over a batch of rows, as a dependency DAG of {@link ExecutionNode}s that
 * name the train kernels they run. {@link PipelineScheduler} executes the nodes in {@link #topologicalOrder()}.
 *
 * <p><b>Forward</b> (every graph), per layer {@code l}: upload {@code W_l} and {@code b_l}
 * ({@code forward_transfer_weights_L<l>}, {@code forward_transfer_biases_L<l>}), {@code z = x * W_l^T}
 * ({@code forward_linear_L<l>}, TR_Gemm_F32) and {@code z += b_l; a = act_l(z)} ({@code forward_compute_L<l>},
 * TR_DenseBiasAct_F32); the input rows are uploaded by {@code transfer_input}.</p>
 *
 * <p><b>Backward</b> (gradient and training graphs): the target upload, the loss gradient
 * {@code dY = (a - t) / rows} of {@code sum((a - t)^2) / (2 * rows)} ({@code backward_loss_grad}, TR_Axpby_F32), then
 * per layer from the last: {@code dZ = dY * act'} in place ({@code backward_act_L<l>}, TR_DenseActBackward_F32),
 * {@code dW = dZ^T * x} ({@code backward_compute_grad_L<l>}, TR_Gemm_F32), {@code db} = column sums of {@code dZ}
 * ({@code backward_bias_grad_L<l>}, TR_ReduceCols_F32) and, above layer 0, {@code dY = dZ * W_l}
 * ({@code backward_input_grad_L<l>}, TR_Gemm_F32). A training graph then applies {@code p -= lr * g} to every weight
 * and bias ({@code update_weights}, TR_SgdStep_F32).</p>
 *
 * <p><b>Readback</b>: {@code output_transfer} reads the output rows (inference and training graphs) and
 * {@code result_transfer} the weights, biases and gradients (gradient and training graphs). Both come last. The
 * uploads depend on no other node, so {@link #topologicalOrder()} places every upload before the first kernel and the
 * host waits for device work only at the readbacks that end a pass.</p>
 *
 * <p>Every read-after-write and write-after-read hazard between nodes is an edge: the layer-{@code l} input gradient
 * overwrites the gradient buffer the layer-{@code l + 1} weight and bias gradients read, so it depends on them, and
 * the update depends on every node that reads the weights. Any topological order is therefore a valid execution
 * order on one in-order queue. GEMM nodes name TR_Gemm_F32; {@code GpuMatmul} runs a matrix-vector shaped product on
 * TR_Gemv_F32 instead.</p>
 */
public final class ExecutionGraph {

    private static final String INPUT_TRANSFER = "transfer_input";
    private static final String TARGET_TRANSFER = "transfer_target";
    private static final String LOSS_GRADIENT = "backward_loss_grad";
    private static final String UPDATE = "update_weights";
    private static final String RESULT_TRANSFER = "result_transfer";
    private static final String OUTPUT_TRANSFER = "output_transfer";

    /** Which nodes a graph contains. */
    private enum Mode {
        INFERENCE, GRADIENTS, TRAINING
    }

    private final Map<String, ExecutionNode> nodes;
    private final List<String> topologicalOrder;
    private final int numLayers;
    private final int batchRows;

    private ExecutionGraph(Map<String, ExecutionNode> nodes, int numLayers, int batchRows) {
        this.nodes = Collections.unmodifiableMap(new LinkedHashMap<>(nodes));
        this.topologicalOrder = computeTopologicalOrder(nodes);
        this.numLayers = numLayers;
        this.batchRows = batchRows;
    }

    /** {@link #buildForTraining(NeuralNetworkData, int)} for {@code networkData.getBatchSize()} rows. */
    public static ExecutionGraph buildForTraining(NeuralNetworkData networkData) {
        return buildForTraining(networkData, Objects.requireNonNull(networkData, "networkData").getBatchSize());
    }

    /**
     * Forward pass, gradients, SGD update of every weight and bias, then readback of the output rows and of the
     * updated parameters and their gradients.
     *
     * @throws IllegalArgumentException when {@code batchRows < 1}
     */
    public static ExecutionGraph buildForTraining(NeuralNetworkData networkData, int batchRows) {
        return build(networkData, batchRows, Mode.TRAINING);
    }

    /**
     * Forward pass and gradients without an update, then readback of the parameters and their gradients.
     *
     * @throws IllegalArgumentException when {@code batchRows < 1}
     */
    public static ExecutionGraph buildForGradients(NeuralNetworkData networkData, int batchRows) {
        return build(networkData, batchRows, Mode.GRADIENTS);
    }

    /** {@link #buildForInference(NeuralNetworkData, int)} for {@code networkData.getBatchSize()} rows. */
    public static ExecutionGraph buildForInference(NeuralNetworkData networkData) {
        return buildForInference(networkData, Objects.requireNonNull(networkData, "networkData").getBatchSize());
    }

    /**
     * Forward pass, then readback of the output rows.
     *
     * @throws IllegalArgumentException when {@code batchRows < 1}
     */
    public static ExecutionGraph buildForInference(NeuralNetworkData networkData, int batchRows) {
        return build(networkData, batchRows, Mode.INFERENCE);
    }

    private static ExecutionGraph build(NeuralNetworkData networkData, int batchRows, Mode mode) {
        Objects.requireNonNull(networkData, "networkData must not be null");
        if (batchRows < 1) {
            throw new IllegalArgumentException("a pass needs at least one row, got " + batchRows);
        }
        int[] sizes = networkData.getLayerSizes();
        int layers = networkData.getNumLayers();
        Map<String, ExecutionNode> nodes = new LinkedHashMap<>();
        String lastForward = addForward(nodes, sizes, layers, batchRows);
        if (mode == Mode.INFERENCE) {
            put(nodes, transfer(OUTPUT_TRANSFER, "Read output rows", ExecutionNodeType.DEVICE_TO_HOST_TRANSFER,
                    Operation.READ_OUTPUT, -1, rowBytes(batchRows, sizes[layers]), true)
                    .addDependency(lastForward));
            return new ExecutionGraph(nodes, layers, batchRows);
        }
        List<String> weightReaders = addBackward(nodes, sizes, layers, batchRows, lastForward);
        String last;
        if (mode == Mode.TRAINING) {
            long params = parameterCount(sizes, layers);
            put(nodes, kernel(UPDATE, "SGD update of every weight and bias", Operation.SGD_UPDATE, -1,
                    Kernels.SGD_STEP_F32, params, false).dependsOn(weightReaders));
            last = UPDATE;
        } else {
            last = null;
        }
        ExecutionNode.Builder results = transfer(RESULT_TRANSFER, "Read parameters and gradients",
                ExecutionNodeType.DEVICE_TO_HOST_TRANSFER, Operation.READ_RESULTS, -1,
                2 * parameterCount(sizes, layers) * Float.BYTES, false);
        put(nodes, last != null ? results.addDependency(last) : results.dependsOn(weightReaders));
        if (mode == Mode.TRAINING) {
            put(nodes, transfer(OUTPUT_TRANSFER, "Read output rows", ExecutionNodeType.DEVICE_TO_HOST_TRANSFER,
                    Operation.READ_OUTPUT, -1, rowBytes(batchRows, sizes[layers]), false)
                    .addDependency(RESULT_TRANSFER));
        }
        return new ExecutionGraph(nodes, layers, batchRows);
    }

    /** Adds the forward nodes; returns the id of the last layer's activation node. */
    private static String addForward(Map<String, ExecutionNode> nodes, int[] sizes, int layers, int rows) {
        put(nodes, transfer(INPUT_TRANSFER, "Upload input rows", ExecutionNodeType.HOST_TO_DEVICE_TRANSFER,
                Operation.UPLOAD_INPUT, -1, rowBytes(rows, sizes[0]), true));
        String previous = INPUT_TRANSFER;
        for (int layer = 0; layer < layers; layer++) {
            int in = sizes[layer];
            int out = sizes[layer + 1];
            String weights = forwardWeightsId(layer);
            String biases = "forward_transfer_biases_L" + layer;
            String linear = "forward_linear_L" + layer;
            String compute = forwardComputeId(layer);
            put(nodes, transfer(weights, "Upload weights layer " + layer, ExecutionNodeType.HOST_TO_DEVICE_TRANSFER,
                    Operation.UPLOAD_WEIGHTS, layer, (long) in * out * Float.BYTES, true));
            put(nodes, transfer(biases, "Upload biases layer " + layer, ExecutionNodeType.HOST_TO_DEVICE_TRANSFER,
                    Operation.UPLOAD_BIASES, layer, (long) out * Float.BYTES, true));
            put(nodes, dispatcher(linear, "x * W^T layer " + layer, Operation.LINEAR, layer, Kernels.GEMM_F32, true)
                    .addDependency(weights).addDependency(previous));
            put(nodes, kernel(compute, "Bias and activation layer " + layer, Operation.BIAS_ACTIVATION, layer,
                    Kernels.DENSE_BIAS_ACT_F32, (long) rows * out, true).addDependency(linear).addDependency(biases));
            previous = compute;
        }
        return previous;
    }

    /**
     * Adds the backward nodes; returns the ids of every node that must complete before the weights change (the
     * gradient nodes and the input-gradient GEMMs that read the weights).
     */
    private static List<String> addBackward(Map<String, ExecutionNode> nodes, int[] sizes, int layers, int rows,
                                            String lastForward) {
        int outputs = sizes[layers];
        put(nodes, transfer(TARGET_TRANSFER, "Upload target rows", ExecutionNodeType.HOST_TO_DEVICE_TRANSFER,
                Operation.UPLOAD_TARGET, -1, rowBytes(rows, outputs), false));
        put(nodes, kernel(LOSS_GRADIENT, "Loss gradient", Operation.LOSS_GRADIENT, layers - 1, Kernels.AXPBY_F32,
                (long) rows * outputs, false).addDependency(lastForward).addDependency(TARGET_TRANSFER));
        List<String> finals = new ArrayList<>();
        String gradientSource = LOSS_GRADIENT;
        for (int layer = layers - 1; layer >= 0; layer--) {
            int out = sizes[layer + 1];
            String act = "backward_act_L" + layer;
            String weightGrad = weightGradientId(layer);
            String biasGrad = biasGradientId(layer);
            put(nodes, kernel(act, "Activation backward layer " + layer, Operation.ACTIVATION_BACKWARD, layer,
                    Kernels.DENSE_ACT_BACKWARD_F32, (long) rows * out, false).addDependency(gradientSource));
            put(nodes, dispatcher(weightGrad, "dZ^T * x layer " + layer, Operation.WEIGHT_GRADIENT, layer,
                    Kernels.GEMM_F32, false).addDependency(act));
            put(nodes, dispatcher(biasGrad, "Column sums of dZ layer " + layer, Operation.BIAS_GRADIENT, layer,
                    Kernels.REDUCE_COLS_F32, false).addDependency(act));
            finals.add(weightGrad);
            finals.add(biasGrad);
            if (layer > 0) {
                String inputGrad = inputGradientId(layer);
                ExecutionNode.Builder b = dispatcher(inputGrad, "dZ * W layer " + layer, Operation.INPUT_GRADIENT,
                        layer, Kernels.GEMM_F32, false).addDependency(act);
                if (layer + 1 < layers) {
                    b.addDependency(weightGradientId(layer + 1)).addDependency(biasGradientId(layer + 1));
                }
                put(nodes, b);
                finals.add(inputGrad);
                gradientSource = inputGrad;
            }
        }
        return finals;
    }

    private static String forwardWeightsId(int layer) {
        return "forward_transfer_weights_L" + layer;
    }

    private static String forwardComputeId(int layer) {
        return "forward_compute_L" + layer;
    }

    private static String weightGradientId(int layer) {
        return "backward_compute_grad_L" + layer;
    }

    private static String biasGradientId(int layer) {
        return "backward_bias_grad_L" + layer;
    }

    private static String inputGradientId(int layer) {
        return "backward_input_grad_L" + layer;
    }

    private static long rowBytes(int rows, int width) {
        return (long) rows * width * Float.BYTES;
    }

    private static long parameterCount(int[] sizes, int layers) {
        long total = 0;
        for (int layer = 0; layer < layers; layer++) {
            total += (long) sizes[layer] * sizes[layer + 1] + sizes[layer + 1];
        }
        return total;
    }

    private static ExecutionNode.Builder transfer(String id, String label, ExecutionNodeType type, Operation op,
                                                  int layer, long bytes, boolean forward) {
        return ExecutionNode.builder().nodeId(id).label(label).type(type).operation(op).layerIndex(layer)
                .transferSizeBytes(bytes).isForward(forward);
    }

    /** A 1-D kernel node of {@code items} work-items. */
    private static ExecutionNode.Builder kernel(String id, String label, Operation op, int layer, String kernel,
                                                long items, boolean forward) {
        return dispatcher(id, label, op, layer, kernel, forward)
                .globalWorkSize(new long[]{items, 1, 1})
                .localWorkSize(new long[]{TrainGpu.LINEAR_LOCAL, 1, 1});
    }

    /** A kernel node whose launch geometry the dispatcher chooses. */
    private static ExecutionNode.Builder dispatcher(String id, String label, Operation op, int layer, String kernel,
                                                    boolean forward) {
        return ExecutionNode.builder().nodeId(id).label(label).type(ExecutionNodeType.KERNEL_DISPATCH).operation(op)
                .layerIndex(layer).kernelName(kernel).isForward(forward);
    }

    private static void put(Map<String, ExecutionNode> nodes, ExecutionNode.Builder builder) {
        ExecutionNode node = builder.build();
        if (nodes.put(node.nodeId(), node) != null) {
            throw new IllegalStateException("duplicate execution node " + node.nodeId());
        }
    }

    public ExecutionNode getNode(String nodeId) {
        return nodes.get(nodeId);
    }

    public List<String> topologicalOrder() {
        return new ArrayList<>(topologicalOrder);
    }

    public List<ExecutionNode> getNodesByType(ExecutionNodeType type) {
        List<ExecutionNode> result = new ArrayList<>();
        for (String nodeId : topologicalOrder) {
            ExecutionNode node = nodes.get(nodeId);
            if (node.type() == type) {
                result.add(node);
            }
        }
        return result;
    }

    /**
     * The host-to-device transfers of the next layer that do not depend on {@code computeNode}, so they may overlap
     * it; empty for a node that is not a kernel dispatch.
     */
    public List<ExecutionNode> getOverlappableTransfers(ExecutionNode computeNode) {
        if (computeNode.type() != ExecutionNodeType.KERNEL_DISPATCH) {
            return Collections.emptyList();
        }

        List<ExecutionNode> overlappable = new ArrayList<>();
        int computeLayerIndex = computeNode.layerIndex();
        boolean isForward = computeNode.isForward();

        for (ExecutionNode node : nodes.values()) {
            if (node.type() != ExecutionNodeType.HOST_TO_DEVICE_TRANSFER) {
                continue;
            }
            if (node.isForward() != isForward) {
                continue;
            }
            if (hasPathTo(node, computeNode)) {
                continue;
            }
            if (node.layerIndex() == computeLayerIndex + 1) {
                overlappable.add(node);
            }
        }

        return overlappable;
    }

    private boolean hasPathTo(ExecutionNode from, ExecutionNode to) {
        if (from.equals(to)) {
            return true;
        }
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        queue.add(from.nodeId());
        visited.add(from.nodeId());

        while (!queue.isEmpty()) {
            String currentId = queue.poll();
            ExecutionNode current = nodes.get(currentId);
            if (current == null) continue;

            for (String depId : current.dependsOn()) {
                if (depId.equals(to.nodeId())) {
                    return true;
                }
                if (!visited.contains(depId)) {
                    visited.add(depId);
                    queue.add(depId);
                }
            }
        }
        return false;
    }

    /** Kahn's algorithm over the nodes in insertion order, so equal graphs give equal orders. */
    private static List<String> computeTopologicalOrder(Map<String, ExecutionNode> nodes) {
        Map<String, Integer> inDegree = new LinkedHashMap<>();
        Map<String, List<String>> adjacency = new LinkedHashMap<>();

        for (String nodeId : nodes.keySet()) {
            inDegree.put(nodeId, 0);
            adjacency.put(nodeId, new ArrayList<>());
        }

        for (ExecutionNode node : nodes.values()) {
            for (String depId : node.dependsOn()) {
                if (!nodes.containsKey(depId)) {
                    throw new IllegalStateException("node " + node.nodeId() + " depends on unknown node " + depId);
                }
                adjacency.get(depId).add(node.nodeId());
                inDegree.merge(node.nodeId(), 1, Integer::sum);
            }
        }

        Queue<String> queue = new LinkedList<>();
        for (Map.Entry<String, Integer> entry : inDegree.entrySet()) {
            if (entry.getValue() == 0) {
                queue.add(entry.getKey());
            }
        }

        List<String> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            String current = queue.poll();
            result.add(current);

            for (String neighbor : adjacency.get(current)) {
                int newDegree = inDegree.get(neighbor) - 1;
                inDegree.put(neighbor, newDegree);
                if (newDegree == 0) {
                    queue.add(neighbor);
                }
            }
        }

        if (result.size() != nodes.size()) {
            throw new IllegalStateException("Cycle detected in execution graph");
        }

        return result;
    }

    public int size() {
        return nodes.size();
    }

    public int numLayers() {
        return numLayers;
    }

    /** The rows of the batch the graph's work sizes and transfers describe. */
    public int batchRows() {
        return batchRows;
    }

    public Set<String> nodeIds() {
        return Collections.unmodifiableSet(nodes.keySet());
    }
}
