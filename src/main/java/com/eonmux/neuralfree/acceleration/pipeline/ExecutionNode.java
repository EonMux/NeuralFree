package com.eonmux.neuralfree.acceleration.pipeline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * One step of an {@link ExecutionGraph}: a host transfer or a device operation of the dense network, the nodes it
 * depends on, and the train kernel it runs. Immutable.
 *
 * <p>{@link #globalWorkSize()} of a 1-D kernel node is its work-item count (the launch rounds it up to a multiple of
 * {@link #localWorkSize()}); nodes whose launch geometry a dispatcher chooses (GEMM, column sums) carry none.</p>
 */
public final class ExecutionNode {

    /** What a node does; {@link PipelineScheduler} executes each graph node by its operation. */
    public enum Operation {
        /** Uploads layer {@code l}'s weight matrix. */
        UPLOAD_WEIGHTS,
        /** Uploads layer {@code l}'s bias vector. */
        UPLOAD_BIASES,
        /** Uploads the input rows. */
        UPLOAD_INPUT,
        /** Uploads the target rows. */
        UPLOAD_TARGET,
        /** {@code z = x * W^T} for layer {@code l} (strided GEMM). */
        LINEAR,
        /** {@code z += b; a = act(z)} for layer {@code l} (TR_DenseBiasAct_F32). */
        BIAS_ACTIVATION,
        /** {@code dY = (a - t) / rows} on the output layer (TR_Axpby_F32). */
        LOSS_GRADIENT,
        /** {@code dZ = dY * act'(z, a)} for layer {@code l}, in place (TR_DenseActBackward_F32). */
        ACTIVATION_BACKWARD,
        /** {@code dW = dZ^T * x} for layer {@code l} (strided GEMM). */
        WEIGHT_GRADIENT,
        /** {@code db = column sums of dZ} for layer {@code l} (TR_ReduceCols_F32). */
        BIAS_GRADIENT,
        /** {@code dY(l - 1) = dZ * W} for layer {@code l > 0} (strided GEMM). */
        INPUT_GRADIENT,
        /** {@code p -= lr * g} over every weight and bias (TR_SgdStep_F32). */
        SGD_UPDATE,
        /** Reads the output rows back. */
        READ_OUTPUT,
        /** Reads the weights, biases and their gradients back. */
        READ_RESULTS
    }

    private final String nodeId;
    private final String label;
    private final ExecutionNodeType type;
    private final Operation operation;
    private final List<String> dependsOn;
    private final long transferSizeBytes;
    private final int layerIndex;
    private final String kernelName;
    private final long[] globalWorkSize;
    private final long[] localWorkSize;
    private final boolean isForward;

    private ExecutionNode(Builder builder) {
        this.nodeId = Objects.requireNonNull(builder.nodeId, "nodeId must not be null");
        this.label = builder.label != null ? builder.label : builder.nodeId;
        this.type = Objects.requireNonNull(builder.type, "type must not be null");
        this.operation = builder.operation;
        this.dependsOn = Collections.unmodifiableList(new ArrayList<>(builder.dependsOn));
        this.transferSizeBytes = builder.transferSizeBytes;
        this.layerIndex = builder.layerIndex;
        this.kernelName = builder.kernelName;
        this.globalWorkSize = builder.globalWorkSize != null ? builder.globalWorkSize.clone() : null;
        this.localWorkSize = builder.localWorkSize != null ? builder.localWorkSize.clone() : null;
        this.isForward = builder.isForward;
    }

    public String nodeId() {
        return nodeId;
    }

    public String label() {
        return label;
    }

    public ExecutionNodeType type() {
        return type;
    }

    /** The operation, or null for a node built without one (such a node cannot be executed). */
    public Operation operation() {
        return operation;
    }

    public List<String> dependsOn() {
        return dependsOn;
    }

    public long transferSizeBytes() {
        return transferSizeBytes;
    }

    /** The layer the node belongs to, or -1 for whole-network nodes. */
    public int layerIndex() {
        return layerIndex;
    }

    /** The train kernel the node runs, or null for transfers. */
    public String kernelName() {
        return kernelName;
    }

    public long[] globalWorkSize() {
        return globalWorkSize != null ? globalWorkSize.clone() : null;
    }

    public long[] localWorkSize() {
        return localWorkSize != null ? localWorkSize.clone() : null;
    }

    public boolean isForward() {
        return isForward;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ExecutionNode that = (ExecutionNode) o;
        return nodeId.equals(that.nodeId);
    }

    @Override
    public int hashCode() {
        return nodeId.hashCode();
    }

    @Override
    public String toString() {
        return "ExecutionNode{" +
               "nodeId='" + nodeId + '\'' +
               ", label='" + label + '\'' +
               ", type=" + type +
               ", operation=" + operation +
               ", kernel=" + kernelName +
               ", layerIndex=" + layerIndex +
               ", isForward=" + isForward +
               '}';
    }

    public static final class Builder {
        private String nodeId;
        private String label;
        private ExecutionNodeType type;
        private Operation operation;
        private List<String> dependsOn = new ArrayList<>();
        private long transferSizeBytes;
        private int layerIndex = -1;
        private String kernelName;
        private long[] globalWorkSize;
        private long[] localWorkSize;
        private boolean isForward = true;

        public Builder nodeId(String nodeId) {
            this.nodeId = nodeId;
            return this;
        }

        public Builder label(String label) {
            this.label = label;
            return this;
        }

        public Builder type(ExecutionNodeType type) {
            this.type = type;
            return this;
        }

        public Builder operation(Operation operation) {
            this.operation = operation;
            return this;
        }

        public Builder addDependency(String nodeId) {
            this.dependsOn.add(nodeId);
            return this;
        }

        public Builder dependsOn(List<String> dependsOn) {
            this.dependsOn = new ArrayList<>(dependsOn);
            return this;
        }

        public Builder transferSizeBytes(long transferSizeBytes) {
            this.transferSizeBytes = transferSizeBytes;
            return this;
        }

        public Builder layerIndex(int layerIndex) {
            this.layerIndex = layerIndex;
            return this;
        }

        public Builder kernelName(String kernelName) {
            this.kernelName = kernelName;
            return this;
        }

        public Builder globalWorkSize(long[] globalWorkSize) {
            this.globalWorkSize = globalWorkSize;
            return this;
        }

        public Builder localWorkSize(long[] localWorkSize) {
            this.localWorkSize = localWorkSize;
            return this;
        }

        public Builder isForward(boolean isForward) {
            this.isForward = isForward;
            return this;
        }

        public ExecutionNode build() {
            return new ExecutionNode(this);
        }
    }
}
