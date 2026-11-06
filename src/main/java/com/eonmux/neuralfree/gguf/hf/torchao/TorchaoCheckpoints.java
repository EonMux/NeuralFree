package com.eonmux.neuralfree.gguf.hf.torchao;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.StTensor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The seam between the checkpoint containers and torchao: a container that read torchao tensor subclasses exposes
 * them through {@link Source}, and {@link #wrap} replaces the flattened inner tensors they serialized with the
 * ordinary checkpoint tensors the quantization schemes read. A container without torchao tensors is returned
 * unchanged, so checkpoints in the native layout take exactly the path they took before.
 *
 * <p>The wrapper keeps the underlying container's virtual address space and side files untouched; only the
 * tensor list and the row sources of the torchao tensors differ. A decoded tensor that would shadow a tensor the
 * checkpoint stores under its own name is refused, and so is addressing a flattened inner tensor by name.</p>
 */
public final class TorchaoCheckpoints {

    /** A container that read torchao tensor subclasses, by checkpoint name. */
    public interface Source {

        /** The torchao tensors the checkpoint stores, in the order its files listed them (unmodifiable). */
        List<TorchaoWeight> torchaoWeights();
    }

    private TorchaoCheckpoints() {
    }

    /**
     * Wraps {@code base} when it exposes torchao tensors, decoding them into ordinary checkpoint tensors.
     *
     * @throws IOException when a torchao tensor names an unsupported subclass, lacks required data, or its shapes
     *                     disagree, or when a decoded tensor would shadow a stored one
     */
    public static CheckpointContainer wrap(CheckpointContainer base) throws IOException {
        Objects.requireNonNull(base, "base");
        if (!(base instanceof Source)) {
            return base;
        }
        List<TorchaoWeight> weights = ((Source) base).torchaoWeights();
        if (weights.isEmpty()) {
            return base;
        }
        TorchaoDecoder.Decoded decoded = TorchaoDecoder.decode(weights, base.description());
        return new Wrapped(base, decoded);
    }

    /** A container over {@code base} whose torchao tensors are the decoded ordinary ones. Immutable. */
    private static final class Wrapped implements CheckpointContainer {

        private final CheckpointContainer base;
        private final TorchaoDecoder.Decoded decoded;
        private final List<StTensor> tensors;

        Wrapped(CheckpointContainer base, TorchaoDecoder.Decoded decoded) throws IOException {
            this.base = base;
            this.decoded = decoded;
            List<StTensor> all = new ArrayList<>();
            for (StTensor t : base.tensors()) {
                if (!decoded.hidden().contains(t.name())) {
                    all.add(t);
                    if (decoded.tensors().containsKey(t.name())) {
                        throw new IOException("Invalid Hugging Face checkpoint '" + base.description()
                                + "': torchao decodes a tensor named '" + t.name() + "', which the checkpoint also"
                                + " stores directly");
                    }
                }
            }
            all.addAll(decoded.tensors().values());
            this.tensors = Collections.unmodifiableList(all);
        }

        @Override
        public String description() {
            return base.description();
        }

        @Override
        public List<StTensor> tensors() {
            return tensors;
        }

        @Override
        public StTensor tensor(String name) {
            Objects.requireNonNull(name, "name");
            StTensor decodedTensor = decoded.tensors().get(name);
            if (decodedTensor != null) {
                return decodedTensor;
            }
            if (decoded.hidden().contains(name)) {
                return null;
            }
            return base.tensor(name);
        }

        @Override
        public RowSource source(String name) {
            Objects.requireNonNull(name, "name");
            RowSource source = decoded.sources().get(name);
            if (source != null) {
                return source;
            }
            if (decoded.hidden().contains(name)) {
                throw new IllegalArgumentException(base.description() + " has no tensor '" + name + "' (torchao"
                        + " subclasses are served under their decoded weight, scale and zero-point names)");
            }
            return base.source(name);
        }

        @Override
        public List<DataSegment> segments() {
            return base.segments();
        }

        @Override
        public long directVirtualOffset(String name) {
            Objects.requireNonNull(name, "name");
            if (decoded.tensors().containsKey(name)) {
                return -1;
            }
            if (decoded.hidden().contains(name)) {
                throw new IllegalArgumentException(base.description() + " has no tensor '" + name + "' (torchao"
                        + " subclasses are served under their decoded weight, scale and zero-point names)");
            }
            return base.directVirtualOffset(name);
        }

        @Override
        public SafetensorsFile openSideFile(String relPath) throws IOException {
            return base.openSideFile(relPath);
        }

        @Override
        public void close() throws IOException {
            base.close();
        }

        @Override
        public String toString() {
            return base.description();
        }
    }
}
