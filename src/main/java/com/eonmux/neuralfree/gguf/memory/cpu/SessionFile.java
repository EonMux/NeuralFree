package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs.GroupPositions;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * The llama.cpp session file ({@code llama_state_save_file}/{@code llama_state_load_file}) of a
 * {@link HeapModelMemory}: the stored tokens with their reuse frontier and wrap flag, every KV group's host
 * position mirror (the {@code kv_pos} of each row, -1 when empty), the K/V row values of every owning cache
 * layer, and the live recurrent state tensors of every state layer. A load first checks the whole plan identity
 * against the memory it opens into (context, ubatch, group and layer geometry, cache dtypes, head shapes, tensor
 * names), then resets the memory and writes everything back, so the next decode continues the saved sequence
 * exactly — attention reads the same rows at the same positions, as llama.cpp's session restore does.
 *
 * <p>Format: the {@code NFST} magic, a version, the plan identity, the sequence record, the per-group position
 * mirrors, and per cache layer its header plus the K/V rows in element order (F32 frames of the row widths,
 * whatever the stored dtype decodes from), then the state tensors. Rows cross the file through the element
 * decode: a block-typed cache restores from the decoded values and the deterministic reference quantizer
 * requantizes them, so its logits match within quantizer rounding.
 *
 * <p>The runtime-only state stays out: recurrent snapshot shadows ({@code snapshot()} holds them in memory),
 * the compressor rollback planes, and the sampler chain. The scope is the single-sequence path: a paged memory
 * (page lists and its sequences), DSV4 compressed caches, the sparse side caches (DSA indexer, MSA, T5
 * cross-attention) and caches that read a target model refuse by name before anything is written or reset.
 */
final class SessionFile {

    private static final int MAGIC = 0x4E465354; // "NFST"
    private static final int VERSION = 1;

    private SessionFile() {
    }

    /** Writes {@code memory}'s session to {@code out} (replaced when it exists), as llama.cpp's save does. */
    static void save(HeapModelMemory memory, Path out) throws IOException {
        String refusal = scopeRefusal(memory);
        if (refusal != null) {
            throw new UnsupportedOperationException("session save refused: " + refusal);
        }
        try (DataOutputStream stream = new DataOutputStream(new BufferedOutputStream(
                Files.newOutputStream(out)))) {
            stream.writeInt(MAGIC);
            stream.writeInt(VERSION);
            writeIdentity(memory, stream);
            writeSequence(memory, stream);
            writePositions(memory, stream);
            writeCaches(memory, stream);
            writeStates(memory, stream);
        }
    }

    /** Reads the session at {@code in} into {@code memory}, resetting it first. */
    static void load(HeapModelMemory memory, Path in) throws IOException {
        String refusal = scopeRefusal(memory);
        if (refusal != null) {
            throw new UnsupportedOperationException("session load refused: " + refusal);
        }
        try (DataInputStream stream = new DataInputStream(new BufferedInputStream(Files.newInputStream(in)))) {
            Reader reader = new Reader(stream, in);
            if (reader.readInt() != MAGIC) {
                throw new IOException(in + " is not a NeuralFree session file");
            }
            int version = reader.readInt();
            if (version != VERSION) {
                throw new IOException(in + " holds session format " + version + ", this build reads " + VERSION);
            }
            checkIdentity(memory, reader);
            memory.reset();
            readSequence(memory, reader);
            readPositions(memory, reader);
            readCaches(memory, reader);
            readStates(memory, reader);
        }
    }

    /**
     * Why {@code memory} is outside the session-file scope, or null when it is covered: the paged page lists,
     * DSV4 compressed caches, the sparse side caches and target-model caches are not serialized.
     */
    private static String scopeRefusal(HeapModelMemory memory) {
        if (memory.pagedSequences() != null) {
            return "the paged page lists of this memory are not serializable yet; the session file covers the"
                    + " single-sequence path";
        }
        if (memory.plan().hasCompressed()) {
            return "the DSV4 compressed caches of this memory are not serializable yet";
        }
        for (int il = 0; il < memory.plan().layers().size(); il++) {
            if (memory.indexer(il) != null || memory.msa(il) != null) {
                return "layer " + il + " runs a sparse-attention side cache, which is not serializable yet";
            }
            if (memory.crossKv(il) != null) {
                return "layer " + il + " plans a cross-attention cache, which is not serializable yet";
            }
            var shared = memory.plan().layers().get(il).sharedFrom();
            if (shared != null && shared.target()) {
                return "layer " + il + " reads a target model's cache, which is not serializable here";
            }
        }
        return null;
    }

    // ------------------------------------------------------------------ identity

    private static void writeIdentity(HeapModelMemory memory, DataOutputStream out) throws IOException {
        var plan = memory.plan();
        out.writeInt(plan.nCtx());
        out.writeInt(plan.ubatch());
        out.writeInt(plan.groups().size());
        out.writeInt(plan.layers().size());
        out.writeInt(plan.snapshotsReserved() ? 1 : 0);
        for (int g = 0; g < plan.groups().size(); g++) {
            out.writeInt(plan.group(g).rows());
        }
    }

    private static void checkIdentity(HeapModelMemory memory, Reader in) throws IOException {
        var plan = memory.plan();
        in.expect(plan.nCtx(), "n_ctx");
        in.expect(plan.ubatch(), "ubatch");
        in.expect(plan.groups().size(), "the KV group count");
        in.expect(plan.layers().size(), "the layer count");
        in.expect(plan.snapshotsReserved() ? 1 : 0, "the snapshot reservation");
        for (int g = 0; g < plan.groups().size(); g++) {
            in.expect(plan.group(g).rows(), "group " + g + " row count");
        }
    }

    // ------------------------------------------------------------------ sequence

    private static void writeSequence(HeapModelMemory memory, DataOutputStream out) throws IOException {
        var sequence = memory.sequence();
        int[] tokens = sequence.tokens();
        out.writeInt(sequence.nPast());
        out.writeInt(sequence.reusable());
        out.writeInt(sequence.wrapped() ? 1 : 0);
        for (int token : tokens) {
            out.writeInt(token);
        }
    }

    private static void readSequence(HeapModelMemory memory, Reader in) throws IOException {
        var sequence = memory.sequence();
        int nPast = in.readInt();
        int reusable = in.readInt();
        boolean wrapped = in.readInt() != 0;
        if (nPast < 0 || nPast > sequence.nCtx()) {
            throw new IOException("the session holds " + nPast + " positions, outside the context of "
                    + sequence.nCtx());
        }
        int[] tokens = new int[nPast];
        for (int i = 0; i < nPast; i++) {
            tokens[i] = in.readInt();
        }
        sequence.append(tokens, 0, nPast);
        sequence.capReusable(Math.min(reusable, nPast));
        if (wrapped) {
            sequence.markWrapped();
        }
    }

    // ------------------------------------------------------------------ position mirrors

    private static void writePositions(HeapModelMemory memory, DataOutputStream out) throws IOException {
        GroupPositions[] positions = memory.groupPositions();
        for (GroupPositions p : positions) {
            int[] mirror = p.snapshot();
            out.writeInt(mirror.length);
            for (int pos : mirror) {
                out.writeInt(pos);
            }
        }
    }

    private static void readPositions(HeapModelMemory memory, Reader in) throws IOException {
        GroupPositions[] positions = memory.groupPositions();
        for (GroupPositions p : positions) {
            int rows = in.readInt();
            if (rows != p.rows()) {
                throw new IOException("the session mirror of group " + p.group().index() + " holds " + rows
                        + " rows, the memory plans " + p.rows());
            }
            for (int r = 0; r < rows; r++) {
                p.set(r, in.readInt());
            }
        }
    }

    // ------------------------------------------------------------------ K/V caches

    private static void writeCaches(HeapModelMemory memory, DataOutputStream out) throws IOException {
        List<CpuKvLayer> layers = memory.kvOwners();
        out.writeInt(layers.size());
        for (CpuKvLayer layer : layers) {
            writeLayerHeader(layer, out);
            int rows = memory.groupPositions()[layer.group()].rows();
            out.writeInt(rows);
            int heads = layer.nHeadKv();
            float[] row = new float[Math.max(layer.layout().widthK(), layer.layout().widthV())];
            for (int r = 0; r < rows; r++) {
                for (int h = 0; h < heads; h++) {
                    layer.readK(r, h, row, h * layer.headDimK());
                }
                for (int i = 0; i < layer.layout().widthK(); i++) {
                    out.writeFloat(row[i]);
                }
                if (layer.layout().vAliasesK()) {
                    continue;
                }
                for (int h = 0; h < heads; h++) {
                    layer.readV(r, h, row, h * layer.headDimV());
                }
                for (int i = 0; i < layer.layout().widthV(); i++) {
                    out.writeFloat(row[i]);
                }
            }
        }
    }

    private static void readCaches(HeapModelMemory memory, Reader in) throws IOException {
        List<CpuKvLayer> layers = memory.kvOwners();
        int count = in.readInt();
        if (count != layers.size()) {
            throw new IOException(in.name() + " holds " + count + " cache layers, the memory plans "
                    + layers.size());
        }
        for (CpuKvLayer layer : layers) {
            checkLayerHeader(layer, in);
            int rows = in.readInt();
            if (rows != memory.groupPositions()[layer.group()].rows()) {
                throw new IOException(in.name() + " holds " + rows + " cache rows for layer " + layer.layer()
                        + ", the memory plans " + memory.groupPositions()[layer.group()].rows());
            }
            int heads = layer.nHeadKv();
            float[] k = new float[layer.layout().widthK()];
            float[] v = new float[layer.layout().widthV()];
            for (int r = 0; r < rows; r++) {
                for (int i = 0; i < k.length; i++) {
                    k[i] = in.readFloat();
                }
                layer.writeK(r, k, 0);
                if (layer.layout().vAliasesK()) {
                    continue;
                }
                for (int i = 0; i < v.length; i++) {
                    v[i] = in.readFloat();
                }
                layer.writeV(r, v, 0);
            }
        }
    }

    private static void writeLayerHeader(CpuKvLayer layer, DataOutputStream out) throws IOException {
        out.writeInt(layer.layer());
        out.writeInt(layer.group());
        out.writeInt(layer.layout().dtypeK().code());
        out.writeInt(layer.layout().dtypeV().code());
        out.writeInt(layer.layout().widthK());
        out.writeInt(layer.layout().widthV());
        out.writeInt(layer.nHeadKv());
        out.writeInt(layer.headDimK());
        out.writeInt(layer.headDimV());
    }

    private static void checkLayerHeader(CpuKvLayer layer, Reader in) throws IOException {
        in.expect(layer.layer(), "cache layer");
        in.expect(layer.group(), "cache layer " + layer.layer() + " group");
        in.expect(layer.layout().dtypeK().code(), "layer " + layer.layer() + " K dtype");
        in.expect(layer.layout().dtypeV().code(), "layer " + layer.layer() + " V dtype");
        in.expect(layer.layout().widthK(), "layer " + layer.layer() + " K width");
        in.expect(layer.layout().widthV(), "layer " + layer.layer() + " V width");
        in.expect(layer.nHeadKv(), "layer " + layer.layer() + " KV head count");
        in.expect(layer.headDimK(), "layer " + layer.layer() + " K head dim");
        in.expect(layer.headDimV(), "layer " + layer.layer() + " V head dim");
    }

    // ------------------------------------------------------------------ recurrent state

    private static void writeStates(HeapModelMemory memory, DataOutputStream out) throws IOException {
        List<CpuRecurrentState> states = memory.stateOwners();
        out.writeInt(states.size());
        for (CpuRecurrentState state : states) {
            out.writeInt(state.layer());
            out.writeInt(state.tensors().size());
            for (var tensor : state.tensors()) {
                out.writeUTF(tensor.name());
                float[] values = state.tensor(tensor.name());
                out.writeInt(values.length);
                for (float v : values) {
                    out.writeFloat(v);
                }
            }
        }
    }

    private static void readStates(HeapModelMemory memory, Reader in) throws IOException {
        List<CpuRecurrentState> states = memory.stateOwners();
        int count = in.readInt();
        if (count != states.size()) {
            throw new IOException(in.name() + " holds " + count + " state layers, the memory plans "
                    + states.size());
        }
        for (CpuRecurrentState state : states) {
            in.expect(state.layer(), "state layer");
            int tensors = in.readInt();
            if (tensors != state.tensors().size()) {
                throw new IOException("state layer " + state.layer() + " holds " + tensors
                        + " tensors in the session, the memory plans " + state.tensors().size());
            }
            for (int t = 0; t < tensors; t++) {
                String name = in.text();
                int floats = in.readInt();
                float[] values = state.tensor(name);
                if (values.length != floats) {
                    throw new IOException("state tensor " + name + " of layer " + state.layer() + " holds "
                            + floats + " floats in the session, the memory plans " + values.length);
                }
                for (int i = 0; i < floats; i++) {
                    values[i] = in.readFloat();
                }
            }
        }
    }

    /**
     * The load-side reader: every value comes through here, so a mismatched count names what the file was saved
     * for and what the memory plans instead.
     */
    private static final class Reader {
        private final DataInputStream in;
        private final Path file;

        Reader(DataInputStream in, Path file) {
            this.in = in;
            this.file = file;
        }

        String name() {
            return file.toString();
        }

        String text() throws IOException {
            return in.readUTF();
        }

        int readInt() throws IOException {
            return in.readInt();
        }

        float readFloat() throws IOException {
            return in.readFloat();
        }

        void expect(int expected, String what) throws IOException {
            int v = in.readInt();
            if (v != expected) {
                throw new IOException(file + " was saved for " + what + " " + v + ", the memory plans " + expected);
            }
        }
    }
}
