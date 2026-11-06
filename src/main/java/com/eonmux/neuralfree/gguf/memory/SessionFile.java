package com.eonmux.neuralfree.gguf.memory;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * The llama.cpp session file ({@code llama_state_save_file} / {@code llama_state_load_file}) of a contiguous
 * {@link ModelMemory}, on either backend: the stored tokens with their reuse frontier and wrap flag, every KV group's
 * position mirror (the {@code kv_pos} of each row, -1 when empty), the raw rows of every owning cache layer and the
 * live recurrent state tensors, all through the memory's {@link StateTransfer}. A load first checks the whole plan
 * identity against the memory it opens into (context, ubatch, group and layer geometry, cache dtypes and widths,
 * tensor names and sizes), then resets the memory and writes everything back, so the next decode continues the saved
 * sequence exactly — attention reads the same rows at the same positions, as llama.cpp's session restore does.
 *
 * <p>Format: the {@code NFST} magic, a version, the plan identity, the sequence record, the per-group position
 * mirrors, and per cache layer its header plus its K and V rows as stored bytes in the cache dtype (little-endian,
 * the layout both backends share) and their side planes (the per-(token, KV head) scales of a per-token-head cache),
 * then the state tensors. A file saved on the CPU loads on the device and back, bit-exact for every cache dtype.
 *
 * <p>The runtime-only state stays out: recurrent snapshot shadows ({@code snapshot()} holds them in memory), the
 * compressor rollback planes and the sampler chain. The scope is the contiguous single-sequence memory: a paged
 * memory (its slots save through the batch engine's slot files), DSV4 compressed caches, the sparse side caches (DSA
 * indexer, MSA), T5 cross-attention caches and caches that read a target model refuse by name before anything is
 * written or reset.
 */
public final class SessionFile {

    private static final int MAGIC = 0x4E465354; // "NFST"
    private static final int VERSION = 3;
    /** Rows a cache moves per transfer: bounded so a large cache never needs one array of its whole size. */
    private static final int CHUNK_BYTES = 1 << 20;

    private SessionFile() {
    }

    /**
     * Writes {@code memory}'s session to {@code out} (replaced when it exists), as llama.cpp's save does.
     *
     * @throws UnsupportedOperationException when the memory is outside the session-file scope
     */
    public static void save(ModelMemory memory, Path out) throws IOException {
        StateTransfer transfer = transferOf(memory, "save");
        try (OutputStream stream = new BufferedOutputStream(Files.newOutputStream(out))) {
            writeSession(memory, transfer, stream);
        }
    }

    /**
     * Writes {@code memory}'s session to an open stream (the format of {@link #save(ModelMemory, Path)}), leaving the
     * stream open for a container that concatenates several sessions.
     */
    public static void save(ModelMemory memory, OutputStream out) throws IOException {
        writeSession(memory, transferOf(memory, "save"), out);
    }

    private static void writeSession(ModelMemory memory, StateTransfer transfer, OutputStream out) throws IOException {
        DataOutputStream stream = new DataOutputStream(out);
        stream.writeInt(MAGIC);
        stream.writeInt(VERSION);
        writeIdentity(memory.plan(), stream);
        writeSequence(memory.sequence(), stream);
        writePositions(memory.plan(), transfer, stream);
        writeCaches(transfer, stream);
        writeStates(transfer, stream);
        stream.flush();
    }

    /**
     * Reads the session at {@code in} into {@code memory}, resetting it first.
     *
     * @throws UnsupportedOperationException when the memory is outside the session-file scope
     * @throws IOException                   for an unreadable file, a foreign format or a plan mismatch
     */
    public static void load(ModelMemory memory, Path in) throws IOException {
        StateTransfer transfer = transferOf(memory, "load");
        try (InputStream stream = new BufferedInputStream(Files.newInputStream(in))) {
            readSession(memory, transfer, stream, in.toString());
        }
    }

    /**
     * Reads one session from an open stream (the format of {@link #save(ModelMemory, Path)}), resetting the memory
     * first; a container may read several sessions one after another from the same stream.
     */
    public static void load(ModelMemory memory, InputStream in, String source) throws IOException {
        readSession(memory, transferOf(memory, "load"), in, source);
    }

    private static void readSession(ModelMemory memory, StateTransfer transfer, InputStream in, String source)
            throws IOException {
        try {
            DataInputStream stream = new DataInputStream(in);
            Reader reader = new Reader(stream, source);
            if (reader.readInt() != MAGIC) {
                throw new IOException(source + " is not a NeuralFree session file");
            }
            int version = reader.readInt();
            if (version != VERSION) {
                throw new IOException(source + " holds session format " + version + ", this build reads " + VERSION);
            }
            checkIdentity(memory.plan(), reader);
            memory.reset();
            readSequence(memory.sequence(), reader);
            readPositions(memory.plan(), transfer, reader);
            readCaches(transfer, reader);
            readStates(transfer, reader);
        } catch (EOFException e) {
            throw new IOException(source + " ends before its session does", e);
        }
    }

    /** The memory's transfer, after the scope check that refuses, by name, what the session file does not cover. */
    private static StateTransfer transferOf(ModelMemory memory, String action) {
        String refusal = scopeRefusal(memory);
        if (refusal != null) {
            throw new UnsupportedOperationException("session " + action + " refused: " + refusal);
        }
        try {
            return memory.stateTransfer();
        } catch (UnsupportedOperationException e) {
            throw new UnsupportedOperationException("session " + action + " refused: " + e.getMessage(), e);
        }
    }

    /**
     * Why {@code memory} is outside the session-file scope, or null when it is covered: the paged page lists, DSV4
     * compressed caches, the sparse side caches, cross-attention caches and target-model caches are not serialized.
     */
    static String scopeRefusal(ModelMemory memory) {
        if (memory.pagedSequences() != null) {
            return "a paged memory holds its sequences in page lists; the session file covers the contiguous"
                    + " single-sequence memory (slots of a paged memory save through the batch engine's slot files)";
        }
        MemoryPlan plan = memory.plan();
        if (plan.hasCompressed()) {
            return "the DSV4 compressed caches of this memory are not serializable";
        }
        for (LayerMemoryPlan lp : plan.layers()) {
            String why = layerRefusal(lp);
            if (why != null) {
                return why;
            }
        }
        return null;
    }

    /** Why a layer's caches are outside the serialized state, or null: side caches, cross caches, target caches. */
    public static String layerRefusal(LayerMemoryPlan lp) {
        int il = lp.layer();
        if (lp.indexer() != null || lp.msa() != null) {
            return "layer " + il + " runs a sparse-attention side cache, which is not serializable";
        }
        if (lp.crossKv() != null) {
            return "layer " + il + " plans a cross-attention cache, which is not serializable";
        }
        LayerMemoryPlan.SharedKv shared = lp.sharedFrom();
        if (shared != null && shared.target()) {
            return "layer " + il + " reads a target model's cache, which is not serializable here";
        }
        return null;
    }

    // ------------------------------------------------------------------ identity

    private static void writeIdentity(MemoryPlan plan, DataOutputStream out) throws IOException {
        out.writeInt(plan.nCtx());
        out.writeInt(plan.ubatch());
        out.writeInt(plan.groups().size());
        out.writeInt(plan.layers().size());
        out.writeInt(plan.snapshotsReserved() ? 1 : 0);
        for (int g = 0; g < plan.groups().size(); g++) {
            out.writeInt(plan.group(g).rows());
        }
    }

    private static void checkIdentity(MemoryPlan plan, Reader in) throws IOException {
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

    private static void writeSequence(SequenceState sequence, DataOutputStream out) throws IOException {
        int[] tokens = sequence.tokens();
        out.writeInt(sequence.nPast());
        out.writeInt(sequence.reusable());
        out.writeInt(sequence.wrapped() ? 1 : 0);
        for (int token : tokens) {
            out.writeInt(token);
        }
    }

    private static void readSequence(SequenceState sequence, Reader in) throws IOException {
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

    private static void writePositions(MemoryPlan plan, StateTransfer transfer, DataOutputStream out)
            throws IOException {
        for (int g = 0; g < plan.groups().size(); g++) {
            int[] mirror = transfer.positions(g);
            out.writeInt(mirror.length);
            for (int pos : mirror) {
                out.writeInt(pos);
            }
        }
    }

    private static void readPositions(MemoryPlan plan, StateTransfer transfer, Reader in) throws IOException {
        for (int g = 0; g < plan.groups().size(); g++) {
            int rows = in.readInt();
            if (rows != plan.group(g).rows()) {
                throw new IOException("the session mirror of group " + g + " holds " + rows + " rows, the memory"
                        + " plans " + plan.group(g).rows());
            }
            int[] mirror = new int[rows];
            for (int r = 0; r < rows; r++) {
                mirror[r] = in.readInt();
            }
            transfer.setPositions(g, mirror);
        }
    }

    // ------------------------------------------------------------------ K/V caches

    private static void writeCaches(StateTransfer transfer, DataOutputStream out) throws IOException {
        List<StateTransfer.Cache> caches = transfer.caches();
        out.writeInt(caches.size());
        for (StateTransfer.Cache cache : caches) {
            writeCacheHeader(cache, out);
            writeRows(cache, false, out);
            if (!cache.layout().vAliasesK()) {
                writeRows(cache, true, out);
            }
            writeSides(cache, false, out);
            if (!cache.layout().vAliasesK()) {
                writeSides(cache, true, out);
            }
        }
    }

    private static void readCaches(StateTransfer transfer, Reader in) throws IOException {
        List<StateTransfer.Cache> caches = transfer.caches();
        int count = in.readInt();
        if (count != caches.size()) {
            throw new IOException(in.name() + " holds " + count + " cache layers, the memory plans "
                    + caches.size());
        }
        for (StateTransfer.Cache cache : caches) {
            checkCacheHeader(cache, in);
            readRows(cache, false, in);
            if (!cache.layout().vAliasesK()) {
                readRows(cache, true, in);
            }
            readSides(cache, false, in);
            if (!cache.layout().vAliasesK()) {
                readSides(cache, true, in);
            }
        }
    }

    /** Every row of K or V, as stored bytes, in chunks of rows. */
    private static void writeRows(StateTransfer.Cache cache, boolean v, DataOutputStream out) throws IOException {
        KvLayout l = cache.layout();
        int rowBytes = Math.toIntExact(v ? l.rowBytesV() : l.rowBytesK());
        int chunk = Math.max(1, CHUNK_BYTES / rowBytes);
        byte[] buf = new byte[Math.toIntExact((long) Math.min(chunk, l.rows()) * rowBytes)];
        for (int row = 0; row < l.rows(); row += chunk) {
            int n = Math.min(chunk, l.rows() - row);
            cache.read(v, row, n, buf);
            out.write(buf, 0, n * rowBytes);
        }
    }

    private static void readRows(StateTransfer.Cache cache, boolean v, Reader in) throws IOException {
        KvLayout l = cache.layout();
        int rowBytes = Math.toIntExact(v ? l.rowBytesV() : l.rowBytesK());
        int chunk = Math.max(1, CHUNK_BYTES / rowBytes);
        byte[] buf = new byte[Math.toIntExact((long) Math.min(chunk, l.rows()) * rowBytes)];
        for (int row = 0; row < l.rows(); row += chunk) {
            int n = Math.min(chunk, l.rows() - row);
            in.readFully(buf, n * rowBytes);
            cache.write(v, row, n, buf);
        }
    }

    /** Every side plane of K or V, row by row in chunks, as stored little-endian values. */
    private static void writeSides(StateTransfer.Cache cache, boolean v, DataOutputStream out) throws IOException {
        List<KvDType.SidePlane> planes = cache.sidePlanes(v);
        for (int p = 0; p < planes.size(); p++) {
            long rowBytes = planes.get(p).rowBytesFor(v ? cache.layout().widthV() : cache.layout().widthK(),
                    cache.layout().nHeadKv());
            int chunk = Math.max(1, (int) (CHUNK_BYTES / rowBytes));
            byte[] buf = new byte[Math.toIntExact(Math.min(chunk, cache.layout().rows()) * rowBytes)];
            for (int row = 0; row < cache.layout().rows(); row += chunk) {
                int n = Math.min(chunk, cache.layout().rows() - row);
                cache.readSide(v, p, row, n, buf);
                out.write(buf, 0, Math.toIntExact(n * rowBytes));
            }
        }
    }

    private static void readSides(StateTransfer.Cache cache, boolean v, Reader in) throws IOException {
        List<KvDType.SidePlane> planes = cache.sidePlanes(v);
        for (int p = 0; p < planes.size(); p++) {
            long rowBytes = planes.get(p).rowBytesFor(v ? cache.layout().widthV() : cache.layout().widthK(),
                    cache.layout().nHeadKv());
            int chunk = Math.max(1, (int) (CHUNK_BYTES / rowBytes));
            byte[] buf = new byte[Math.toIntExact(Math.min(chunk, cache.layout().rows()) * rowBytes)];
            for (int row = 0; row < cache.layout().rows(); row += chunk) {
                int n = Math.min(chunk, cache.layout().rows() - row);
                in.readFully(buf, Math.toIntExact(n * rowBytes));
                cache.writeSide(v, p, row, n, buf);
            }
        }
    }

    private static void writeCacheHeader(StateTransfer.Cache cache, DataOutputStream out) throws IOException {
        KvLayout l = cache.layout();
        out.writeInt(cache.layer());
        out.writeInt(cache.group());
        out.writeInt(l.rows());
        out.writeInt(l.dtypeK().code());
        out.writeInt(l.dtypeV().code());
        out.writeInt(l.widthK());
        out.writeInt(l.widthV());
        out.writeInt(l.vAliasesK() ? 1 : 0);
        out.writeInt(l.nHeadKv());
    }

    private static void checkCacheHeader(StateTransfer.Cache cache, Reader in) throws IOException {
        KvLayout l = cache.layout();
        String layer = "layer " + cache.layer();
        in.expect(cache.layer(), "cache layer");
        in.expect(cache.group(), layer + " group");
        in.expect(l.rows(), layer + " row count");
        in.expect(l.dtypeK().code(), layer + " K dtype");
        in.expect(l.dtypeV().code(), layer + " V dtype");
        in.expect(l.widthK(), layer + " K width");
        in.expect(l.widthV(), layer + " V width");
        in.expect(l.vAliasesK() ? 1 : 0, layer + " latent V");
        in.expect(l.nHeadKv(), layer + " KV head count");
    }

    // ------------------------------------------------------------------ recurrent state

    private static void writeStates(StateTransfer transfer, DataOutputStream out) throws IOException {
        List<StateTransfer.State> states = transfer.states();
        out.writeInt(states.size());
        for (StateTransfer.State state : states) {
            out.writeInt(state.layer());
            out.writeInt(state.tensors().size());
            for (LayerMemoryPlan.StateTensor tensor : state.tensors()) {
                out.writeUTF(tensor.name());
                float[] values = state.read(tensor.name());
                out.writeInt(values.length);
                for (float v : values) {
                    out.writeFloat(v);
                }
            }
        }
    }

    private static void readStates(StateTransfer transfer, Reader in) throws IOException {
        List<StateTransfer.State> states = transfer.states();
        int count = in.readInt();
        if (count != states.size()) {
            throw new IOException(in.name() + " holds " + count + " state layers, the memory plans "
                    + states.size());
        }
        for (StateTransfer.State state : states) {
            in.expect(state.layer(), "state layer");
            int tensors = in.readInt();
            if (tensors != state.tensors().size()) {
                throw new IOException("state layer " + state.layer() + " holds " + tensors
                        + " tensors in the session, the memory plans " + state.tensors().size());
            }
            for (LayerMemoryPlan.StateTensor tensor : state.tensors()) {
                String name = in.text();
                if (!name.equals(tensor.name())) {
                    throw new IOException("state layer " + state.layer() + " holds tensor " + name
                            + " in the session where the memory plans " + tensor.name());
                }
                long planned = tensor.bytes() / Float.BYTES;
                int floats = in.readInt();
                if (floats != planned) {
                    throw new IOException("state tensor " + name + " of layer " + state.layer() + " holds "
                            + floats + " floats in the session, the memory plans " + planned);
                }
                float[] values = new float[floats];
                for (int i = 0; i < floats; i++) {
                    values[i] = in.readFloat();
                }
                state.write(name, values);
            }
        }
    }

    /**
     * The load-side reader: every value comes through here, so a mismatched count names what the file was saved
     * for and what the memory plans instead.
     */
    private static final class Reader {
        private final DataInputStream in;
        private final String source;

        Reader(DataInputStream in, String source) {
            this.in = in;
            this.source = source;
        }

        String name() {
            return source;
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

        void readFully(byte[] dst, int n) throws IOException {
            in.readFully(dst, 0, n);
        }

        void expect(int expected, String what) throws IOException {
            int v = in.readInt();
            if (v != expected) {
                throw new IOException(source + " was saved for " + what + " " + v + ", the memory plans " + expected);
            }
        }
    }
}
