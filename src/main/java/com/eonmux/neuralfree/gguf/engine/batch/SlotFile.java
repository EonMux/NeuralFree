package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.StateTransfer;

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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The slot file of a paged memory (llama.cpp's {@code llama_state_seq_save_file} / {@code _load_file} of one
 * sequence): the prefix-cache chain that holds a slot's history, as host copies of its blocks ({@link BlockImage}).
 * Format: the {@code NFSL} magic, a version, the block layout's identity ({@link BlockLayout#identity()}), the
 * chain's extra keys (adapter identity and names, cache salt), then per block its row count, whether it is whole,
 * the token and stored position of every row and, per cache layer, the rows' K and V bytes in the cache dtype and
 * their side planes (the per-(token, KV head) scales of a per-token-head cache). The bytes are the ones both
 * backends store, so a slot saved on the CPU restores on the device and back.
 *
 * <p>A read checks the identity and every block's shape before it answers, naming the field that differs, so a
 * file of another model, cache type or page size never reaches the memory.
 */
final class SlotFile {

    private static final int MAGIC = 0x4E46534C; // "NFSL"
    private static final int VERSION = 2;

    /** A read slot: the chain's extra keys and its blocks, root first. Immutable. */
    static final class Contents {
        final PrefixKey key;
        final List<BlockImage> blocks;

        Contents(PrefixKey key, List<BlockImage> blocks) {
            this.key = Objects.requireNonNull(key, "key");
            this.blocks = Collections.unmodifiableList(new ArrayList<>(blocks));
        }

        /** Rows the chain holds. */
        int tokens() {
            int n = 0;
            for (BlockImage b : blocks) {
                n += b.rows();
            }
            return n;
        }
    }

    private SlotFile() {
    }

    /** Writes the chain {@code blocks} under {@code key} to {@code out} (replaced when it exists). */
    static void write(Path out, BlockLayout layout, PrefixKey key, List<BlockImage> blocks) throws IOException {
        try (DataOutputStream s = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(out)))) {
            write(s, layout, key, blocks);
        }
    }

    /**
     * Writes the chain to an open stream (the format of {@link #write(Path, BlockLayout, PrefixKey, List)}),
     * leaving the stream open for a container: a KV transfer payload frames the same bytes.
     */
    static void write(OutputStream out, BlockLayout layout, PrefixKey key, List<BlockImage> blocks)
            throws IOException {
        DataOutputStream s = new DataOutputStream(out);
        s.writeInt(MAGIC);
        s.writeInt(VERSION);
        int[] identity = layout.identity();
        s.writeInt(identity.length);
        for (int v : identity) {
            s.writeInt(v);
        }
        writeText(s, key.adapters());
        writeText(s, key.loraName());
        writeText(s, key.salt());
        s.writeInt(blocks.size());
        for (BlockImage b : blocks) {
            writeBlock(s, b);
        }
        s.flush();
    }

    private static void writeBlock(DataOutputStream s, BlockImage b) throws IOException {
        s.writeInt(b.rows());
        s.writeBoolean(b.full());
        for (int t : b.tokens()) {
            s.writeInt(t);
        }
        for (int p : b.positions()) {
            s.writeInt(p);
        }
        for (int c = 0; c < b.layers(); c++) {
            s.write(b.keys(c));
            writeSideArrays(s, b.kSides(c));
            if (b.values(c) != null) {
                s.write(b.values(c));
                writeSideArrays(s, b.vSides(c));
            }
        }
    }

    private static void writeSideArrays(DataOutputStream s, byte[][] sides) throws IOException {
        for (byte[] side : sides) {
            s.write(side);
        }
    }

    /**
     * Reads the slot at {@code in} for {@code layout}.
     *
     * @throws IOException for an unreadable or truncated file, a foreign format, another layout's identity or a
     *                     malformed chain
     */
    static Contents read(Path in, BlockLayout layout) throws IOException {
        try (DataInputStream s = new DataInputStream(new BufferedInputStream(Files.newInputStream(in)))) {
            return read(s, in.toString(), layout);
        }
    }

    /**
     * Reads a chain from an open stream (the format of {@link #write(Path, BlockLayout, PrefixKey, List)}), leaving
     * the stream open for a container; {@code source} names the stream in every refusal.
     */
    static Contents read(InputStream in, String source, BlockLayout layout) throws IOException {
        DataInputStream s = new DataInputStream(in);
        try {
            if (s.readInt() != MAGIC) {
                throw new IOException(source + " is not a NeuralFree slot file");
            }
            int version = s.readInt();
            if (version != VERSION) {
                throw new IOException(source + " holds slot format " + version + ", this build reads " + VERSION);
            }
            checkIdentity(s, source, layout);
            String adapters = readText(s);
            String loraName = readText(s);
            String salt = readText(s);
            int count = s.readInt();
            if (count < 0) {
                throw new IOException(source + " holds " + count + " blocks");
            }
            List<BlockImage> blocks = new ArrayList<>(Math.min(count, 1 << 16));
            for (int i = 0; i < count; i++) {
                BlockImage b = readBlock(s, source, layout, i);
                if (!b.full() && i != count - 1) {
                    throw new IOException(source + " holds partial block " + i + " before the chain's end");
                }
                blocks.add(b);
            }
            return new Contents(PrefixKey.restored(adapters, loraName, salt), blocks);
        } catch (EOFException e) {
            throw new IOException(source + " ends before its slot does", e);
        }
    }

    private static void checkIdentity(DataInputStream s, String source, BlockLayout layout) throws IOException {
        int[] want = layout.identity();
        int n = s.readInt();
        if (n != want.length) {
            int layers = (n - 2) / 8;
            throw new IOException(source + " was saved for " + layers + " cache layers, the memory holds "
                    + layout.caches().size());
        }
        for (int i = 0; i < n; i++) {
            int v = s.readInt();
            if (v != want[i]) {
                throw new IOException(source + " was saved for " + layout.identityField(i) + " " + v
                        + ", the memory holds " + want[i]);
            }
        }
    }

    private static BlockImage readBlock(DataInputStream s, String source, BlockLayout layout, int index)
            throws IOException {
        int rows = s.readInt();
        boolean full = s.readBoolean();
        int pageSize = layout.pageSize();
        if (full ? rows != pageSize : rows < 1 || rows >= pageSize) {
            throw new IOException(source + " holds " + (full ? "whole" : "partial") + " block " + index + " of "
                    + rows + " rows for pages of " + pageSize);
        }
        int[] tokens = new int[rows];
        for (int r = 0; r < rows; r++) {
            tokens[r] = s.readInt();
            if (tokens[r] < 0) {
                throw new IOException(source + " holds token " + tokens[r] + " in block " + index);
            }
        }
        int[] positions = new int[rows];
        for (int r = 0; r < rows; r++) {
            positions[r] = s.readInt();
            if (positions[r] < 0 || (r > 0 && positions[r] < positions[r - 1])) {
                throw new IOException(source + " holds position " + positions[r] + " at row " + r + " of block "
                        + index);
            }
        }
        List<StateTransfer.Cache> caches = layout.caches();
        byte[][] k = new byte[caches.size()][];
        byte[][] v = new byte[caches.size()][];
        byte[][][] kSides = new byte[caches.size()][][];
        byte[][][] vSides = new byte[caches.size()][][];
        for (int c = 0; c < caches.size(); c++) {
            StateTransfer.Cache cache = caches.get(c);
            KvLayout l = cache.layout();
            k[c] = new byte[Math.toIntExact(rows * l.rowBytesK())];
            s.readFully(k[c]);
            kSides[c] = readSideArrays(s, cache, false, rows);
            if (!l.vAliasesK()) {
                v[c] = new byte[Math.toIntExact(rows * l.rowBytesV())];
                s.readFully(v[c]);
                vSides[c] = readSideArrays(s, cache, true, rows);
            }
        }
        return new BlockImage(tokens, positions, full, k, v, kSides, vSides);
    }

    private static byte[][] readSideArrays(DataInputStream s, StateTransfer.Cache cache, boolean v, int rows)
            throws IOException {
        List<com.eonmux.neuralfree.gguf.memory.KvDType.SidePlane> planes = cache.sidePlanes(v);
        byte[][] out = new byte[planes.size()][];
        for (int p = 0; p < planes.size(); p++) {
            long rowBytes = planes.get(p).rowBytesFor(v ? cache.layout().widthV() : cache.layout().widthK(),
                    cache.layout().nHeadKv());
            out[p] = new byte[Math.toIntExact(rows * rowBytes)];
            s.readFully(out[p]);
        }
        return out;
    }

    private static void writeText(DataOutputStream s, String text) throws IOException {
        s.writeBoolean(text != null);
        if (text != null) {
            s.writeUTF(text);
        }
    }

    private static String readText(DataInputStream s) throws IOException {
        return s.readBoolean() ? s.readUTF() : null;
    }
}
