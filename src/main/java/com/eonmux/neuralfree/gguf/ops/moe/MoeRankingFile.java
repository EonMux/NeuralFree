package com.eonmux.neuralfree.gguf.ops.moe;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * The {@code moe-cache-ranking.bin} file: an expert ranking, hottest first, in Strata's format
 * ({@code "STRP"}, then version, layers, experts, slots, pairs as little-endian {@code uint32}; version 1 pairs are
 * (layer {@code uint16}, expert {@code uint16}, heat 0), version 2 pairs carry the counted routed rows as
 * (layer {@code uint16}, expert {@code uint16}, heat {@code uint32})). {@link #read} keeps the file's order — the
 * hottest first, which the preload follows — and {@link #write} sorts by heat, then layer, then expert, the order
 * Avalon's profile save uses, and replaces the file atomically through a {@code .tmp} sibling.
 *
 * <p>On read, bytes after the declared pairs are ignored, as Avalon's {@code moe_cache_ranking_read} reads only the
 * declared pairs. On write, the header's layers count is the entries' largest layer plus one and its experts count
 * the model's (or the entries' largest expert plus one); the cache's preload compares both against the model, the
 * profile save's merge the experts count. Static.</p>
 */
public final class MoeRankingFile {

    /** File magic. */
    public static final String MAGIC = "STRP";
    /** Version 1: Strata's {@code data/expert-profile.bin}, pairs without heats. */
    public static final int VERSION_1 = 1;
    /** Version 2: pairs carry the routed rows counted for each expert. */
    public static final int VERSION_2 = 2;

    private static final int HEADER_INTS = 5;
    private static final int HEADER_BYTES = 4 + 4 * HEADER_INTS;
    private static final int V1_ENTRY_BYTES = 4;
    private static final int V2_ENTRY_BYTES = 8;

    private MoeRankingFile() {
    }

    /**
     * One read ranking: its pairs in file order and the header's layers and experts counts. Immutable.
     */
    public static final class Parsed {
        private final List<RankedExpert> entries;
        private final int nLayers;
        private final int nExperts;

        Parsed(List<RankedExpert> entries, int nLayers, int nExperts) {
            this.entries = Collections.unmodifiableList(entries);
            this.nLayers = nLayers;
            this.nExperts = nExperts;
        }

        /** The ranking, hottest first as the file holds it. */
        public List<RankedExpert> entries() {
            return entries;
        }

        /** The header's layers count. */
        public int nLayers() {
            return nLayers;
        }

        /** The header's experts count per layer. */
        public int nExperts() {
            return nExperts;
        }
    }

    /**
     * Reads a ranking of either version.
     *
     * @throws IOException when the file cannot be read, is not an expert ranking (naming the magic or the
     *                     version) or is truncated (naming the missing bytes)
     */
    public static Parsed read(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        byte[] bytes;
        try {
            bytes = Files.readAllBytes(file);
        } catch (IOException e) {
            throw new IOException("expert ranking " + file + " cannot be read: " + e.getMessage(), e);
        }
        if (bytes.length < HEADER_BYTES) {
            throw new IOException("expert ranking " + file + " is not an expert ranking: " + bytes.length
                    + " bytes, needs at least " + HEADER_BYTES);
        }
        ByteBuffer in = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        byte[] magic = new byte[4];
        in.get(magic);
        if (!MAGIC.equals(new String(magic, StandardCharsets.US_ASCII))) {
            throw new IOException("expert ranking " + file + " is not an expert ranking: magic '"
                    + new String(magic, StandardCharsets.US_ASCII) + "', needs '" + MAGIC + "'");
        }
        int version = in.getInt();
        if (version != VERSION_1 && version != VERSION_2) {
            throw new IOException("expert ranking " + file + " has unsupported version " + version + ", needs "
                    + VERSION_1 + " or " + VERSION_2);
        }
        int nLayers = in.getInt();
        int nExperts = in.getInt();
        in.getInt();   // slots: informational, Avalon writes the pair count
        int pairs = in.getInt();
        if (pairs < 0) {
            throw new IOException("expert ranking " + file + " declares " + pairs + " pairs");
        }
        int entryBytes = version == VERSION_1 ? V1_ENTRY_BYTES : V2_ENTRY_BYTES;
        long needed = HEADER_BYTES + (long) pairs * entryBytes;
        if (bytes.length < needed) {
            throw new IOException("expert ranking " + file + " is truncated: " + bytes.length + " bytes for "
                    + pairs + " version " + version + " pairs, needs " + needed);
        }
        List<RankedExpert> entries = new ArrayList<>(pairs);
        for (int i = 0; i < pairs; i++) {
            int layer = in.getShort() & 0xFFFF;
            int expert = in.getShort() & 0xFFFF;
            long heat = version == VERSION_1 ? 0L : in.getInt() & 0xFFFFFFFFL;
            entries.add(new RankedExpert(layer, expert, heat));
        }
        return new Parsed(entries, nLayers, nExperts);
    }

    /**
     * Writes a version 2 ranking, sorted by heat, then layer, then expert, replacing {@code file} atomically
     * through {@code file.tmp}; the header names the entries' largest layer and expert plus one. An empty list
     * writes the header alone.
     *
     * @throws IOException when the file or its replacement cannot be written
     */
    public static void write(Path file, List<RankedExpert> entries) throws IOException {
        write(file, entries, maxPart(entries, RankedExpert::expert) + 1);
    }

    /**
     * Writes a version 2 ranking like {@link #write(Path, List)}, with the header naming {@code nExperts} experts
     * per layer — the model's expert count, which the profile save's shape check compares (Avalon writes its
     * uniform expert count there, not the entries' maximum).
     *
     * @throws IOException                     when the file or its replacement cannot be written
     * @throws IllegalArgumentException when {@code nExperts} does not hold the largest entry's expert
     */
    public static void write(Path file, List<RankedExpert> entries, int nExperts) throws IOException {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(entries, "entries");
        if (nExperts <= maxPart(entries, RankedExpert::expert)) {
            throw new IllegalArgumentException("the header expert count " + nExperts + " must hold the largest"
                    + " entry's expert " + maxPart(entries, RankedExpert::expert));
        }
        List<RankedExpert> sorted = sorted(entries);
        ByteBuffer out = ByteBuffer.allocate(HEADER_BYTES + sorted.size() * V2_ENTRY_BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        out.put(MAGIC.getBytes(StandardCharsets.US_ASCII));
        out.putInt(VERSION_2);
        out.putInt(maxPart(sorted, RankedExpert::layer) + 1);
        out.putInt(nExperts);
        out.putInt(sorted.size());   // slots
        out.putInt(sorted.size());   // pairs
        for (RankedExpert e : sorted) {
            out.putShort((short) e.layer());
            out.putShort((short) e.expert());
            out.putInt((int) e.heat());
        }
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        Files.write(tmp, out.array());
        try {
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new IOException("expert ranking " + tmp + " cannot replace " + file + ": " + e.getMessage(), e);
        }
    }

    /** The entries by heat, then layer, then expert — Avalon's profile save order (hottest first). */
    public static List<RankedExpert> sorted(List<RankedExpert> entries) {
        List<RankedExpert> out = new ArrayList<>(entries);
        out.sort(Comparator.comparingLong(RankedExpert::heat).reversed()
                .thenComparingInt(RankedExpert::layer)
                .thenComparingInt(RankedExpert::expert));
        return out;
    }

    private static int maxPart(List<RankedExpert> entries, java.util.function.ToIntFunction<RankedExpert> part) {
        int max = -1;
        for (RankedExpert e : entries) {
            max = Math.max(max, part.applyAsInt(e));
        }
        return max;
    }
}
