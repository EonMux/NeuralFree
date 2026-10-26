package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The row parts of a {@link com.eonmux.neuralfree.gguf.memory.MemoryPlan}'s KV groups (the row-part rule, prefix
 * cache design T-8): every datum a group's cell index addresses is part of its row, and every copy, move, seal,
 * restore or comparison of the rows iterates these parts, never only K and V. The parts are derived from two
 * declarations, never handed-listed:
 * <ul>
 *   <li>every {@code LayerMemoryPlan} buffer declares its {@link LayerMemoryPlan.Addressing addressing}: a KV group's
 *       cell index ({@code CELL}: the K and V or latent rows, the cross-attention K/V, the DSA indexer keys and the
 *       MSA keys), a recurrent slot ({@code SLOT}) or neither ({@code NEITHER}: the DSV4 compressed caches);</li>
 *   <li>every {@link KvDType} declares the per-row planes it stores outside the row ({@link KvDType#sidePlanes()},
 *       empty for every type of today's set).</li>
 * </ul>
 * {@code kv_pos} is never a copied row part: an operation writes it from the row's position. Stateless and
 * thread-safe.
 */
public final class RowParts {

    /** One row part of a KV group: the buffer's rows of {@code width} values of {@code dtype}. Immutable. */
    public static final class RowPart {

        private final int group;
        private final String buffer;
        private final KvDType dtype;
        private final int width;

        private RowPart(int group, String buffer, KvDType dtype, int width) {
            this.group = group;
            this.buffer = buffer;
            this.dtype = dtype;
            this.width = width;
        }

        /** Index of the KV group the rows belong to. */
        public int group() { return group; }

        /** Buffer name: {@code K}, {@code V}, a side plane's name, the indexer or MSA keys or the cross K/V. */
        public String buffer() { return buffer; }
        public KvDType dtype() { return dtype; }
        /** Values per row ({@link KvDType#rowBytes(long)} byte-sizes one row of a part). */
        public int width() { return width; }
        /** Bytes of one row of the part. */
        public long rowBytes() { return dtype.rowBytes(width); }

        @Override
        public String toString() {
            return buffer + "[group=" + group + ", " + width + " " + dtype + ", " + rowBytes() + " B/row]";
        }
    }

    /**
     * The row parts of every KV group of {@code plan}, group order preserved and parts in a fixed order per group:
     * the K or latent rows, the V rows (0 when V aliases K), then the side planes of the declared dtypes, the
     * indexer keys, the MSA keys and the cross-attention K/V.
     *
     * @throws IllegalArgumentException when a layer's own caches declare no group while they address one
     */
    public static List<RowPart> of(MemoryPlan plan) {
        Objects.requireNonNull(plan, "plan");
        List<RowPart> out = new ArrayList<>();
        for (MemoryPlan.KvGroup g : plan.groups()) {
            out.addAll(of(plan, g.index()));
        }
        return out;
    }

    /**
     * The row parts of KV group {@code g}: the K/V or latent rows of every layer owning a cache in the group, the
     * group's side caches, the cross-attention K/V of the group's layers, and the planes the layer dtypes declare
     * outside the row.
     *
     * @throws IllegalArgumentException when {@code g} is outside the plan
     */
    public static List<RowPart> of(MemoryPlan plan, int g) {
        Objects.requireNonNull(plan, "plan");
        if (g < 0 || g >= plan.groups().size()) {
            throw new IllegalArgumentException("KV group " + g + " outside the plan's " + plan.groups().size()
                    + " groups");
        }
        List<RowPart> out = new ArrayList<>();
        for (LayerMemoryPlan l : plan.layers()) {
            if (l.kvGroup() != g || l.kv() == null) {
                continue;
            }
            addBuffer(out, g, "K", l.kv().dtypeK(), l.kv().widthK());
            addSidePlanes(out, g, l.kv().dtypeK());
            if (!l.kv().vAliasesK()) {
                addBuffer(out, g, "V", l.kv().dtypeV(), l.kv().widthV());
                addSidePlanes(out, g, l.kv().dtypeV());
            }
            if (l.indexer() != null && l.indexer().addressing() == LayerMemoryPlan.Addressing.CELL) {
                addBuffer(out, g, "indexer", l.indexer().dtype(), l.indexer().width());
            }
            if (l.msa() != null && l.msa().addressing() == LayerMemoryPlan.Addressing.CELL) {
                addBuffer(out, g, "msa", l.msa().dtype(), l.msa().width());
            }
            if (l.crossKv() != null) {
                addBuffer(out, g, "crossK", l.crossKv().dtypeK(), l.crossKv().widthK());
                if (!l.crossKv().vAliasesK()) {
                    addBuffer(out, g, "crossV", l.crossKv().dtypeV(), l.crossKv().widthV());
                }
            }
        }
        return out;
    }

    /**
     * True when {@link #of(MemoryPlan)} finds any row part: true for every plan whose layers own a cache.
     */
    public static boolean hasRowParts(MemoryPlan plan) {
        return !of(plan).isEmpty();
    }

    private static void addBuffer(List<RowPart> out, int group, String name, KvDType dtype, int width) {
        if (width <= 0) {
            throw new IllegalArgumentException("row part " + name + " of group " + group + " has width " + width);
        }
        out.add(new RowPart(group, name, dtype, width));
    }

    private static void addSidePlanes(List<RowPart> out, int group, KvDType dtype) {
        int i = 0;
        for (KvDType.SidePlane plane : dtype.sidePlanes()) {
            out.add(new RowPart(group, "side:" + plane.name() + (i++ == 0 ? "" : ":" + i), plane.dtype(),
                    plane.width()));
        }
    }

    private RowParts() {
    }
}
