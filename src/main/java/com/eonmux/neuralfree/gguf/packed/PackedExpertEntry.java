package com.eonmux.neuralfree.gguf.packed;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The packed-expert layout of one MoE layer, as Avalon's {@code pack-experts.py} writes it and its
 * loader reads it (Avalon/src/llama-model-loader.cpp:718-767): the layer's three routed-expert
 * tensors are stored one after the other, expert by expert, in one I8 tensor
 * {@code blk.N.ffn_exps_packed} of shape {@code [stride, n_expert]}. One expert's entry holds, per
 * role in the order up, gate, down: the role's {@code ne[0] x ne[1]} data at a 256-byte-aligned
 * offset, followed by the role's pad ({@link #rolePad}, what MMQ reads past a row when {@code ne[0]}
 * is not a multiple of 512; the pad repeats the next expert's first bytes of the same role, the
 * last expert its own). The key {@code avalon.experts_packed.blk.N} describes the layout as an
 * array of {@value #METADATA_VALUES} UINT64 values: stride, expert count, then per role the ggml
 * type id, {@code ne[0]}, {@code ne[1]} and the byte offset of the role's data in the entry.
 * Immutable.
 */
public final class PackedExpertEntry {

    /** Alignment of every role's data inside an entry, in bytes ({@code pack-experts.py ALIGN}). */
    public static final int ALIGN = 256;

    /** Values of one {@code avalon.experts_packed.blk.N} metadata array. */
    public static final int METADATA_VALUES = 14;

    /** Metadata key prefix; the key of layer {@code N} is the prefix plus {@code blk.N}. */
    public static final String KEY_PREFIX = "avalon.experts_packed.";

    /** Tensor suffix; the packed tensor of layer {@code N} is {@code blk.N.<suffix>}. */
    public static final String TENSOR_SUFFIX = "ffn_exps_packed";

    /** Role suffixes in storage order ({@code pack-experts.py ROLES}). */
    static final String[] ROLE_SUFFIXES = {"up", "gate", "down"};

    private final int layer;
    private final long stride;
    private final long nExpert;
    private final List<Role> roles;

    private PackedExpertEntry(int layer, long stride, long nExpert, List<Role> roles) {
        this.layer = layer;
        this.stride = stride;
        this.nExpert = nExpert;
        this.roles = roles;
    }

    /** One role of an entry: its tensor geometry and its place in the entry. Immutable. */
    public static final class Role {
        private final String name;
        private final GGMLType type;
        private final long ne0;
        private final long ne1;
        private final long offset;
        private final long rowBytes;
        private final long pad;

        Role(String name, GGMLType type, long ne0, long ne1, long offset) {
            this.name = name;
            this.type = type;
            this.ne0 = ne0;
            this.ne1 = ne1;
            this.offset = offset;
            this.rowBytes = BlockLayout.of(type).rowBytes(ne0);
            this.pad = rolePad(type, ne0);
        }

        /** Role name: {@code up}, {@code gate} or {@code down}. */
        public String name() {
            return name;
        }

        /** Storage type of the role's expert tensor. */
        public GGMLType type() {
            return type;
        }

        /** {@code ne[0]} of the role's expert tensor (its row length in elements). */
        public long ne0() {
            return ne0;
        }

        /** {@code ne[1]} of the role's expert tensor (its row count). */
        public long ne1() {
            return ne1;
        }

        /** Byte offset of the role's data inside one expert's entry. */
        public long offset() {
            return offset;
        }

        /** Packed bytes of one row ({@code ne[0]} elements of the type). */
        public long rowBytes() {
            return rowBytes;
        }

        /** Bytes of one expert's role data ({@code ne[1]} rows). */
        public long entryBytes() {
            return rowBytes * ne1;
        }

        /** Pad bytes that follow the role's data in the entry ({@link PackedExpertEntry#rolePad}). */
        public long pad() {
            return pad;
        }

        /** Standard tensor name of the role: {@code blk.<layer>.ffn_<name>_exps.weight}. */
        public String standardTensorName(int layer) {
            return "blk." + layer + ".ffn_" + name + "_exps.weight";
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof Role)) {
                return false;
            }
            Role other = (Role) o;
            return name.equals(other.name) && type == other.type && ne0 == other.ne0 && ne1 == other.ne1
                    && offset == other.offset;
        }

        @Override
        public int hashCode() {
            return Objects.hash(name, type, ne0, ne1, offset);
        }

        @Override
        public String toString() {
            return "ffn_" + name + "_exps " + type + " [" + ne0 + ", " + ne1 + "] @" + offset
                    + " pad " + pad;
        }
    }

    /** The layer the entry belongs to. */
    public int layer() {
        return layer;
    }

    /** Bytes of one expert's entry; the step from one expert's role data to the next expert's. */
    public long stride() {
        return stride;
    }

    /** Experts per entry ({@code ne[2]} of every role). */
    public long nExpert() {
        return nExpert;
    }

    /** The roles in storage order (up, gate, down); unmodifiable. */
    public List<Role> roles() {
        return roles;
    }

    /** Name of the layer's packed tensor: {@code blk.<layer>.ffn_exps_packed}. */
    public String packedTensorName() {
        return "blk." + layer + "." + TENSOR_SUFFIX;
    }

    /** Metadata key of the layer: {@code avalon.experts_packed.blk.<layer>}. */
    public String metadataKey() {
        return KEY_PREFIX + "blk." + layer;
    }

    /** Role whose standard tensor is {@code blk.<layer>.<suffix>}, or null when {@code suffix} is none. */
    public Role roleForSuffix(String suffix) {
        for (Role r : roles) {
            if (("ffn_" + r.name + "_exps").equals(suffix)) {
                return r;
            }
        }
        return null;
    }

    /**
     * The {@code avalon.experts_packed.blk.<layer>} values of the entry: stride, expert count, then
     * per role the ggml type id, {@code ne[0]}, {@code ne[1]} and the byte offset, in storage order.
     */
    public long[] metadataValues() {
        long[] v = new long[METADATA_VALUES];
        v[0] = stride;
        v[1] = nExpert;
        for (int r = 0; r < roles.size(); r++) {
            Role role = roles.get(r);
            v[2 + 4 * r] = typeValue(role.type);
            v[3 + 4 * r] = role.ne0;
            v[4 + 4 * r] = role.ne1;
            v[5 + 4 * r] = role.offset;
        }
        return v;
    }

    /**
     * The entry described by one {@code avalon.experts_packed.blk.<layer>} metadata array, checked
     * against the {@code blk.<layer>.ffn_exps_packed} tensor it describes, with the checks of
     * Avalon's loader (llama-model-loader.cpp:742-761): the byte size of the packed tensor must be
     * {@code stride * n_expert}, and every role's data must fit inside the entry.
     *
     * @param layer        the layer the key and tensor carry
     * @param values       the decoded UINT64 values of the key, in file order
     * @param packedTensor the layer's packed tensor description
     * @throws IOException with the key or tensor named when the array or the layout is malformed
     */
    public static PackedExpertEntry fromMetadata(int layer, long[] values, GGUFTensorInfo packedTensor)
            throws IOException {
        String key = KEY_PREFIX + "blk." + layer;
        if (values.length != METADATA_VALUES) {
            throw new IOException("packed expert layer " + layer + ": key " + key + " holds " + values.length
                    + " values, expected " + METADATA_VALUES);
        }
        long stride = values[0];
        long nExpert = values[1];
        if (stride <= 0 || nExpert <= 0 || nExpert > Integer.MAX_VALUE) {
            throw new IOException("packed expert layer " + layer + ": key " + key + " describes stride "
                    + stride + " and " + nExpert + " experts; both must be positive and the expert count"
                    + " must fit an int");
        }
        if (stride > Long.MAX_VALUE / nExpert || packedTensor.sizeInBytes() != stride * nExpert) {
            throw new IOException("packed expert layer " + layer + ": tensor " + packedTensor.name() + " holds "
                    + packedTensor.sizeInBytes() + " bytes, expected stride " + stride + " x " + nExpert
                    + " experts");
        }
        List<Role> roles = parseRoles(layer, key, values, stride);
        return new PackedExpertEntry(layer, stride, nExpert, roles);
    }

    /**
     * The roles of one {@code avalon.experts_packed} array: per role in storage order the type,
     * {@code ne[0]}, {@code ne[1]} and byte offset, each checked against the entry stride.
     */
    private static List<Role> parseRoles(int layer, String key, long[] values, long stride) throws IOException {
        Role[] parsed = new Role[ROLE_SUFFIXES.length];
        for (int r = 0; r < parsed.length; r++) {
            GGMLType type = typeOf(layer, key, values[2 + 4 * r]);
            long ne0 = values[3 + 4 * r];
            long ne1 = values[4 + 4 * r];
            long offset = values[5 + 4 * r];
            BlockLayout layout = BlockLayout.of(type);
            if (ne0 <= 0 || ne1 <= 0 || ne0 % layout.blockElements() != 0) {
                throw new IOException("packed expert layer " + layer + ": key " + key + " role " + ROLE_SUFFIXES[r]
                        + " has ne0 " + ne0 + ", not a positive multiple of the " + type + " block size "
                        + layout.blockElements());
            }
            long bytes = ne1 > Long.MAX_VALUE / layout.rowBytes(ne0) ? Long.MAX_VALUE : layout.rowBytes(ne0) * ne1;
            if (offset < 0 || bytes > stride || offset > stride - bytes) {
                throw new IOException("packed expert layer " + layer + ": key " + key + " role " + ROLE_SUFFIXES[r]
                        + " data of " + bytes + " bytes at offset " + offset + " does not fit the entry stride "
                        + stride);
            }
            if (bytes > Integer.MAX_VALUE - 8L) {
                throw new IOException("packed expert layer " + layer + ": key " + key + " role " + ROLE_SUFFIXES[r]
                        + " holds " + bytes + " bytes per expert, more than one region serves");
            }
            parsed[r] = new Role(ROLE_SUFFIXES[r], type, ne0, ne1, offset);
        }
        return Collections.unmodifiableList(Arrays.asList(parsed));
    }

    /**
     * The entry that packs the three routed-expert tensors of one layer, laid out as
     * {@code pack-experts.py} lays it out: per role in storage order, its per-expert bytes at the
     * next 256-byte-aligned offset, followed by the role's pad.
     *
     * @param layer the layer the three tensors belong to
     * @param experts the three expert tensors by role, in storage order (up, gate, down); each must
     *                be a 3-D {@code [ne0, ne1, n_expert]} tensor with the same expert count
     * @throws IOException with the layer or tensor named when a tensor is not a 3-D expert tensor
     *                     or the expert counts disagree
     */
    public static PackedExpertEntry fromStandardTensors(int layer, List<GGUFTensorInfo> experts)
            throws IOException {
        if (experts.size() != ROLE_SUFFIXES.length) {
            throw new IOException("packed expert layer " + layer + ": needs " + ROLE_SUFFIXES.length
                    + " expert tensors, got " + experts.size());
        }
        long nExpert = 0;
        for (int r = 0; r < experts.size(); r++) {
            GGUFTensorInfo t = experts.get(r);
            if (t.rank() != 3) {
                throw new IOException("packed expert layer " + layer + ": tensor " + t.name() + " has "
                        + t.rank() + " dimensions, expected the 3-D [ne0, ne1, n_expert] shape");
            }
            if (t.dim(2) <= 0) {
                throw new IOException("packed expert layer " + layer + ": tensor " + t.name() + " holds "
                        + t.dim(2) + " experts");
            }
            if (nExpert != 0 && nExpert != t.dim(2)) {
                throw new IOException("packed expert layer " + layer + ": tensor " + t.name() + " holds "
                        + t.dim(2) + " experts, but an earlier role holds " + nExpert);
            }
            nExpert = t.dim(2);
            BlockLayout layout = BlockLayout.of(t.type());
            if (t.dim(0) <= 0 || t.dim(1) <= 0 || t.dim(0) % layout.blockElements() != 0) {
                throw new IOException("packed expert layer " + layer + ": tensor " + t.name() + " of type "
                        + t.type() + " has ne0 " + t.dim(0) + ", not a positive multiple of the block size "
                        + layout.blockElements());
            }
        }
        Role[] parsed = new Role[ROLE_SUFFIXES.length];
        long offset = 0;
        for (int r = 0; r < experts.size(); r++) {
            GGUFTensorInfo t = experts.get(r);
            Role role = new Role(ROLE_SUFFIXES[r], t.type(), t.dim(0), t.dim(1), offset);
            parsed[r] = role;
            offset = alignUp(offset + role.entryBytes(), ALIGN) + role.pad;
        }
        return new PackedExpertEntry(layer, offset, nExpert, Collections.unmodifiableList(Arrays.asList(parsed)));
    }

    /**
     * The pad bytes a role's entry data is followed by ({@code pack-experts.py role_pad}, the MoE
     * cache's {@code moe_cache_role_pad}): what MMQ reads past one row when {@code ne[0]} is
     * rounded up to 512 elements, in whole blocks, aligned up to {@value #ALIGN} bytes.
     */
    public static long rolePad(GGMLType type, long ne0) {
        BlockLayout layout = BlockLayout.of(type);
        long over = (ne0 + 511) / 512 * 512 - ne0;
        long bytes = over / layout.blockElements() * layout.blockBytes();
        return alignUp(bytes, ALIGN);
    }

    private static long alignUp(long v, long alignment) {
        return (v + alignment - 1) / alignment * alignment;
    }

    private static GGMLType typeOf(int layer, String key, long id) throws IOException {
        if (id < 0 || id > Integer.MAX_VALUE) {
            throw new IOException("packed expert layer " + layer + ": key " + key + " holds type id " + id
                    + ", outside the ggml type range");
        }
        GGMLType type = GGMLType.fromId((int) id);
        if (!BlockLayout.isDefined(type)) {
            throw new IOException("packed expert layer " + layer + ": key " + key + " holds type id " + id
                    + ", which is unknown or was removed from ggml");
        }
        return type;
    }

    private static long typeValue(GGMLType type) {
        return type.getId() & 0xFFFFFFFFL;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof PackedExpertEntry)) {
            return false;
        }
        PackedExpertEntry other = (PackedExpertEntry) o;
        return layer == other.layer && stride == other.stride && nExpert == other.nExpert
                && roles.equals(other.roles);
    }

    @Override
    public int hashCode() {
        return Objects.hash(layer, stride, nExpert, roles);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("blk.").append(layer).append('.').append(TENSOR_SUFFIX)
                .append(" stride ").append(stride).append(" x ").append(nExpert).append(" experts:");
        for (Role r : roles) {
            sb.append(' ').append(r);
        }
        return sb.toString();
    }
}
