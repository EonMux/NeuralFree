package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Immutable reflection result of one compute SPIR-V module: entry point,
 * storage-buffer bindings with merged access flags, push-constant members,
 * local size (fixed or by SpecId), specialisation defaults and capabilities.
 *
 * <p>Bindings are unique by number. Aliased typed views of one binding (for example 8-, 16- and
 * 32-bit views of a weight buffer) are merged into one {@link BindingInfo} by
 * {@link SpirvReflector}: the binding is readable when any view is readable and writable when any
 * view is writable. Binding numbers may skip a binding the optimiser stripped; the argument of
 * such a binding is still counted by the build-time
 * {@link com.eonmux.neuralfree.acceleration.backend.KernelSignature}.</p>
 */
public final class SpirvModuleInfo {

    /** SPIR-V capability {@code Float16}. */
    public static final int CAP_FLOAT16 = 9;
    /** SPIR-V capability {@code Float64}. */
    public static final int CAP_FLOAT64 = 10;
    /** SPIR-V capability {@code Int64}. */
    public static final int CAP_INT64 = 11;
    /** SPIR-V capability {@code Int16}. */
    public static final int CAP_INT16 = 22;
    /** SPIR-V capability {@code Int8}. */
    public static final int CAP_INT8 = 39;
    /** SPIR-V capability {@code GroupNonUniformArithmetic}. */
    public static final int CAP_GROUP_NON_UNIFORM_ARITHMETIC = 63;
    /** SPIR-V capability {@code StorageBuffer16BitAccess}. */
    public static final int CAP_STORAGE_BUFFER_16BIT_ACCESS = 4433;
    /** SPIR-V capability {@code StorageBuffer8BitAccess}. */
    public static final int CAP_STORAGE_BUFFER_8BIT_ACCESS = 4448;

    /** Byte size of a 32-bit specialisation constant. */
    public static final int SPEC_SIZE_32 = 4;
    /** Byte size of a 64-bit specialisation constant. */
    public static final int SPEC_SIZE_64 = 8;

    /** One storage-buffer binding of descriptor set 0; aliased views of a binding are merged. */
    public static final class BindingInfo {
        private final int binding;
        private final boolean readable;
        private final boolean writable;

        public BindingInfo(int binding, boolean readable, boolean writable) {
            if (binding < 0) {
                throw new IllegalArgumentException("binding number must not be negative: " + binding);
            }
            this.binding = binding;
            this.readable = readable;
            this.writable = writable;
        }

        public int binding() {
            return binding;
        }

        /** False only when every view of the binding is {@code NonReadable} ({@code writeonly}). */
        public boolean readable() {
            return readable;
        }

        /** False only when every view of the binding is {@code NonWritable} ({@code readonly}). */
        public boolean writable() {
            return writable;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof BindingInfo)) {
                return false;
            }
            BindingInfo that = (BindingInfo) o;
            return binding == that.binding && readable == that.readable && writable == that.writable;
        }

        @Override
        public int hashCode() {
            return (binding * 31 + (readable ? 1 : 0)) * 31 + (writable ? 1 : 0);
        }

        @Override
        public String toString() {
            return "binding " + binding + (readable ? (writable ? " (read-write)" : " (read-only)") : " (write-only)");
        }
    }

    /** One member of the push-constant block. */
    public static final class PushMember {

        /** Member type. */
        public enum Kind {
            I32, U32, F32, I64, F64, BYTES
        }

        private final int offset;
        private final int size;
        private final Kind kind;

        public PushMember(int offset, int size, Kind kind) {
            if (offset < 0 || size <= 0) {
                throw new IllegalArgumentException("invalid push-constant member: offset " + offset + ", size " + size);
            }
            this.offset = offset;
            this.size = size;
            this.kind = Objects.requireNonNull(kind, "kind");
            int expected = expectedSize(kind);
            if (expected > 0 && size != expected) {
                throw new IllegalArgumentException("push-constant member of kind " + kind + " must be " + expected
                        + " bytes, got " + size);
            }
        }

        public int offset() {
            return offset;
        }

        public int size() {
            return size;
        }

        public Kind kind() {
            return kind;
        }

        /** Short type name for messages, for example {@code int32}. */
        public String typeName() {
            switch (kind) {
                case I32:
                    return "int32";
                case U32:
                    return "uint32";
                case F32:
                    return "float32";
                case I64:
                    return "int64";
                case F64:
                    return "float64";
                default:
                    return size + " bytes";
            }
        }

        private static int expectedSize(Kind kind) {
            switch (kind) {
                case I32:
                case U32:
                case F32:
                    return 4;
                case I64:
                case F64:
                    return 8;
                default:
                    return 0;
            }
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof PushMember)) {
                return false;
            }
            PushMember that = (PushMember) o;
            return offset == that.offset && size == that.size && kind == that.kind;
        }

        @Override
        public int hashCode() {
            return (offset * 31 + size) * 31 + kind.hashCode();
        }

        @Override
        public String toString() {
            return typeName() + " at offset " + offset;
        }
    }

    private final String entryPoint;
    private final List<BindingInfo> bindings;
    private final BindingInfo[] bindingByNumber;
    private final List<PushMember> pushMembers;
    private final int pushConstantSize;
    private final int[] localSize;
    private final int[] localSizeSpecIds;
    private final Map<Integer, Integer> specDefaults;
    private final Map<Integer, Integer> specSizes;
    private final Set<Integer> capabilities;

    /**
     * Same as the full constructor with every specialisation constant 32 bits wide.
     */
    public SpirvModuleInfo(String entryPoint, List<BindingInfo> bindings, List<PushMember> pushMembers,
                           int pushConstantSize, int[] localSize, int[] localSizeSpecIds,
                           Map<Integer, Integer> specDefaults, Set<Integer> capabilities) {
        this(entryPoint, bindings, pushMembers, pushConstantSize, localSize, localSizeSpecIds, specDefaults,
                Collections.<Integer, Integer>emptyMap(), capabilities);
    }

    /**
     * @param localSize        fixed local size per dimension (the default where specialisable)
     * @param localSizeSpecIds SpecId per dimension, or -1 where the size is fixed
     * @param bindings         storage-buffer bindings of descriptor set 0; numbers are unique but may
     *                         skip a binding the optimiser stripped, whose argument the build-time
     *                         {@link com.eonmux.neuralfree.acceleration.backend.KernelSignature} still counts
     * @param specDefaults     default value of each declared SpecId (64-bit constants: the low 32 bits)
     * @param specSizes        byte size of each SpecId that is not 32 bits wide ({@link #SPEC_SIZE_64});
     *                         SpecIds absent from this map are 32 bits wide
     * @throws UnsupportedKernelException when a binding number repeats
     * @throws IllegalArgumentException   when the local size or push layout is malformed
     */
    public SpirvModuleInfo(String entryPoint, List<BindingInfo> bindings, List<PushMember> pushMembers,
                           int pushConstantSize, int[] localSize, int[] localSizeSpecIds,
                           Map<Integer, Integer> specDefaults, Map<Integer, Integer> specSizes,
                           Set<Integer> capabilities) {
        if (entryPoint == null || entryPoint.isEmpty()) {
            throw new IllegalArgumentException("entry point name must not be empty");
        }
        Objects.requireNonNull(bindings, "bindings");
        Objects.requireNonNull(pushMembers, "pushMembers");
        Objects.requireNonNull(specDefaults, "specDefaults");
        Objects.requireNonNull(specSizes, "specSizes");
        Objects.requireNonNull(capabilities, "capabilities");
        this.entryPoint = entryPoint;

        List<BindingInfo> sortedBindings = new ArrayList<>(bindings);
        sortedBindings.sort(Comparator.comparingInt(BindingInfo::binding));
        int slots = sortedBindings.isEmpty() ? 0 : sortedBindings.get(sortedBindings.size() - 1).binding() + 1;
        BindingInfo[] byNumber = new BindingInfo[slots];
        for (BindingInfo b : sortedBindings) {
            if (byNumber[b.binding()] != null) {
                throw new UnsupportedKernelException("SPIR-V module " + entryPoint + " declares binding "
                        + b.binding() + " twice after merging aliased views");
            }
            byNumber[b.binding()] = b;
        }
        this.bindings = Collections.unmodifiableList(sortedBindings);
        this.bindingByNumber = byNumber;

        List<PushMember> sortedMembers = new ArrayList<>(pushMembers);
        sortedMembers.sort(Comparator.comparingInt(PushMember::offset));
        int end = 0;
        for (PushMember m : sortedMembers) {
            if (m.offset() < end) {
                throw new IllegalArgumentException("push-constant member at offset " + m.offset()
                        + " overlaps the previous member ending at " + end);
            }
            end = m.offset() + m.size();
        }
        if (pushConstantSize < end || pushConstantSize % 4 != 0) {
            throw new IllegalArgumentException("push-constant size " + pushConstantSize
                    + " must be a multiple of 4 covering every member (last member ends at " + end + ")");
        }
        this.pushMembers = Collections.unmodifiableList(sortedMembers);
        this.pushConstantSize = pushConstantSize;

        if (localSize == null || localSize.length != 3 || localSizeSpecIds == null || localSizeSpecIds.length != 3) {
            throw new IllegalArgumentException("local size and its SpecIds must have three entries");
        }
        for (int d = 0; d < 3; d++) {
            if (localSize[d] < 1) {
                throw new IllegalArgumentException("local size " + Arrays.toString(localSize) + " must be positive");
            }
            if (localSizeSpecIds[d] < -1) {
                throw new IllegalArgumentException("invalid local-size SpecId " + localSizeSpecIds[d]);
            }
        }
        this.localSize = localSize.clone();
        this.localSizeSpecIds = localSizeSpecIds.clone();

        Map<Integer, Integer> sizes = new HashMap<>();
        for (Map.Entry<Integer, Integer> e : specSizes.entrySet()) {
            int size = e.getValue();
            if (size != SPEC_SIZE_32 && size != SPEC_SIZE_64) {
                throw new IllegalArgumentException("SpecId " + e.getKey() + " has unsupported size " + size);
            }
            if (!specDefaults.containsKey(e.getKey())) {
                throw new IllegalArgumentException("SpecId " + e.getKey() + " has a size but no default value");
            }
            if (size != SPEC_SIZE_32) {
                sizes.put(e.getKey(), size);
            }
        }
        this.specDefaults = Collections.unmodifiableMap(new HashMap<>(specDefaults));
        this.specSizes = Collections.unmodifiableMap(sizes);
        this.capabilities = Collections.unmodifiableSet(new HashSet<>(capabilities));
    }

    public String entryPoint() {
        return entryPoint;
    }

    /** Unmodifiable bindings present in the module, sorted by binding number (gaps allowed). */
    public List<BindingInfo> bindings() {
        return bindings;
    }

    /** Number of bindings present in the module. */
    public int bindingCount() {
        return bindings.size();
    }

    /** Highest binding number plus one; 0 when the module has no storage buffers. */
    public int bindingSlots() {
        return bindingByNumber.length;
    }

    /** The binding with this number, or null when the module does not declare it. */
    public BindingInfo binding(int number) {
        return number >= 0 && number < bindingByNumber.length ? bindingByNumber[number] : null;
    }

    public boolean hasBinding(int number) {
        return binding(number) != null;
    }

    /** Unmodifiable push-constant members sorted by offset. */
    public List<PushMember> pushMembers() {
        return pushMembers;
    }

    /** Size of the push-constant block in bytes; 0 when there is none. */
    public int pushConstantSize() {
        return pushConstantSize;
    }

    /** Local size per dimension (copy). */
    public int[] localSize() {
        return localSize.clone();
    }

    /** Local size of dimension {@code d}: the fixed value, or the default where specialisable. */
    public int localSize(int d) {
        return localSize[d];
    }

    /** SpecId per local-size dimension, -1 where fixed (copy). */
    public int[] localSizeSpecIds() {
        return localSizeSpecIds.clone();
    }

    /** SpecId of local-size dimension {@code d}, or -1 where it is fixed. */
    public int localSizeSpecId(int d) {
        return localSizeSpecIds[d];
    }

    /** True when every local-size dimension is specialisable. */
    public boolean hasSpecLocalSize() {
        return localSizeSpecIds[0] >= 0 && localSizeSpecIds[1] >= 0 && localSizeSpecIds[2] >= 0;
    }

    /** True when at least one local-size dimension is specialisable. */
    public boolean hasAnySpecLocalSize() {
        return localSizeSpecIds[0] >= 0 || localSizeSpecIds[1] >= 0 || localSizeSpecIds[2] >= 0;
    }

    /** True when {@code specId} sizes one of the local-size dimensions. */
    public boolean isLocalSizeSpecId(int specId) {
        return specId >= 0
                && (localSizeSpecIds[0] == specId || localSizeSpecIds[1] == specId || localSizeSpecIds[2] == specId);
    }

    /** Unmodifiable default value of each declared SpecId. */
    public Map<Integer, Integer> specDefaults() {
        return specDefaults;
    }

    public boolean declaresSpecId(int specId) {
        return specDefaults.containsKey(specId);
    }

    /** Byte size of a declared SpecId: {@link #SPEC_SIZE_32} or {@link #SPEC_SIZE_64}. */
    public int specConstantSize(int specId) {
        if (!specDefaults.containsKey(specId)) {
            throw new IllegalArgumentException("SpecId " + specId + " is not declared by " + entryPoint);
        }
        Integer size = specSizes.get(specId);
        return size == null ? SPEC_SIZE_32 : size;
    }

    /** Unmodifiable set of declared SPIR-V capabilities. */
    public Set<Integer> capabilities() {
        return capabilities;
    }

    public boolean requiresInt64() {
        return capabilities.contains(CAP_INT64);
    }

    public boolean requiresInt8() {
        return capabilities.contains(CAP_INT8);
    }

    public boolean requiresInt16() {
        return capabilities.contains(CAP_INT16);
    }

    public boolean requiresFloat16() {
        return capabilities.contains(CAP_FLOAT16);
    }

    public boolean requiresFloat64() {
        return capabilities.contains(CAP_FLOAT64);
    }

    public boolean requiresStorage8() {
        return capabilities.contains(CAP_STORAGE_BUFFER_8BIT_ACCESS);
    }

    public boolean requiresStorage16() {
        return capabilities.contains(CAP_STORAGE_BUFFER_16BIT_ACCESS);
    }

    public boolean requiresSubgroupArithmetic() {
        return capabilities.contains(CAP_GROUP_NON_UNIFORM_ARITHMETIC);
    }

    /**
     * True when {@code other} has the same argument interface: the same bindings with the same
     * access, the same push-constant members and size, and the same specialisable local-size
     * dimensions. Two variants of one kernel (plain and subgroup) must agree on it, because they
     * share one argument state.
     */
    public boolean sameInterface(SpirvModuleInfo other) {
        return other != null
                && bindings.equals(other.bindings)
                && pushMembers.equals(other.pushMembers)
                && pushConstantSize == other.pushConstantSize
                && Arrays.equals(localSizeSpecIds, other.localSizeSpecIds);
    }

    @Override
    public String toString() {
        return "SpirvModuleInfo{entry=" + entryPoint + ", bindings=" + bindings + ", push=" + pushMembers
                + " (" + pushConstantSize + " bytes), local=" + Arrays.toString(localSize)
                + ", localSpecIds=" + Arrays.toString(localSizeSpecIds) + ", specs=" + specDefaults.keySet() + '}';
    }
}
