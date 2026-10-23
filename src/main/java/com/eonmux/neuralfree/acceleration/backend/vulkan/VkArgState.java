package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArgs;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.backend.StickyArgs;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

/**
 * Sticky arguments of one Vulkan kernel handle, materialised at dispatch into
 * buffer bindings, a push-constant image, and the read and write tracking ids the
 * hazard tracker needs. The argument layout is validated against the build-time
 * {@link KernelSignature} (when present) and the reflected module, and re-derived
 * only when an argument kind changes.
 *
 * <p>Mapping: the k-th {@link GpuBuffer} argument binds to binding k and the
 * s-th scalar argument fills push-constant member s.</p>
 * <ul>
 *   <li><b>With a signature</b> (LLM kernels) the Java arguments must match the signature's buffer
 *   count, scalar count and scalar kinds. Scalar s lives at byte offset {@code 4*s}. A binding the
 *   optimiser stripped from the module still consumes its buffer argument but is neither bound nor
 *   tracked, and scalars beyond the module's push range are not pushed, so the remaining arguments
 *   never shift.</li>
 *   <li><b>Without a signature</b> (legacy shaders) the module itself is the signature: the buffer
 *   arguments must cover bindings {@code 0..bindingSlots-1}, and the scalar arguments must match the
 *   push-constant members one to one in kind and size, at the members' reflected offsets.</li>
 * </ul>
 * <p>A binding is tracked as read when the module may read it and as written when it may write it;
 * the same buffer bound twice is tracked once per access kind.</p>
 */
@NotThreadSafe("Owned by one kernel handle; callers serialise dispatches of a handle.")
public final class VkArgState {

    private static final long[] NO_IDS = new long[0];

    private final String kernelName;
    private final SpirvModuleInfo info;
    private final KernelSignature signature;
    private final StickyArgs sticky;
    private final int pushSize;
    private final ByteBuffer pushImage;

    /** Per argument index: buffer ordinal (>= 0) or {@code -1 - pushOffset} for a scalar; rebuilt with the layout. */
    private int[] slots = new int[0];
    private long derivedLayoutVersion = -1;
    private long writtenVersion = -1;
    private VkGpuBuffer[] bindings = new VkGpuBuffer[0];
    private long bindingVersion;
    private long[] readIds = NO_IDS;
    private long[] writeIds = NO_IDS;

    /**
     * @param signature build-time signature, or null for legacy shaders validated by reflection only
     * @throws UnsupportedKernelException when the module and the signature disagree (a binding or
     *                                    push-constant member the signature does not describe)
     */
    public VkArgState(String kernelName, SpirvModuleInfo info, KernelSignature signature) {
        if (kernelName == null || kernelName.isEmpty()) {
            throw new IllegalArgumentException("kernelName must not be empty");
        }
        if (info == null) {
            throw new IllegalArgumentException("module info of " + kernelName + " must not be null");
        }
        this.kernelName = kernelName;
        this.info = info;
        this.signature = signature;
        this.sticky = new StickyArgs(kernelName);
        if (signature != null) {
            checkModuleAgainstSignature();
        }
        this.pushSize = info.pushConstantSize();
        this.pushImage = ByteBuffer.allocateDirect(Math.max(pushSize, 4)).order(ByteOrder.LITTLE_ENDIAN);
        this.pushImage.limit(pushSize);
    }

    public String kernelName() {
        return kernelName;
    }

    /** The reflected module the layout is derived from. */
    public SpirvModuleInfo info() {
        return info;
    }

    /** The build-time signature, or null for a legacy shader. */
    public KernelSignature signature() {
        return signature;
    }

    /** Sets one argument with snapshot semantics. */
    public void set(int index, Object arg) {
        sticky.set(index, arg);
    }

    /** Replaces every argument, as {@code enqueueKernel(..., args)} does. */
    public void setAll(Object[] args) {
        sticky.setAll(args);
    }

    /** The underlying argument snapshots. */
    public StickyArgs sticky() {
        return sticky;
    }

    /** True when every argument the kernel takes has been set. */
    public boolean isComplete() {
        return sticky.count() == expectedArgCount() && sticky.isComplete();
    }

    /**
     * Validates and rewrites the dirty parts of the bindings and push image.
     *
     * @throws IllegalArgumentException naming the kernel and the mismatching argument, or a bound
     *                                  buffer that is closed or not a Vulkan buffer
     * @throws IllegalStateException    when an argument was never set
     */
    public void materialize() {
        sticky.requireComplete();
        boolean full = false;
        if (derivedLayoutVersion != sticky.layoutVersion()) {
            deriveLayout();
            derivedLayoutVersion = sticky.layoutVersion();
            full = true;
        }
        boolean bindingsChanged = full;
        if (full || writtenVersion != sticky.version()) {
            BitSet dirty = sticky.dirty();
            int count = sticky.count();
            for (int i = full ? 0 : dirty.nextSetBit(0); i >= 0 && i < count;
                 i = full ? i + 1 : dirty.nextSetBit(i + 1)) {
                bindingsChanged |= write(i);
            }
            sticky.clearDirty();
            writtenVersion = sticky.version();
        }
        for (VkGpuBuffer b : bindings) {
            if (b != null && b.isClosed()) {
                throw new IllegalArgumentException(kernelName + ": a bound buffer (" + b + ") is closed");
            }
        }
        if (bindingsChanged) {
            bindingVersion++;
            rebuildTracking();
        }
    }

    /** Buffer per binding number; null for a binding absent from the optimised module. */
    public VkGpuBuffer[] bindings() {
        return bindings;
    }

    /** Increases whenever a bound buffer changes; lets pooled descriptors reuse a set. */
    public long bindingVersion() {
        return bindingVersion;
    }

    /**
     * Direct buffer holding the push-constant block, {@link #pushSize()} bytes, little-endian,
     * position 0 and limit {@link #pushSize()}. Callers must not change its position or limit.
     */
    public ByteBuffer pushImage() {
        return pushImage;
    }

    public int pushSize() {
        return pushSize;
    }

    /** Tracking ids of buffers the kernel reads (each once). */
    public long[] readIds() {
        return readIds;
    }

    /** Tracking ids of buffers the kernel writes (each once). */
    public long[] writeIds() {
        return writeIds;
    }

    private int expectedArgCount() {
        return signature != null ? signature.argCount() : info.bindingSlots() + info.pushMembers().size();
    }

    /** Classifies every argument, validates the layout and sizes the binding array. */
    private void deriveLayout() {
        int count = sticky.count();
        int[] newSlots = new int[count];
        int buffers = 0;
        int scalars = 0;
        List<SpirvModuleInfo.PushMember> members = info.pushMembers();
        for (int i = 0; i < count; i++) {
            KernelArgs.ArgKind kind = sticky.kind(i);
            if (kind == KernelArgs.ArgKind.BUFFER) {
                newSlots[i] = buffers++;
                continue;
            }
            int offset;
            if (signature != null) {
                signature.checkScalar(kernelName, i, scalars, kind, describe(i));
                offset = scalars * KernelSignature.SCALAR_BYTES;
            } else {
                if (scalars >= members.size()) {
                    throw new IllegalArgumentException(kernelName + ": arg " + i + " (" + describe(i)
                            + ") exceeds the " + members.size() + " push-constant members of the module");
                }
                SpirvModuleInfo.PushMember member = members.get(scalars);
                if (!matches(kind, i, member)) {
                    throw new IllegalArgumentException(kernelName + ": arg " + i + " (" + describe(i)
                            + ") does not match push-constant member " + scalars + " (" + member + ")");
                }
                offset = member.offset();
            }
            newSlots[i] = -1 - offset;
            scalars++;
        }
        if (signature != null) {
            signature.checkCounts(kernelName, buffers, scalars);
        } else if (buffers != info.bindingSlots() || scalars != members.size()) {
            throw new IllegalArgumentException(kernelName + ": got " + buffers + " buffers and " + scalars
                    + " scalars, module expects " + info.bindingSlots() + " buffers and " + members.size()
                    + " scalars");
        }
        slots = newSlots;
        if (bindings.length != buffers) {
            bindings = new VkGpuBuffer[buffers];
        }
    }

    /** Writes argument {@code i}; returns true when a bound buffer changed. */
    private boolean write(int i) {
        int slot = slots[i];
        Object value = sticky.get(i);
        if (slot >= 0) {
            if (!(value instanceof VkGpuBuffer)) {
                throw new IllegalArgumentException(kernelName + ": arg " + i + " is a "
                        + value.getClass().getSimpleName() + ", not a Vulkan buffer of this backend");
            }
            VkGpuBuffer buffer = info.hasBinding(slot) ? (VkGpuBuffer) value : null;
            if (bindings[slot] != buffer) {
                bindings[slot] = buffer;
                return true;
            }
            return false;
        }
        int offset = -1 - slot;
        int size = KernelArgs.byteSize(value);
        if (offset + size <= pushSize) {
            KernelArgs.writeLE(pushImage, offset, value);
        } else if (offset < pushSize) {
            throw new IllegalArgumentException(kernelName + ": arg " + i + " (" + describe(i) + ") at offset "
                    + offset + " straddles the end of the " + pushSize + "-byte push-constant range");
        }
        return false;
    }

    private void rebuildTracking() {
        long[] reads = new long[bindings.length];
        long[] writes = new long[bindings.length];
        int nr = 0;
        int nw = 0;
        for (int b = 0; b < bindings.length; b++) {
            VkGpuBuffer buffer = bindings[b];
            if (buffer == null) {
                continue;
            }
            SpirvModuleInfo.BindingInfo binding = info.binding(b);
            long id = buffer.trackingId();
            if (binding.readable() && !contains(reads, nr, id)) {
                reads[nr++] = id;
            }
            if (binding.writable() && !contains(writes, nw, id)) {
                writes[nw++] = id;
            }
        }
        readIds = nr == 0 ? NO_IDS : Arrays.copyOf(reads, nr);
        writeIds = nw == 0 ? NO_IDS : Arrays.copyOf(writes, nw);
    }

    private static boolean contains(long[] ids, int n, long id) {
        for (int i = 0; i < n; i++) {
            if (ids[i] == id) {
                return true;
            }
        }
        return false;
    }

    private boolean matches(KernelArgs.ArgKind kind, int index, SpirvModuleInfo.PushMember member) {
        switch (member.kind()) {
            case I32:
            case U32:
                return kind == KernelArgs.ArgKind.I32;
            case F32:
                return kind == KernelArgs.ArgKind.F32;
            case I64:
                return kind == KernelArgs.ArgKind.I64;
            case F64:
                return kind == KernelArgs.ArgKind.F64;
            default:
                return kind == KernelArgs.ArgKind.BYTES && KernelArgs.byteSize(sticky.get(index)) == member.size();
        }
    }

    private String describe(int index) {
        return KernelArgs.describe(sticky.get(index));
    }

    /**
     * Rejects a module that declares a binding or push-constant member the signature does not
     * describe; such a module and signature come from different builds.
     */
    private void checkModuleAgainstSignature() {
        if (info.bindingSlots() > signature.bufferCount()) {
            throw new UnsupportedKernelException(kernelName + ": the SPIR-V module declares binding "
                    + (info.bindingSlots() - 1) + " but the shader index signature has " + signature.bufferCount()
                    + " buffers; the module and index.properties come from different builds");
        }
        if (info.pushConstantSize() > signature.pushBytes()) {
            throw new UnsupportedKernelException(kernelName + ": the SPIR-V push-constant block is "
                    + info.pushConstantSize() + " bytes but the shader index signature has " + signature.scalarCount()
                    + " scalars (" + signature.pushBytes() + " bytes)");
        }
        List<SpirvModuleInfo.PushMember> members = info.pushMembers();
        for (int m = 0; m < members.size(); m++) {
            SpirvModuleInfo.PushMember member = members.get(m);
            int scalar = member.offset() / KernelSignature.SCALAR_BYTES;
            char kind = member.offset() % KernelSignature.SCALAR_BYTES == 0 && scalar < signature.scalarCount()
                    ? signature.pushKind(scalar) : 0;
            boolean ok = (kind == 'i' && member.kind() == SpirvModuleInfo.PushMember.Kind.I32)
                    || (kind == 'u' && member.kind() == SpirvModuleInfo.PushMember.Kind.U32)
                    || (kind == 'f' && member.kind() == SpirvModuleInfo.PushMember.Kind.F32);
            if (!ok) {
                throw new UnsupportedKernelException(kernelName + ": SPIR-V push-constant member " + m + " ("
                        + member + ") does not match the shader index signature " + signature);
            }
        }
    }
}
