package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.utils.LoggingUtil;
import com.sun.jna.Function;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.NativeLibrary;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;
import org.apache.logging.log4j.Logger;
import org.jocl.Sizeof;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.CL_DEVICE_PLATFORM;
import static org.jocl.CL.CL_SUCCESS;
import static org.jocl.CL.clGetDeviceInfo;

/**
 * {@link OclCommandBufferApi} over the extension functions JOCL does not bind, reached through
 * {@code clGetExtensionFunctionAddressForPlatform} and invoked with JNA.
 *
 * <p>The functions are loaded once per backend and only when the device advertises
 * {@code cl_khr_command_buffer}; a device without it, or a platform whose ICD loader does not
 * export the extension entry points, gets no instance and keeps dispatching directly. Mutable
 * dispatch is enabled only when the device advertises {@code cl_khr_command_buffer_mutable_dispatch}
 * and the {@code CL_MUTABLE_DISPATCH_ARGUMENTS_KHR} capability, so a captured dispatch's scalar
 * arguments can be replaced in place; otherwise a changed argument forces a re-capture.</p>
 *
 * <p>Host values follow the platform's pointer and {@code size_t} sizes. The recorded command
 * sequence is chained with sync points, because commands of a command buffer carry no implicit
 * order.</p>
 */
final class JnaOclCommandBufferApi implements OclCommandBufferApi {

    private static final Logger LOGGER = LoggingUtil.getLogger(JnaOclCommandBufferApi.class);

    private static final String COMMAND_BUFFER_EXTENSION = "cl_khr_command_buffer";
    private static final String MUTABLE_DISPATCH_EXTENSION = "cl_khr_command_buffer_mutable_dispatch";

    private static final int CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR = 0x12A9;
    private static final int CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR = 0x12AA;
    private static final int CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR = 0x12B0;
    private static final long CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR = 1L << 2;
    private static final long CL_MUTABLE_DISPATCH_ARGUMENTS_KHR = 1L << 3;
    private static final long CL_COMMAND_BUFFER_FLAGS_KHR = 0x1293L;
    private static final long CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR = 1L;
    private static final long CL_COMMAND_BUFFER_MUTABLE_KHR = 1L << 1;
    private static final int CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR = 0;

    private final Function createCommandBuffer;
    private final Function finalizeCommandBuffer;
    private final Function releaseCommandBuffer;
    private final Function enqueueCommandBuffer;
    private final Function commandNDRangeKernel;
    private final Function commandCopyBuffer;
    private final Function commandFillBuffer;
    private final Function updateMutableCommands;
    private final boolean simultaneousUse;
    private final boolean mutableArguments;

    private JnaOclCommandBufferApi(Function createCommandBuffer, Function finalizeCommandBuffer,
                                   Function releaseCommandBuffer, Function enqueueCommandBuffer,
                                   Function commandNDRangeKernel, Function commandCopyBuffer,
                                   Function commandFillBuffer, Function updateMutableCommands,
                                   boolean simultaneousUse, boolean mutableArguments) {
        this.createCommandBuffer = createCommandBuffer;
        this.finalizeCommandBuffer = finalizeCommandBuffer;
        this.releaseCommandBuffer = releaseCommandBuffer;
        this.enqueueCommandBuffer = enqueueCommandBuffer;
        this.commandNDRangeKernel = commandNDRangeKernel;
        this.commandCopyBuffer = commandCopyBuffer;
        this.commandFillBuffer = commandFillBuffer;
        this.updateMutableCommands = updateMutableCommands;
        this.simultaneousUse = simultaneousUse;
        this.mutableArguments = mutableArguments;
    }

    /**
     * Loads the extension for {@code device}, or returns null when it is unavailable: the device
     * does not advertise {@code cl_khr_command_buffer}, its queues would need properties this
     * backend does not create, or the entry points cannot be resolved.
     *
     * @param device          the device extension functions are resolved for
     * @param extensions      the device's {@code CL_DEVICE_EXTENSIONS} string
     * @param queueProperties the properties of the queues this backend creates
     */
    static OclCommandBufferApi tryLoad(cl_device_id device, String extensions, long queueProperties) {
        if (device == null || !OclComputeBackend.hasExtension(extensions, COMMAND_BUFFER_EXTENSION)) {
            return null;
        }
        long required = queryDeviceULong(device, CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR);
        if (required > 0 && (required & ~queueProperties) != 0) {
            LOGGER.info("OpenCL command buffers are unavailable: the device requires queue properties 0x{} "
                    + "that this backend does not create", Long.toHexString(required));
            return null;
        }
        try {
            NativeLibrary library = NativeLibrary.getInstance("OpenCL");
            Function addressOf = library.getFunction("clGetExtensionFunctionAddressForPlatform");
            long platform = platformOf(device);
            Function create = extension(addressOf, platform, "clCreateCommandBufferKHR");
            Function finalize = extension(addressOf, platform, "clFinalizeCommandBufferKHR");
            Function release = extension(addressOf, platform, "clReleaseCommandBufferKHR");
            Function enqueue = extension(addressOf, platform, "clEnqueueCommandBufferKHR");
            Function kernel = extension(addressOf, platform, "clCommandNDRangeKernelKHR");
            Function copy = extension(addressOf, platform, "clCommandCopyBufferKHR");
            Function fill = extension(addressOf, platform, "clCommandFillBufferKHR");
            if (create == null || finalize == null || release == null || enqueue == null
                    || kernel == null || copy == null || fill == null) {
                LOGGER.info("OpenCL command buffers are unavailable: the ICD loader did not resolve the "
                        + "extension entry points");
                return null;
            }
            Function update = null;
            long mutableCapabilities = queryDeviceULong(device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR);
            if (OclComputeBackend.hasExtension(extensions, MUTABLE_DISPATCH_EXTENSION)
                    && (mutableCapabilities & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR) != 0) {
                update = extension(addressOf, platform, "clUpdateMutableCommandsKHR");
            }
            long capabilities = queryDeviceULong(device, CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR);
            boolean simultaneous = (capabilities & CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR) != 0;
            return new JnaOclCommandBufferApi(create, finalize, release, enqueue, kernel, copy, fill,
                    update, simultaneous, update != null);
        } catch (RuntimeException | LinkageError e) {
            LOGGER.info("OpenCL command buffers are unavailable: {}", e.toString());
            return null;
        }
    }

    @Override
    public boolean simultaneousUse() {
        return simultaneousUse;
    }

    @Override
    public boolean mutableArguments() {
        return mutableArguments;
    }

    @Override
    public long createCommandBuffer(long queue, boolean withSimultaneousUse, boolean mutable) {
        Memory queues = new Memory(Native.POINTER_SIZE);
        queues.setPointer(0, new Pointer(queue));
        long flags = (withSimultaneousUse ? CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR : 0L)
                | (mutable ? CL_COMMAND_BUFFER_MUTABLE_KHR : 0L);
        Memory properties = null;
        if (flags != 0) {
            properties = new Memory(3L * Long.BYTES);
            properties.setLong(0, CL_COMMAND_BUFFER_FLAGS_KHR);
            properties.setLong(Long.BYTES, flags);
            properties.setLong(2L * Long.BYTES, 0L);
        }
        IntByReference error = new IntByReference();
        Pointer commandBuffer = createCommandBuffer.invokePointer(
                new Object[]{Integer.valueOf(1), queues, pointerOrNull(properties), error});
        check(error.getValue(), "clCreateCommandBufferKHR", null);
        if (commandBuffer == null) {
            throw new IllegalStateException("clCreateCommandBufferKHR reported CL_SUCCESS but returned no buffer");
        }
        return Pointer.nativeValue(commandBuffer);
    }

    @Override
    public void finalizeCommandBuffer(long commandBuffer) {
        check(finalizeCommandBuffer.invokeInt(new Object[]{handle(commandBuffer)}),
                "clFinalizeCommandBufferKHR", null);
    }

    @Override
    public void releaseCommandBuffer(long commandBuffer) {
        check(releaseCommandBuffer.invokeInt(new Object[]{handle(commandBuffer)}),
                "clReleaseCommandBufferKHR", null);
    }

    @Override
    public void enqueueCommandBuffer(long commandBuffer, long queue, long[] waitEvents) {
        Memory queues = new Memory(Native.POINTER_SIZE);
        queues.setPointer(0, new Pointer(queue));
        int count = waitEvents == null ? 0 : waitEvents.length;
        Memory waits = null;
        if (count > 0) {
            waits = new Memory((long) count * Native.POINTER_SIZE);
            for (int i = 0; i < count; i++) {
                waits.setPointer((long) i * Native.POINTER_SIZE, new Pointer(waitEvents[i]));
            }
        }
        int status = enqueueCommandBuffer.invokeInt(new Object[]{Integer.valueOf(1), queues,
                handle(commandBuffer), Integer.valueOf(count), pointerOrNull(waits), Pointer.NULL});
        check(status, "clEnqueueCommandBufferKHR", null);
    }

    @Override
    public Recorded commandNDRangeKernel(long commandBuffer, long queue, long kernel, int workDim,
                                         long[] globalWorkSize, long[] localWorkSize, int waitSyncPoint) {
        Memory global = sizeTArray(globalWorkSize);
        Memory local = localWorkSize == null ? null : sizeTArray(localWorkSize);
        SyncWait wait = new SyncWait(waitSyncPoint);
        IntByReference syncPoint = new IntByReference();
        PointerByReference mutableCommand = mutableArguments ? new PointerByReference() : null;
        int status = commandNDRangeKernel.invokeInt(new Object[]{
                handle(commandBuffer), handle(queue), Pointer.NULL, handle(kernel), Integer.valueOf(workDim),
                Pointer.NULL, global, pointerOrNull(local), Integer.valueOf(wait.count()), wait.list(),
                syncPoint, mutableCommand == null ? Pointer.NULL : mutableCommand});
        check(status, "clCommandNDRangeKernelKHR", null);
        long mutable = mutableCommand == null || mutableCommand.getValue() == null
                ? 0L : Pointer.nativeValue(mutableCommand.getValue());
        return new Recorded(syncPoint.getValue(), mutable);
    }

    @Override
    public int commandCopyBuffer(long commandBuffer, long queue, long src, long dst, long srcOffset,
                                 long dstOffset, long size, int waitSyncPoint) {
        SyncWait wait = new SyncWait(waitSyncPoint);
        IntByReference syncPoint = new IntByReference();
        int status = commandCopyBuffer.invokeInt(new Object[]{
                handle(commandBuffer), handle(queue), Pointer.NULL, handle(src), handle(dst),
                sizeT(srcOffset), sizeT(dstOffset), sizeT(size), Integer.valueOf(wait.count()), wait.list(),
                syncPoint, Pointer.NULL});
        check(status, "clCommandCopyBufferKHR", null);
        return syncPoint.getValue();
    }

    @Override
    public int commandFillBuffer(long commandBuffer, long queue, long dst, int pattern, long offset,
                                 long size, int waitSyncPoint) {
        Memory patternBytes = new Memory(Integer.BYTES);
        // the direct fill path lays the 32-bit pattern out least-significant byte first
        for (int i = 0; i < Integer.BYTES; i++) {
            patternBytes.setByte(i, (byte) (pattern >>> (8 * i)));
        }
        SyncWait wait = new SyncWait(waitSyncPoint);
        IntByReference syncPoint = new IntByReference();
        // clCommandFillBufferKHR(command_buffer, queue, properties, buffer, pattern, pattern_size,
        //                        offset, size, num_sync_points, sync_point_wait_list, sync_point, mutable)
        int status = commandFillBuffer.invokeInt(new Object[]{
                handle(commandBuffer), handle(queue), Pointer.NULL, handle(dst), patternBytes,
                sizeT(Integer.BYTES), sizeT(offset), sizeT(size), Integer.valueOf(wait.count()), wait.list(),
                syncPoint, Pointer.NULL});
        check(status, "clCommandFillBufferKHR", null);
        return syncPoint.getValue();
    }

    @Override
    public void updateMutableArguments(long commandBuffer, List<MutableDispatch> updates) {
        if (updates == null || updates.isEmpty()) {
            return;
        }
        ConfigLayout layout = new ConfigLayout();
        int count = updates.size();
        Memory types = new Memory((long) count * Integer.BYTES);
        Memory pointers = new Memory((long) count * layout.pointer);
        // JNA passes the nested Memory objects as raw pointers, which do not keep them alive
        List<Memory> keepAlive = new ArrayList<>();
        try {
            for (int i = 0; i < count; i++) {
                types.setInt((long) i * Integer.BYTES, CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR);
                pointers.setPointer((long) i * layout.pointer,
                        layout.config(updates.get(i), keepAlive));
            }
            int status = updateMutableCommands.invokeInt(
                    new Object[]{handle(commandBuffer), Integer.valueOf(count), types, pointers});
            check(status, "clUpdateMutableCommandsKHR", null);
        } finally {
            keepAlive.clear();
        }
    }

    /**
     * The byte layout of {@code cl_mutable_dispatch_config_khr} and {@code cl_mutable_dispatch_arg_khr}
     * on this platform, and the builders that fill them in.
     */
    private static final class ConfigLayout {
        private final int pointer = Native.POINTER_SIZE;
        private final int sizeT = Native.SIZE_T_SIZE;
        private final long argIndexOffset = 0;
        private final long argSizeOffset = align(Integer.BYTES, sizeT);
        private final long argValueOffset = argSizeOffset + sizeT;
        private final long argSize = align(argValueOffset + pointer, Math.max(pointer, sizeT));
        private final long commandOffset = 0;
        private final long numArgsOffset = pointer;
        private final long argListOffset;
        private final long configSize;

        ConfigLayout() {
            long numSvmArgsOffset = numArgsOffset + Integer.BYTES;
            long numExecInfosOffset = numSvmArgsOffset + Integer.BYTES;
            long workDimOffset = numExecInfosOffset + Integer.BYTES;
            long argSvmListOffset = align(workDimOffset + Integer.BYTES, pointer) + pointer;
            long execInfoListOffset = argSvmListOffset + pointer;
            long globalOffsetOffset = execInfoListOffset + pointer;
            long globalSizeOffset = globalOffsetOffset + pointer;
            long localSizeOffset = globalSizeOffset + pointer;
            argListOffset = align(workDimOffset + Integer.BYTES, pointer);
            configSize = align(localSizeOffset + pointer, pointer);
        }

        /** One {@code cl_mutable_dispatch_config_khr} for {@code update}, kept alive by the caller. */
        Memory config(MutableDispatch update, List<Memory> keepAlive) {
            Memory config = new Memory(configSize);
            config.clear();
            config.setPointer(commandOffset, new Pointer(update.mutableCommand));
            config.setInt(numArgsOffset, update.arguments.size());
            config.setPointer(argListOffset, arguments(update.arguments, keepAlive));
            keepAlive.add(config);
            return config;
        }

        /** The {@code cl_mutable_dispatch_arg_khr} table of one dispatch, kept alive by the caller. */
        private Memory arguments(List<MutableArg> args, List<Memory> keepAlive) {
            Memory table = new Memory((long) args.size() * argSize);
            for (int i = 0; i < args.size(); i++) {
                MutableArg argument = args.get(i);
                Memory value = new Memory(Math.max(1, argument.value.length));
                value.write(0, argument.value, 0, argument.value.length);
                keepAlive.add(value);
                long base = (long) i * argSize;
                table.setInt(base + argIndexOffset, argument.index);
                putSizeT(table, base + argSizeOffset, argument.value.length);
                table.setPointer(base + argValueOffset, value);
            }
            keepAlive.add(table);
            return table;
        }
    }

    /** The device's required queue properties are already satisfied; a failed query imposes none. */
    private static long queryDeviceULong(cl_device_id device, int parameter) {
        long[] value = new long[1];
        try {
            int status = clGetDeviceInfo(device, parameter, Sizeof.cl_ulong, org.jocl.Pointer.to(value), null);
            return status == CL_SUCCESS ? value[0] : 0L;
        } catch (RuntimeException e) {
            return 0L;
        }
    }

    /** The platform handle of {@code device}, needed by the platform-scoped extension lookup. */
    private static long platformOf(cl_device_id device) {
        cl_platform_id platform = new cl_platform_id();
        int status = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, Sizeof.cl_platform_id,
                org.jocl.Pointer.to(platform), null);
        check(status, "clGetDeviceInfo(CL_DEVICE_PLATFORM)", null);
        return platform.getNativePointer();
    }

    /** Resolves one extension function for {@code platform}, or null when the loader does not know it. */
    private static Function extension(Function addressOf, long platform, String name) {
        Pointer address = addressOf.invokePointer(new Object[]{new Pointer(platform), name});
        return address == null ? null : Function.getFunction(address);
    }

    private static Memory sizeTArray(long[] values) {
        Memory memory = new Memory((long) values.length * Native.SIZE_T_SIZE);
        for (int i = 0; i < values.length; i++) {
            putSizeT(memory, (long) i * Native.SIZE_T_SIZE, values[i]);
        }
        return memory;
    }

    /** One native handle argument; a null handle is still a pointer argument, not a missing one. */
    private static Pointer handle(long nativeHandle) {
        return nativeHandle == 0 ? Pointer.NULL : new Pointer(nativeHandle);
    }

    /** One {@code size_t} argument, marshalled by the platform's size rather than the Java type. */
    private static Object sizeT(long value) {
        return Native.SIZE_T_SIZE == Integer.BYTES ? (Object) Integer.valueOf((int) value)
                : (Object) Long.valueOf(value);
    }

    /** A nullable pointer argument; JNA needs an explicit {@link Pointer#NULL} to marshal it. */
    private static Pointer pointerOrNull(Memory memory) {
        return memory == null ? Pointer.NULL : memory;
    }

    private static void putSizeT(Memory memory, long offset, long value) {
        if (Native.SIZE_T_SIZE == Long.BYTES) {
            memory.setLong(offset, value);
        } else {
            memory.setInt(offset, (int) value);
        }
    }

    /** A size_t rounded up to {@code alignment}, for the field offsets of a structure. */
    private static long align(long value, long alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    private static void check(int status, String operation, String subject) {
        OclErrors.check(status, operation, subject);
    }

    /** The sync point wait list of one command, or none for the first command of a buffer. */
    private static final class SyncWait {
        private final int syncPoint;
        private final Pointer list;

        SyncWait(int syncPoint) {
            this.syncPoint = syncPoint;
            if (syncPoint == NO_SYNC_POINT) {
                this.list = Pointer.NULL;
            } else {
                Memory memory = new Memory(Integer.BYTES);
                memory.setInt(0, syncPoint);
                this.list = memory;
            }
        }

        int count() {
            return syncPoint == NO_SYNC_POINT ? 0 : 1;
        }

        Pointer list() {
            return list;
        }
    }
}
