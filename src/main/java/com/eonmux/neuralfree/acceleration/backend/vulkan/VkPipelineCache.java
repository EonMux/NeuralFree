package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPipelineCacheCreateInfo;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;

import static org.lwjgl.vulkan.VK10.VK_INCOMPLETE;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_PIPELINE_CACHE_HEADER_VERSION_ONE;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkCreatePipelineCache;
import static org.lwjgl.vulkan.VK10.vkDestroyPipelineCache;
import static org.lwjgl.vulkan.VK10.nvkGetPipelineCacheData;

/**
 * A {@code VkPipelineCache}, optionally persisted per device and driver.
 *
 * <p>The persistent form loads
 * {@code ${user.home}/.cache/neuralfree/vulkan/<vendorId>-<deviceId>-<driverVersion>-<pipelineCacheUUID>.bin}
 * at creation and writes it back on {@link #save()} and {@link #close()}. Loaded data is used only
 * when its Vulkan header (version one) names the same vendor, device and pipeline-cache UUID.
 * Persistence is best-effort: IO failures and rejected data are logged at DEBUG and the cache
 * starts empty. {@code -Dneuralfree.vulkan.pipelineCache=false} disables persistence.</p>
 *
 * <p>Pipeline creation may use the cache from several threads; {@link #save()} and
 * {@link #close()} are synchronised. Saves replace the file atomically from a private temporary
 * file, so readers and concurrent writers never see a partly written image.</p>
 */
public final class VkPipelineCache implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(VkPipelineCache.class);

    /** System property enabling the on-disk cache; {@code true} by default. */
    public static final String PERSIST_PROPERTY = "neuralfree.vulkan.pipelineCache";

    private static final int HEADER_BYTES = 32;
    private static final int UUID_BYTES = 16;
    private static final long MAX_FILE_BYTES = 512L * 1024 * 1024;
    private static final String TEMP_SUFFIX = ".tmp";

    private final VkDevice device;
    private final long handle;
    private final Path file;
    private final int vendorId;
    private final int deviceId;
    private final byte[] uuid;
    private boolean closed;

    /** An in-memory cache that is never persisted. */
    public VkPipelineCache(VkDevice device) {
        this(device, null, null);
    }

    /**
     * A cache persisted for {@code info}'s device and driver, unless persistence is disabled by
     * {@value #PERSIST_PROPERTY}.
     *
     * @throws GpuOutOfMemoryException when {@code vkCreatePipelineCache} without initial data runs out of memory
     * @throws GpuBackendException     when {@code vkCreatePipelineCache} fails otherwise even without initial data
     */
    public VkPipelineCache(VkDevice device, VkPhysicalDeviceInfo info) {
        this(device, info, persistenceEnabled() && info != null ? cacheFile(info) : null);
    }

    private VkPipelineCache(VkDevice device, VkPhysicalDeviceInfo info, Path file) {
        if (device == null) {
            throw new IllegalArgumentException("VkDevice must not be null");
        }
        this.device = device;
        this.file = file;
        this.vendorId = info != null ? info.vendorId() : 0;
        this.deviceId = info != null ? info.deviceId() : 0;
        this.uuid = info != null ? info.pipelineCacheUuid() : new byte[UUID_BYTES];
        byte[] initial = file != null ? load(file) : null;
        long created = initial != null ? create(initial) : VK_NULL_HANDLE;
        this.handle = created != VK_NULL_HANDLE ? created : create(null);
    }

    /** The {@code VkPipelineCache} handle. */
    public long handle() {
        if (closed) {
            throw new IllegalStateException("pipeline cache is closed");
        }
        return handle;
    }

    /** The cache file, or null when the cache is not persisted. */
    public Path file() {
        return file;
    }

    public boolean isPersistent() {
        return file != null;
    }

    /**
     * The current cache contents as returned by {@code vkGetPipelineCacheData}.
     *
     * @throws IllegalStateException    when the cache is closed, or its contents exceed the size a cache file may have
     * @throws GpuOutOfMemoryException when the driver call reports exhausted host or device memory
     * @throws GpuBackendException     when the driver call fails otherwise
     */
    public synchronized byte[] data() {
        if (closed) {
            throw new IllegalStateException("pipeline cache is closed");
        }
        return readData();
    }

    /**
     * Writes the cache contents to the cache file; does nothing when not persisted. Best-effort. The contents go to
     * a temporary file of their own in the cache directory, which then replaces the cache file in one rename, so
     * concurrent savers of the same device's cache (other processes, other backends on the device) each publish a
     * complete image and a failed save leaves no temporary file behind.
     */
    public synchronized void save() {
        if (closed || file == null) {
            return;
        }
        Path tmp = null;
        try {
            byte[] data = readData();
            if (data.length < HEADER_BYTES) {
                return;
            }
            Path directory = file.getParent();
            Files.createDirectories(directory);
            tmp = Files.createTempFile(directory, file.getFileName() + ".", TEMP_SUFFIX);
            Files.write(tmp, data);
            try {
                Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
            }
            tmp = null;
            LOGGER.debug("Vulkan pipeline cache saved: {} bytes to {}", data.length, file);
        } catch (IOException | RuntimeException e) {
            LOGGER.debug("Vulkan pipeline cache not saved to {}: {}", file, e.toString());
        } finally {
            if (tmp != null) {
                deleteTemporary(tmp);
            }
        }
    }

    private static void deleteTemporary(Path tmp) {
        try {
            Files.deleteIfExists(tmp);
        } catch (IOException e) {
            LOGGER.debug("Vulkan pipeline cache temporary file {} not deleted: {}", tmp, e.toString());
        }
    }

    /** Saves (when persisted) and destroys the cache; idempotent. */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        save();
        vkDestroyPipelineCache(device, handle, null);
        closed = true;
    }

    /**
     * Creates the cache, seeded with {@code initial} when non-null.
     *
     * @return the handle, or {@code VK_NULL_HANDLE} when the driver rejects the initial data
     * @throws GpuOutOfMemoryException when creating an empty cache runs out of host or device memory
     * @throws GpuBackendException     when creating an empty cache fails otherwise
     */
    private long create(byte[] initial) {
        ByteBuffer data = null;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPipelineCacheCreateInfo info = VkPipelineCacheCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO);
            if (initial != null) {
                data = MemoryUtil.memAlloc(initial.length);
                data.put(initial).flip();
                info.pInitialData(data);
            }
            LongBuffer pCache = stack.mallocLong(1);
            int result = vkCreatePipelineCache(device, info, null, pCache);
            if (result != VK_SUCCESS) {
                if (initial == null) {
                    throw VkErrors.translate(result, "vkCreatePipelineCache");
                }
                LOGGER.debug("Vulkan pipeline cache data from {} rejected by the driver ({}); starting empty",
                        file, result);
                return VK_NULL_HANDLE;
            }
            if (initial != null) {
                LOGGER.debug("Vulkan pipeline cache loaded: {} bytes from {}", initial.length, file);
            }
            return pCache.get(0);
        } finally {
            if (data != null) {
                MemoryUtil.memFree(data);
            }
        }
    }

    private byte[] readData() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer pSize = stack.mallocPointer(1);
            int result = nvkGetPipelineCacheData(device, handle, MemoryUtil.memAddress(pSize), MemoryUtil.NULL);
            VkErrors.check(result, "vkGetPipelineCacheData (size)");
            long size = pSize.get(0);
            if (size <= 0) {
                return new byte[0];
            }
            if (size > MAX_FILE_BYTES) {
                throw new IllegalStateException("Vulkan pipeline cache of " + size + " bytes exceeds the "
                        + MAX_FILE_BYTES + "-byte limit");
            }
            ByteBuffer buffer = MemoryUtil.memAlloc((int) size);
            try {
                result = nvkGetPipelineCacheData(device, handle, MemoryUtil.memAddress(pSize),
                        MemoryUtil.memAddress(buffer));
                if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
                    throw VkErrors.translate(result, "vkGetPipelineCacheData");
                }
                byte[] out = new byte[(int) Math.min(pSize.get(0), size)];
                buffer.get(out);
                return out;
            } finally {
                MemoryUtil.memFree(buffer);
            }
        }
    }

    /** The file's bytes when it holds a compatible version-one header, else null. */
    private byte[] load(Path path) {
        try {
            if (!Files.isRegularFile(path)) {
                return null;
            }
            long size = Files.size(path);
            if (size < HEADER_BYTES || size > MAX_FILE_BYTES) {
                LOGGER.debug("Vulkan pipeline cache {} ignored: {} bytes", path, size);
                return null;
            }
            byte[] data = Files.readAllBytes(path);
            ByteBuffer header = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
            int headerSize = header.getInt(0);
            int headerVersion = header.getInt(4);
            byte[] fileUuid = Arrays.copyOfRange(data, 16, 16 + UUID_BYTES);
            if (headerSize < HEADER_BYTES || headerSize > data.length
                    || headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE
                    || header.getInt(8) != vendorId || header.getInt(12) != deviceId
                    || !Arrays.equals(fileUuid, uuid)) {
                LOGGER.debug("Vulkan pipeline cache {} ignored: header does not match this device and driver", path);
                return null;
            }
            return data;
        } catch (IOException | RuntimeException e) {
            LOGGER.debug("Vulkan pipeline cache {} not loaded: {}", path, e.toString());
            return null;
        }
    }

    private static boolean persistenceEnabled() {
        String value = System.getProperty(PERSIST_PROPERTY);
        return value == null || !value.trim().equalsIgnoreCase("false");
    }

    private static Path cacheFile(VkPhysicalDeviceInfo info) {
        StringBuilder uuidHex = new StringBuilder(2 * UUID_BYTES);
        for (byte b : info.pipelineCacheUuid()) {
            uuidHex.append(String.format("%02x", b & 0xFF));
        }
        String name = String.format("%04x-%04x-%08x-%s.bin", info.vendorId(), info.deviceId(),
                info.driverVersion(), uuidHex);
        return Paths.get(System.getProperty("user.home"), ".cache", "neuralfree", "vulkan", name);
    }
}
