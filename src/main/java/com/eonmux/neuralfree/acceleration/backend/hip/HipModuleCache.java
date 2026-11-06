package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Loads each HIP code-object module once, from classpath resources by path. Modules are
 * unloaded by {@link #close()}.
 *
 * <p>One module per code object means one driver load however many kernels the module provides.
 * A code object holds machine code for explicit targets ({@code gfx1100}, {@code gfx942}), so
 * the hipcc build compiles one object per target architecture and the index of the kernel
 * library lists the one that covers the device ({@code HipKernelLibrary}). Thread-safe.</p>
 */
public final class HipModuleCache implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(HipModuleCache.class);

    private static final int READ_CHUNK = 64 * 1024;

    private final HipContext context;
    private final com.eonmux.neuralfree.acceleration.hip.HipDriver driver;
    private final Map<String, Long> byResource = new HashMap<>();
    private boolean closed;

    /**
     * @param context the device context modules are loaded under
     * @param driver  the runtime operations modules are loaded and unloaded with
     * @throws IllegalArgumentException when an argument is null
     */
    public HipModuleCache(HipContext context, com.eonmux.neuralfree.acceleration.hip.HipDriver driver) {
        if (context == null) {
            throw new IllegalArgumentException("HIP context must not be null");
        }
        if (driver == null) {
            throw new IllegalArgumentException("HIP driver must not be null");
        }
        this.context = context;
        this.driver = driver;
    }

    /**
     * The module of a classpath code object, loaded on first use. The resource is read once.
     *
     * @param resourcePath classpath path of a {@code .hsaco}, {@code .co} or {@code .hipfb} file
     * @throws UnsupportedKernelException when the resource is missing or the image does not support this device
     * @throws IllegalStateException      after {@link #close()}
     */
    public synchronized long get(String resourcePath) {
        ensureOpen();
        String path = normalize(resourcePath);
        Long module = byResource.get(path);
        if (module != null) {
            return module;
        }
        byte[] image = readResource(path);
        context.makeCurrent(driver);
        long loaded = driver.moduleLoadData(image);
        byResource.put(path, loaded);
        LOGGER.debug("Loaded HIP module {} ({} bytes)", path, image.length);
        return loaded;
    }

    /** Number of loaded modules. */
    public synchronized int size() {
        return byResource.size();
    }

    /**
     * Unloads every module; idempotent. Failures are logged and the remaining modules are still
     * unloaded.
     */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (byResource.isEmpty()) {
            return;
        }
        context.makeCurrent(driver);
        for (Long module : byResource.values()) {
            try {
                driver.moduleUnload(module);
            } catch (RuntimeException e) {
                LOGGER.warn("Failed to unload HIP module: {}", e.getMessage());
            }
        }
        byResource.clear();
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("HIP module cache is closed");
        }
    }

    private static String normalize(String resourcePath) {
        if (resourcePath == null || resourcePath.trim().isEmpty()) {
            throw new IllegalArgumentException("HIP module resource path must not be null or empty");
        }
        String path = resourcePath.trim();
        while (path.startsWith("/")) {
            path = path.substring(1);
        }
        return path;
    }

    private static byte[] readResource(String path) {
        InputStream in = open(path);
        if (in == null) {
            throw new UnsupportedKernelException("HIP module resource " + path
                    + " is not on the classpath (HIP kernels were not built, or the build is incomplete)", null);
        }
        try (InputStream stream = in) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] chunk = new byte[READ_CHUNK];
            int n;
            while ((n = stream.read(chunk)) > 0) {
                out.write(chunk, 0, n);
            }
            byte[] bytes = out.toByteArray();
            if (bytes.length == 0) {
                throw new UnsupportedKernelException("HIP module resource " + path + " is empty", null);
            }
            return bytes;
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read HIP module resource " + path, e);
        }
    }

    private static InputStream open(String path) {
        ClassLoader own = HipModuleCache.class.getClassLoader();
        InputStream in = own == null ? ClassLoader.getSystemResourceAsStream(path) : own.getResourceAsStream(path);
        if (in != null) {
            return in;
        }
        ClassLoader contextLoader = Thread.currentThread().getContextClassLoader();
        return contextLoader == null || contextLoader == own ? null : contextLoader.getResourceAsStream(path);
    }
}
