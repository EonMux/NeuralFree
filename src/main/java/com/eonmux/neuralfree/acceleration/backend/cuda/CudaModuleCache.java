package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.driver.CUmodule;
import org.apache.logging.log4j.Logger;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.driver.JCudaDriver.cuModuleUnload;

/**
 * Loads each CUDA module once: classpath fatbin/cubin/PTX resources by path and
 * legacy PTX text by content hash. Modules are unloaded by {@link #close()}.
 *
 * <p>One module per fatbin means one copy of its {@code __constant__} tables and one
 * driver load however many kernels the module provides. The driver picks the SASS
 * matching the device, or JIT-compiles the embedded PTX (cached in
 * {@code CUDA_CACHE_PATH}). Thread-safe.</p>
 */
public final class CudaModuleCache implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(CudaModuleCache.class);

    private static final int READ_CHUNK = 64 * 1024;

    private final CudaContext context;
    private final Map<String, CUmodule> byResource = new HashMap<>();
    private final Map<String, CUmodule> byPtxHash = new HashMap<>();
    private boolean closed;

    /**
     * @param context the context modules are loaded into
     * @throws IllegalArgumentException when {@code context} is null
     */
    public CudaModuleCache(CudaContext context) {
        if (context == null) {
            throw new IllegalArgumentException("CUDA context must not be null");
        }
        this.context = context;
    }

    /**
     * The module of a classpath image, loaded on first use. The resource is read once.
     *
     * @param resourcePath classpath path of a {@code .fatbin}, {@code .cubin} or {@code .ptx} file
     * @throws UnsupportedKernelException when the resource is missing or the image does not support this device
     * @throws IllegalStateException      after {@link #close()}
     */
    public synchronized CUmodule get(String resourcePath) {
        ensureOpen();
        String path = normalize(resourcePath);
        CUmodule module = byResource.get(path);
        if (module != null) {
            return module;
        }
        byte[] image = readResource(path);
        if (path.endsWith(".ptx")) {
            image = nulTerminated(image);
        }
        module = load(image, "cuModuleLoadData(" + path + ")");
        byResource.put(path, module);
        LOGGER.debug("Loaded CUDA module {} ({} bytes)", path, image.length);
        return module;
    }

    /**
     * The module of PTX source text, loaded on first use and shared by every kernel of that text.
     *
     * @throws IllegalArgumentException   when the text is null or empty
     * @throws UnsupportedKernelException when the driver cannot JIT the PTX for this device
     * @throws IllegalStateException      after {@link #close()}
     */
    public synchronized CUmodule getPtx(String ptxText) {
        ensureOpen();
        if (ptxText == null || ptxText.isEmpty()) {
            throw new IllegalArgumentException("PTX text must not be null or empty");
        }
        String key = sha256(ptxText);
        CUmodule module = byPtxHash.get(key);
        if (module != null) {
            return module;
        }
        module = load(nulTerminated(ptxText.getBytes(StandardCharsets.UTF_8)), "cuModuleLoadData(PTX " + key + ")");
        byPtxHash.put(key, module);
        return module;
    }

    /** Number of loaded modules. */
    public synchronized int size() {
        return byResource.size() + byPtxHash.size();
    }

    /**
     * Unloads every module; idempotent. Failures are logged and the remaining modules are still unloaded.
     */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        List<CUmodule> modules = new ArrayList<>(byResource.values());
        modules.addAll(byPtxHash.values());
        byResource.clear();
        byPtxHash.clear();
        if (modules.isEmpty() || context.isReleased()) {
            return;
        }
        context.makeCurrent();
        for (CUmodule module : modules) {
            try {
                CudaErrors.check("cuModuleUnload", () -> cuModuleUnload(module));
            } catch (RuntimeException e) {
                LOGGER.warn("Failed to unload CUDA module: {}", e.getMessage());
            }
        }
    }

    private CUmodule load(byte[] image, String op) {
        context.makeCurrent();
        CUmodule module = new CUmodule();
        CudaErrors.check(op, () -> cuModuleLoadData(module, image));
        return module;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("CUDA module cache is closed");
        }
    }

    private static String normalize(String resourcePath) {
        if (resourcePath == null || resourcePath.trim().isEmpty()) {
            throw new IllegalArgumentException("CUDA module resource path must not be null or empty");
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
            throw new UnsupportedKernelException("CUDA module resource " + path
                    + " is not on the classpath (CUDA kernels were not built, or the build is incomplete)");
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
                throw new UnsupportedKernelException("CUDA module resource " + path + " is empty");
            }
            return bytes;
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read CUDA module resource " + path, e);
        }
    }

    private static InputStream open(String path) {
        ClassLoader own = CudaModuleCache.class.getClassLoader();
        InputStream in = own == null ? ClassLoader.getSystemResourceAsStream(path) : own.getResourceAsStream(path);
        if (in != null) {
            return in;
        }
        ClassLoader contextLoader = Thread.currentThread().getContextClassLoader();
        return contextLoader == null || contextLoader == own ? null : contextLoader.getResourceAsStream(path);
    }

    private static byte[] nulTerminated(byte[] text) {
        if (text.length > 0 && text[text.length - 1] == 0) {
            return text;
        }
        return Arrays.copyOf(text, text.length + 1);
    }

    /** Lowercase hex SHA-256 of the UTF-8 text; the identity of a legacy PTX source. */
    static String sha256(String text) {
        try {
            byte[] hash = MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder(hash.length * 2);
            for (byte b : hash) {
                hex.append(Character.forDigit((b >> 4) & 0xF, 16)).append(Character.forDigit(b & 0xF, 16));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable in this JVM", e);
        }
    }
}
