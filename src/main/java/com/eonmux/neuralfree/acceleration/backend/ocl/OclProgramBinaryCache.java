package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.ThreadSafe;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.jocl.CLException;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_program;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.FileTime;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

import static org.jocl.CL.*;

/**
 * On-disk cache of compiled OpenCL program binaries, so a program that was built once is not
 * rebuilt from source on the next model load.
 *
 * <p><b>Key.</b> One file per (device, driver, source, options):
 * {@code <dir>/<vendor>-<device>-<driver>-<sha256(source, options)>.bin}. The vendor, device and
 * driver strings come from {@code CL_DEVICE_VENDOR}, {@code CL_DEVICE_NAME} and
 * {@code CL_DRIVER_VERSION}; a driver update therefore never reuses a binary of the old driver.
 * The readable prefix is shortened for the file name, so the digest also covers the three full
 * strings: it hashes the UTF-8 vendor, device, driver, source and build options, each followed by
 * a NUL separator.</p>
 *
 * <p><b>Format.</b> A file holds a header (magic, format version, payload length, SHA-256 of
 * the payload) followed by the driver's {@code CL_PROGRAM_BINARIES} bytes. A file whose header
 * or digest does not verify is deleted and reported as a miss.</p>
 *
 * <p><b>Fallback.</b> {@link #createProgram} returns null when the driver rejects the binary
 * ({@code CL_INVALID_BINARY} or any other creation error); the caller then invalidates the entry
 * and builds from source. Writes are atomic (temporary file, then move), so concurrent processes
 * never observe a partial file. The directory is pruned oldest-first above
 * {@link #MAX_CACHE_BYTES}; hits refresh the file time.</p>
 *
 * <p>Location {@code ~/.cache/neuralfree/opencl}. Disabled with
 * {@code -Dneuralfree.opencl.binaryCache=false}. Every I/O failure degrades to a cache miss and
 * never fails a kernel load.</p>
 */
@ThreadSafe("Holds only immutable state; file updates are atomic moves and concurrent prunes tolerate missing files.")
public final class OclProgramBinaryCache {
    private static final Logger LOGGER = LoggingUtil.getLogger(OclProgramBinaryCache.class);

    /** System property that disables the cache when set to {@code false}. */
    public static final String ENABLED_PROPERTY = "neuralfree.opencl.binaryCache";
    /** Upper bound on the total size of cached binaries; the oldest files are deleted above it. */
    static final long MAX_CACHE_BYTES = 1L << 30;

    private static final byte[] MAGIC = "NFCLBIN\u0001".getBytes(StandardCharsets.US_ASCII);
    private static final int FORMAT_VERSION = 1;
    private static final int DIGEST_BYTES = 32;
    private static final int HEADER_BYTES = MAGIC.length + Integer.BYTES + Long.BYTES + DIGEST_BYTES;
    private static final String SUFFIX = ".bin";
    private static final int MAX_LABEL_CHARS = 48;
    private static final String TEMP_PREFIX = "nfclbin";
    private static final String TEMP_SUFFIX = ".tmp";
    /** Age after which a temporary file is an orphan of an interrupted store and is deleted. */
    private static final long STALE_TEMP_MILLIS = 60L * 60L * 1000L;

    private final Path directory;
    private final String devicePrefix;
    private final byte[] deviceFingerprint;
    private final boolean enabled;

    /**
     * @param directory     cache directory; created on the first store
     * @param vendor        {@code CL_DEVICE_VENDOR}
     * @param deviceName    {@code CL_DEVICE_NAME}
     * @param driverVersion {@code CL_DRIVER_VERSION}
     * @param enabled       false turns every operation into a no-op miss
     */
    public OclProgramBinaryCache(Path directory, String vendor, String deviceName, String driverVersion,
                                 boolean enabled) {
        if (enabled && directory == null) {
            throw new IllegalArgumentException("binary cache directory must not be null when the cache is enabled");
        }
        this.directory = directory;
        this.devicePrefix = label(vendor) + "-" + label(deviceName) + "-" + label(driverVersion);
        this.deviceFingerprint = (nullToEmpty(vendor) + '\0' + nullToEmpty(deviceName) + '\0'
                + nullToEmpty(driverVersion) + '\0').getBytes(StandardCharsets.UTF_8);
        this.enabled = enabled;
    }

    /**
     * The cache for one device at the default location, enabled unless
     * {@code -Dneuralfree.opencl.binaryCache=false}.
     */
    public static OclProgramBinaryCache forDevice(String vendor, String deviceName, String driverVersion) {
        boolean enabled = !"false".equalsIgnoreCase(System.getProperty(ENABLED_PROPERTY, "true").trim());
        Path dir = null;
        if (enabled) {
            String home = System.getProperty("user.home");
            if (home == null || home.isEmpty()) {
                LOGGER.warn("OpenCL program binary cache disabled: user.home is not set");
                enabled = false;
            } else {
                dir = Paths.get(home, ".cache", "neuralfree", "opencl");
            }
        }
        return new OclProgramBinaryCache(dir, vendor, deviceName, driverVersion, enabled);
    }

    public boolean isEnabled() {
        return enabled;
    }

    /** Cache directory, or null when the cache is disabled at construction. */
    public Path directory() {
        return directory;
    }

    /**
     * File name (without directory) for a program built from {@code source} with {@code options}.
     * Deterministic for equal inputs on the same device and driver.
     */
    public String keyFor(String source, String options) {
        if (source == null) {
            throw new IllegalArgumentException("program source must not be null");
        }
        MessageDigest md = sha256();
        md.update(deviceFingerprint);
        md.update(source.getBytes(StandardCharsets.UTF_8));
        md.update((byte) 0);
        md.update(nullToEmpty(options).getBytes(StandardCharsets.UTF_8));
        md.update((byte) 0);
        return devicePrefix + "-" + hex(md.digest()) + SUFFIX;
    }

    /**
     * Returns the verified binary stored under {@code key}, or null on a miss, when disabled, or
     * when the file is unreadable or corrupt (a corrupt file is deleted).
     */
    public byte[] load(String key) {
        if (!enabled) {
            return null;
        }
        Path file = resolve(key);
        if (!Files.isRegularFile(file)) {
            return null;
        }
        byte[] data;
        try {
            data = Files.readAllBytes(file);
        } catch (IOException e) {
            LOGGER.warn("OpenCL binary cache: cannot read {}: {}", file, e.toString());
            return null;
        }
        byte[] payload = decode(data);
        if (payload == null) {
            LOGGER.warn("OpenCL binary cache: discarding corrupt entry {}", file);
            delete(file);
            return null;
        }
        try {
            Files.setLastModifiedTime(file, FileTime.fromMillis(System.currentTimeMillis()));
        } catch (IOException e) {
            LOGGER.debug("OpenCL binary cache: cannot refresh the time of {}: {}", file, e.toString());
        }
        return payload;
    }

    /**
     * Stores {@code binary} under {@code key} atomically, then prunes the directory. Failures are
     * logged and otherwise ignored.
     */
    public void store(String key, byte[] binary) {
        if (!enabled || binary == null || binary.length == 0) {
            return;
        }
        Path file = resolve(key);
        Path tmp = null;
        try {
            Files.createDirectories(directory);
            tmp = Files.createTempFile(directory, TEMP_PREFIX, TEMP_SUFFIX);
            Files.write(tmp, encode(binary));
            try {
                Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (java.nio.file.AtomicMoveNotSupportedException e) {
                Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
            }
            tmp = null;
            LOGGER.debug("OpenCL binary cache: stored {} ({} bytes)", file, binary.length);
        } catch (IOException | SecurityException e) {
            LOGGER.warn("OpenCL binary cache: cannot write {}: {}", file, e.toString());
        } finally {
            if (tmp != null) {
                delete(tmp);
            }
        }
        prune();
    }

    /** Deletes the entry stored under {@code key}, if any. */
    public void invalidate(String key) {
        if (enabled) {
            delete(resolve(key));
        }
    }

    /**
     * Creates a program for {@code device} from a cached binary. The caller still has to call
     * {@code clBuildProgram} on the result.
     *
     * @return the program, or null when the driver rejects the binary
     */
    public static cl_program createProgram(cl_context context, cl_device_id device, byte[] binary) {
        int[] binaryStatus = {CL_SUCCESS};
        int[] error = {CL_SUCCESS};
        cl_program program;
        try {
            program = clCreateProgramWithBinary(context, 1, new cl_device_id[]{device},
                    new long[]{binary.length}, new byte[][]{binary}, binaryStatus, error);
        } catch (CLException e) {
            LOGGER.info("OpenCL binary cache: driver rejected the cached binary ({}); rebuilding from source",
                    stringFor_errorCode(e.getStatus()));
            return null;
        }
        if (error[0] != CL_SUCCESS || binaryStatus[0] != CL_SUCCESS || program == null) {
            int code = error[0] != CL_SUCCESS ? error[0] : binaryStatus[0];
            LOGGER.info("OpenCL binary cache: driver rejected the cached binary ({}); rebuilding from source",
                    stringFor_errorCode(code));
            if (program != null) {
                releaseQuietly(program);
            }
            return null;
        }
        return program;
    }

    /**
     * Extracts the built binary of {@code program} for {@code device} through
     * {@code CL_PROGRAM_BINARIES}.
     *
     * @return the binary, or null when the driver provides none or a query fails
     */
    public static byte[] extractBinary(cl_program program, cl_device_id device) {
        try {
            int[] numDevices = new int[1];
            int status = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, Sizeof.cl_uint,
                    Pointer.to(numDevices), null);
            if (status != CL_SUCCESS || numDevices[0] <= 0) {
                return null;
            }
            int n = numDevices[0];
            cl_device_id[] devices = new cl_device_id[n];
            for (int i = 0; i < n; i++) {
                devices[i] = new cl_device_id();
            }
            status = clGetProgramInfo(program, CL_PROGRAM_DEVICES, (long) n * Sizeof.cl_device_id,
                    Pointer.to(devices), null);
            if (status != CL_SUCCESS) {
                return null;
            }
            int index = -1;
            for (int i = 0; i < n; i++) {
                if (device.equals(devices[i])) {
                    index = i;
                    break;
                }
            }
            if (index < 0) {
                return null;
            }
            long[] sizes = sizeTArray(program, CL_PROGRAM_BINARY_SIZES, n);
            if (sizes == null || sizes[index] <= 0 || sizes[index] > Integer.MAX_VALUE - HEADER_BYTES) {
                return null;
            }
            byte[][] binaries = new byte[n][];
            Pointer[] targets = new Pointer[n];
            for (int i = 0; i < n; i++) {
                binaries[i] = new byte[(int) Math.max(1, sizes[i])];
                targets[i] = Pointer.to(binaries[i]);
            }
            status = clGetProgramInfo(program, CL_PROGRAM_BINARIES, (long) n * Sizeof.POINTER,
                    Pointer.to(targets), null);
            if (status != CL_SUCCESS) {
                return null;
            }
            return binaries[index].length == sizes[index]
                    ? binaries[index]
                    : Arrays.copyOf(binaries[index], (int) sizes[index]);
        } catch (CLException e) {
            LOGGER.debug("OpenCL binary cache: CL_PROGRAM_BINARIES unavailable: {}",
                    stringFor_errorCode(e.getStatus()));
            return null;
        }
    }

    private static long[] sizeTArray(cl_program program, int param, int n) {
        ByteBuffer buf = ByteBuffer.allocate(n * Sizeof.size_t).order(ByteOrder.nativeOrder());
        int status = clGetProgramInfo(program, param, (long) n * Sizeof.size_t, Pointer.to(buf), null);
        if (status != CL_SUCCESS) {
            return null;
        }
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            out[i] = Sizeof.size_t == 4 ? (buf.getInt(i * 4) & 0xFFFFFFFFL) : buf.getLong(i * 8);
        }
        return out;
    }

    static byte[] encode(byte[] payload) {
        ByteBuffer out = ByteBuffer.allocate(HEADER_BYTES + payload.length).order(ByteOrder.LITTLE_ENDIAN);
        out.put(MAGIC);
        out.putInt(FORMAT_VERSION);
        out.putLong(payload.length);
        out.put(sha256().digest(payload));
        out.put(payload);
        return out.array();
    }

    /** Returns the verified payload, or null when the header, length or digest does not match. */
    static byte[] decode(byte[] data) {
        if (data == null || data.length <= HEADER_BYTES) {
            return null;
        }
        ByteBuffer in = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        byte[] magic = new byte[MAGIC.length];
        in.get(magic);
        if (!Arrays.equals(magic, MAGIC) || in.getInt() != FORMAT_VERSION) {
            return null;
        }
        long length = in.getLong();
        if (length != data.length - HEADER_BYTES) {
            return null;
        }
        byte[] digest = new byte[DIGEST_BYTES];
        in.get(digest);
        byte[] payload = Arrays.copyOfRange(data, HEADER_BYTES, data.length);
        return MessageDigest.isEqual(digest, sha256().digest(payload)) ? payload : null;
    }

    private void prune() {
        deleteStaleTempFiles();
        List<Path> files = new ArrayList<>();
        long total = 0;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(directory, "*" + SUFFIX)) {
            for (Path p : stream) {
                try {
                    total += Files.size(p);
                    files.add(p);
                } catch (IOException e) {
                    LOGGER.debug("OpenCL binary cache: cannot stat {}: {}", p, e.toString());
                }
            }
        } catch (IOException | SecurityException e) {
            LOGGER.debug("OpenCL binary cache: cannot list {}: {}", directory, e.toString());
            return;
        }
        if (total <= MAX_CACHE_BYTES) {
            return;
        }
        files.sort(Comparator.comparingLong(OclProgramBinaryCache::modifiedMillis));
        for (Path p : files) {
            if (total <= MAX_CACHE_BYTES) {
                break;
            }
            try {
                long size = Files.size(p);
                if (Files.deleteIfExists(p)) {
                    total -= size;
                    LOGGER.debug("OpenCL binary cache: evicted {}", p);
                }
            } catch (IOException e) {
                LOGGER.debug("OpenCL binary cache: cannot evict {}: {}", p, e.toString());
            }
        }
    }

    /** Deletes temporary files left behind by a store that was interrupted before its move. */
    private void deleteStaleTempFiles() {
        long cutoff = System.currentTimeMillis() - STALE_TEMP_MILLIS;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(directory, TEMP_PREFIX + "*" + TEMP_SUFFIX)) {
            for (Path p : stream) {
                if (modifiedMillis(p) < cutoff) {
                    delete(p);
                }
            }
        } catch (IOException | SecurityException e) {
            LOGGER.debug("OpenCL binary cache: cannot list {}: {}", directory, e.toString());
        }
    }

    private static long modifiedMillis(Path p) {
        try {
            return Files.getLastModifiedTime(p).toMillis();
        } catch (IOException e) {
            return Long.MIN_VALUE;
        }
    }

    private Path resolve(String key) {
        if (key == null || key.isEmpty() || key.indexOf('/') >= 0 || key.indexOf('\\') >= 0
                || !key.endsWith(SUFFIX)) {
            throw new IllegalArgumentException("invalid OpenCL binary cache key: " + key);
        }
        return directory.resolve(key);
    }

    private static void delete(Path file) {
        try {
            Files.deleteIfExists(file);
        } catch (IOException | SecurityException e) {
            LOGGER.debug("OpenCL binary cache: cannot delete {}: {}", file, e.toString());
        }
    }

    private static void releaseQuietly(cl_program program) {
        try {
            clReleaseProgram(program);
        } catch (CLException e) {
            LOGGER.debug("clReleaseProgram failed: {}", stringFor_errorCode(e.getStatus()));
        }
    }

    /** Lower-case file-name-safe form of a driver string, bounded in length. */
    static String label(String raw) {
        String s = raw == null ? "" : raw.trim().toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9.]+", "_");
        s = s.replaceAll("^_+|_+$", "");
        if (s.isEmpty()) {
            return "unknown";
        }
        return s.length() > MAX_LABEL_CHARS ? s.substring(0, MAX_LABEL_CHARS) : s;
    }

    private static String nullToEmpty(String s) {
        return s == null ? "" : s;
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is required by every Java platform", e);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(Character.forDigit((b >> 4) & 0xF, 16)).append(Character.forDigit(b & 0xF, 16));
        }
        return sb.toString();
    }
}
