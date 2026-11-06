package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvVendorCodec;

import java.util.Objects;

/**
 * The device geometry of the vLLM vendor KV cache formats: the side buffer of a cache ({@link GpuKvLayer#vendorSide()})
 * holding its checkpoint KV scales, the Lloyd-Max table of a TurboQuant key stream and the packed side-plane rows,
 * with the head widths the device store kernels support.
 *
 * <p>Every cache type has its device store ({@code LLM_KV_Store_<type>}) and flash attention variants; the vendor
 * formats decode through the planes of their side buffer.</p>
 */
public final class DeviceKvFormats {

    /** Largest TurboQuant key head width the device store rotates (the kernel's private work-item buffer). */
    public static final int TQ_MAX_HEAD_DIM = 256;

    /** F32 values of the Lloyd-Max centroid table at the head of a vendor side buffer (the levels of a 4-bit stream). */
    public static final int SIDE_CTR_FLOATS = 16;

    private DeviceKvFormats() {
    }

    /**
     * Refuses a vendor cache type in a device cache that keeps no side planes (the DFlash draft cache builds its own
     * rows from the main cache's row format): such a cache would decode the packed codes without their scales.
     *
     * @param dtype the K or V row type of the cache
     * @param what  the cache the message names, e.g. {@code "the DFlash draft KV cache"}
     * @throws UnsupportedOperationException naming the type and the host-backed alternatives
     */
    public static void requireKernel(KvDType dtype, String what) {
        Objects.requireNonNull(dtype, "dtype");
        if (!dtype.vendorFormat()) {
            return;
        }
        throw new UnsupportedOperationException(what + " uses the KV cache type " + name(dtype) + ", whose side"
                + " planes that cache does not keep; run it on the CPU backend, or keep the KV caches in host RAM"
                + " (neuralfree.kv.offload=false / --no-kv-offload)");
    }

    /**
     * Checks the device-side geometry of a cache layout.
     *
     * @param layout the layout of the cache
     * @param what   the cache the message names, e.g. {@code "KV cache of layer 3"}
     * @throws IllegalArgumentException naming the field the device kernels cannot address
     */
    public static void requireLayout(KvLayout layout, String what) {
        Objects.requireNonNull(layout, "layout");
        requireDtype(layout.dtypeK(), layout.widthK(), layout.nHeadKv(), what);
        if (!layout.vAliasesK()) {
            requireDtype(layout.dtypeV(), layout.widthV(), layout.nHeadKv(), what);
        }
    }

    /**
     * Checks one side of a cache.
     *
     * @throws IllegalArgumentException for a vendor geometry the device kernels do not address
     */
    public static void requireDtype(KvDType dtype, int width, int heads, String what) {
        Objects.requireNonNull(dtype, "dtype");
        if (!dtype.vendorFormat()) {
            return;
        }
        if (heads <= 0 || width % heads != 0) {
            throw new IllegalArgumentException(what + " uses the KV cache type "
                    + name(dtype) + " over " + width + " values and " + heads + " KV heads, which do not tile");
        }
        int headDim = width / heads;
        if (dtype.nvfp4() && headDim % KvVendorCodec.NVFP4_GROUP != 0) {
            throw new IllegalArgumentException(what + " uses the KV cache type " + name(dtype) + "; its head width "
                    + headDim + " is not a multiple of the " + KvVendorCodec.NVFP4_GROUP + "-value NVFP4 scale"
                    + " group");
        }
        if (dtype.turboQuantMse()) {
            if (!KvVendorCodec.hadamardWidth(headDim)) {
                throw new IllegalArgumentException(what + " uses the KV cache type " + name(dtype) + "; its head"
                        + " width " + headDim + " is not a power of two (the Hadamard rotation)");
            }
            if (headDim > TQ_MAX_HEAD_DIM) {
                throw new IllegalArgumentException(what + " uses the KV cache type " + name(dtype) + "; the device"
                        + " store rotates at most " + TQ_MAX_HEAD_DIM + " values per head, got " + headDim);
            }
        }
        if ((dtype.turboQuantKey() || dtype.turboQuantValue()) && headDim % 8 != 0) {
            throw new IllegalArgumentException(what + " uses the KV cache type " + name(dtype) + "; its TurboQuant"
                    + " streams pack codes in groups of 8, got a head width of " + headDim);
        }
        if (dtype.latentFormat() && width != KvDType.DS_MLA_ROW_VALUES) {
            throw new IllegalArgumentException(what + " uses the KV cache type " + name(dtype) + "; its latent rows"
                    + " hold exactly " + KvDType.DS_MLA_ROW_VALUES + " values, got " + width);
        }
    }

    /**
     * Bytes of one row of the packed side planes of a vendor cache side, the sum of
     * {@link KvDType.SidePlane#rowBytesFor(int, int)} over its planes; 0 for a type without planes and for every
     * non-vendor type.
     */
    public static int planeRowBytes(KvDType dtype, int width, int heads) {
        if (!dtype.vendorFormat()) {
            return 0;
        }
        long bytes = 0;
        for (KvDType.SidePlane plane : dtype.sidePlanes()) {
            bytes += plane.rowBytesFor(width, heads);
        }
        return Math.toIntExact(bytes);
    }

    /** Bytes of one row of the packed K side planes of {@code layout}. */
    public static int kPlaneRowBytes(KvLayout layout) {
        return planeRowBytes(layout.dtypeK(), layout.widthK(), layout.nHeadKv());
    }

    /** Bytes of one row of the packed V side planes of {@code layout}; 0 when V aliases K. */
    public static int vPlaneRowBytes(KvLayout layout) {
        return layout.vAliasesK() ? 0 : planeRowBytes(layout.dtypeV(), layout.widthV(), layout.nHeadKv());
    }

    /**
     * Byte offset of the packed plane rows in the side buffer of a cache with {@code heads} KV heads: behind the
     * {@code 2 * heads} checkpoint scale values and the {@link #SIDE_CTR_FLOATS}-value Lloyd-Max table.
     */
    public static int sidePlaneBase(int heads) {
        return 4 * (2 * heads + SIDE_CTR_FLOATS);
    }

    /** Bytes of one side-buffer row of {@code layout}: the K plane row followed by the V plane row, if separate. */
    public static int sideRowBytes(KvLayout layout) {
        return Math.addExact(kPlaneRowBytes(layout), vPlaneRowBytes(layout));
    }

    /** Byte offset of the K ({@code v} false) or V plane row inside the rows of the side buffer. */
    public static int sidePlaneOffset(KvLayout layout, boolean v) {
        return v ? kPlaneRowBytes(layout) : 0;
    }

    /** Bytes of the whole side buffer of a vendor {@code layout}: the scales and centroid header and every row. */
    public static long sideBytes(KvLayout layout) {
        return sidePlaneBase(layout.nHeadKv()) + (long) layout.rows() * sideRowBytes(layout);
    }

    /**
     * The F32 header of the side buffer of a vendor {@code layout}: {@code 2 * nHeadKv} unit checkpoint scales then
     * the Lloyd-Max table (zeros for the rest of its slot and for a cache that needs none).
     *
     * @throws IllegalArgumentException for a TurboQuant key stream whose head width is not a power of two
     */
    public static float[] sideHeader(KvLayout layout) {
        int heads = layout.nHeadKv();
        float[] header = new float[2 * heads + SIDE_CTR_FLOATS];
        java.util.Arrays.fill(header, 0, 2 * heads, 1.0f);
        float[] table = centroids(layout);
        if (table != null) {
            System.arraycopy(table, 0, header, 2 * heads, table.length);
        }
        return header;
    }

    /**
     * The sorted Lloyd-Max centroids of the K cache of {@code layout}, or null when it needs none.
     *
     * @throws IllegalArgumentException for a TurboQuant key stream whose head width is not a power of two
     */
    public static float[] centroids(KvLayout layout) {
        KvDType dtype = layout.dtypeK();
        if (!dtype.turboQuantMse()) {
            return null;
        }
        int headDim = layout.widthK() / layout.nHeadKv();
        int bits = dtype == KvDType.TQ_3BIT_KEYS ? 3 : 4;
        return KvVendorCodec.centroids(headDim, bits);
    }

    private static String name(KvDType dtype) {
        return dtype.name().toLowerCase(java.util.Locale.ROOT);
    }
}
