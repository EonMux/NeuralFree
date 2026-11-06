package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.sun.jna.Function;
import com.sun.jna.Memory;
import com.sun.jna.NativeLibrary;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Free VRAM from WDDM's adapter segment statistics on Windows.
 *
 * <p>{@code hipMemGetInfo} and an OpenCL device's memory size count only what this process may
 * allocate; on WDDM other processes (the desktop compositor, a browser) already hold part of the
 * card, and past the physical VRAM WDDM moves allocations into system memory. The kernel-mode
 * {@code D3DKMTQueryStatistics} call reports every process: the local (non-aperture) segments'
 * commit limit minus their resident bytes is the VRAM that is actually free.</p>
 *
 * <p>The adapter is matched to the device's PCI address through {@code D3DKMTEnumAdapters2} and
 * {@code D3DKMTQueryAdapterInfo} ({@code KMTQAITYPE_ADAPTERADDRESS}); all three entry points are
 * taken from {@code gdi32} at run time, so the class loads and degrades to -1 on non-Windows
 * systems and on Windows versions that do not export them.</p>
 */
public final class WddmAdapterStatistics {

    private static final Logger LOGGER = LogManager.getLogger(WddmAdapterStatistics.class);

    /** A query that produced no figure; a completed query may legitimately report zero free bytes. */
    private static final long UNAVAILABLE = -1;

    /** {@code D3DKMT_QUERYSTATISTICS_ADAPTER}. */
    private static final int QUERY_ADAPTER = 0;
    /** {@code D3DKMT_QUERYSTATISTICS_SEGMENT}. */
    private static final int QUERY_SEGMENT = 3;
    /** {@code KMTQAITYPE_ADAPTERADDRESS}. */
    private static final int QUERY_ADAPTER_ADDRESS = 6;
    /** {@code STATUS_SUCCESS}. */
    private static final int STATUS_SUCCESS = 0;

    /** {@code sizeof(D3DKMT_ADAPTERINFO)} on x64: handle, LUID, source count, preference flag. */
    private static final int ADAPTER_INFO_BYTES = 20;
    /** {@code sizeof(D3DKMT_QUERYSTATISTICS_RESULT)}: its largest member is the adapter information union. */
    private static final int QUERY_RESULT_BYTES = 776;
    /** {@code sizeof(DXGK_ADAPTER_ADDRESS)}: bus, device and function numbers. */
    private static final int ADAPTER_ADDRESS_BYTES = 12;
    /** {@code offsetof(D3DKMT_QUERYSTATISTICS_SEGMENT_INFORMATION, BytesResident)}. */
    private static final int SEGMENT_BYTES_RESIDENT_OFFSET = 16;
    /** {@code offsetof(D3DKMT_QUERYSTATISTICS_SEGMENT_INFORMATION, Aperture)}. */
    private static final int SEGMENT_APERTURE_OFFSET = 40;
    /** At most this many adapters are read from {@code D3DKMTEnumAdapters2}. */
    private static final int MAX_ADAPTERS = 64;

    /** {@code gdi32}, or null when this is not Windows or the library cannot be loaded. */
    private static final NativeLibrary GDI32 = loadGdi32();

    private WddmAdapterStatistics() {
    }

    /**
     * Free VRAM of {@code identity}'s adapter with every process counted, or -1 when this is not
     * Windows, the identity has no PCI bus id, the entry points are missing, no adapter matches the
     * address, or a query fails. A completed reading may legitimately be 0.
     */
    public static long freeBytes(DeviceIdentity identity) {
        if (GDI32 == null || identity == null) {
            return UNAVAILABLE;
        }
        try {
            return query(identity);
        } catch (RuntimeException e) {
            LOGGER.debug("D3DKMT: query for {} failed: {}", identity, e.toString());
            return UNAVAILABLE;
        }
    }

    private static long query(DeviceIdentity identity) {
        int[] wanted = parsePciAddress(identity.pciBusId());
        if (wanted == null) {
            LOGGER.debug("D3DKMT: '{}' has no usable PCI address", identity);
            return UNAVAILABLE;
        }
        Function enumerate = function("D3DKMTEnumAdapters2");
        Function queryAdapter = function("D3DKMTQueryAdapterInfo");
        Function queryStatistics = function("D3DKMTQueryStatistics");
        if (enumerate == null || queryAdapter == null || queryStatistics == null) {
            return UNAVAILABLE;
        }
        List<AdapterInfo> adapters = enumerateAdapters(enumerate);
        for (AdapterInfo adapter : adapters) {
            int[] address = adapterAddress(queryAdapter, adapter);
            if (address == null || !sameAddress(address, wanted)) {
                continue;
            }
            long free = segments(queryStatistics, adapter.adapterLuid);
            if (free >= 0) {
                LOGGER.debug("D3DKMT: adapter {} ({}) reports {} MiB free",
                        adapter.adapterLuid, identity.pciBusId(), free >> 20);
                return free;
            }
        }
        LOGGER.debug("D3DKMT: no adapter statistics for PCI address {}", identity.pciBusId());
        return UNAVAILABLE;
    }

    private static NativeLibrary loadGdi32() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (!os.contains("win")) {
            return null;
        }
        try {
            return NativeLibrary.getInstance("gdi32");
        } catch (Throwable t) {
            LOGGER.debug("D3DKMT: gdi32 is unavailable: {}", t.toString());
            return null;
        }
    }

    private static Function function(String name) {
        try {
            return GDI32.getFunction(name);
        } catch (Throwable t) {
            LOGGER.debug("D3DKMT: gdi32 does not export {}: {}", name, t.toString());
            return null;
        }
    }

    /** Every adapter the kernel enumerates; empty when the first or second call fails. */
    private static List<AdapterInfo> enumerateAdapters(Function enumerate) {
        EnumAdapters2 head = new EnumAdapters2();
        head.numAdapters = 0;
        head.adapters = null;
        head.write();
        int status = enumerate.invokeInt(new Object[]{head.getPointer()});
        head.read();
        int count = head.numAdapters;
        if (count <= 0 || count > MAX_ADAPTERS) {
            LOGGER.debug("D3DKMTEnumAdapters2 reported {} adapters (status 0x{})",
                    count, Integer.toHexString(status));
            return Arrays.<AdapterInfo>asList();
        }
        Memory array = new Memory((long) count * ADAPTER_INFO_BYTES);
        head.adapters = array;
        head.write();
        status = enumerate.invokeInt(new Object[]{head.getPointer()});
        if (status != STATUS_SUCCESS) {
            LOGGER.debug("D3DKMTEnumAdapters2 failed with status 0x{}", Integer.toHexString(status));
            return Arrays.<AdapterInfo>asList();
        }
        AdapterInfo[] adapters = new AdapterInfo[count];
        for (int i = 0; i < count; i++) {
            adapters[i] = new AdapterInfo(array.share((long) i * ADAPTER_INFO_BYTES));
        }
        return Arrays.asList(adapters);
    }

    /** The adapter's PCI address, or null when the driver does not answer the address query. */
    private static int[] adapterAddress(Function queryAdapter, AdapterInfo adapter) {
        Memory address = new Memory(ADAPTER_ADDRESS_BYTES);
        address.clear();
        QueryAdapterInfo query = new QueryAdapterInfo();
        query.hAdapter = adapter.hAdapter;
        query.type = QUERY_ADAPTER_ADDRESS;
        query.privateDriverData = address;
        query.privateDriverDataSize = ADAPTER_ADDRESS_BYTES;
        query.write();
        if (queryAdapter.invokeInt(new Object[]{query.getPointer()}) != STATUS_SUCCESS) {
            return null;
        }
        return new int[]{address.getInt(0), address.getInt(4), address.getInt(8)};
    }

    /**
     * Free bytes over the adapter's local segments: their commit limit minus resident bytes. The
     * aperture segments (system memory seen through the GPU) are skipped. -1 when a query fails or
     * the adapter reports no local segment.
     */
    private static long segments(Function queryStatistics, Luid luid) {
        QueryStatistics adapter = new QueryStatistics();
        adapter.type = QUERY_ADAPTER;
        adapter.adapterLuid.lowPart = luid.lowPart;
        adapter.adapterLuid.highPart = luid.highPart;
        adapter.write();
        if (queryStatistics.invokeInt(new Object[]{adapter.getPointer()}) != STATUS_SUCCESS) {
            return UNAVAILABLE;
        }
        adapter.read();
        int count = ByteBuffer.wrap(adapter.queryResult).order(ByteOrder.nativeOrder()).getInt(0);
        if (count <= 0) {
            return UNAVAILABLE;
        }
        long limit = 0;
        long resident = 0;
        for (int i = 0; i < count; i++) {
            QueryStatistics segment = new QueryStatistics();
            segment.type = QUERY_SEGMENT;
            segment.adapterLuid.lowPart = luid.lowPart;
            segment.adapterLuid.highPart = luid.highPart;
            segment.querySegmentId = i;
            segment.write();
            if (queryStatistics.invokeInt(new Object[]{segment.getPointer()}) != STATUS_SUCCESS) {
                return UNAVAILABLE;
            }
            segment.read();
            ByteBuffer result = ByteBuffer.wrap(segment.queryResult).order(ByteOrder.nativeOrder());
            if (result.getInt(SEGMENT_APERTURE_OFFSET) != 0) {
                continue;
            }
            limit += result.getLong(0);
            resident += result.getLong(SEGMENT_BYTES_RESIDENT_OFFSET);
        }
        if (limit <= 0) {
            return UNAVAILABLE;
        }
        return resident < limit ? limit - resident : 0;
    }

    /**
     * Free bytes encoded in one segment's statistics result: commit limit minus resident bytes, or
     * -1 for an aperture segment.
     */
    static long freeFromSegmentResult(byte[] result) {
        ByteBuffer buffer = ByteBuffer.wrap(result).order(ByteOrder.nativeOrder());
        if (buffer.getInt(SEGMENT_APERTURE_OFFSET) != 0) {
            return UNAVAILABLE;
        }
        long limit = buffer.getLong(0);
        long resident = buffer.getLong(SEGMENT_BYTES_RESIDENT_OFFSET);
        return resident < limit ? limit - resident : 0;
    }

    /**
     * Splits a normalised {@code [domain:]bus:device.function} address into bus, device and
     * function, or returns null when it is malformed or out of range.
     */
    static int[] parsePciAddress(String pciBusId) {
        if (pciBusId == null) {
            return null;
        }
        String[] parts = pciBusId.trim().split(":");
        if (parts.length < 2 || parts.length > 3) {
            return null;
        }
        String[] deviceFunction = parts[parts.length - 1].split("\\.");
        if (deviceFunction.length != 2) {
            return null;
        }
        try {
            if (parts.length == 3) {
                int domain = Integer.parseInt(parts[0], 16);
                if (domain < 0 || domain > 0xFFFF) {
                    return null;
                }
            }
            int bus = Integer.parseInt(parts[parts.length - 2], 16);
            int device = Integer.parseInt(deviceFunction[0], 16);
            int function = Integer.parseInt(deviceFunction[1], 16);
            if (bus < 0 || bus > 0xFF || device < 0 || device > 0x1F || function < 0 || function > 0x7) {
                return null;
            }
            return new int[]{bus, device, function};
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static boolean sameAddress(int[] adapter, int[] wanted) {
        return adapter[0] == wanted[0] && adapter[1] == wanted[1] && adapter[2] == wanted[2];
    }

    /** Base for the native structures; exposes the computed field offset to the layout tests. */
    public abstract static class WindowsStructure extends Structure {
        public WindowsStructure() {
        }

        public WindowsStructure(Pointer pointer) {
            super(pointer);
        }

        int offsetOf(String field) {
            return fieldOffset(field);
        }
    }

    /** {@code LUID} as the kernel uses it: two 32-bit words. */
    public static final class Luid extends WindowsStructure {
        public int lowPart;
        public int highPart;

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("lowPart", "highPart");
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "%08x:%08x", highPart, lowPart);
        }
    }

    /** {@code D3DKMT_ENUMADAPTERS2}: in/out count and a pointer to the caller's array. */
    public static final class EnumAdapters2 extends WindowsStructure {
        public int numAdapters;
        public Pointer adapters;

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("numAdapters", "adapters");
        }
    }

    /** {@code D3DKMT_ADAPTERINFO}: one entry of the enum array. */
    public static final class AdapterInfo extends WindowsStructure {
        public int hAdapter;
        public Luid adapterLuid = new Luid();
        public int numOfSources;
        public int precisePresentRegionsPreferred;

        public AdapterInfo() {
        }

        public AdapterInfo(Pointer pointer) {
            super(pointer);
            read();
        }

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("hAdapter", "adapterLuid", "numOfSources", "precisePresentRegionsPreferred");
        }
    }

    /** {@code D3DKMT_QUERYADAPTERINFO}. */
    public static final class QueryAdapterInfo extends WindowsStructure {
        public int hAdapter;
        public int type;
        public Pointer privateDriverData;
        public int privateDriverDataSize;

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("hAdapter", "type", "privateDriverData", "privateDriverDataSize");
        }
    }

    /**
     * {@code D3DKMT_QUERYSTATISTICS}. The result union is held as its raw bytes because the
     * struct is queried with several result types; the trailing union's first member is the
     * segment id.
     */
    public static final class QueryStatistics extends WindowsStructure {
        public int type;
        public Luid adapterLuid = new Luid();
        public Pointer hProcess;
        public byte[] queryResult = new byte[QUERY_RESULT_BYTES];
        public int querySegmentId;

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("type", "adapterLuid", "hProcess", "queryResult", "querySegmentId");
        }
    }
}
