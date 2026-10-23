package com.eonmux.neuralfree.acceleration.backend;

import java.util.Locale;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Immutable cross-API identity of one physical device.
 *
 * <p>The same GPU is usually visible through OpenCL, CUDA and Vulkan at once.
 * Backend selection uses {@link #samePhysicalDevice(DeviceIdentity)} to recognise
 * those views as one device. The PCI bus id is the strongest key, then the
 * device UUID, then the exact device name.</p>
 *
 * <p>Formats are normalised at construction: the PCI bus id is
 * {@code "dddd:bb:dd.f"} in lowercase hex, and the UUID is 32 lowercase hex
 * characters without dashes.</p>
 */
public final class DeviceIdentity {

    /** Value of {@link #vendorId()} when the PCI vendor id is unknown. */
    public static final int UNKNOWN_VENDOR = -1;

    private static final Pattern PCI_BUS_ID = Pattern.compile(
            "^(?:([0-9a-fA-F]{1,8}):)?([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,2})\\.([0-7])$");
    private static final Pattern UUID_HEX = Pattern.compile("^[0-9a-f]{32}$");
    private static final String ALL_ZERO_UUID = "00000000000000000000000000000000";
    private static final char[] HEX = "0123456789abcdef".toCharArray();

    private final String name;
    private final int vendorId;
    private final String pciBusId;
    private final String uuid;

    /**
     * @param name     device name as the driver reports it; never null
     * @param vendorId vendor id as the API reports it: a 16-bit PCI vendor id, a Khronos vendor id
     *                 above {@code 0xFFFF} for non-PCI devices (for example Mesa {@code 0x10005}), or
     *                 {@link #UNKNOWN_VENDOR}
     * @param pciBusId PCI address as {@code [domain:]bus:device.function} in hex, or null when unknown
     * @param uuid     device UUID as 32 hex characters (dashes allowed), or null when unknown;
     *                 an all-zero UUID counts as unknown
     * @throws IllegalArgumentException when the name is null, the vendor id is negative (other than
     *                                  {@link #UNKNOWN_VENDOR}), or an address or UUID is malformed
     */
    public DeviceIdentity(String name, int vendorId, String pciBusId, String uuid) {
        if (name == null) {
            throw new IllegalArgumentException("device name must not be null");
        }
        if (vendorId < UNKNOWN_VENDOR) {
            throw new IllegalArgumentException("vendor id must be non-negative or UNKNOWN_VENDOR: " + vendorId);
        }
        this.name = name.trim();
        this.vendorId = vendorId;
        this.pciBusId = normalizePciBusId(pciBusId);
        this.uuid = normalizeUuid(uuid);
    }

    /** An identity that knows only the device name. */
    public static DeviceIdentity ofName(String name) {
        return new DeviceIdentity(name, UNKNOWN_VENDOR, null, null);
    }

    /**
     * Formats a PCI address as {@code "dddd:bb:dd.f"}.
     *
     * @throws IllegalArgumentException when a component is out of range
     */
    public static String formatPciBusId(int domain, int bus, int device, int function) {
        if (domain < 0 || domain > 0xFFFF || bus < 0 || bus > 0xFF
                || device < 0 || device > 0x1F || function < 0 || function > 0x7) {
            throw new IllegalArgumentException(String.format(Locale.ROOT,
                    "PCI address out of range: domain=%d bus=%d device=%d function=%d",
                    domain, bus, device, function));
        }
        return String.format(Locale.ROOT, "%04x:%02x:%02x.%x", domain, bus, device, function);
    }

    /**
     * Encodes a 16-byte device UUID as 32 lowercase hex characters.
     *
     * @throws IllegalArgumentException when the array is not 16 bytes long
     */
    public static String hexUuid(byte[] uuidBytes) {
        if (uuidBytes == null || uuidBytes.length != 16) {
            throw new IllegalArgumentException("device UUID must be 16 bytes, got "
                    + (uuidBytes == null ? "null" : uuidBytes.length + " bytes"));
        }
        char[] out = new char[32];
        for (int i = 0; i < 16; i++) {
            out[2 * i] = HEX[(uuidBytes[i] >> 4) & 0xF];
            out[2 * i + 1] = HEX[uuidBytes[i] & 0xF];
        }
        return new String(out);
    }

    public String name() {
        return name;
    }

    public int vendorId() {
        return vendorId;
    }

    /** Normalised PCI address, or null when unknown. */
    public String pciBusId() {
        return pciBusId;
    }

    /** Normalised 32-character hex UUID, or null when unknown. */
    public String uuid() {
        return uuid;
    }

    /**
     * True when both identities describe the same physical device.
     *
     * <p>When both PCI bus ids are known they alone decide. Otherwise, when both
     * UUIDs are known they alone decide. Otherwise the non-empty names must be
     * equal. Two identities with nothing comparable are never the same device.</p>
     */
    public boolean samePhysicalDevice(DeviceIdentity other) {
        if (other == null) {
            return false;
        }
        if (pciBusId != null && other.pciBusId != null) {
            return pciBusId.equals(other.pciBusId);
        }
        if (uuid != null && other.uuid != null) {
            return uuid.equals(other.uuid);
        }
        return !name.isEmpty() && name.equals(other.name);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof DeviceIdentity)) {
            return false;
        }
        DeviceIdentity that = (DeviceIdentity) o;
        return vendorId == that.vendorId
                && name.equals(that.name)
                && Objects.equals(pciBusId, that.pciBusId)
                && Objects.equals(uuid, that.uuid);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, vendorId, pciBusId, uuid);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("DeviceIdentity{name='").append(name).append('\'');
        if (vendorId != UNKNOWN_VENDOR) {
            sb.append(", vendor=0x").append(String.format(Locale.ROOT, "%04x", vendorId));
        }
        if (pciBusId != null) {
            sb.append(", pci=").append(pciBusId);
        }
        if (uuid != null) {
            sb.append(", uuid=").append(uuid);
        }
        return sb.append('}').toString();
    }

    private static String normalizePciBusId(String raw) {
        if (raw == null) {
            return null;
        }
        String trimmed = raw.trim();
        if (trimmed.isEmpty()) {
            return null;
        }
        Matcher m = PCI_BUS_ID.matcher(trimmed);
        if (!m.matches()) {
            throw new IllegalArgumentException("malformed PCI bus id '" + raw
                    + "', expected [domain:]bus:device.function in hex");
        }
        long domain = m.group(1) == null ? 0 : Long.parseLong(m.group(1), 16);
        int bus = Integer.parseInt(m.group(2), 16);
        int device = Integer.parseInt(m.group(3), 16);
        int function = Integer.parseInt(m.group(4), 16);
        if (domain > 0xFFFF) {
            throw new IllegalArgumentException("PCI domain out of range in '" + raw + "'");
        }
        return formatPciBusId((int) domain, bus, device, function);
    }

    private static String normalizeUuid(String raw) {
        if (raw == null) {
            return null;
        }
        String compact = raw.trim().replace("-", "").toLowerCase(Locale.ROOT);
        if (compact.isEmpty()) {
            return null;
        }
        if (!UUID_HEX.matcher(compact).matches()) {
            throw new IllegalArgumentException("malformed device UUID '" + raw
                    + "', expected 32 hex characters");
        }
        // Drivers that cannot identify the device report an all-zero UUID; treat it as unknown.
        return ALL_ZERO_UUID.equals(compact) ? null : compact;
    }
}
