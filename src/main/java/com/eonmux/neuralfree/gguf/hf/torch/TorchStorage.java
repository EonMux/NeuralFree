package com.eonmux.neuralfree.gguf.hf.torch;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.Objects;

/**
 * One storage of a torch checkpoint as the pickle's persistent ids reference it: the storage's dtype (its storage
 * class name, {@code torch.FloatStorage} and friends, or {@code torch.storage.UntypedStorage} for a view that names
 * its dtype itself), its {@code <archive>/data/<k>} zip entry, its location, and positional reads of its bytes.
 * Nothing is executed and nothing is trusted from the pickle beyond the class name (the closed dtype set below), the
 * key (a child name of {@code <archive>/data}) and the location ({@code cpu}). Relinquishing a storage is the
 * checkpoint's {@code close}; a storage itself is immutable and thread-safe while the checkpoint is open.
 */
public final class TorchStorage {

    /** The storage classes torch pickles as persistent ids, to the safetensors dtype of their element bytes
     *  (package-visible: the restricted unpickler's {@code GLOBAL} allow-list intersects it). */
    static final Map<String, StDType> TYPES = new java.util.HashMap<>();

    static {
        TYPES.put("torch.ByteStorage", StDType.U8);
        TYPES.put("torch.BoolStorage", StDType.BOOL);
        TYPES.put("torch.CharStorage", StDType.I8);
        TYPES.put("torch.ShortStorage", StDType.I16);
        TYPES.put("torch.IntStorage", StDType.I32);
        TYPES.put("torch.LongStorage", StDType.I64);
        TYPES.put("torch.FloatStorage", StDType.F32);
        TYPES.put("torch.DoubleStorage", StDType.F64);
        TYPES.put("torch.HalfStorage", StDType.F16);
        TYPES.put("torch.BFloat16Storage", StDType.BF16);
    }

    /** The only location a storage may live on; a downloaded checkpoint read here keeps its weights addressable. */
    public static final String READABLE_LOCATION = "cpu";

    /** The untyped storage class; a {@code _rebuild_tensor_v3} view names its dtype. */
    public static final String UNTYPED = "torch.storage.UntypedStorage";
    /** Default archive directory of a container whose pickle is {@code archive/data.pkl}. */
    public static final String ARCHIVE = "archive/";

    private final TorchZipReader zip;
    private final TorchZipReader.Entry entry;
    private final String typeName;
    private final String key;
    private final String location;
    private final StDType dtype;
    private final long absoluteOffset;
    private final ByteBuffer deflatedBytes;

    private TorchStorage(TorchZipReader zip, TorchZipReader.Entry entry, String typeName, String key,
                         String location, StDType dtype, long absoluteOffset, ByteBuffer deflatedBytes) {
        this.zip = zip;
        this.entry = entry;
        this.typeName = typeName;
        this.key = key;
        this.location = location;
        this.dtype = dtype;
        this.absoluteOffset = absoluteOffset;
        this.deflatedBytes = deflatedBytes;
    }

    /** The safetensors dtype of the storage's element bytes, or null for an untyped storage. */
    public StDType dtype() {
        return dtype;
    }

    /** The storage's class name, for example {@code torch.FloatStorage}. */
    public String typeName() {
        return typeName;
    }

    /** The storage's key, which names its {@code archive/data/<k>} entry. */
    public String key() {
        return key;
    }

    /** The storage's location, always {@link #READABLE_LOCATION}. */
    public String location() {
        return location;
    }

    /** The entry holding the storage's bytes. */
    public TorchZipReader.Entry entry() {
        return entry;
    }

    /** The saving torch version's element byte count, or 0 for an untyped storage. */
    public int elementBytes() {
        return dtype == null ? 0 : dtype.bits() / 8;
    }

    /** The storage's byte length. */
    public long length() {
        return entry.size();
    }

    /** The absolute file offset of the storage's first byte when its entry is stored (mapped in place); -1 when
     *  the entry is deflated, whose bytes were read at the checkpoint's open. */
    public long absoluteOffset() {
        return absoluteOffset;
    }

    /**
     * Reads {@code dst.remaining()} bytes of the storage from its byte {@code relOff} into {@code dst}; the
     * position advances.
     *
     * @throws IllegalArgumentException when the range lies outside the storage
     * @throws IOException              when the bytes cannot be read
     */
    public void read(long relOff, ByteBuffer dst) throws IOException {
        Objects.requireNonNull(dst, "dst");
        if (relOff < 0 || relOff > entry.size() - dst.remaining()) {
            throw new IllegalArgumentException("bytes [" + relOff + ", +" + dst.remaining() + ") lie outside storage '"
                    + entry.name() + "' of " + entry.size() + " bytes");
        }
        if (deflatedBytes != null) {
            ByteBuffer slice = deflatedBytes.duplicate();
            slice.position((int) relOff);
            slice.limit((int) Math.addExact(relOff, dst.remaining()));
            dst.put(slice);
            return;
        }
        zip.read(entry, relOff, dst);
    }

    /** The storage @code data segment placed at {@code virtualBase} of a checkpoint's virtual address space;
     *  null when the entry is deflated (its bytes are not mapped in place). */
    public DataSegment segment(long virtualBase) {
        if (deflatedBytes != null || entry.dataOffset() < 0) {
            return null;
        }
        return new DataSegment(zip.path().toString(), entry.dataOffset(), virtualBase, entry.size());
    }

    /**
     * The storage a pickle's persistent id ({@code ('storage', storageType, key, location)}) names in {@code zip}.
     * The class name must be one of {@link #TYPES}, the key must be a safe child name of {@code archive/data} and
     * the location must be {@link #READABLE_LOCATION}; anything else is refused, named.
     *
     * @throws IllegalArgumentException when the persistent id names a storage type, key or location outside the
     *                                  accepted set
     * @throws IOException              when the storage's entry cannot be read
     */
    static TorchStorage of(TorchZipReader zip, String typeName, String key, String location, String dataPrefix)
            throws IOException {
        Objects.requireNonNull(zip);
        boolean untyped = UNTYPED.equals(typeName);
        StDType dtype = untyped ? null : typeName == null ? null : TYPES.get(typeName);
        if (!untyped && dtype == null) {
            throw new IllegalArgumentException("Invalid PyTorch checkpoint '" + zip.path() + "': the pickle"
                    + " references storage type '" + typeName + "', which is not one of torch's plain storage types");
        }
        if (key == null || key.isEmpty()) {
            throw new IllegalArgumentException("Invalid PyTorch checkpoint '" + zip.path() + "': storage '" + typeName
                    + "' has no key");
        }
        for (int i = 0; i < key.length(); i++) {
            char c = key.charAt(i);
            boolean ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                    || c == '-' || c == '_' || c == '.';
            if (!ok) {
                throw new IllegalArgumentException("Invalid PyTorch checkpoint '" + zip.path() + "': storage key '"
                        + key + "' holds the character '" + c + "', so its entry path would leave '"
                        + dataPrefix + "data'");
            }
        }
        if (key.startsWith(".") || key.endsWith(".")) {
            throw new IllegalArgumentException("Invalid PyTorch checkpoint '" + zip.path() + "': storage key '"
                    + key + "' names no entry inside '" + dataPrefix + "data'");
        }
        if (!READABLE_LOCATION.equals(location)) {
            throw new IllegalArgumentException("Invalid PyTorch checkpoint '" + zip.path() + "': storage '" + key
                    + "' of '" + typeName + "' is on '" + location + "'; NeuralFree reads " + READABLE_LOCATION
                    + " storages");
        }
        TorchZipReader.Entry entry = zip.entry(dataEntryName(dataPrefix, key));
        if (entry == null || entry.directory()) {
            throw new IllegalArgumentException("Invalid PyTorch checkpoint '" + zip.path() + "': the pickle"
                    + " references storage '" + key + "', which is not an entry of '" + dataPrefix + "data'");
        }
        if (!untyped && entry.size() % elementBytes(dtype) != 0) {
            throw new IllegalArgumentException("Invalid PyTorch checkpoint '" + zip.path() + "': storage '" + key
                    + "' has " + entry.size() + " bytes, not a whole number of " + dtype.headerName() + " elements");
        }
        long offset = entry.stored() ? entry.dataOffset() : -1;
        ByteBuffer deflated = entry.stored() ? null : zip.readAll(entry);
        return new TorchStorage(zip, entry, typeName, key, location, dtype, offset, deflated);
    }

    private static int elementBytes(StDType dtype) {
        if (dtype.bits() % 8 != 0) {
            throw new IllegalArgumentException("storage dtype " + dtype.headerName() + " is not a whole number of"
                    + " bytes");
        }
        return dtype.bits() / 8;
    }

    /** The zip entry name of a storage key: {@code <prefix>data/} + key, as torch writes the entry. */
    public static String dataEntryName(String dataPrefix, String key) {
        return dataPrefix + "data/" + key;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof TorchStorage
                && key.equals(((TorchStorage) o).key)
                && zip.path().equals(((TorchStorage) o).zip.path());
    }

    @Override
    public int hashCode() {
        return 31 * key.hashCode() + zip.path().hashCode();
    }

    @Override
    public String toString() {
        return "TorchStorage{" + key + ", " + typeName + ", " + entry.size() + " bytes}";
    }
}
