package com.eonmux.neuralfree.gguf.inspect;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.utils.json.Json;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The report of {@code nftool dump}: a GGUF file's key/value store with each value's GGUF type, element count and
 * value, followed by its tensor table with each tensor's element count, shape (padded to four dimensions), type
 * and name — the text form of llama.cpp's {@code gguf_dump.py}, and the same content as JSON.
 *
 * <p>Values print as the file stores them: integers as unsigned for the UINT types, strings quoted, arrays with
 * their first six elements. A string longer than 60 characters keeps its first 57 and an array longer than six
 * elements ends with {@code ...}, as the reference truncates them. The three header facts the reference dumps as
 * synthetic fields — {@code GGUF.version}, {@code GGUF.tensor_count} and {@code GGUF.kv_count} — lead the report,
 * with their file offsets. The JSON form maps each key to its type, its element count and the file offset of the
 * key/value pair; a tensor maps to its shape, its type and its data offset relative to the data section. Array
 * values appear only when the caller asks for them.</p>
 */
public final class GgufDump {

    private static final int PAD_DIMS = 4;
    private static final int ARRAY_PREVIEW = 6;
    private static final int STRING_PREVIEW = 60;
    /** The header rows the reference dumps before the key/value store: version, tensor count, pair count. */
    private static final int HEADER_FIELDS = 3;

    private GgufDump() {
    }

    /** What the report includes; immutable. */
    public static final class Options {

        private final boolean noTensors;
        private final boolean json;
        private final boolean jsonArray;

        private Options(boolean noTensors, boolean json, boolean jsonArray) {
            this.noTensors = noTensors;
            this.json = json;
            this.jsonArray = jsonArray;
        }

        /** The text report with its tensor table. */
        public static Options text() {
            return new Options(false, false, false);
        }

        /** Whether the tensor table is omitted. */
        public Options noTensors(boolean value) {
            return value == noTensors ? this : new Options(value, json, jsonArray);
        }

        /** Whether the report is JSON instead of text. */
        public Options json(boolean value) {
            return value == json ? this : new Options(noTensors, value, jsonArray);
        }

        /** Whether the JSON report carries array values in full. */
        public Options jsonArray(boolean value) {
            return value == jsonArray ? this : new Options(noTensors, json, value);
        }

        public boolean isNoTensors() {
            return noTensors;
        }

        public boolean isJson() {
            return json;
        }

        public boolean isJsonArray() {
            return jsonArray;
        }
    }

    /** Writes the report of {@code src} in the form {@code options} selects. */
    public static void write(GGUFFileParser src, Options options, Appendable out) throws IOException {
        if (options.isJson()) {
            Json.write(json(src, options), Json.Style.PYTHON_DEFAULT, out);
            out.append('\n');
        } else {
            text(src, options, out);
        }
    }

    /** The text report. */
    public static void text(GGUFFileParser src, Options options, Appendable out) throws IOException {
        out.append("* File is ").append(endianName()).append(" endian, script is running on a ")
                .append(hostEndianName()).append(" endian host.\n");
        out.append("* Dumping ").append(Long.toString(src.getKVCount() + HEADER_FIELDS))
                .append(" key/value pair(s)\n");
        long index = 1;
        index = headerField(out, index, "UINT32", "GGUF.version", Long.toString(src.getVersion()));
        index = headerField(out, index, "UINT64", "GGUF.tensor_count", Long.toString(src.getTensorCount()));
        index = headerField(out, index, "UINT64", "GGUF.kv_count", Long.toString(src.getKVCount()));
        for (Map.Entry<String, Object> e : src.getMetadata().entrySet()) {
            String key = e.getKey();
            GGUFValueType type = src.getValueType(key);
            GGUFValueType elementType = type == GGUFValueType.ARRAY ? src.getArrayElementType(key) : null;
            out.append(String.format(Locale.ROOT, "  %5d: %-10s | %8d | %s", index++, prettyType(type, elementType),
                    count(type, e.getValue()), key));
            out.append(" = ").append(valueText(type, elementType, e.getValue())).append('\n');
        }
        if (options.isNoTensors()) {
            return;
        }
        out.append("* Dumping ").append(Integer.toString(src.getTensors().size())).append(" tensor(s)\n");
        index = 1;
        for (GGUFTensorInfo t : src.getTensors()) {
            out.append(String.format(Locale.ROOT, "  %5d: %10d | %s | %-7s | %s", index++, t.numElements(),
                    dims(t), t.type().name(), t.name())).append('\n');
        }
    }

    /** Appends one of the three header rows the reference dumps as {@code GGUF.*} fields. */
    private static long headerField(Appendable out, long index, String type, String key, String value)
            throws IOException {
        out.append(String.format(Locale.ROOT, "  %5d: %-10s | %8d | %s = %s", index, type, 1, key, value))
                .append('\n');
        return index + 1;
    }

    /** The JSON report as the value model {@link Json} writes. */
    public static Map<String, Object> json(GGUFFileParser src, Options options) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("filename", src.getFilePath());
        report.put("endian", endianName());
        Map<String, Object> metadata = new LinkedHashMap<>();
        report.put("metadata", metadata);
        long index = 0;
        index = headerEntry(metadata, index, "GGUF.version", "UINT32", 4L, Long.valueOf(src.getVersion()));
        index = headerEntry(metadata, index, "GGUF.tensor_count", "UINT64", 8L,
                Long.valueOf(src.getTensorCount()));
        index = headerEntry(metadata, index, "GGUF.kv_count", "UINT64", 16L, Long.valueOf(src.getKVCount()));
        for (Map.Entry<String, Object> e : src.getMetadata().entrySet()) {
            String key = e.getKey();
            GGUFValueType type = src.getValueType(key);
            GGUFValueType elementType = type == GGUFValueType.ARRAY ? src.getArrayElementType(key) : null;
            Map<String, Object> field = new LinkedHashMap<>();
            field.put("index", Long.valueOf(index++));
            field.put("type", type == null ? "UNKNOWN" : type.name());
            Long offset = src.getKeyOffset(key);
            field.put("offset", offset == null ? Long.valueOf(-1L) : offset);
            if (type == GGUFValueType.ARRAY) {
                List<Object> types = new ArrayList<>(1);
                types.add(elementType == null ? "UNKNOWN" : elementType.name());
                field.put("array_types", types);
                if (options.isJsonArray()) {
                    field.put("value", jsonValue(type, elementType, e.getValue()));
                }
            } else {
                field.put("value", jsonValue(type, elementType, e.getValue()));
            }
            metadata.put(key, field);
        }
        Map<String, Object> tensors = new LinkedHashMap<>();
        report.put("tensors", tensors);
        if (!options.isNoTensors()) {
            index = 0;
            for (GGUFTensorInfo t : src.getTensors()) {
                List<Object> shape = new ArrayList<>(t.rank());
                for (long d : t.dims()) {
                    shape.add(Long.valueOf(d));
                }
                Map<String, Object> tensor = new LinkedHashMap<>();
                tensor.put("index", Long.valueOf(index++));
                tensor.put("shape", shape);
                tensor.put("type", t.type().name());
                tensor.put("offset", Long.valueOf(t.offset()));
                tensors.put(t.name(), tensor);
            }
        }
        return report;
    }

    /** Adds one of the three header rows the reference dumps as {@code GGUF.*} fields. */
    private static long headerEntry(Map<String, Object> metadata, long index, String key, String type, Object offset,
                                    Object value) {
        Map<String, Object> field = new LinkedHashMap<>();
        field.put("index", Long.valueOf(index));
        field.put("type", type);
        field.put("offset", offset);
        field.put("value", value);
        metadata.put(key, field);
        return index + 1;
    }

    /** {@code LITTLE} or {@code BIG}: the order {@link GGUFFileParser} reads, which is little-endian. */
    private static String endianName() {
        return "LITTLE";
    }

    private static String hostEndianName() {
        return ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN ? "LITTLE" : "BIG";
    }

    private static String prettyType(GGUFValueType type, GGUFValueType elementType) {
        if (type == null) {
            return "N/A";
        }
        return type == GGUFValueType.ARRAY ? "[" + (elementType == null ? "UNKNOWN" : elementType.name()) + "]"
                : type.name();
    }

    /** The element count a dump line shows: 1 for a scalar or a string, the list size for an array. */
    private static long count(GGUFValueType type, Object value) {
        return type == GGUFValueType.ARRAY && value instanceof List ? ((List<?>) value).size() : 1;
    }

    private static String dims(GGUFTensorInfo t) {
        StringBuilder s = new StringBuilder();
        for (int i = 0; i < PAD_DIMS; i++) {
            if (i > 0) {
                s.append(", ");
            }
            s.append(String.format(Locale.ROOT, "%5d", t.dim(i)));
        }
        return s.toString();
    }

    private static String valueText(GGUFValueType type, GGUFValueType elementType, Object value) {
        if (type == GGUFValueType.STRING) {
            String s = (String) value;
            if (s.length() > STRING_PREVIEW) {
                s = s.substring(0, STRING_PREVIEW - 3) + "...";
            }
            return quote(s);
        }
        if (type == GGUFValueType.ARRAY) {
            List<?> list = (List<?>) value;
            StringBuilder s = new StringBuilder("[");
            int n = Math.min(list.size(), ARRAY_PREVIEW);
            for (int i = 0; i < n; i++) {
                if (i > 0) {
                    s.append(", ");
                }
                s.append(elementText(elementType, list.get(i)));
            }
            if (list.size() > ARRAY_PREVIEW) {
                s.append(", ...");
            }
            return s.append(']').toString();
        }
        return elementText(type, value);
    }

    private static String elementText(GGUFValueType type, Object value) {
        if (value == null || type == null) {
            return "None";
        }
        switch (type) {
            case STRING:
                return quote((String) value);
            case BOOL:
                return ((Boolean) value).booleanValue() ? "True" : "False";
            case UINT8:
            case UINT16:
                return Integer.toString(((Number) value).intValue());
            case UINT32:
                return Long.toString(Integer.toUnsignedLong(((Number) value).intValue()));
            case UINT64:
                return Long.toUnsignedString(((Number) value).longValue());
            case FLOAT32:
                return Float.toString(((Number) value).floatValue());
            case FLOAT64:
                return Double.toString(((Number) value).doubleValue());
            default:
                return String.valueOf(value);
        }
    }

    /** Python {@code repr} of a string: single quotes, escaped backslash, quote and control characters. */
    private static String quote(String s) {
        StringBuilder b = new StringBuilder(s.length() + 2).append('\'');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '\\': b.append("\\\\"); break;
                case '\'': b.append("\\'"); break;
                case '\n': b.append("\\n"); break;
                case '\r': b.append("\\r"); break;
                case '\t': b.append("\\t"); break;
                default:
                    if (c < 0x20 || c == 0x7F) {
                        b.append(String.format(Locale.ROOT, "\\x%02x", (int) c));
                    } else {
                        b.append(c);
                    }
            }
        }
        return b.append('\'').toString();
    }

    /** A metadata value as JSON writes it, integers unsigned for the UINT types. */
    private static Object jsonValue(GGUFValueType type, GGUFValueType elementType, Object value) {
        if (type == GGUFValueType.ARRAY) {
            List<?> list = (List<?>) value;
            List<Object> out = new ArrayList<>(list.size());
            for (Object e : list) {
                out.add(jsonScalar(elementType, e));
            }
            return out;
        }
        return jsonScalar(type, value);
    }

    private static Object jsonScalar(GGUFValueType type, Object value) {
        if (type == null || value == null) {
            return null;
        }
        switch (type) {
            case UINT8:
            case UINT16:
                return Long.valueOf(((Number) value).longValue());
            case UINT32:
                return Long.valueOf(Integer.toUnsignedLong(((Number) value).intValue()));
            case UINT64: {
                long v = ((Number) value).longValue();
                return v >= 0 ? Long.valueOf(v) : BigInteger.valueOf(v).add(BigInteger.ONE.shiftLeft(64));
            }
            case INT8:
            case INT16:
            case INT32:
                return Long.valueOf(((Number) value).longValue());
            case INT64:
                return value;
            case FLOAT32:
                return Double.valueOf(((Number) value).doubleValue());
            case FLOAT64:
                return value;
            default:
                return value;
        }
    }
}
