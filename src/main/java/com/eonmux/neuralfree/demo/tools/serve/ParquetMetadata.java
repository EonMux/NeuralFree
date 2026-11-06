package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * The Thrift compact protocol reader and the Parquet footer model: the file metadata, its schema
 * elements, the row groups and their column chunks. Only the fields the benchmark's row reader
 * needs are kept; every other field is skipped by type, so a newer writer's metadata still parses.
 * The footer is untrusted input: the reader bounds-checks its buffers and refuses a truncated or
 * malformed footer naming the file.
 */
final class ParquetMetadata {

    static final int TYPE_BOOLEAN = 0;
    static final int TYPE_INT32 = 1;
    static final int TYPE_INT64 = 2;
    static final int TYPE_INT96 = 3;
    static final int TYPE_FLOAT = 4;
    static final int TYPE_DOUBLE = 5;
    static final int TYPE_BYTE_ARRAY = 6;
    static final int TYPE_FIXED_LEN_BYTE_ARRAY = 7;

    static final int REPETITION_REQUIRED = 0;
    static final int REPETITION_OPTIONAL = 1;
    static final int REPETITION_REPEATED = 2;

    static final int CONVERTED_UTF8 = 0;
    static final int CONVERTED_MAP = 1;
    static final int CONVERTED_MAP_KEY_VALUE = 2;
    static final int CONVERTED_LIST = 3;
    static final int CONVERTED_ENUM = 4;
    static final int CONVERTED_DECIMAL = 5;
    static final int CONVERTED_JSON = 19;
    static final int CONVERTED_BSON = 20;

    static final int ENCODING_PLAIN = 0;
    static final int ENCODING_PLAIN_DICTIONARY = 2;
    static final int ENCODING_RLE = 3;
    static final int ENCODING_BIT_PACKED = 4;
    static final int ENCODING_DELTA_BINARY_PACKED = 5;
    static final int ENCODING_DELTA_LENGTH_BYTE_ARRAY = 6;
    static final int ENCODING_DELTA_BYTE_ARRAY = 7;
    static final int ENCODING_RLE_DICTIONARY = 8;
    static final int ENCODING_BYTE_STREAM_SPLIT = 9;

    static final int PAGE_DATA = 0;
    static final int PAGE_INDEX = 1;
    static final int PAGE_DICTIONARY = 2;
    static final int PAGE_DATA_V2 = 3;

    /** One schema element of the file's flat schema list. */
    static final class SchemaElement {

        int type = -1;
        int typeLength;
        int repetition = REPETITION_REQUIRED;
        String name = "";
        int numChildren;
        int convertedType = -1;
        int scale;
        int precision;

        boolean isLeaf() {
            return type >= 0;
        }

        boolean isList() {
            return convertedType == CONVERTED_LIST;
        }
    }

    /** One column chunk of a row group. */
    static final class ColumnChunk {

        int type = -1;
        int codec;
        long numValues;
        long dataPageOffset = -1;
        long dictionaryPageOffset = -1;
        long totalCompressedSize;
        String[] path = new String[0];
        String filePath = "";
    }

    /** One row group: its columns and row count. */
    static final class RowGroup {

        final List<ColumnChunk> columns = new ArrayList<>();
        long numRows;
    }

    /** The footer a reader needs. */
    static final class FileMeta {

        final List<SchemaElement> schema = new ArrayList<>();
        final List<RowGroup> rowGroups = new ArrayList<>();
        long numRows;
    }

    private ParquetMetadata() {
    }

    /** Reads the footer of {@code file}, whose last eight bytes hold its length and {@code PAR1}. */
    static FileMeta read(java.io.RandomAccessFile file) throws IOException {
        long length = file.length();
        if (length < 12) {
            throw new IllegalArgumentException("the parquet file is shorter than its footer");
        }
        file.seek(length - 8);
        byte[] tail = new byte[8];
        file.readFully(tail);
        if (tail[4] != 'P' || tail[5] != 'A' || tail[6] != 'R' || tail[7] != '1') {
            throw new IllegalArgumentException("the parquet file does not end with the PAR1 magic");
        }
        long metadataLength = uint32(tail, 0);
        if (metadataLength < 0 || metadataLength > length - 12) {
            throw new IllegalArgumentException("the parquet footer names " + metadataLength
                    + " bytes, beyond the file");
        }
        byte[] footer = new byte[(int) metadataLength];
        file.seek(length - 8 - metadataLength);
        file.readFully(footer);
        Compact in = new Compact(new ByteInput(footer));
        return readFileMeta(in);
    }

    private static FileMeta readFileMeta(Compact in) throws IOException {
        FileMeta meta = new FileMeta();
        int last = 0;
        while (true) {
            int header = in.readByte();
            if (header == 0) {
                return meta;
            }
            int type = header & 0x0f;
            int delta = (header >>> 4) & 0x0f;
            int id = delta != 0 ? last + delta : in.readZigZag32();
            last = id;
            switch (id) {
                case 2:
                    int count = in.readListHeader();
                    int elementType = in.elementType();
                    for (int i = 0; i < count; i++) {
                        meta.schema.add(readSchemaElement(in, elementType));
                    }
                    break;
                case 3:
                    meta.numRows = in.readZigZag64();
                    break;
                case 4:
                    int groups = in.readListHeader();
                    int groupType = in.elementType();
                    for (int i = 0; i < groups; i++) {
                        meta.rowGroups.add(readRowGroup(in, groupType));
                    }
                    break;
                default:
                    in.skip(type);
            }
        }
    }

    private static SchemaElement readSchemaElement(Compact in, int type) throws IOException {
        if (type != Compact.STRUCT) {
            in.skip(type);
            return new SchemaElement();
        }
        SchemaElement element = new SchemaElement();
        int last = 0;
        while (true) {
            int header = in.readByte();
            if (header == 0) {
                return element;
            }
            int fieldType = header & 0x0f;
            int delta = (header >>> 4) & 0x0f;
            int id = delta != 0 ? last + delta : in.readZigZag32();
            last = id;
            switch (id) {
                case 1:
                    element.type = in.readZigZag32();
                    break;
                case 2:
                    element.typeLength = in.readZigZag32();
                    break;
                case 3:
                    element.repetition = in.readZigZag32();
                    break;
                case 4:
                    element.name = in.readBinaryText();
                    break;
                case 5:
                    element.numChildren = in.readZigZag32();
                    break;
                case 6:
                    element.convertedType = in.readZigZag32();
                    break;
                case 7:
                    element.scale = in.readZigZag32();
                    break;
                case 8:
                    element.precision = in.readZigZag32();
                    break;
                default:
                    in.skip(fieldType);
            }
        }
    }

    private static RowGroup readRowGroup(Compact in, int type) throws IOException {
        RowGroup group = new RowGroup();
        if (type != Compact.STRUCT) {
            in.skip(type);
            return group;
        }
        int last = 0;
        while (true) {
            int header = in.readByte();
            if (header == 0) {
                return group;
            }
            int fieldType = header & 0x0f;
            int delta = (header >>> 4) & 0x0f;
            int id = delta != 0 ? last + delta : in.readZigZag32();
            last = id;
            switch (id) {
                case 1:
                    int count = in.readListHeader();
                    int elementType = in.elementType();
                    for (int i = 0; i < count; i++) {
                        group.columns.add(readColumnChunk(in, elementType));
                    }
                    break;
                case 3:
                    group.numRows = in.readZigZag64();
                    break;
                default:
                    in.skip(fieldType);
            }
        }
    }

    private static ColumnChunk readColumnChunk(Compact in, int type) throws IOException {
        ColumnChunk chunk = new ColumnChunk();
        if (type != Compact.STRUCT) {
            in.skip(type);
            return chunk;
        }
        int last = 0;
        while (true) {
            int header = in.readByte();
            if (header == 0) {
                return chunk;
            }
            int fieldType = header & 0x0f;
            int delta = (header >>> 4) & 0x0f;
            int id = delta != 0 ? last + delta : in.readZigZag32();
            last = id;
            if (id == 1) {
                chunk.filePath = fieldType == Compact.BINARY ? in.readBinaryText()
                        : skipString(in, fieldType);
            } else if (id == 3) {
                readColumnMeta(in, fieldType, chunk);
            } else {
                in.skip(fieldType);
            }
        }
    }

    private static void readColumnMeta(Compact in, int type, ColumnChunk chunk) throws IOException {
        if (type != Compact.STRUCT) {
            in.skip(type);
            return;
        }
        int last = 0;
        while (true) {
            int header = in.readByte();
            if (header == 0) {
                return;
            }
            int fieldType = header & 0x0f;
            int delta = (header >>> 4) & 0x0f;
            int id = delta != 0 ? last + delta : in.readZigZag32();
            last = id;
            switch (id) {
                case 1:
                    chunk.type = in.readZigZag32();
                    break;
                case 3:
                    int count = in.readListHeader();
                    int elementType = in.elementType();
                    String[] path = new String[count];
                    for (int i = 0; i < count; i++) {
                        path[i] = elementType == Compact.BINARY ? in.readBinaryText() : skipString(in, elementType);
                    }
                    chunk.path = path;
                    break;
                case 4:
                    chunk.codec = in.readZigZag32();
                    break;
                case 5:
                    chunk.numValues = in.readZigZag64();
                    break;
                case 7:
                    chunk.totalCompressedSize = in.readZigZag64();
                    break;
                case 9:
                    chunk.dataPageOffset = in.readZigZag64();
                    break;
                case 11:
                    chunk.dictionaryPageOffset = in.readZigZag64();
                    break;
                default:
                    in.skip(fieldType);
            }
        }
    }

    private static String skipString(Compact in, int type) throws IOException {
        in.skip(type);
        return "";
    }

    static long uint32(byte[] bytes, int offset) {
        return (bytes[offset] & 0xffL) | ((bytes[offset + 1] & 0xffL) << 8)
                | ((bytes[offset + 2] & 0xffL) << 16) | ((bytes[offset + 3] & 0xffL) << 24);
    }

    /** A byte source the compact reader consumes. */
    interface Input {

        int read() throws IOException;
    }

    /** The in-memory footer bytes. */
    private static final class ByteInput implements Input {

        private final byte[] bytes;
        private int position;

        ByteInput(byte[] bytes) {
            this.bytes = bytes;
        }

        @Override
        public int read() throws IOException {
            if (position >= bytes.length) {
                throw new IOException("the parquet footer is truncated");
            }
            return bytes[position++] & 0xff;
        }
    }

    /** The Thrift compact protocol over one input. */
    static final class Compact {

        static final int STOP = 0;
        static final int BOOLEAN_TRUE = 1;
        static final int BOOLEAN_FALSE = 2;
        static final int BYTE = 3;
        static final int I16 = 4;
        static final int I32 = 5;
        static final int I64 = 6;
        static final int DOUBLE = 7;
        static final int BINARY = 8;
        static final int LIST = 9;
        static final int SET = 10;
        static final int MAP = 11;
        static final int STRUCT = 12;

        private final Input in;
        private int elementType;

        Compact(Input in) {
            this.in = in;
        }

        int elementType() {
            return elementType;
        }

        int readByte() throws IOException {
            return in.read();
        }

        long readVarint() throws IOException {
            long value = 0;
            int shift = 0;
            while (true) {
                if (shift > 63) {
                    throw new IOException("a Thrift varint is longer than 64 bits");
                }
                int b = in.read();
                value |= ((long) (b & 0x7f)) << shift;
                if ((b & 0x80) == 0) {
                    return value;
                }
                shift += 7;
            }
        }

        int readZigZag32() throws IOException {
            long raw = readVarint();
            return (int) ((raw >>> 1) ^ -(raw & 1));
        }

        long readZigZag64() throws IOException {
            long raw = readVarint();
            return (raw >>> 1) ^ -(raw & 1);
        }

        double readDouble() throws IOException {
            long bits = 0;
            for (int i = 0; i < 8; i++) {
                bits |= ((long) in.read()) << (8 * i);
            }
            return Double.longBitsToDouble(bits);
        }

        byte[] readBinaryBytes() throws IOException {
            long length = readVarint();
            if (length < 0 || length > Integer.MAX_VALUE - 8) {
                throw new IOException("a Thrift binary field is too long");
            }
            byte[] out = new byte[(int) length];
            for (int i = 0; i < out.length; i++) {
                out[i] = (byte) in.read();
            }
            return out;
        }

        String readBinaryText() throws IOException {
            return new String(readBinaryBytes(), java.nio.charset.StandardCharsets.UTF_8);
        }

        /** Reads a list or set header and returns its size; {@link #elementType()} holds the type. */
        int readListHeader() throws IOException {
            int first = in.read();
            int size = (first >>> 4) & 0x0f;
            elementType = first & 0x0f;
            if (size == 15) {
                long extended = readVarint();
                if (extended < 0 || extended > Integer.MAX_VALUE - 8) {
                    throw new IOException("a Thrift list is too long");
                }
                size = (int) extended;
            }
            return size;
        }

        void skip(int type) throws IOException {
            switch (type) {
                case STOP:
                case BOOLEAN_TRUE:
                case BOOLEAN_FALSE:
                    return;
                case BYTE:
                    in.read();
                    return;
                case I16:
                case I32:
                case I64:
                    readVarint();
                    return;
                case DOUBLE:
                    for (int i = 0; i < 8; i++) {
                        in.read();
                    }
                    return;
                case BINARY:
                    readBinaryBytes();
                    return;
                case LIST:
                case SET: {
                    int size = readListHeader();
                    int element = elementType;
                    for (int i = 0; i < size; i++) {
                        skip(element);
                    }
                    return;
                }
                case MAP: {
                    long size = readVarint();
                    if (size <= 0) {
                        return;
                    }
                    int types = in.read();
                    int keyType = (types >>> 4) & 0x0f;
                    int valueType = types & 0x0f;
                    for (long i = 0; i < size; i++) {
                        skip(keyType);
                        skip(valueType);
                    }
                    return;
                }
                case STRUCT: {
                    int last = 0;
                    while (true) {
                        int header = in.read();
                        if (header == 0) {
                            return;
                        }
                        int fieldType = header & 0x0f;
                        int delta = (header >>> 4) & 0x0f;
                        if (delta == 0) {
                            last = readZigZag32();
                        } else {
                            last += delta;
                        }
                        skip(fieldType);
                    }
                }
                default:
                    throw new IOException("unknown Thrift type " + type);
            }
        }
    }
}
