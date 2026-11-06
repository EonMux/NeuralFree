package com.eonmux.neuralfree.demo.tools.serve;

import java.io.EOFException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The Parquet row reader of the benchmark's Hugging Face datasets: the footer's schema and row
 * groups drive the page decoding ({@link ParquetValues}) and the nested records are reassembled
 * from their repetition and definition levels. Flat primitive columns, structs and the
 * {@code LIST<primitive>} / {@code LIST<STRUCT>} shapes the reference's datasets use are read;
 * a MAP, a nested list or another construction refuses by name. The file is untrusted input, so
 * every length is bounds-checked and every page must stay inside the file.
 */
final class ParquetReader {

    private ParquetReader() {
    }

    /** The records of {@code path}, one map per row in file order. */
    static List<Map<String, Object>> read(Path path) throws IOException {
        try (RandomAccessFile file = new RandomAccessFile(path.toFile(), "r")) {
            ParquetMetadata.FileMeta meta = ParquetMetadata.read(file);
            if (meta.schema.isEmpty()) {
                throw new IllegalArgumentException("the parquet file " + path + " has no schema");
            }
            Node root = buildTree(meta.schema, path);
            List<Map<String, Object>> rows = new ArrayList<>();
            for (ParquetMetadata.RowGroup group : meta.rowGroups) {
                if (group.numRows < 0 || group.numRows > Integer.MAX_VALUE - 8) {
                    throw new IllegalArgumentException("the parquet file " + path + " names a row group of "
                            + group.numRows + " rows");
                }
                int count = (int) group.numRows;
                Map<String, Leveled> columns = readColumns(file, group, root, path);
                rows.addAll(reconstruct(root, columns, count, path));
            }
            return rows;
        }
    }

    // ================================================================== schema

    /** One node of the schema tree, with its accumulated definition and repetition levels. */
    static final class Node {

        final ParquetMetadata.SchemaElement element;
        final Node parent;
        final List<Node> children = new ArrayList<>();
        int maxDef;
        int maxRep;
        String path;

        Node(ParquetMetadata.SchemaElement element, Node parent) {
            this.element = element;
            this.parent = parent;
        }

        boolean isLeaf() {
            return element.isLeaf();
        }

        boolean isList() {
            return !isLeaf() && element.isList();
        }
    }

    private static Node buildTree(List<ParquetMetadata.SchemaElement> flat, Path file) throws IOException {
        int[] index = {0};
        return build(flat, index, null, 0, 0, file);
    }

    private static Node build(List<ParquetMetadata.SchemaElement> flat, int[] index, Node parent,
                              int def, int rep, Path file) throws IOException {
        if (index[0] >= flat.size()) {
            throw new IllegalArgumentException("the parquet schema of " + file + " ends early");
        }
        ParquetMetadata.SchemaElement element = flat.get(index[0]++);
        Node node = new Node(element, parent);
        node.maxDef = def + (parent != null && element.repetition != ParquetMetadata.REPETITION_REQUIRED ? 1 : 0);
        node.maxRep = rep + (element.repetition == ParquetMetadata.REPETITION_REPEATED ? 1 : 0);
        node.path = parent == null ? "" : parent.path.isEmpty() ? element.name : parent.path + "." + element.name;
        for (int i = 0; i < element.numChildren; i++) {
            node.children.add(build(flat, index, node, node.maxDef, node.maxRep, file));
        }
        return node;
    }

    /** Every leaf under {@code node}, in schema order. */
    private static List<Node> leaves(Node node) {
        List<Node> out = new ArrayList<>();
        collectLeaves(node, out);
        return out;
    }

    private static void collectLeaves(Node node, List<Node> out) {
        if (node.isLeaf()) {
            out.add(node);
            return;
        }
        for (Node child : node.children) {
            collectLeaves(child, out);
        }
    }

    // ================================================================== columns

    /** One leaf's entries: the levels and the non-null values, in entry order. */
    static final class Leveled {

        final int[] rep;
        final int[] def;
        final List<Object> values;

        Leveled(int[] rep, int[] def, List<Object> values) {
            this.rep = rep;
            this.def = def;
            this.values = values;
        }
    }

    private static Map<String, Leveled> readColumns(RandomAccessFile file, ParquetMetadata.RowGroup group,
                                                    Node root, Path path) throws IOException {
        Map<String, Leveled> out = new LinkedHashMap<>();
        for (ParquetMetadata.ColumnChunk chunk : group.columns) {
            String key = String.join(".", chunk.path);
            Node leaf = findLeaf(root, key);
            if (leaf == null) {
                throw new IllegalArgumentException("the parquet file " + path + " carries the column '"
                        + key + "', which its schema does not declare");
            }
            out.put(key, readColumn(file, chunk, leaf, path));
        }
        return out;
    }

    private static Node findLeaf(Node node, String path) {
        if (node.isLeaf()) {
            return node.path.equals(path) ? node : null;
        }
        for (Node child : node.children) {
            Node found = findLeaf(child, path);
            if (found != null) {
                return found;
            }
        }
        return null;
    }

    /** One column chunk's entries, across its dictionary and data pages. */
    private static Leveled readColumn(RandomAccessFile file, ParquetMetadata.ColumnChunk chunk,
                                      Node leaf, Path path) throws IOException {
        if (chunk.dataPageOffset < 0) {
            throw new IllegalArgumentException("the parquet column '" + String.join(".", chunk.path)
                    + "' of " + path + " has no data page");
        }
        if (chunk.filePath != null && !chunk.filePath.isEmpty()) {
            throw new IllegalArgumentException("the parquet column '" + String.join(".", chunk.path)
                    + "' of " + path + " lives in the external file '" + chunk.filePath
                    + "', which the harness does not read");
        }
        long start = chunk.dictionaryPageOffset >= 0
                ? Math.min(chunk.dictionaryPageOffset, chunk.dataPageOffset) : chunk.dataPageOffset;
        file.seek(start);
        List<Object> dictionary = null;
        List<int[]> reps = new ArrayList<>();
        List<int[]> defs = new ArrayList<>();
        List<Object> values = new ArrayList<>();
        long seen = 0;
        long guard = 0;
        while (seen < chunk.numValues) {
            if (++guard > chunk.numValues + 1024) {
                throw new IllegalArgumentException("the parquet column '"
                        + String.join(".", chunk.path) + "' of " + path + " does not end");
            }
            PageHeader header = readPageHeader(file, path);
            byte[] body = readBody(file, header.compressedPageSize, path);
            if (header.type == ParquetMetadata.PAGE_DICTIONARY) {
                byte[] data = ParquetValues.decompress(chunk.codec, body, header.uncompressedPageSize);
                dictionary = ParquetValues.plain(chunk.type, leaf.element.typeLength, isText(leaf),
                        data, 0, data.length, header.dictionaryCount);
                continue;
            }
            if (header.type == ParquetMetadata.PAGE_DATA) {
                requireLevelEncoding(header.repetitionEncoding, leaf.maxRep, "repetition", chunk, path);
                requireLevelEncoding(header.definitionEncoding, leaf.maxDef, "definition", chunk, path);
                byte[] data = ParquetValues.decompress(chunk.codec, body, header.uncompressedPageSize);
                int[] position = {0};
                int[] rep = chunkRep(header, leaf, data, position);
                int[] def = chunkDef(header, leaf, data, position);
                int nulls = countNulls(def, leaf.maxDef);
                reps.add(rep);
                defs.add(def);
                values.addAll(ParquetValues.decode(header.encoding, data, position[0], data.length,
                        header.numValues - nulls, chunk.type, leaf.element.typeLength, isText(leaf),
                        dictionary));
                seen += header.numValues;
                continue;
            }
            if (header.type == ParquetMetadata.PAGE_DATA_V2) {
                int[] rep = new int[header.numValues];
                int[] def = new int[header.numValues];
                int levelBytes = header.repetitionLength + header.definitionLength;
                if (levelBytes < 0 || levelBytes > body.length) {
                    throw new IllegalArgumentException("the parquet data page of '"
                            + String.join(".", chunk.path) + "' names " + levelBytes + " level bytes");
                }
                int[] position = {0};
                if (leaf.maxRep > 0) {
                    rep = ParquetValues.hybrid(body, position, header.repetitionLength,
                            ParquetValues.bitWidth(leaf.maxRep), header.numValues);
                }
                position[0] = header.repetitionLength;
                if (leaf.maxDef > 0) {
                    def = ParquetValues.hybrid(body, position, header.repetitionLength + header.definitionLength,
                            ParquetValues.bitWidth(leaf.maxDef), header.numValues);
                }
                int valueBytes = body.length - levelBytes;
                byte[] data = new byte[valueBytes];
                System.arraycopy(body, levelBytes, data, 0, valueBytes);
                if (header.isCompressed) {
                    data = ParquetValues.decompress(chunk.codec, data,
                            header.uncompressedPageSize - levelBytes);
                }
                reps.add(rep);
                defs.add(def);
                values.addAll(ParquetValues.decode(header.encoding, data, 0, data.length,
                        header.numValues - header.numNulls, chunk.type, leaf.element.typeLength,
                        isText(leaf), dictionary));
                seen += header.numValues;
                continue;
            }
            // an index page carries no values; its bytes were read above and are dropped
        }
        return new Leveled(concat(reps, chunk.numValues), concat(defs, chunk.numValues), values);
    }

    /** Refuses a level section encoded with anything but the RLE/bit-packed hybrid. */
    private static void requireLevelEncoding(int encoding, int maxLevel, String what,
                                             ParquetMetadata.ColumnChunk chunk, Path path) {
        if (maxLevel > 0 && encoding != ParquetMetadata.ENCODING_RLE) {
            throw new IllegalArgumentException("the parquet column '" + String.join(".", chunk.path)
                    + "' of " + path + " encodes its " + what + " levels with " + encoding
                    + ", which the harness does not read");
        }
    }

    private static int[] chunkRep(PageHeader header, Node leaf, byte[] data, int[] position) {
        if (leaf.maxRep <= 0) {
            return new int[header.numValues];
        }
        return ParquetValues.hybridWithLength(data, position, ParquetValues.bitWidth(leaf.maxRep),
                header.numValues);
    }

    private static int[] chunkDef(PageHeader header, Node leaf, byte[] data, int[] position) {
        if (leaf.maxDef <= 0) {
            return new int[header.numValues];
        }
        return ParquetValues.hybridWithLength(data, position, ParquetValues.bitWidth(leaf.maxDef),
                header.numValues);
    }

    private static int countNulls(int[] def, int maxDef) {
        int nulls = 0;
        for (int level : def) {
            if (level < maxDef) {
                nulls++;
            }
        }
        return nulls;
    }

    /** The levels of every data page, concatenated in page order. */
    private static int[] concat(List<int[]> parts, long expected) {
        long total = 0;
        for (int[] part : parts) {
            total += part.length;
        }
        if (total != expected) {
            throw new IllegalArgumentException("a parquet column decoded " + total + " of " + expected
                    + " values");
        }
        int[] out = new int[(int) total];
        int offset = 0;
        for (int[] part : parts) {
            System.arraycopy(part, 0, out, offset, part.length);
            offset += part.length;
        }
        return out;
    }

    private static boolean isText(Node leaf) {
        int converted = leaf.element.convertedType;
        return converted == ParquetMetadata.CONVERTED_UTF8 || converted == ParquetMetadata.CONVERTED_ENUM
                || converted == ParquetMetadata.CONVERTED_JSON;
    }

    private static byte[] readBody(RandomAccessFile file, int size, Path path) throws IOException {
        if (size < 0) {
            throw new IllegalArgumentException("a parquet page of " + path + " names " + size + " bytes");
        }
        byte[] body = new byte[size];
        file.readFully(body);
        return body;
    }

    // ================================================================== page headers

    /** The header fields the reader consumes. */
    private static final class PageHeader {

        int type;
        int uncompressedPageSize;
        int compressedPageSize;
        int numValues;
        int encoding;
        int dictionaryCount;
        int definitionEncoding = ParquetMetadata.ENCODING_RLE;
        int repetitionEncoding = ParquetMetadata.ENCODING_RLE;
        int numNulls;
        int definitionLength;
        int repetitionLength;
        boolean isCompressed = true;
    }

    private static PageHeader readPageHeader(RandomAccessFile file, Path path) throws IOException {
        PageHeader header = new PageHeader();
        ParquetMetadata.Compact in = new ParquetMetadata.Compact(() -> {
            int b = file.read();
            if (b < 0) {
                throw new EOFException("the parquet file " + path + " ends inside a page header");
            }
            return b;
        });
        int last = 0;
        while (true) {
            int raw = in.readByte();
            if (raw == 0) {
                return header;
            }
            int type = raw & 0x0f;
            int delta = (raw >>> 4) & 0x0f;
            int id = delta != 0 ? last + delta : in.readZigZag32();
            last = id;
            switch (id) {
                case 1:
                    header.type = in.readZigZag32();
                    break;
                case 2:
                    header.uncompressedPageSize = in.readZigZag32();
                    break;
                case 3:
                    header.compressedPageSize = in.readZigZag32();
                    break;
                case 5:
                    readDataHeader(in, type, header, false);
                    break;
                case 7:
                    readDictionaryHeader(in, type, header);
                    break;
                case 8:
                    readDataHeader(in, type, header, true);
                    break;
                default:
                    in.skip(type);
            }
        }
    }

    private static void readDataHeader(ParquetMetadata.Compact in, int type, PageHeader header,
                                       boolean v2) throws IOException {
        if (type != ParquetMetadata.Compact.STRUCT) {
            in.skip(type);
            return;
        }
        int last = 0;
        while (true) {
            int raw = in.readByte();
            if (raw == 0) {
                return;
            }
            int fieldType = raw & 0x0f;
            int delta = (raw >>> 4) & 0x0f;
            int id = delta != 0 ? last + delta : in.readZigZag32();
            last = id;
            switch (id) {
                case 1:
                    header.numValues = in.readZigZag32();
                    break;
                case 2:
                    if (v2) {
                        header.numNulls = in.readZigZag32();
                    } else {
                        header.encoding = in.readZigZag32();
                    }
                    break;
                case 3:
                    if (v2) {
                        in.skip(fieldType);
                    } else {
                        header.definitionEncoding = in.readZigZag32();
                    }
                    break;
                case 4:
                    if (v2) {
                        header.encoding = in.readZigZag32();
                    } else {
                        header.repetitionEncoding = in.readZigZag32();
                    }
                    break;
                case 5:
                    if (v2) {
                        header.definitionLength = in.readZigZag32();
                    } else {
                        in.skip(fieldType);
                    }
                    break;
                case 6:
                    if (v2) {
                        header.repetitionLength = in.readZigZag32();
                    } else {
                        in.skip(fieldType);
                    }
                    break;
                case 7:
                    if (v2) {
                        header.isCompressed = fieldType == ParquetMetadata.Compact.BOOLEAN_TRUE;
                    } else {
                        in.skip(fieldType);
                    }
                    break;
                default:
                    in.skip(fieldType);
            }
        }
    }

    private static void readDictionaryHeader(ParquetMetadata.Compact in, int type, PageHeader header)
            throws IOException {
        if (type != ParquetMetadata.Compact.STRUCT) {
            in.skip(type);
            return;
        }
        int last = 0;
        while (true) {
            int raw = in.readByte();
            if (raw == 0) {
                return;
            }
            int fieldType = raw & 0x0f;
            int delta = (raw >>> 4) & 0x0f;
            int id = delta != 0 ? last + delta : in.readZigZag32();
            last = id;
            if (id == 1) {
                header.dictionaryCount = in.readZigZag32();
            } else if (id == 2) {
                header.encoding = in.readZigZag32();
            } else {
                in.skip(fieldType);
            }
        }
    }

    // ================================================================== rows

    private static List<Map<String, Object>> reconstruct(Node root, Map<String, Leveled> columns,
                                                         int rows, Path path) {
        List<Map<String, Object>> out = new ArrayList<>(rows);
        for (int i = 0; i < rows; i++) {
            out.add(new LinkedHashMap<>());
        }
        for (Node field : root.children) {
            Object[] values = valuesOf(field, columns, rows, path);
            for (int i = 0; i < rows; i++) {
                out.get(i).put(field.element.name, values[i]);
            }
        }
        return out;
    }

    /** One top-level field's value per row. */
    private static Object[] valuesOf(Node node, Map<String, Leveled> columns, int rows, Path path) {
        if (node.isLeaf()) {
            return flat(node, columns.get(node.path), rows, path);
        }
        if (node.isList()) {
            return listValues(node, columns, rows, path);
        }
        if (node.children.isEmpty()) {
            throw new IllegalArgumentException("the parquet column '" + node.path + "' of " + path
                    + " is a group the harness does not read");
        }
        return structValues(node, columns, rows, path);
    }

    private static Object[] flat(Node leaf, Leveled leveled, int rows, Path path) {
        if (leveled == null) {
            throw new IllegalArgumentException("the parquet file " + path + " has no column '"
                    + leaf.path + "'");
        }
        if (leveled.def.length < rows) {
            throw new IllegalArgumentException("the parquet column '" + leaf.path + "' of " + path
                    + " carries " + leveled.def.length + " entries for " + rows + " rows");
        }
        Object[] out = new Object[rows];
        int valueIndex = 0;
        for (int i = 0; i < rows; i++) {
            if (leveled.def[i] >= leaf.maxDef) {
                if (valueIndex >= leveled.values.size()) {
                    throw new IllegalArgumentException("the parquet column '" + leaf.path + "' of " + path
                            + " carries too few values");
                }
                out[i] = leveled.values.get(valueIndex++);
            }
        }
        return out;
    }

    private static Object[] structValues(Node node, Map<String, Leveled> columns, int rows, Path path) {
        Object[] out = new Object[rows];
        List<Object[]> fields = new ArrayList<>();
        for (Node child : node.children) {
            fields.add(valuesOf(child, columns, rows, path));
        }
        for (int i = 0; i < rows; i++) {
            Map<String, Object> value = new LinkedHashMap<>();
            for (int f = 0; f < node.children.size(); f++) {
                value.put(node.children.get(f).element.name, fields.get(f)[i]);
            }
            out[i] = value;
        }
        return out;
    }

    private static Object[] listValues(Node node, Map<String, Leveled> columns, int rows, Path path) {
        Node repeated = null;
        for (Node child : node.children) {
            if (child.element.repetition == ParquetMetadata.REPETITION_REPEATED) {
                repeated = child;
                break;
            }
        }
        if (repeated == null) {
            throw new IllegalArgumentException("the parquet list '" + node.path + "' of " + path
                    + " has no repeated element");
        }
        Node element = repeated.isLeaf() ? repeated
                : repeated.children.size() == 1 ? repeated.children.get(0) : null;
        if (element == null) {
            throw new IllegalArgumentException("the parquet list '" + node.path + "' of " + path
                    + " has a nested element the harness does not read");
        }
        List<Node> leafList = leaves(repeated);
        List<Cursor> cursors = new ArrayList<>();
        for (Node leaf : leafList) {
            Leveled leveled = columns.get(leaf.path);
            if (leveled == null) {
                throw new IllegalArgumentException("the parquet file " + path + " has no column '"
                        + leaf.path + "'");
            }
            cursors.add(new Cursor(leaf, leveled));
        }
        Cursor first = cursors.get(0);
        Object[] out = new Object[rows];
        for (int row = 0; row < rows; row++) {
            if (first.entry >= first.size()) {
                throw new IllegalArgumentException("the parquet list '" + node.path + "' of " + path
                        + " ends before row " + row);
            }
            int entry = first.entry;
            int def = first.defAt(entry);
            if (def < node.maxDef) {
                out[row] = null;
                advance(cursors);
                continue;
            }
            if (def < repeated.maxDef) {
                out[row] = new ArrayList<>();
                advance(cursors);
                continue;
            }
            List<Object> list = new ArrayList<>();
            while (first.entry < first.size()) {
                if (!list.isEmpty() && first.repAt(first.entry) == 0) {
                    break;
                }
                list.add(elementValue(element, cursors, first.entry));
                advance(cursors);
            }
            out[row] = list;
        }
        return out;
    }

    private static void advance(List<Cursor> cursors) {
        for (Cursor cursor : cursors) {
            cursor.entry++;
        }
    }

    /** One element of a list: a primitive's value, or a struct assembled from its leaves. */
    private static Object elementValue(Node element, List<Cursor> cursors, int entry) {
        if (element.isLeaf()) {
            return cursor(cursors, element).valueAt(entry);
        }
        if (entry >= 0 && defAt(cursors, element, entry) < element.maxDef) {
            return null;
        }
        Map<String, Object> value = new LinkedHashMap<>();
        for (Node child : element.children) {
            value.put(child.element.name, structField(child, cursors, entry));
        }
        return value;
    }

    private static Object structField(Node node, List<Cursor> cursors, int entry) {
        if (node.isLeaf()) {
            return cursor(cursors, node).valueAt(entry);
        }
        if (node.children.isEmpty()) {
            return null;
        }
        if (entry >= 0 && defAt(cursors, node, entry) < node.maxDef) {
            return null;
        }
        Map<String, Object> value = new LinkedHashMap<>();
        for (Node child : node.children) {
            value.put(child.element.name, structField(child, cursors, entry));
        }
        return value;
    }

    private static int defAt(List<Cursor> cursors, Node node, int entry) {
        Cursor cursor = cursor(cursors, node);
        return cursor.defAt(entry);
    }

    private static Cursor cursor(List<Cursor> cursors, Node node) {
        for (Cursor cursor : cursors) {
            if (cursor.node == node) {
                return cursor;
            }
        }
        // a group's definition is the first leaf's; every leaf shares the repeated structure
        for (Cursor cursor : cursors) {
            if (cursor.node.path.startsWith(node.path + ".")) {
                return cursor;
            }
        }
        throw new IllegalArgumentException("the parquet list element '" + node.path
                + "' has no column");
    }

    /** One leaf's entries and the position of the next value to take. */
    private static final class Cursor {

        final Node node;
        final Leveled leveled;
        int entry;
        int valueIndex;

        Cursor(Node node, Leveled leveled) {
            this.node = node;
            this.leveled = leveled;
        }

        int size() {
            return leveled.def.length;
        }

        int defAt(int entry) {
            return entry < leveled.def.length ? leveled.def[entry] : 0;
        }

        int repAt(int entry) {
            return entry < leveled.rep.length ? leveled.rep[entry] : 0;
        }

        Object valueAt(int entry) {
            if (entry >= leveled.def.length || leveled.def[entry] < node.maxDef) {
                return null;
            }
            if (valueIndex >= leveled.values.size()) {
                throw new IllegalArgumentException("the parquet column '" + node.path
                        + "' carries too few values");
            }
            return leveled.values.get(valueIndex++);
        }
    }
}
