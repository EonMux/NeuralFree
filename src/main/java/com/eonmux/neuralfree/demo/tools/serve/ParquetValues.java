package com.eonmux.neuralfree.demo.tools.serve;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * The Parquet value decoders: PLAIN, dictionary indices, the delta encodings, the RLE/bit-packed
 * hybrid of levels, and the page compression codecs. Every decoder works over a byte array and a
 * mutable position, and refuses a truncated or malformed page naming the encoding.
 */
final class ParquetValues {

    private ParquetValues() {
    }

    /** Decodes {@code count} non-null values of one page encoding. */
    static List<Object> decode(int encoding, byte[] data, int offset, int limit, int count,
                               int physicalType, int typeLength, boolean text, List<Object> dictionary) {
        switch (encoding) {
            case ParquetMetadata.ENCODING_PLAIN:
                return plain(physicalType, typeLength, text, data, offset, limit, count);
            case ParquetMetadata.ENCODING_PLAIN_DICTIONARY:
            case ParquetMetadata.ENCODING_RLE_DICTIONARY:
                return dictionary(data, offset, limit, count, dictionary);
            case ParquetMetadata.ENCODING_DELTA_BINARY_PACKED:
                return deltaBinary(physicalType, data, offset, limit, count);
            case ParquetMetadata.ENCODING_DELTA_LENGTH_BYTE_ARRAY:
                return deltaLengthByteArray(data, offset, limit, count);
            case ParquetMetadata.ENCODING_DELTA_BYTE_ARRAY:
                return deltaByteArray(data, offset, limit, count);
            default:
                throw new IllegalArgumentException("the parquet page uses encoding " + encoding
                        + ", which the harness does not read");
        }
    }

    /** {@code ceil(log2(level + 1))} bits, the width of a level or dictionary index. */
    static int bitWidth(int maxValue) {
        return maxValue <= 0 ? 0 : 32 - Integer.numberOfLeadingZeros(maxValue);
    }

    /** The RLE/bit-packed hybrid of a level section, preceded by its four-byte length. */
    static int[] hybridWithLength(byte[] data, int[] position, int bitWidth, int count) {
        require(data, position[0], 4);
        long length = ParquetMetadata.uint32(data, position[0]);
        position[0] += 4;
        if (length < 0 || position[0] + length > data.length) {
            throw new IllegalArgumentException("a parquet level section names " + length
                    + " bytes, beyond its page");
        }
        return hybrid(data, position, position[0] + (int) length, bitWidth, count);
    }

    /** The RLE/bit-packed hybrid between {@code offset} and {@code limit}. */
    static int[] hybrid(byte[] data, int[] position, int limit, int bitWidth, int count) {
        int[] out = new int[count];
        int produced = 0;
        while (produced < count) {
            long header = varint(data, position, limit);
            if ((header & 1) == 0) {
                int run = (int) (header >>> 1);
                int bytes = (bitWidth + 7) / 8;
                int value = 0;
                require(position[0], limit, bytes);
                for (int i = 0; i < bytes; i++) {
                    value |= (data[position[0]++] & 0xff) << (8 * i);
                }
                for (int i = 0; i < run && produced < count; i++) {
                    out[produced++] = value;
                }
            } else {
                int groups = (int) (header >>> 1);
                long total = (long) groups * 8;
                int take = (int) Math.min(total, count - produced);
                int bytes = (int) (((long) groups * 8 * bitWidth + 7) / 8);
                require(position[0], limit, bytes);
                int bit = 0;
                for (int i = 0; i < take; i++) {
                    out[produced++] = bits(data, position[0], bit, bitWidth);
                    bit += bitWidth;
                }
                position[0] += bytes;
            }
        }
        return out;
    }

    private static int bits(byte[] data, int offset, int bit, int width) {
        return (int) bitsLong(data, offset, bit, width);
    }

    /** {@code width} bits at {@code bit} of the little-endian bit stream, up to 64 bits wide. */
    private static long bitsLong(byte[] data, int offset, int bit, int width) {
        if (width == 0) {
            return 0;
        }
        long value = 0;
        for (int i = 0; i < width; i++) {
            int position = bit + i;
            int index = offset + position / 8;
            if (index >= data.length) {
                throw new IllegalArgumentException("a parquet bit-packed run is truncated");
            }
            int b = (data[index] >>> (position % 8)) & 1;
            value |= ((long) b) << i;
        }
        return value;
    }

    /** PLAIN values of one physical type. */
    static List<Object> plain(int physicalType, int typeLength, boolean text, byte[] data,
                              int offset, int limit, int count) {
        List<Object> out = new ArrayList<>(count);
        int[] position = {offset};
        switch (physicalType) {
            case ParquetMetadata.TYPE_BOOLEAN:
                require(position[0], limit, (count + 7) / 8);
                for (int i = 0; i < count; i++) {
                    out.add(((data[position[0] + i / 8] >>> (i % 8)) & 1) != 0);
                }
                position[0] += (count + 7) / 8;
                break;
            case ParquetMetadata.TYPE_INT32:
                require(position[0], limit, count * 4L);
                for (int i = 0; i < count; i++) {
                    out.add(int32(data, position[0]));
                    position[0] += 4;
                }
                break;
            case ParquetMetadata.TYPE_INT64:
                require(position[0], limit, count * 8L);
                for (int i = 0; i < count; i++) {
                    out.add(int64(data, position[0]));
                    position[0] += 8;
                }
                break;
            case ParquetMetadata.TYPE_FLOAT:
                require(position[0], limit, count * 4L);
                for (int i = 0; i < count; i++) {
                    out.add(Float.intBitsToFloat((int) ParquetMetadata.uint32(data, position[0])));
                    position[0] += 4;
                }
                break;
            case ParquetMetadata.TYPE_DOUBLE:
                require(position[0], limit, count * 8L);
                for (int i = 0; i < count; i++) {
                    out.add(Double.longBitsToDouble(int64(data, position[0])));
                    position[0] += 8;
                }
                break;
            case ParquetMetadata.TYPE_BYTE_ARRAY:
                for (int i = 0; i < count; i++) {
                    require(position[0], limit, 4);
                    long length = ParquetMetadata.uint32(data, position[0]);
                    position[0] += 4;
                    require(position[0], limit, length);
                    if (length > Integer.MAX_VALUE - 8) {
                        throw new IllegalArgumentException("a parquet byte array is too long");
                    }
                    out.add(bytes(data, position[0], (int) length, text));
                    position[0] += (int) length;
                }
                break;
            case ParquetMetadata.TYPE_FIXED_LEN_BYTE_ARRAY:
                require(position[0], limit, (long) count * typeLength);
                for (int i = 0; i < count; i++) {
                    out.add(bytes(data, position[0], typeLength, text));
                    position[0] += typeLength;
                }
                break;
            default:
                throw new IllegalArgumentException("the parquet page uses physical type " + physicalType
                        + ", which the harness does not read");
        }
        return out;
    }

    /** Dictionary indices: the first byte is the bit width, then the hybrid stream. */
    static List<Object> dictionary(byte[] data, int offset, int limit, int count, List<Object> dictionary) {
        if (dictionary == null || dictionary.isEmpty()) {
            throw new IllegalArgumentException("a parquet dictionary-indexed page has no dictionary page");
        }
        List<Object> out = new ArrayList<>(count);
        if (count == 0) {
            return out;
        }
        require(offset, limit, 1);
        int width = data[offset] & 0xff;
        int[] position = {offset + 1};
        int[] indices = width == 0 ? new int[count]
                : hybrid(data, position, limit, width, count);
        for (int index : indices) {
            if (index < 0 || index >= dictionary.size()) {
                throw new IllegalArgumentException("a parquet dictionary index " + index
                        + " is outside the dictionary of " + dictionary.size() + " values");
            }
            out.add(dictionary.get(index));
        }
        return out;
    }

    /** DELTA_BINARY_PACKED values of an INT32 or INT64 column. */
    static List<Object> deltaBinary(int physicalType, byte[] data, int offset, int limit, int count) {
        if (count == 0) {
            return new ArrayList<>();
        }
        int[] position = {offset};
        long blockSize = varint(data, position, limit);
        long miniBlocks = varint(data, position, limit);
        long total = varint(data, position, limit);
        if (blockSize <= 0 || miniBlocks <= 0 || blockSize % miniBlocks != 0
                || blockSize > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("a parquet delta page has block " + blockSize
                    + " and " + miniBlocks + " miniblocks");
        }
        long first = zigzag(varint(data, position, limit));
        List<Object> out = new ArrayList<>((int) Math.min(total, count));
        out.add(number(physicalType, first));
        long previous = first;
        int valuesPerMini = (int) (blockSize / miniBlocks);
        while (out.size() < total) {
            long minDelta = zigzag(varint(data, position, limit));
            int[] widths = new int[(int) miniBlocks];
            require(position[0], limit, miniBlocks);
            for (int i = 0; i < widths.length; i++) {
                widths[i] = data[position[0]++] & 0xff;
            }
            for (int mini = 0; mini < miniBlocks && out.size() < total; mini++) {
                int width = widths[mini];
                int bytes = (int) (((long) valuesPerMini * width + 7) / 8);
                require(position[0], limit, bytes);
                int bit = 0;
                for (int i = 0; i < valuesPerMini && out.size() < total; i++) {
                    long delta = bitsLong(data, position[0], bit, width);
                    bit += width;
                    previous += minDelta + delta;
                    out.add(number(physicalType, previous));
                }
                position[0] += bytes;
            }
        }
        return out.size() > count ? new ArrayList<>(out.subList(0, count)) : out;
    }

    /** DELTA_LENGTH_BYTE_ARRAY values: the delta-packed lengths, then the concatenated bytes. */
    static List<Object> deltaLengthByteArray(byte[] data, int offset, int limit, int count) {
        int[] position = {offset};
        List<Long> lengths = deltaLongs(data, position, limit, count);
        List<Object> out = new ArrayList<>(lengths.size());
        for (long length : lengths) {
            require(position[0], limit, length);
            out.add(bytes(data, position[0], (int) length, true));
            position[0] += (int) length;
        }
        return out;
    }

    /** DELTA_BYTE_ARRAY values: each value is its prefix of the previous one plus a suffix. */
    static List<Object> deltaByteArray(byte[] data, int offset, int limit, int count) {
        int[] position = {offset};
        List<Long> prefixes = deltaLongs(data, position, limit, count);
        List<Long> lengths = deltaLongs(data, position, limit, count);
        List<Object> out = new ArrayList<>(lengths.size());
        byte[] previous = new byte[0];
        for (int i = 0; i < lengths.size(); i++) {
            int prefix = (int) (long) prefixes.get(i);
            int suffix = (int) (long) lengths.get(i);
            if (prefix < 0 || prefix > previous.length) {
                throw new IllegalArgumentException("a parquet delta byte array has prefix " + prefix
                        + " over " + previous.length + " bytes");
            }
            require(position[0], limit, suffix);
            byte[] value = new byte[prefix + suffix];
            System.arraycopy(previous, 0, value, 0, prefix);
            System.arraycopy(data, position[0], value, prefix, suffix);
            position[0] += suffix;
            out.add(new String(value, StandardCharsets.UTF_8));
            previous = value;
        }
        return out;
    }

    private static List<Long> deltaLongs(byte[] data, int[] position, int limit, int count) {
        if (count == 0) {
            return new ArrayList<>();
        }
        long blockSize = varint(data, position, limit);
        long miniBlocks = varint(data, position, limit);
        long total = varint(data, position, limit);
        if (blockSize <= 0 || miniBlocks <= 0 || blockSize % miniBlocks != 0) {
            throw new IllegalArgumentException("a parquet delta length page has block " + blockSize
                    + " and " + miniBlocks + " miniblocks");
        }
        List<Long> out = new ArrayList<>((int) Math.min(total, count));
        long previous = zigzag(varint(data, position, limit));
        out.add(previous);
        int valuesPerMini = (int) (blockSize / miniBlocks);
        while (out.size() < total) {
            long minDelta = zigzag(varint(data, position, limit));
            int[] widths = new int[(int) miniBlocks];
            require(position[0], limit, miniBlocks);
            for (int i = 0; i < widths.length; i++) {
                widths[i] = data[position[0]++] & 0xff;
            }
            for (int mini = 0; mini < miniBlocks && out.size() < total; mini++) {
                int width = widths[mini];
                int bytes = (int) (((long) valuesPerMini * width + 7) / 8);
                require(position[0], limit, bytes);
                int bit = 0;
                for (int i = 0; i < valuesPerMini && out.size() < total; i++) {
                    long delta = bitsLong(data, position[0], bit, width);
                    bit += width;
                    previous += minDelta + delta;
                    out.add(previous);
                }
                position[0] += bytes;
            }
        }
        return out.size() > count ? new ArrayList<>(out.subList(0, count)) : out;
    }

    /** Decompresses one page body with the column's codec. */
    static byte[] decompress(int codec, byte[] compressed, int uncompressedSize) {
        switch (codec) {
            case 0:
                return compressed;
            case 1:
                return ParquetSnappy.decompress(compressed);
            case 2:
                try (GZIPInputStream in = new GZIPInputStream(new ByteArrayInputStream(compressed))) {
                    byte[] out = in.readAllBytes();
                    return out;
                } catch (IOException e) {
                    throw new IllegalArgumentException("a parquet GZIP page cannot be decompressed: "
                            + e.getMessage(), e);
                }
            default:
                throw new IllegalArgumentException("the parquet file uses " + codecName(codec)
                        + " compression, which the harness does not read");
        }
    }

    /** The compression codec's name, for a refusal. */
    static String codecName(int codec) {
        switch (codec) {
            case 0:
                return "UNCOMPRESSED";
            case 1:
                return "SNAPPY";
            case 2:
                return "GZIP";
            case 3:
                return "LZO";
            case 4:
                return "BROTLI";
            case 5:
                return "LZ4";
            case 6:
                return "ZSTD";
            case 7:
                return "LZ4_RAW";
            default:
                return "codec " + codec;
        }
    }

    private static Object number(int physicalType, long value) {
        return physicalType == ParquetMetadata.TYPE_INT32 ? (Object) (int) value : (Object) value;
    }

    private static Object bytes(byte[] data, int offset, int length, boolean text) {
        if (text) {
            return new String(data, offset, length, StandardCharsets.UTF_8);
        }
        byte[] out = new byte[length];
        System.arraycopy(data, offset, out, 0, length);
        return out;
    }

    private static int int32(byte[] data, int offset) {
        return (data[offset] & 0xff) | ((data[offset + 1] & 0xff) << 8)
                | ((data[offset + 2] & 0xff) << 16) | ((data[offset + 3] & 0xff) << 24);
    }

    private static long int64(byte[] data, int offset) {
        return (data[offset] & 0xffL) | ((data[offset + 1] & 0xffL) << 8)
                | ((data[offset + 2] & 0xffL) << 16) | ((data[offset + 3] & 0xffL) << 24)
                | ((data[offset + 4] & 0xffL) << 32) | ((data[offset + 5] & 0xffL) << 40)
                | ((data[offset + 6] & 0xffL) << 48) | ((data[offset + 7] & 0xffL) << 56);
    }

    static long varint(byte[] data, int[] position, int limit) {
        long value = 0;
        int shift = 0;
        while (true) {
            if (position[0] >= limit || shift > 63) {
                throw new IllegalArgumentException("a parquet page ends inside a varint");
            }
            int b = data[position[0]++] & 0xff;
            value |= ((long) (b & 0x7f)) << shift;
            if ((b & 0x80) == 0) {
                return value;
            }
            shift += 7;
        }
    }

    private static long zigzag(long raw) {
        return (raw >>> 1) ^ -(raw & 1);
    }

    private static void require(byte[] data, int position, long count) {
        if (count < 0 || position + count > data.length) {
            throw new IllegalArgumentException("a parquet page is truncated");
        }
    }

    private static void require(int position, int limit, long count) {
        if (count < 0 || position + count > limit) {
            throw new IllegalArgumentException("a parquet page is truncated");
        }
    }
}
