package com.eonmux.neuralfree.demo.tools.serve;

/**
 * The raw Snappy block format Parquet's SNAPPY codec stores: a varint uncompressed length followed
 * by literal and copy tags. The decoder is the only piece of Snappy the benchmark needs, so it
 * lives here rather than as a dependency; a malformed stream is refused by name.
 */
final class ParquetSnappy {

    private ParquetSnappy() {
    }

    /**
     * Decompresses one Snappy block.
     *
     * @throws IllegalArgumentException when the block is truncated or carries an unknown tag
     */
    static byte[] decompress(byte[] input) {
        int[] position = {0};
        long length = varint(input, position);
        if (length < 0 || length > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("a Snappy block names an impossible length");
        }
        byte[] out = new byte[(int) length];
        int produced = 0;
        while (position[0] < input.length) {
            int tag = input[position[0]++] & 0xff;
            int kind = tag & 0x03;
            if (kind == 0) {
                int literal = tag >>> 2;
                if (literal < 60) {
                    literal += 1;
                } else {
                    int bytes = literal - 59;
                    literal = 0;
                    for (int i = 0; i < bytes; i++) {
                        literal |= (input[position[0]++] & 0xff) << (8 * i);
                    }
                    literal += 1;
                }
                require(input, position[0], literal);
                require(out, produced, literal);
                System.arraycopy(input, position[0], out, produced, literal);
                position[0] += literal;
                produced += literal;
                continue;
            }
            int lengthValue;
            int offset;
            if (kind == 1) {
                lengthValue = ((tag >>> 2) & 0x07) + 4;
                require(input, position[0], 1);
                offset = ((tag >>> 5) << 8) | (input[position[0]++] & 0xff);
            } else if (kind == 2) {
                lengthValue = (tag >>> 2) + 1;
                require(input, position[0], 2);
                offset = (input[position[0]] & 0xff) | ((input[position[0] + 1] & 0xff) << 8);
                position[0] += 2;
            } else {
                lengthValue = (tag >>> 2) + 1;
                require(input, position[0], 4);
                offset = (input[position[0]] & 0xff) | ((input[position[0] + 1] & 0xff) << 8)
                        | ((input[position[0] + 2] & 0xff) << 16)
                        | ((input[position[0] + 3] & 0xff) << 24);
                position[0] += 4;
            }
            if (offset <= 0 || offset > produced) {
                throw new IllegalArgumentException("a Snappy copy names offset " + offset
                        + " with only " + produced + " bytes produced");
            }
            require(out, produced, lengthValue);
            for (int i = 0; i < lengthValue; i++) {
                out[produced + i] = out[produced + i - offset];
            }
            produced += lengthValue;
        }
        if (produced != out.length) {
            throw new IllegalArgumentException("a Snappy block produced " + produced + " of "
                    + out.length + " bytes");
        }
        return out;
    }

    private static long varint(byte[] input, int[] position) {
        long value = 0;
        int shift = 0;
        while (true) {
            if (position[0] >= input.length || shift > 63) {
                throw new IllegalArgumentException("a Snappy block ends inside its length varint");
            }
            int b = input[position[0]++] & 0xff;
            value |= ((long) (b & 0x7f)) << shift;
            if ((b & 0x80) == 0) {
                return value;
            }
            shift += 7;
        }
    }

    private static void require(byte[] data, int position, int count) {
        if (count < 0 || position + count > data.length) {
            throw new IllegalArgumentException("a Snappy block is truncated");
        }
    }
}
