package com.eonmux.neuralfree.gguf.engine.batch;

import org.junit.Test;

import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertArrayEquals;

/**
 * {@link PickleWriter} and {@link CborWriter} byte for byte against the bytes Python's
 * {@code pickle.dumps(value, protocol=5)} (the C pickler) and {@code cbor2.dumps(value, canonical=True)} write:
 * every opcode form of the block-hash inputs — short and long strings and byte strings, ints of every encoding
 * class, tuples of 0, 1, 2, 3 and 5 elements, {@code None}, and the pickle frame header when 4 or more bytes follow.
 */
public class PickleWriterCborWriterTest {

    private static byte[] utf8(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }

    @Test
    public void aShortStringRidesTheShortUnicodeForm() {
        PickleWriter w = new PickleWriter();
        w.string("vllm-none-hash");
        assertArrayEquals(hex("80059512000000000000008c0e766c6c6d2d6e6f6e652d68617368942e"), w.finish());
        CborWriter c = new CborWriter();
        c.string("vllm-none-hash");
        assertArrayEquals(hex("6e766c6c6d2d6e6f6e652d68617368"), c.finish());
    }

    @Test
    public void aLongStringRidesTheLongUnicodeForm() {
        String value = repeat('s', 300);
        PickleWriter w = new PickleWriter();
        w.string(value);
        byte[] expected = hex("8005953301000000000000" + "582c010000"
                + hexOf(utf8(value)) + "942e");
        assertArrayEquals(expected, w.finish());
        CborWriter c = new CborWriter();
        c.string(value);
        assertArrayEquals(hex("79012c" + hexOf(utf8(value))), c.finish());
    }

    @Test
    public void byteValuesRideTheBinbytesForms() {
        PickleWriter w = new PickleWriter();
        w.bytes(range32());
        assertArrayEquals(hex("80059524000000000000004320000102030405060708090a0b0c0d0e0f101112131415161718191a1b"
                + "1c1d1e1f942e"), w.finish());
        CborWriter c = new CborWriter();
        c.bytes(range32());
        assertArrayEquals(hex("5820000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"), c.finish());

        byte[] longBytes = new byte[512];
        for (int i = 0; i < longBytes.length; i++) {
            longBytes[i] = (byte) (i % 256);
        }
        PickleWriter big = new PickleWriter();
        big.bytes(longBytes);
        // BINBYTES (0x42) with the 4-byte little-endian length 0x200, then MEMOIZE
        assertArrayEquals(hex("80059507020000000000004200020000" + hexOf(longBytes) + "942e"), big.finish());
        CborWriter bigCbor = new CborWriter();
        bigCbor.bytes(longBytes);
        assertArrayEquals(hex("590200" + hexOf(longBytes)), bigCbor.finish());
    }

    @Test
    public void noneAndTheIntClassesTakeTheirOwnOpcodes() {
        PickleWriter none = new PickleWriter();
        none.nullValue();
        assertArrayEquals(hex("80054e2e"), none.finish());

        assertPickleInt(5, "80054b052e");
        assertPickleInt(0, "80054b002e");
        assertPickleInt(255, "80054bff2e");
        assertPickleInt(300, "80059504000000000000004d2c012e");
        assertPickleInt(65535, "80059504000000000000004dffff2e");
        assertPickleInt(-1, "80059506000000000000004affffffff2e");
        assertPickleInt(Integer.MAX_VALUE, "80059506000000000000004affffff7f2e");
        assertPickleInt(Integer.MIN_VALUE, "80059506000000000000004a000000802e");

        CborWriter c = new CborWriter();
        c.nullValue();
        assertArrayEquals(hex("f6"), c.finish());
        CborWriter i5 = new CborWriter();
        i5.integer(5);
        assertArrayEquals(hex("05"), i5.finish());
        CborWriter i300 = new CborWriter();
        i300.integer(300);
        assertArrayEquals(hex("19012c"), i300.finish());
        CborWriter iNeg = new CborWriter();
        iNeg.integer(-1);
        assertArrayEquals(hex("20"), iNeg.finish());
        CborWriter iBig = new CborWriter();
        iBig.integer(Integer.MIN_VALUE);
        assertArrayEquals(hex("3a7fffffff"), iBig.finish());
    }

    private static void assertPickleInt(int value, String expected) {
        PickleWriter w = new PickleWriter();
        w.integer(value);
        assertArrayEquals("pickle of " + value, hex(expected.replace(" ", "")), w.finish());
    }

    @Test
    public void tuplesTakeTheFormTheirLengthAsksFor() {
        PickleWriter empty = new PickleWriter();
        empty.tupleOpen(0);
        empty.tupleClose(0);
        assertArrayEquals(hex("8005292e"), empty.finish());
        CborWriter emptyCbor = new CborWriter();
        emptyCbor.array(0);
        assertArrayEquals(hex("80"), emptyCbor.finish());

        PickleWriter one = new PickleWriter();
        one.tupleOpen(1);
        one.string("solo");
        one.tupleClose(1);
        assertArrayEquals(hex("8005950a000000000000008c04736f6c6f9485942e"), one.finish());
        CborWriter oneCbor = new CborWriter();
        oneCbor.array(1);
        oneCbor.string("solo");
        assertArrayEquals(hex("8164736f6c6f"), oneCbor.finish());

        PickleWriter two = new PickleWriter();
        two.tupleOpen(2);
        two.string("lora");
        two.string("ada");
        two.tupleClose(2);
        assertArrayEquals(hex("80059510000000000000008c046c6f7261948c036164619486942e"), two.finish());
        CborWriter twoCbor = new CborWriter();
        twoCbor.array(2);
        twoCbor.string("lora");
        twoCbor.string("ada");
        assertArrayEquals(hex("82646c6f726163616461"), twoCbor.finish());

        PickleWriter three = new PickleWriter();
        three.tupleOpen(3);
        three.bytes(range8());
        three.tupleOpen(2);
        three.integer(1);
        three.integer(2);
        three.tupleClose(2);
        three.nullValue();
        three.tupleClose(3);
        assertArrayEquals(hex("800595150000000000000043080001020304050607944b014b0286944e87942e"), three.finish());
        CborWriter threeCbor = new CborWriter();
        threeCbor.array(3);
        threeCbor.bytes(range8());
        threeCbor.array(2);
        threeCbor.integer(1);
        threeCbor.integer(2);
        threeCbor.nullValue();
        assertArrayEquals(hex("83480001020304050607820102f6"), threeCbor.finish());

        PickleWriter five = new PickleWriter();
        five.tupleOpen(5);
        for (String s : new String[] {"a", "b", "c", "d", "e"}) {
            five.string(s);
        }
        five.tupleClose(5);
        assertArrayEquals(hex("8005951800000000000000288c0161948c0162948c0163948c0164948c01659474942e"), five.finish());
        CborWriter fiveCbor = new CborWriter();
        fiveCbor.array(5);
        for (String s : new String[] {"a", "b", "c", "d", "e"}) {
            fiveCbor.string(s);
        }
        assertArrayEquals(hex("8561616162616361646165"), fiveCbor.finish());
    }

    /** A 3-byte tail rides without the pickle frame header ({@code _FRAME_SIZE_MIN} is 4). */
    @Test
    public void aShortTailNeedsNoFrame() {
        PickleWriter w = new PickleWriter();
        w.integer(255);
        assertArrayEquals(hex("80054bff2e"), w.finish());
    }

    private static String repeat(char c, int n) {
        char[] chars = new char[n];
        java.util.Arrays.fill(chars, c);
        return new String(chars);
    }

    private static byte[] range32() {
        byte[] bytes = new byte[32];
        for (int i = 0; i < 32; i++) {
            bytes[i] = (byte) i;
        }
        return bytes;
    }

    private static byte[] range8() {
        byte[] bytes = new byte[8];
        for (int i = 0; i < 8; i++) {
            bytes[i] = (byte) i;
        }
        return bytes;
    }

    private static String hexOf(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(Character.forDigit((b >> 4) & 0xf, 16)).append(Character.forDigit(b & 0xf, 16));
        }
        return sb.toString();
    }

    private static byte[] hex(String hex) {
        byte[] out = new byte[hex.length() / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) Integer.parseInt(hex.substring(i * 2, i * 2 + 2), 16);
        }
        return out;
    }
}
