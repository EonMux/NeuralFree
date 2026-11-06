package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

import static org.junit.Assert.assertNotNull;

/**
 * The shared half of the vLLM online-quantization parity tests: the golden checkpoints' LCG BF16 weights, the
 * golden files' reader and the canonical digests of a planned tensor ({@code vllm-online-parity.txt} documents
 * them).
 */
final class VllmParityDigests {

    private VllmParityDigests() {
    }

    /** The golden's BF16 weight: an LCG-drawn mantissa and sign over exponents near {@code base}. */
    static byte[] lcgBf16(int rows, int cols, long seed, int base) {
        byte[] out = new byte[2 * rows * cols];
        long s = seed & 0xFFFFFFFFL;
        for (int i = 0; i < rows * cols; i++) {
            s = (s * 1664525L + 1013904223L) & 0xFFFFFFFFL;
            long a = s;
            s = (s * 1664525L + 1013904223L) & 0xFFFFFFFFL;
            long b = s;
            int e = (int) ((a >>> 24) & 7);
            e = e >= 6 ? e - 2 : e;
            if ((b >>> 22) == 0) {
                e = 6 + (int) ((b >>> 21) & 1);
            }
            int bits = (int) (((a >>> 31) << 15) | ((long) (base + e) << 7) | ((a >>> 8) & 0x7F));
            out[2 * i] = (byte) bits;
            out[2 * i + 1] = (byte) (bits >>> 8);
        }
        return out;
    }

    /** The non-comment lines of the test resource {@code name}, split on single spaces. */
    static List<String[]> lines(String name) throws IOException {
        List<String[]> out = new ArrayList<>();
        try (InputStream in = VllmParityDigests.class.getResourceAsStream(name)) {
            assertNotNull(name, in);
            BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
            for (String line = r.readLine(); line != null; line = r.readLine()) {
                if (!line.isEmpty() && !line.startsWith("#")) {
                    out.add(line.split(" "));
                }
            }
        }
        return out;
    }

    /** Every stored byte of {@code src}. */
    static byte[] bytes(RowSource src) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(src.rows() * src.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        src.readRows(0, Math.toIntExact(src.rows()), b);
        return b.array();
    }

    static String sha256(byte[] data) {
        try {
            byte[] d = MessageDigest.getInstance("SHA-256").digest(data);
            StringBuilder hex = new StringBuilder();
            for (byte x : d) {
                hex.append(String.format("%02x", x & 0xFF));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /**
     * The golden's digests of a planned tensor: its codes, its block scales and, for NVFP4, its global scales; for
     * a float tensor, its stored bytes ({@code values}).
     */
    static Map<String, String> digests(RowSource src) throws IOException {
        byte[] all = bytes(src);
        GGMLType type = src.format().ggmlType();
        Map<String, String> out = new LinkedHashMap<>();
        if (type == GGMLType.BF16 || type == GGMLType.F16 || type == GGMLType.F32) {
            out.put("values", sha256(all));
            return out;
        }
        ByteArrayOutputStream codes = new ByteArrayOutputStream();
        ByteArrayOutputStream scales = new ByteArrayOutputStream();
        int blockBytes = type.getTypeSize();
        for (int off = 0; off < all.length; off += blockBytes) {
            block(type, all, off, codes, scales);
        }
        out.put("codes", sha256(codes.toByteArray()));
        out.put("scales", sha256(scales.toByteArray()));
        float[] global = src.outputScale();
        if (global != null) {
            StringJoiner j = new StringJoiner(" ");
            for (float g : global) {
                j.add(String.format("%08x", Float.floatToRawIntBits(g)));
            }
            out.put("global", j.toString());
        }
        return out;
    }

    private static void block(GGMLType type, byte[] all, int off, ByteArrayOutputStream codes,
                              ByteArrayOutputStream scales) {
        switch (type) {
            case F8_E4M3_B128:
                scales.write(all, off + BlockLayout.F8B128_D, 4);
                codes.write(all, off + BlockLayout.F8B128_QS, BlockLayout.QK_F8B128);
                break;
            case F8_E4M3_B32:
                scales.write(all, off + BlockLayout.F8B32_D, 4);
                codes.write(all, off + BlockLayout.F8B32_QS, BlockLayout.QK_F8B32);
                break;
            case MXFP4:
                scales.write(all[off + BlockLayout.MXFP4_E]);
                nibbles(all, off + BlockLayout.MXFP4_QS, BlockLayout.QK_MXFP4 / 2, false, codes);
                break;
            case NVFP4:
                int subs = BlockLayout.QK_NVFP4 / BlockLayout.QK_NVFP4_SUB;
                for (int s = 0; s < subs; s++) {
                    scales.write(all[off + BlockLayout.NVFP4_D + s]);
                    nibbles(all, off + BlockLayout.NVFP4_QS + s * BlockLayout.QK_NVFP4_SUB / 2,
                            BlockLayout.QK_NVFP4_SUB / 2, true, codes);
                }
                break;
            default:
                throw new AssertionError("no golden layout for " + type);
        }
    }

    /** The split-nibble codes of {@code half} bytes: the low nibbles, then the high ones. */
    private static void nibbles(byte[] all, int at, int half, boolean zeroForNegativeZero, ByteArrayOutputStream out) {
        for (int shift = 0; shift <= 4; shift += 4) {
            for (int j = 0; j < half; j++) {
                int code = (all[at + j] >>> shift) & 0x0F;
                out.write(zeroForNegativeZero && code == 0x8 ? 0 : code);
            }
        }
    }
}
