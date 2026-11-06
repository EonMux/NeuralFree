package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * The safetensors dtype of a tensorizer dtype string.
 *
 * <p>Tensorizer stores the numpy dtype string ({@code <f4}, {@code |u1}, ...) and, for a torch dtype numpy cannot
 * represent, that dtype after a NUL ({@code <V2\0torch.bfloat16}). The torch name is authoritative when present;
 * otherwise the numpy name is mapped directly. Big-endian, packed and sub-byte dtypes are refused by name.</p>
 */
final class TensorizerDtype {

    private static final char OPAQUE_SEPARATOR = '\0';

    private static final Map<String, StDType> TORCH = torch();
    private static final Map<String, StDType> NUMPY = numpy();

    private TensorizerDtype() {
    }

    private static Map<String, StDType> torch() {
        Map<String, StDType> m = new HashMap<>();
        m.put("torch.bool", StDType.BOOL);
        m.put("torch.uint8", StDType.U8);
        m.put("torch.int8", StDType.I8);
        m.put("torch.int16", StDType.I16);
        m.put("torch.int32", StDType.I32);
        m.put("torch.int64", StDType.I64);
        m.put("torch.uint16", StDType.U16);
        m.put("torch.uint32", StDType.U32);
        m.put("torch.uint64", StDType.U64);
        m.put("torch.float16", StDType.F16);
        m.put("torch.bfloat16", StDType.BF16);
        m.put("torch.float32", StDType.F32);
        m.put("torch.float64", StDType.F64);
        m.put("torch.complex64", StDType.C64);
        m.put("torch.float8_e5m2", StDType.F8_E5M2);
        m.put("torch.float8_e4m3fn", StDType.F8_E4M3);
        m.put("torch.float8_e5m2fnuz", StDType.F8_E5M2FNUZ);
        m.put("torch.float8_e4m3fnuz", StDType.F8_E4M3FNUZ);
        m.put("torch.float8_e8m0fnu", StDType.F8_E8M0);
        return m;
    }

    private static Map<String, StDType> numpy() {
        Map<String, StDType> m = new HashMap<>();
        m.put("|b1", StDType.BOOL);
        m.put("|i1", StDType.I8);
        m.put("|u1", StDType.U8);
        m.put("<i2", StDType.I16);
        m.put("<u2", StDType.U16);
        m.put("<i4", StDType.I32);
        m.put("<u4", StDType.U32);
        m.put("<i8", StDType.I64);
        m.put("<u8", StDType.U64);
        m.put("<f2", StDType.F16);
        m.put("<f4", StDType.F32);
        m.put("<f8", StDType.F64);
        m.put("<c8", StDType.C64);
        return m;
    }

    /**
     * The dtype of the tensorizer dtype string {@code dtype}.
     *
     * @throws IOException when the string is not a dtype NeuralFree reads, naming the tensor and the dtype
     */
    static StDType parse(Path path, String name, String dtype) throws IOException {
        String numpy = dtype;
        String torch = null;
        int at = dtype.indexOf(OPAQUE_SEPARATOR);
        if (at >= 0) {
            numpy = dtype.substring(0, at);
            torch = dtype.substring(at + 1);
            if (torch.indexOf(OPAQUE_SEPARATOR) >= 0) {
                throw invalid(path, name, dtype, "it holds more than one dtype separator");
            }
        }
        StDType byTorch = torch == null ? null : TORCH.get(torch);
        if (torch != null && byTorch == null) {
            throw invalid(path, name, dtype, "torch dtype '" + torch + "' has no safetensors equivalent");
        }
        StDType byNumpy = NUMPY.get(normalize(numpy));
        if (byNumpy == null && byTorch == null) {
            throw invalid(path, name, dtype, "the numpy dtype '" + numpy + "' has no safetensors equivalent");
        }
        StDType chosen = byTorch != null ? byTorch : byNumpy;
        if (byTorch != null && byNumpy != null && byTorch.bits() != byNumpy.bits()) {
            throw invalid(path, name, dtype, "the numpy and torch dtypes disagree on the element size");
        }
        if (torch != null && byNumpy == null) {
            // the numpy dtype of an opaque dtype is a byte blob: check its element size against the torch dtype
            int bytes = opaqueBytes(numpy);
            if (bytes > 0 && bytes * 8 != chosen.bits()) {
                throw invalid(path, name, dtype, "the opaque numpy dtype holds " + bytes + " bytes but " + torch
                        + " needs " + chosen.bits() / 8);
            }
        }
        return chosen;
    }

    /** A numpy dtype string with a native ({@code =}) byte order made little-endian. */
    private static String normalize(String numpy) {
        if (numpy.startsWith("=")) {
            return "<" + numpy.substring(1);
        }
        return numpy;
    }

    /** The element size of an opaque numpy dtype ({@code <V2}), or -1. */
    private static int opaqueBytes(String numpy) {
        String n = normalize(numpy);
        if (n.length() < 3 || (n.charAt(0) != '<' && n.charAt(0) != '|') || n.charAt(1) != 'V') {
            return -1;
        }
        try {
            return Integer.parseInt(n.substring(2));
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    private static IOException invalid(Path path, String name, String dtype, String reason) {
        return new IOException("Invalid tensorizer file '" + path + "': tensor '" + name + "' has dtype '"
                + dtype + "': " + reason);
    }
}
