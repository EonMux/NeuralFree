package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EmbeddingNorm;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;

/**
 * How one embeddings request shapes its vectors, with the llama.cpp and vLLM fields: {@code embd_normalize}
 * (llama.cpp's modes, {@link EmbeddingNorm}; default 2, euclidean), {@code use_activation} or {@code normalize}
 * false for raw vectors (vLLM), {@code dimensions} (the first {@code n} values, normalized after the cut, as vLLM's
 * matryoshka truncation), {@code truncate_prompt_tokens} (keep the last {@code n} tokens, -1 for the context
 * length), {@code add_special_tokens} (default true) and {@code encoding_format} {@code float} or {@code base64}
 * (float32 bytes, little-endian unless {@code endianness} is {@code big}). Immutable.
 *
 * @param norm          the normalization mode
 * @param dimensions    the kept leading values, or 0 for all
 * @param truncate      the kept trailing prompt tokens, or 0 for all
 * @param addSpecial    whether text inputs get the vocabulary's BOS/EOS (CLS/SEP)
 * @param base64        whether vectors are written as base64 float32 bytes
 * @param order         the byte order of base64 vectors
 */
record EmbeddingOptions(int norm, int dimensions, int truncate, boolean addSpecial, boolean base64,
                        ByteOrder order) {

    /** The options of {@code body} for a model whose prompts hold at most {@code contextLength} tokens. */
    static EmbeddingOptions of(JsonView body, int contextLength) {
        return new EmbeddingOptions(norm(body), dimensions(body), truncate(body, contextLength),
                body.optBool("add_special_tokens", true), base64(body), order(body));
    }

    private static int norm(JsonView body) {
        if (body.raw("embd_normalize") != null) {
            int mode = body.i32("embd_normalize");
            if (mode < EmbeddingNorm.NONE) {
                throw ServerJson.badRequest("embd_normalize must be -1 (none), 0 (max absolute), 1 (taxicab),"
                        + " 2 (euclidean) or a larger p-norm, got " + mode);
            }
            return mode;
        }
        boolean activation = body.optBool("use_activation", body.optBool("normalize", true));
        return activation ? EmbeddingNorm.EUCLIDEAN : EmbeddingNorm.NONE;
    }

    private static int dimensions(JsonView body) {
        if (body.raw("dimensions") == null) {
            return 0;
        }
        int n = body.i32("dimensions");
        if (n < 1) {
            throw ServerJson.badRequest("dimensions must be at least 1, got " + n);
        }
        return n;
    }

    private static int truncate(JsonView body, int contextLength) {
        if (body.raw("truncate_prompt_tokens") == null) {
            return 0;
        }
        int n = body.i32("truncate_prompt_tokens");
        if (n == -1) {
            return contextLength;
        }
        if (n < 1) {
            throw ServerJson.badRequest("truncate_prompt_tokens must be -1 or at least 1, got " + n);
        }
        return n;
    }

    private static boolean base64(JsonView body) {
        String format = body.optStr("encoding_format", "float");
        if (!format.equals("float") && !format.equals("base64")) {
            throw ServerJson.badRequest("encoding_format must be float or base64, got '" + format + "'");
        }
        String dtype = body.optStr("embed_dtype", "float32");
        if (!dtype.equals("float32")) {
            throw ServerJson.badRequest("embed_dtype '" + dtype + "' is not supported; the embeddings are float32");
        }
        return format.equals("base64");
    }

    private static ByteOrder order(JsonView body) {
        String endianness = body.optStr("endianness", "native");
        switch (endianness) {
            case "native":
            case "little":
                return ByteOrder.LITTLE_ENDIAN;
            case "big":
                return ByteOrder.BIG_ENDIAN;
            default:
                throw ServerJson.badRequest("endianness must be native, little or big, got '" + endianness + "'");
        }
    }

    /** The prompt after {@code truncate_prompt_tokens}: its last tokens. */
    int[] truncated(int[] tokens) {
        return truncate == 0 || tokens.length <= truncate ? tokens
                : Arrays.copyOfRange(tokens, tokens.length - truncate, tokens.length);
    }

    /**
     * A pooled vector cut to {@code dimensions} and normalized.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 when {@code dimensions} exceeds the vector
     */
    float[] shaped(float[] pooled) {
        float[] kept = pooled;
        if (dimensions > 0) {
            if (dimensions > pooled.length) {
                throw ServerJson.badRequest("dimensions " + dimensions + " exceeds the model's " + pooled.length
                        + "-value embedding");
            }
            kept = Arrays.copyOf(pooled, dimensions);
        }
        return EmbeddingNorm.normalize(kept, norm);
    }

    /** The vector as the response writes it: a number array, or base64 bytes. */
    Object encoded(float[] v) {
        if (base64) {
            ByteBuffer bytes = ByteBuffer.allocate(Float.BYTES * v.length).order(order);
            for (float x : v) {
                bytes.putFloat(x);
            }
            return Base64.getEncoder().encodeToString(bytes.array());
        }
        return values(v);
    }

    /** The values of {@code v} as JSON numbers. */
    static List<Object> values(float[] v) {
        List<Object> out = new ArrayList<>(v.length);
        for (float x : v) {
            out.add((double) x);
        }
        return out;
    }
}
