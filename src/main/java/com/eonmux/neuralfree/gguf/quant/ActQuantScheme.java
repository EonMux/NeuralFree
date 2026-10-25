package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

/**
 * How a quantized checkpoint quantizes the input activations of one projection, emulated as fake-quantization
 * {@code x' = dequant(quant(x))} before the unchanged weight kernel ({@link ActFakeQuant} on the CPU,
 * {@code LLM_ActFakeQuant} on the GPU). Each scheme quantizes a row of {@code x} in runs of {@link #runLength(int)}
 * elements, with the operation order of the vLLM kernel it reproduces. Notation: {@code amax = max|x|} over the run;
 * {@code /} and {@code 1/s} are correctly rounded float operations; {@code rne} rounds to nearest even (NaN to 0);
 * {@code sat8} clamps to {@code [-128, 127]}; {@code e4m3(v)} is the value of the E4M3FN round-to-nearest-even code
 * of {@code clamp(v, -448, 448)}; {@code e2m1(v)} the E2M1 round-to-nearest-even value. {@code s} is the dequant
 * multiplier, and the static parameters are the weight's sibling tensors {@code <base>.input_scale} and
 * {@code <base>.input_zero_point}:
 * <ul>
 *   <li>{@link #INT8_TOKEN}: {@code s = amax/127}; {@code inv = (amax == 0) ? 0 : 127/amax};
 *       {@code q = sat8(rne(x*inv))}; {@code x' = q*s}; no floor (vLLM csrc/quantization/w8a8/int8/scaled_quant.cu
 *       :164-176).</li>
 *   <li>{@link #INT8_STATIC}: {@code s = input_scale}; {@code q = sat8(rne(x/s))}; {@code x' = q*s}
 *       (scaled_quant.cu:99-116).</li>
 *   <li>{@link #INT8_STATIC_ASYM}: {@code s = input_scale}, {@code azp = input_zero_point};
 *       {@code q = sat8(rne(x*(1/s)) + azp)}; {@code x' = (q - azp)*s} (scaled_quant.cu:119-139).</li>
 *   <li>{@link #INT8_GROUP128}: per 128-element run {@code s = max(amax, 1e-10f)/127};
 *       {@code q = trunc(clamp(x/s, -128, 127))} (the Triton cast truncates toward zero); {@code x' = q*s} (vLLM
 *       model_executor/layers/quantization/utils/int8_utils.py:123-125).</li>
 *   <li>{@link #FP8_TOKEN}: {@code a = min(amax, ub)} with {@code ub} the model's
 *       {@code neuralfree.act_quant.scale_ub} ({@code +inf} when absent); {@code s = max(a/448, 1/(448*512))}
 *       (the float {@code 0x36924925}); {@code x' = e4m3(x/s)*s} (vLLM csrc/quantization/utils.cuh:50-54,
 *       w8a8/fp8/common.cu:204-243; fbgemm_fp8.py:76, :142-146).</li>
 *   <li>{@link #FP8_STATIC}: {@code s = input_scale}; {@code x' = e4m3(x*(1/s))*s} (common.cu:184-201).</li>
 *   <li>{@link #FP8_GROUP128}: per 128-element run {@code s = max(amax, 1e-10f)/448};
 *       {@code x' = e4m3(x*(1/s))*s} (vLLM w8a8/fp8/per_token_group_quant.cu:46-90).</li>
 *   <li>{@link #NVFP4}: global scale {@code g = input_scale}; per 16-element run {@code s16 = e4m3(amax/6/g)} and
 *       {@code x' = (e2m1(x/(s16*g)) * s16) * g}, the products left to right.</li>
 *   <li>{@link #MXFP4}: per 32-element run, {@code r = amax} rounded to a 3-bit mantissa (RNE,
 *       {@code (bits(amax) + 0x200000) & 0x7F800000}), {@code r = 2^-126} when that is 0,
 *       {@code e = clamp(floor(log2(r)) - 2, -127, 127)} ({@code e = 127} when {@code r} is infinite),
 *       {@code x' = e2m1(x/2^e)*2^e} (vLLM mxfp4_utils.py:255-280).</li>
 *   <li>{@link #MXFP8}: per 32-element run,
 *       {@code e = clamp(ceil(log2(max(amax, FLT_MIN)/448)) + 127, 0, 254) - 127},
 *       {@code x' = e4m3(x/2^e)*2^e} (vLLM mxfp8_utils.py:55-70).</li>
 * </ul>
 * A zero dequant multiplier gives {@code x' = +0}. The {@link #code()} values are stable: they are stored in the
 * {@code neuralfree.act_quant.schemes} metadata array and passed to the GPU fake-quant kernels.
 */
public enum ActQuantScheme {
    /** Dynamic symmetric int8, one scale per token row. */
    INT8_TOKEN(0, Run.ROW, Param.NONE),
    /** Static symmetric int8, scale {@code input_scale}. */
    INT8_STATIC(1, Run.ROW, Param.SCALE),
    /** Static asymmetric int8, scale {@code input_scale} and zero point {@code input_zero_point}. */
    INT8_STATIC_ASYM(2, Run.ROW, Param.SCALE_AND_ZERO_POINT),
    /** Dynamic symmetric int8, one scale per 128-element run (SGLang {@code blockwise_int8}). */
    INT8_GROUP128(3, Run.GROUP_128, Param.NONE),
    /** Dynamic E4M3 FP8, one scale per token row. */
    FP8_TOKEN(4, Run.ROW, Param.NONE),
    /** Static E4M3 FP8, scale {@code input_scale}. */
    FP8_STATIC(5, Run.ROW, Param.SCALE),
    /** Dynamic E4M3 FP8, one scale per 128-element run (DeepSeek block FP8). */
    FP8_GROUP128(6, Run.GROUP_128, Param.NONE),
    /** NVFP4: E2M1 elements, an E4M3 scale per 16-element run, global scale {@code input_scale}. */
    NVFP4(7, Run.GROUP_16, Param.GLOBAL_SCALE),
    /** MXFP4: E2M1 elements, a power-of-two scale per 32-element run. */
    MXFP4(8, Run.GROUP_32, Param.NONE),
    /** MXFP8: E4M3 elements, a power-of-two scale per 32-element run. */
    MXFP8(9, Run.GROUP_32, Param.NONE);

    /** Elements sharing one scale. */
    private enum Run {
        ROW(0), GROUP_16(16), GROUP_32(32), GROUP_128(128);

        /** Run length in elements, or 0 for the whole row. */
        final int length;

        Run(int length) {
            this.length = length;
        }
    }

    /** Checkpoint parameters the scheme reads from the weight's sibling tensors. */
    private enum Param {
        NONE, SCALE, SCALE_AND_ZERO_POINT, GLOBAL_SCALE
    }

    private static final ActQuantScheme[] BY_CODE = buildByCode();

    private final int code;
    private final Run run;
    private final Param param;

    ActQuantScheme(int code, Run run, Param param) {
        this.code = code;
        this.run = run;
        this.param = param;
    }

    private static ActQuantScheme[] buildByCode() {
        ActQuantScheme[] byCode = new ActQuantScheme[values().length];
        for (ActQuantScheme s : values()) {
            if (s.code < 0 || s.code >= byCode.length || byCode[s.code] != null) {
                throw new IllegalStateException("activation quantization codes must be dense and unique: " + s);
            }
            byCode[s.code] = s;
        }
        return byCode;
    }

    /** The stable scheme code (0-9) of the metadata array and the GPU kernels. */
    public int code() {
        return code;
    }

    /**
     * The scheme with code {@code code}.
     *
     * @throws GgufFormatException naming the code when no scheme has it
     */
    public static ActQuantScheme fromCode(int code) {
        if (code < 0 || code >= BY_CODE.length) {
            throw new GgufFormatException("unknown activation quantization scheme code " + code
                    + "; known codes are 0.." + (BY_CODE.length - 1));
        }
        return BY_CODE[code];
    }

    /**
     * Elements quantized with one scale in a row of {@code rowLen} elements: the whole row for the per-token and
     * static schemes, otherwise 16, 32 or 128.
     *
     * @throws IllegalArgumentException when {@code rowLen} is not positive or not a whole number of runs
     */
    public int runLength(int rowLen) {
        if (rowLen <= 0) {
            throw new IllegalArgumentException(this + " activation rows must hold at least one element, got "
                    + rowLen);
        }
        if (run == Run.ROW) {
            return rowLen;
        }
        if (rowLen % run.length != 0) {
            throw new IllegalArgumentException(this + " quantizes activations in " + run.length
                    + "-element runs; a row of " + rowLen + " elements is not a whole number of runs");
        }
        return run.length;
    }

    /** True when the quantization scale is the checkpoint's {@code input_scale} rather than computed from {@code x}. */
    public boolean isStatic() {
        return param == Param.SCALE || param == Param.SCALE_AND_ZERO_POINT;
    }

    /** True when the scheme reads the zero point {@code input_zero_point} (asymmetric int8). */
    public boolean needsZeroPoint() {
        return param == Param.SCALE_AND_ZERO_POINT;
    }

    /** True when the scheme reads {@code input_scale} as the global scale of a two-level scheme (NVFP4). */
    public boolean needsGlobalScale() {
        return param == Param.GLOBAL_SCALE;
    }
}
