package com.eonmux.neuralfree.gguf.engine;

/**
 * How the W12 streaming ({@code neuralfree.weights.w12}) treats an F16 tensor's values, which a
 * {@link com.eonmux.neuralfree.gguf.GGMLType#W12} record can store only on the bfloat16 grid:
 * <ul>
 *   <li>{@link #REFUSE} (the default): the load refuses an F16 weight off the grid, naming the tensor, the
 *       first offending element and {@link SpeculativeOptions#PROP_W12_F16}. The W12 run keeps the lossless
 *       contract: its logits are the stored weights'.</li>
 *   <li>{@link #ROUND}: every weight is rounded onto the grid at upload (round-to-nearest-even), so each
 *       off-grid weight loses its 3 low mantissa bits and the output can differ from the F16 run; the load
 *       reports how many weights moved and their largest relative change.</li>
 * </ul>
 */
public enum W12F16Mode {

    /** Refuse an off-grid F16 weight by name: the W12 streaming stays lossless. */
    REFUSE,
    /** Round off-grid F16 weights onto the bfloat16 grid at upload: lossy, 3 low mantissa bits per weight. */
    ROUND;

    /** The default: {@link #REFUSE}. */
    public static final W12F16Mode DEFAULT = REFUSE;

    /**
     * The mode {@code value} names, case-insensitively.
     *
     * @throws IllegalArgumentException naming {@code property} for any other value
     */
    public static W12F16Mode parse(String property, String value) {
        String v = value.trim().toLowerCase(java.util.Locale.ROOT);
        if (v.equals("refuse")) {
            return REFUSE;
        }
        if (v.equals("round")) {
            return ROUND;
        }
        throw new IllegalArgumentException(property + "=" + value + ": expected refuse or round");
    }

    /** The lowercase name a configuration or command line writes. */
    public String propertyNameValue() {
        return name().toLowerCase(java.util.Locale.ROOT);
    }
}
