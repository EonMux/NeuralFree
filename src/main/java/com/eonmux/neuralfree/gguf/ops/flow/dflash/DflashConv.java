package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import java.util.Objects;

/**
 * DFlash2 dynamic depthwise convolution over the positions of one noise block, {@code build_dflash2_conv}
 * (dflash.cpp:483-546). For output token {@code t}, channel {@code c} and side {@code s}:
 * <pre>
 *   weight(t, tap, c) = dynamic[t][g + G*(tap + K*s)] + base[c + E*(tap + K*s)],   g = c / groupSize
 *   out[t][c]         = sum_{tap < min(K, B)} weight(t, tap, c) * (t - tap >= 0 ? hidden[t - tap][c] : 0)
 * </pre>
 * where {@code E} is the hidden size, {@code K} the kernel size, {@code G = E / groupSize} the group count and
 * {@code B} the block length. {@code dynamic} is the per-token {@code conv_proj} output ({@code 2*K*G} values,
 * ggml shape {@code [G, K, 2]}); {@code base} is the {@code conv_base} tensor (ggml shape {@code [E, K, 2]}).
 * Taps at or past the block length only read the zero left padding and are skipped, as the graph does. Stateless.
 */
public final class DflashConv {

    private final int hidden;
    private final int kernel;
    private final int groupSize;
    private final int groups;

    /**
     * @throws IllegalArgumentException for non-positive sizes or a hidden size the group size does not divide
     */
    public DflashConv(int hidden, int kernel, int groupSize) {
        if (hidden <= 0 || kernel <= 0 || groupSize <= 0 || hidden % groupSize != 0) {
            throw new IllegalArgumentException("DFlash2 conv needs positive sizes with conv_group_size dividing the"
                    + " hidden size, got hidden " + hidden + ", kernel " + kernel + ", group " + groupSize);
        }
        this.hidden = hidden;
        this.kernel = kernel;
        this.groupSize = groupSize;
        this.groups = hidden / groupSize;
    }

    /** Values per token of the dynamic coefficients ({@code conv_proj} rows): {@code 2 * kernel * groups}. */
    public int dynamicWidth() {
        return 2 * kernel * groups;
    }

    /** Values of the {@code conv_base} tensor: {@code hidden * kernel * 2}. */
    public int baseLength() {
        return hidden * kernel * 2;
    }

    /**
     * Convolves one block of {@code nTokens} rows. {@code out} must not overlap {@code x}.
     *
     * @param x       block rows {@code x[xOff + t*hidden + c]}
     * @param dynamic coefficient rows {@code dynamic[dOff + t*dynamicWidth() + i]}
     * @param base    the {@code conv_base} values ({@link #baseLength()} floats)
     * @param side    0 for the input side, 1 for the output side
     * @param out     result rows {@code out[outOff + t*hidden + c]}
     * @throws IllegalArgumentException for an invalid side, a negative token count or a range outside its array
     */
    public void apply(float[] x, int xOff, float[] dynamic, int dOff, float[] base, int side, int nTokens,
                      float[] out, int outOff) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(dynamic, "dynamic");
        Objects.requireNonNull(base, "base");
        Objects.requireNonNull(out, "out");
        if (side != 0 && side != 1) {
            throw new IllegalArgumentException("conv side must be 0 or 1, got " + side);
        }
        if (nTokens < 0) {
            throw new IllegalArgumentException("token count must be non-negative, got " + nTokens);
        }
        final int dw = dynamicWidth();
        requireRange("conv input", x.length, xOff, (long) nTokens * hidden);
        requireRange("conv coefficients", dynamic.length, dOff, (long) nTokens * dw);
        requireRange("conv base", base.length, 0, baseLength());
        requireRange("conv output", out.length, outOff, (long) nTokens * hidden);
        if (x == out && Math.abs((long) xOff - outOff) < (long) nTokens * hidden) {
            throw new IllegalArgumentException("conv output overlaps its input");
        }
        final int nTaps = Math.min(kernel, nTokens);
        for (int t = 0; t < nTokens; t++) {
            final int o = outOff + t * hidden;
            final int d = dOff + t * dw;
            for (int c = 0; c < hidden; c++) {
                out[o + c] = 0.0f;
            }
            for (int tap = 0; tap < nTaps; tap++) {
                final int src = t - tap;
                if (src < 0) {
                    continue;
                }
                final int coeffBase = d + groups * (tap + kernel * side);
                final int baseOff = hidden * (tap + kernel * side);
                final int xRow = xOff + src * hidden;
                for (int c = 0; c < hidden; c++) {
                    final float w = dynamic[coeffBase + c / groupSize] + base[baseOff + c];
                    out[o + c] += w * x[xRow + c];
                }
            }
        }
    }

    private static void requireRange(String what, int length, int off, long n) {
        if (off < 0 || off + n > length) {
            throw new IllegalArgumentException(what + " range [" + off + ", " + (off + n) + ") exceeds length "
                    + length);
        }
    }
}
