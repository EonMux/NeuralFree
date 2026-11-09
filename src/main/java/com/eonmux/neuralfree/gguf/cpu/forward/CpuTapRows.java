package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.engine.HiddenTap;

/**
 * The rows of the wanted tap points over one call, gathered step by step and delivered once per tap point after
 * the call ({@link HiddenTap}: {@code nRows} is every token of the call, in call order; a batched step stores
 * each of its rows at the call row the ubatch's {@code batchRows} names).
 */
final class CpuTapRows {
    private final HiddenTap tap;
    private final int total;
    private final float[][] rows;
    private final int[] widths;

    private CpuTapRows(HiddenTap tap, int total, int points) {
        this.tap = tap;
        this.total = total;
        this.rows = new float[points][];
        this.widths = new int[points];
    }

    /** Collector of one call over tap points {@code [0, nLayer]}; null when the tap wants none of them. */
    static CpuTapRows open(HiddenTap tap, int total, int nLayer) {
        for (int p = 0; p <= nLayer; p++) {
            if (tap.wants(p)) {
                return new CpuTapRows(tap, total, nLayer + 1);
            }
        }
        return null;
    }

    boolean wants(int point) {
        return tap.wants(point);
    }

    /** Copies the step's {@code n} rows, which are rows {@code [off, off + n)} of the call. */
    void capture(int point, float[] x, int off, int n, int width) {
        ensure(point, width);
        System.arraycopy(x, 0, rows[point], off * width, n * width);
    }

    /**
     * Copies the step's {@code n} rows (row {@code r} of {@code x} is row {@code callRows[r]} of the call, the
     * ubatch's batch row mapping).
     */
    void capture(int point, float[] x, int[] callRows, int n, int width) {
        ensure(point, width);
        for (int r = 0; r < n; r++) {
            System.arraycopy(x, r * width, rows[point], callRows[r] * width, width);
        }
    }

    /** The call-wide row buffer of {@code point}: every step of the call writes its own rows into it. */
    private void ensure(int point, int width) {
        if (rows[point] == null) {
            rows[point] = new float[Math.multiplyExact(total, width)];
            widths[point] = width;
        } else if (widths[point] != width) {
            throw new IllegalStateException("tap point " + point + " changed width from " + widths[point]
                    + " to " + width + " within a call");
        }
    }

    void deliver() {
        for (int p = 0; p < rows.length; p++) {
            if (rows[p] != null) {
                tap.onHidden(p, rows[p], total, widths[p]);
            }
        }
    }
}
