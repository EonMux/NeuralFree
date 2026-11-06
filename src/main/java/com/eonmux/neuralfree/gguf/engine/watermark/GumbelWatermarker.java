package com.eonmux.neuralfree.gguf.engine.watermark;

import java.util.Objects;
import java.util.function.DoubleSupplier;

/**
 * The single-key Gumbel-max watermarker of vLLM's {@code GumbelWatermarker}: every candidate token's draw comes from
 * one {@link WatermarkPrf} keyed by the request's secret key. Immutable; the PRF is stateless, so one instance may
 * serve any number of sequences as long as each call carries its own context.
 */
public final class GumbelWatermarker extends Watermarker {

    private final WatermarkPrf prf;
    private final int contextWidth;

    /** A Philox-keyed watermarker ({@code key} is the 64-bit secret as a {@code long} bit pattern). */
    public GumbelWatermarker(long key, int contextWidth) {
        this(key, contextWidth, WatermarkPrfName.PHILOX);
    }

    /**
     * A watermarker over the named built-in PRF.
     *
     * @throws IllegalArgumentException when the context width is below 1
     */
    public GumbelWatermarker(long key, int contextWidth, WatermarkPrfName prfName) {
        this(new PhiloxPrf(key), contextWidth);
        Objects.requireNonNull(prfName, "prf");
        if (prfName != WatermarkPrfName.PHILOX) {
            throw new IllegalArgumentException("unsupported watermark PRF " + prfName.label());
        }
    }

    /**
     * A watermarker over an explicit PRF (the key travels in the PRF itself).
     *
     * @throws IllegalArgumentException when the context width is below 1
     */
    public static GumbelWatermarker over(WatermarkPrf prf, int contextWidth) {
        return new GumbelWatermarker(prf, contextWidth);
    }

    private GumbelWatermarker(WatermarkPrf prf, int contextWidth) {
        Objects.requireNonNull(prf, "prf");
        if (contextWidth < 1) {
            throw new IllegalArgumentException("watermark context width must be 1 or more, got " + contextWidth);
        }
        this.prf = prf;
        this.contextWidth = contextWidth;
    }

    /** The pseudorandom function this watermarker draws from. */
    public WatermarkPrf prf() {
        return prf;
    }

    @Override
    public int contextWidth() {
        return contextWidth;
    }

    @Override
    public int sample(int[] tokenIds, float[] logits, int count, int[] context, DoubleSupplier routing) {
        return gumbelSelect(prf, tokenIds, logits, count, context, contextWidth);
    }

    @Override
    public String toString() {
        return "GumbelWatermarker{contextWidth=" + contextWidth + ", prf=" + prf.version() + '}';
    }
}
