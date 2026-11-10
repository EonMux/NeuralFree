package com.eonmux.neuralfree.gguf.cpu.forward;

/**
 * The tap mixes of limite's MUDD dense connections (limite.cpp:120-140), the host reference of the
 * {@code LLM_MuddMix} kernel: for one token, the attention input and the residual are weighted sums of the banked
 * layer inputs of the layer's taps, {@code cur[i] = sum_j w[j] * h_j[i]} and {@code resid[i] = sum_j w[n_taps + j]
 * * h_j[i]}, accumulated in tap order in single precision. The bank holds one slot of {@code bankStride} floats per
 * trunk layer, each slot the step's rows of that layer's input.
 */
public final class MuddMath {

    private MuddMath() {
    }

    /**
     * Mixes token {@code t}: {@code h_j[i] = bank[bankOff + taps[j] * bankStride + t * nEmbd + i]}, the weights
     * {@code w[wBase, wBase + 2 * taps.length)}, the outputs at {@code cur[outBase + i]} and
     * {@code resid[outBase + i]} for {@code i < nEmbd}.
     */
    public static void mixRow(float[] bank, int bankOff, int bankStride, int[] taps, int t, int nEmbd, float[] w,
                              int wBase, float[] cur, float[] resid, int outBase) {
        int nTaps = taps.length;
        int rowOff = t * nEmbd;
        for (int i = 0; i < nEmbd; i++) {
            float c = 0f;
            float r = 0f;
            for (int j = 0; j < nTaps; j++) {
                float h = bank[bankOff + taps[j] * bankStride + rowOff + i];
                c += w[wBase + j] * h;
                r += w[wBase + nTaps + j] * h;
            }
            cur[outBase + i] = c;
            resid[outBase + i] = r;
        }
    }
}
