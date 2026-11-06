package com.eonmux.neuralfree.acceleration.hip;

import java.util.Locale;
import java.util.Objects;

/**
 * One AMD processor target of the HIP backend, parsed from its GCN architecture name
 * ({@code gcnArchName}: {@code gfx1100}, {@code gfx90a:xnack-}), with the instruction-set
 * availability llama.cpp's CUDA kernels key their per-family tuning on ({@code common.cuh}).
 *
 * <p>The families follow llama.cpp's compute-capability macros: a target encodes as
 * {@code 0x1000000 | gfx} (gfx942 &rarr; 0x1000942, gfx1100 &rarr; 0x1001100), so the AMD targets
 * share one number line with NVIDIA's {@code major * 10 + minor} compute capabilities. The
 * instruction sets decide what the kernels run:</p>
 * <ul>
 *   <li><b>CDNA</b> (gfx908 MI100 and newer, plus gfx1250, which vLLM classifies as CDNA):
 *       MFMA matrix instructions (fp16, bf16 and fp32), fp16 from CDNA1 on.</li>
 *   <li><b>RDNA3 and newer</b> (gfx1100+, including RDNA3.5 gfx115x, RDNA4-M gfx117x and RDNA4
 *       gfx120x): WMMA matrix instructions (fp16 and bf16).</li>
 *   <li><b>RDNA2</b> (gfx1030) and CDNA (gfx906 Vega20 and newer): packed {@code dp4a} integer
 *       dot products.</li>
 *   <li><b>GCN</b> (everything older, including the gfx909/gfx90c APUs): scalar paths only.</li>
 * </ul>
 *
 * <p>Instances are immutable; {@link #parse(String)} memoizes nothing, so callers hold one per
 * device.</p>
 */
public final class HipGfxTarget {

    /**
     * The processor family of a gfx target, the AMD range of llama.cpp's
     * {@code GGML_CUDA_CC_IS_*} macros.
     */
    public enum Family {
        /** Pre-Vega GCN and the GCN APUs (gfx700&ndash;gfx90c, gfx909): scalar datapaths. */
        GCN,
        /** Vega 20 (gfx906): GCN datapaths plus {@code dp4a}. */
        GCN_DP4A,
        /** MI100 and newer Instinct accelerators (gfx908, gfx90a, gfx942, gfx950, gfx1250): MFMA. */
        CDNA,
        /** RX 5000 (gfx1010&ndash;gfx101x). */
        RDNA1,
        /** RX 6000 (gfx1030&ndash;gfx103x): {@code dp4a}. */
        RDNA2,
        /** RX 7000 (gfx1100&ndash;gfx114x): WMMA. */
        RDNA3,
        /** Strix Point and Halo APUs (gfx115x): WMMA. */
        RDNA3_5,
        /** gfx117x: WMMA. */
        RDNA4_M,
        /** RX 9000 (gfx120x and newer): WMMA. */
        RDNA4
    }

    /** llama.cpp {@code GGML_CUDA_CC_OFFSET_AMD}: the AMD range of the compute-capability line. */
    public static final int AMD_CC_OFFSET = 0x1000000;
    /** PCI vendor id of AMD, the only vendor HIP devices come from. */
    public static final int AMD_VENDOR_ID = 0x1002;

    private static final int GFX_906 = 0x906;
    private static final int GFX_908 = 0x908;
    private static final int GFX_909 = 0x909;
    private static final int GFX_90C = 0x90c;
    private static final int GFX_950 = 0x950;
    private static final int GFX_1250 = 0x1250;
    private static final int GFX_RDNA1_MIN = 0x1010;
    private static final int GFX_RDNA2_MIN = 0x1030;
    private static final int GFX_RDNA3_MIN = 0x1100;
    private static final int GFX_RDNA3_5_MIN = 0x1150;
    private static final int GFX_RDNA4_M_MIN = 0x1170;
    private static final int GFX_RDNA4_MIN = 0x1200;

    private final String gcnArchName;
    private final int gfxId;
    private final Family family;

    private HipGfxTarget(String gcnArchName, int gfxId, Family family) {
        this.gcnArchName = gcnArchName;
        this.gfxId = gfxId;
        this.family = family;
    }

    /**
     * Parses a GCN architecture name, ignoring case and any capability suffix
     * ({@code gfx90a:xnack-}, {@code gfx942:sramecc+:xnack+}).
     *
     * @param gcnArchName the driver's processor name, for example {@code gfx1100}
     * @throws IllegalArgumentException when the name is null or not a {@code gfx} string of digits
     */
    public static HipGfxTarget parse(String gcnArchName) {
        if (gcnArchName == null) {
            throw new IllegalArgumentException("gcnArchName must not be null");
        }
        String name = gcnArchName.trim().toLowerCase(Locale.ROOT);
        int colon = name.indexOf(':');
        String base = colon >= 0 ? name.substring(0, colon) : name;
        if (!base.startsWith("gfx") || base.length() <= 3) {
            throw new IllegalArgumentException("gcnArchName '" + gcnArchName + "' is not a gfx processor name"
                    + " (expected gfx plus digits, for example gfx1100)");
        }
        int gfxId;
        try {
            gfxId = Integer.parseInt(base.substring(3), 16);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("gcnArchName '" + gcnArchName + "' is not a gfx processor name"
                    + " (expected gfx plus hex digits)", e);
        }
        return new HipGfxTarget(base, gfxId, familyOf(gfxId));
    }

    /**
     * The family of the AMD compute capability {@code 0x1000000 | gfx} of a HIP device
     * ({@link #amdComputeCapability()} of its target), the form
     * {@link com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities} exposes.
     *
     * @param computeCapability an AMD compute capability, for example 0x1001100 for gfx1100
     * @throws IllegalArgumentException when the value does not carry the AMD range offset
     */
    public static Family familyOfComputeCapability(int computeCapability) {
        if ((computeCapability & AMD_CC_OFFSET) == 0
                || (computeCapability & ~(AMD_CC_OFFSET | 0xFFFFFF)) != 0) {
            throw new IllegalArgumentException("AMD compute capability " + Integer.toHexString(computeCapability)
                    + " is not the 0x" + Integer.toHexString(AMD_CC_OFFSET) + " range offset plus a gfx id");
        }
        return familyOf(computeCapability & ~AMD_CC_OFFSET);
    }

    /** The family of a gfx id, the llama.cpp {@code GGML_CUDA_CC_IS_*} ranges. */
    private static Family familyOf(int gfxId) {
        if (gfxId == GFX_1250) {
            return Family.CDNA;
        }
        if (gfxId >= GFX_RDNA4_MIN) {
            return Family.RDNA4;
        }
        if (gfxId >= GFX_RDNA4_M_MIN) {
            return Family.RDNA4_M;
        }
        if (gfxId >= GFX_RDNA3_5_MIN) {
            return Family.RDNA3_5;
        }
        if (gfxId >= GFX_RDNA3_MIN) {
            return Family.RDNA3;
        }
        if (gfxId >= GFX_RDNA2_MIN) {
            return Family.RDNA2;
        }
        if (gfxId >= GFX_RDNA1_MIN) {
            return Family.RDNA1;
        }
        // The GCN APUs gfx909/gfx90c sit numerically inside the CDNA range; llama.cpp's
        // GGML_CUDA_CC_IS_GCN_APU carves them out, and so does this classification.
        if (gfxId == GFX_909 || gfxId == GFX_90C) {
            return Family.GCN;
        }
        if (gfxId >= GFX_908 && gfxId <= GFX_950) {
            return Family.CDNA;
        }
        if (gfxId >= GFX_906) {
            return Family.GCN_DP4A;
        }
        return Family.GCN;
    }

    /** The normalized processor name without capability suffix, for example {@code gfx90a}. */
    public String gcnArchName() {
        return gcnArchName;
    }

    /** The gfx id of the name, for example 0x90a for gfx90a. */
    public int gfxId() {
        return gfxId;
    }

    /** The processor family. */
    public Family family() {
        return family;
    }

    /**
     * The target on llama.cpp's AMD compute-capability line: {@code 0x1000000 | gfx} (gfx1100
     * &rarr; 0x1001100), comparable with NVIDIA's {@code major * 10 + minor} values.
     */
    public int amdComputeCapability() {
        return AMD_CC_OFFSET | gfxId;
    }

    /** True for every Instinct accelerator family (CDNA), whose MMQ kernels run the MFMA form. */
    public boolean isCdna() {
        return family == Family.CDNA;
    }

    /** True for RDNA3, RDNA3.5, RDNA4-M and RDNA4, whose kernels run the WMMA form. */
    public boolean isRdna3OrNewer() {
        return family == Family.RDNA3 || family == Family.RDNA3_5 || family == Family.RDNA4_M
                || family == Family.RDNA4;
    }

    /** {@code amd_mfma_available}: the fp16/bf16/fp32 MFMA matrix instructions (CDNA). */
    public boolean mfmaAvailable() {
        return isCdna();
    }

    /** {@code amd_wmma_available}: the fp16/bf16 WMMA matrix instructions (RDNA3 and newer). */
    public boolean wmmaAvailable() {
        return isRdna3OrNewer();
    }

    /** True when a matrix instruction set exists at all: MFMA on CDNA or WMMA on RDNA3+. */
    public boolean matrixInstructionsAvailable() {
        return mfmaAvailable() || wmmaAvailable();
    }

    /** {@code dp4a_available}: packed integer dot products (Vega 20, RDNA2 and everything newer). */
    public boolean dp4aAvailable() {
        return gfxId >= GFX_906;
    }

    /** {@code bf16_mma_hardware_available}: bfloat16 matrix arithmetic (CDNA, RDNA3 and newer). */
    public boolean bf16MmaAvailable() {
        return isCdna() || isRdna3OrNewer();
    }

    /** True when the kernels may run 16-bit float arithmetic: every HIP target since GCN4. */
    public boolean float16Available() {
        return true;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof HipGfxTarget)) {
            return false;
        }
        HipGfxTarget that = (HipGfxTarget) o;
        return gfxId == that.gfxId && gcnArchName.equals(that.gcnArchName);
    }

    @Override
    public int hashCode() {
        return Objects.hash(gcnArchName, gfxId);
    }

    @Override
    public String toString() {
        return gcnArchName + " (" + family + ")";
    }
}
