package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * beam's depth-scaled residual {@code beam_residual_add(h, r, i) = h / sqrt(i) + r * sqrt((i - 1) / i)} with
 * {@code i = 2 (il + 1)} for the attention add and {@code i + 1} for the FFN add (models/beam.cpp:109-113, 203-204,
 * 237), the {@link ResidualSpec} contract that carries it, and the glm5-next NextN blocks' plain residual
 * (glm5-next.cpp:85-92, 626, 656).
 */
public class ResidualResolverTest {

    @Test
    public void beamScalesFollowTheResidualIndex() {
        assertEquals((float) (1.0 / Math.sqrt(2.0)), ResidualResolver.beamBranchScale(2), 0f);
        assertEquals((float) Math.sqrt(0.5), ResidualResolver.beamStreamScale(2), 0f);
        assertEquals((float) (1.0 / Math.sqrt(3.0)), ResidualResolver.beamBranchScale(3), 0f);
        assertEquals((float) Math.sqrt(2.0 / 3.0), ResidualResolver.beamStreamScale(3), 0f);
    }

    @Test
    public void beamRowCarriesTheDepthScaledTrait() {
        assertEquals(ResidualKind.DEPTH_SCALED, ArchRegistry.traits(Arch.BEAM).residualKind());
        assertEquals(ResidualKind.SCALED, ArchRegistry.traits(Arch.MINICPM3).residualKind());
    }

    @Test
    public void depthScaledSpecSeparatesMixerAndFfnScales() {
        ResidualSpec r = ResidualSpec.builder(ResidualKind.DEPTH_SCALED, 64)
                .depthScales(0.5f, 0.75f, 0.25f, 0.875f)
                .build();
        assertEquals(0.5f, r.branchScale(false), 0f);
        assertEquals(0.75f, r.streamScale(false), 0f);
        assertEquals(0.25f, r.branchScale(true), 0f);
        assertEquals(0.875f, r.streamScale(true), 0f);
    }

    @Test
    public void depthScalesBelongToTheDepthScaledKind() {
        assertThrows(IllegalArgumentException.class, () -> ResidualSpec.builder(ResidualKind.STANDARD, 64)
                .depthScales(0.5f, 0.75f, 0.25f, 0.875f).build());
        assertThrows(IllegalArgumentException.class, () -> ResidualSpec.builder(ResidualKind.DEPTH_SCALED, 64)
                .depthScales(0f, 1f, 1f, 1f).build());
    }

    @Test
    public void glm5NextNextnBlocksAddBranchesWithoutHyperConnections() {
        int nEmbd = 8;
        int hc = 4;
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.GLM5_NEXT)).layerCounts(3, 1)
                .nEmbd(nEmbd).normRmsEps(1e-6f).dsv4HcMult(hc).build();
        ResidualResolver resolver = new ResidualResolver(hp,
                TensorCatalog.of(Arch.GLM5_NEXT.ggufName(), Collections.emptyList()));
        for (int il = 0; il < hp.nLayer(); il++) {
            ResidualSpec trunk = resolver.resolve(il);
            assertEquals(ResidualKind.HYPER_CONNECTIONS, trunk.kind());
            assertEquals(hc, trunk.hcMult());
        }
        ResidualSpec nextn = resolver.resolve(hp.nLayer());
        assertEquals(ResidualKind.STANDARD, nextn.kind());
        assertEquals(nEmbd, nextn.streamWidth());
    }
}
