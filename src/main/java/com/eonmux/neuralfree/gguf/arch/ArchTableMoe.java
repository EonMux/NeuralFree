package com.eonmux.neuralfree.gguf.arch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Architecture table for the MoE, MLA and DSA architectures: 49 generative decoders split into
 * {@link ArchTableMoeClassic} (MoE with plain multi-head attention), {@link ArchTableMoeMla} (latent and
 * sparse attention) and {@link ArchTableMoeSwa} (sliding-window MoE). {@link ArchRegistry} merges it with the
 * other tables; an architecture appears in exactly one table.
 */
public final class ArchTableMoe {

    private ArchTableMoe() {
    }

    /** The table rows. */
    public static List<ArchEntry> entries() {
        List<ArchEntry> out = new ArrayList<>();
        out.addAll(ArchTableMoeClassic.entries());
        out.addAll(ArchTableMoeMla.entries());
        out.addAll(ArchTableMoeSwa.entries());
        return Collections.unmodifiableList(out);
    }

    /** Generative decoder with a standard KV cache and the given RoPE type. */
    static ArchTraits.Builder decoder(Arch arch, RopeType rope) {
        return decoder(arch, rope, MemoryKind.KV);
    }

    /** Generative decoder with the given RoPE type and memory kind. */
    static ArchTraits.Builder decoder(Arch arch, RopeType rope, MemoryKind memory) {
        return ArchTraits.builder(arch).ropeType(rope).memoryKind(memory).support(SupportClass.GEN);
    }
}
