package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerRegistry;
import org.junit.Test;

import java.util.EnumSet;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The counts the root guides state, pinned to the registries they describe: the architecture table split, the
 * native GPU and CPU weight types, the NF extension ids, the KV cache types and the BPE pre-tokenizer names.
 */
public class DocumentedFactsTest {

    @Test
    public void architectureCountsMatchTheGuides() {
        assertEquals(192, Arch.values().length);
        int generative = 0;
        int refused = 0;
        for (Arch arch : Arch.values()) {
            SupportClass support = ArchRegistry.traits(arch).support();
            if (support == SupportClass.GEN) {
                generative++;
            } else if (support == SupportClass.NA) {
                refused++;
            }
        }
        assertEquals("generative architectures", 160, generative);
        assertEquals("refused architectures", 2, refused);
        assertEquals(SupportClass.NA, ArchRegistry.traits(Arch.CLIP).support());
        assertEquals(SupportClass.NA, ArchRegistry.traits(Arch.UNKNOWN).support());
        assertTrue("architectures without a table entry: " + ArchRegistry.missing(),
                ArchRegistry.missing().isEmpty());
    }

    @Test
    public void weightTypeCountsMatchTheGuides() {
        Set<GGMLType> gpu = EnumSet.noneOf(GGMLType.class);
        Set<GGMLType> cpu = EnumSet.noneOf(GGMLType.class);
        int extensionTypes = 0;
        for (GGMLType type : GGMLType.values()) {
            if (type.isDeviceOnly()) {
                continue;
            }
            if (GpuOps.supportsWeightType(type)) {
                gpu.add(type);
            }
            if (CpuMatVec.supports(type)) {
                cpu.add(type);
            }
            if (type.getId() >= GGMLType.EXTENSION_ID_MIN && type.getId() <= GGMLType.EXTENSION_ID_MAX) {
                extensionTypes++;
            }
        }
        assertEquals("native GPU weight types", 42, gpu.size());
        assertEquals("native CPU weight types", 41, cpu.size());
        assertEquals("the CPU engine runs the GPU set except Q8_K",
                EnumSet.of(GGMLType.Q8_K), difference(gpu, cpu));
        assertTrue("the CPU engine runs no type the GPU set lacks", difference(cpu, gpu).isEmpty());
        assertEquals("NF extension weight types", 10, extensionTypes);
    }

    @Test
    public void cacheAndTokenizerCountsMatchTheGuides() {
        assertEquals("KV cache types", 23, KvDType.values().length);
        assertEquals("tokenizer.ggml.pre names", 98, PreTokenizerRegistry.names().size());
    }

    private static Set<GGMLType> difference(Set<GGMLType> a, Set<GGMLType> b) {
        Set<GGMLType> out = EnumSet.copyOf(a);
        out.removeAll(b);
        return out;
    }
}
