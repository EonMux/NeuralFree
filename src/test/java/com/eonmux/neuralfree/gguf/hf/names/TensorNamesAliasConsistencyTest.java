package com.eonmux.neuralfree.gguf.hf.names;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import org.junit.Test;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The Hugging Face aliases {@link TensorNames} keeps for GGUF files that store Hugging Face names agree with the
 * converter's tensor name map: a model-level alias maps to its GGUF name under every architecture that maps it at all,
 * and a layer alias, which describes the Llama layout, maps to its GGUF name under the Llama-layout reference
 * architectures that list the tensor.
 */
public class TensorNamesAliasConsistencyTest {

    /** Llama-layout architectures: LLAMA has no q/k norms, QWEN3 adds them. */
    private static final List<Arch> LAYOUT_REFERENCES = Arrays.asList(Arch.LLAMA, Arch.QWEN3);

    private static final String[] SUFFIXES = {"." + TensorNames.WEIGHT, "." + TensorNames.BIAS};

    private static String[][] table(String field) throws ReflectiveOperationException {
        Field f = TensorNames.class.getDeclaredField(field);
        f.setAccessible(true);
        return (String[][]) f.get(null);
    }

    private static List<HfTensorNameMap> allMaps() {
        List<HfTensorNameMap> out = new ArrayList<>();
        for (Arch a : Arch.values()) {
            if (a != Arch.UNKNOWN && !HfNameTables.archTensors(a.ggufName()).isEmpty()) {
                out.add(HfTensorNameMap.forArch(a, 1));
            }
        }
        return out;
    }

    @Test
    public void modelAliasesMapToTheirNameUnderEveryArchitecture() throws ReflectiveOperationException {
        String[][] rows = table("MODEL_ALIASES");
        assertTrue(rows.length > 0);
        List<HfTensorNameMap> maps = allMaps();
        for (String[] row : rows) {
            String gguf = row[0];
            assertEquals(Arrays.asList(row).subList(1, row.length), TensorNames.modelAliases(gguf));
            for (String alias : TensorNames.modelAliases(gguf)) {
                int mapped = 0;
                for (HfTensorNameMap m : maps) {
                    Optional<HfTensorNameMap.Mapped> r = m.map(alias, SUFFIXES);
                    if (r.isPresent()) {
                        assertEquals(alias + " under " + m.arch(), gguf, r.get().ggufName());
                        mapped++;
                    }
                }
                assertTrue(alias + " is mapped by no architecture", mapped > 0);
            }
        }
    }

    @Test
    public void layerAliasesMapToTheirNameUnderTheLlamaLayout() throws ReflectiveOperationException {
        String[][] bases = table("LAYER_ALIAS_BASES");
        assertTrue(bases.length > 0);
        for (String[] base : bases) {
            for (String suffix : SUFFIXES) {
                String gguf = base[0] + suffix;
                List<String> aliases = TensorNames.layerAliases(gguf);
                assertEquals(gguf, 1, aliases.size());
                assertEquals(base[1] + suffix, aliases.get(0));
                String hf = TensorNames.HF_LAYER_PREFIX + ".0." + aliases.get(0);
                int checked = 0;
                for (Arch a : LAYOUT_REFERENCES) {
                    Optional<HfTensorNameMap.Mapped> r = HfTensorNameMap.forArch(a, 1).map(hf, SUFFIXES);
                    if (r.isPresent()) {
                        assertEquals(hf + " under " + a, TensorNames.layerName(0, gguf), r.get().ggufName());
                        checked++;
                    }
                }
                assertTrue(hf + " is mapped by no Llama-layout reference architecture", checked > 0);
            }
        }
    }

    @Test
    public void namesWithoutAliasesHaveNone() {
        assertTrue(TensorNames.layerAliases("ssm_conv1d.weight").isEmpty());
        assertTrue(TensorNames.modelAliases("rope_freqs.weight").isEmpty());
        assertFalse(TensorNames.modelAliases(TensorNames.weight(TensorNames.OUTPUT)).isEmpty());
    }
}
