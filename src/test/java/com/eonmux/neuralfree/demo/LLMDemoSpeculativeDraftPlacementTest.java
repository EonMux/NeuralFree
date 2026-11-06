package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The draft model's placement and thread flags of {@link LLMDemo} ({@link SpeculativeFlags}) in llama.cpp's
 * spellings reach the engine configuration: the draft devices ({@code -devd}, {@code none} alone keeps the draft
 * on the host), the draft layer count ({@code -ngld}), the draft tensor overrides ({@code -otd}), the draft expert
 * placement ({@code -cmoed}, {@code -ncmoed}) and the draft thread flags ({@code -td}, {@code -tbd}), which size
 * the one CPU thread pool and must agree with each other and with {@code -t}. Invalid values are usage errors
 * naming their flag, and the placement settings need a draft model.
 */
public class LLMDemoSpeculativeDraftPlacementTest {

    private static final String[] SPECULATIVE_PROPS = {
            EngineConfig.PROP_DRAFT,
            EngineConfig.PROP_DRAFT_GPU_DEVICES,
            EngineConfig.PROP_DRAFT_N_GPU_LAYERS,
            EngineConfig.PROP_DRAFT_TENSOR_OVERRIDES,
            EngineConfig.PROP_DRAFT_CPU_MOE,
            EngineConfig.PROP_DRAFT_N_CPU_MOE,
            CpuParallel.THREADS_PROPERTY,
    };

    private final Map<String, String> saved = new HashMap<>();

    @Before
    public void clearSpeculativeProperties() {
        for (String key : SPECULATIVE_PROPS) {
            saved.put(key, System.getProperty(key));
            System.clearProperty(key);
        }
    }

    @After
    public void restoreSpeculativeProperties() {
        for (Map.Entry<String, String> e : saved.entrySet()) {
            if (e.getValue() == null) {
                System.clearProperty(e.getKey());
            } else {
                System.setProperty(e.getKey(), e.getValue());
            }
        }
    }

    private static EngineConfig engine(String... args) throws UsageException {
        return LLMDemoOptions.parse(args).engine;
    }

    private static void assertRefused(String flag, String... args) {
        UsageException e = assertThrows(UsageException.class, () -> engine(args));
        assertTrue(e.getMessage(), e.getMessage().startsWith(flag));
    }

    @Test
    public void theDraftDeviceListReachesTheDraftDevices() throws UsageException {
        assertEquals(List.of("0", "A100"), engine("--draft", "d.gguf", "-devd", "0, A100").draftDevices());
        assertEquals(List.of("1"), engine("--draft", "d.gguf", "--spec-draft-device", "1").draftDevices());
        assertEquals(List.of("A100"), engine("--draft", "d.gguf", "--device-draft", "A100").draftDevices());
    }

    @Test
    public void noneAloneKeepsTheDraftOnTheHost() throws UsageException {
        EngineConfig c = engine("--draft", "d.gguf", "-devd", "none");
        assertEquals(List.of(EngineConfig.DRAFT_NO_DEVICE), c.draftDevices());
        assertTrue(c.hasDraftPlacement());
    }

    @Test
    public void unusableDraftDeviceListsAreRefusedNamingTheFlag() {
        assertRefused("-devd", "--draft", "d.gguf", "-devd", "none,0");
        assertRefused("-devd", "--draft", "d.gguf", "-devd", "0,,1");
        assertRefused("--device-draft", "--draft", "d.gguf", "--device-draft", "none,0");
    }

    @Test
    public void theDraftLayerCountTakesLlamaCppsValues() throws UsageException {
        assertEquals(4, engine("--draft", "d.gguf", "-ngld", "4").draftGpuLayers());
        assertEquals(EngineConfig.GPU_LAYERS_AUTO, engine("--draft", "d.gguf", "-ngld", "auto").draftGpuLayers());
        assertEquals(EngineConfig.GPU_LAYERS_AUTO, engine("--draft", "d.gguf", "--spec-draft-ngl", "-1")
                .draftGpuLayers());
        assertEquals(EngineConfig.GPU_LAYERS_ALL, engine("--draft", "d.gguf", "--gpu-layers-draft", "all")
                .draftGpuLayers());
        assertRefused("--n-gpu-layers-draft", "--draft", "d.gguf", "--n-gpu-layers-draft", "many");
        assertRefused("-ngld", "--draft", "d.gguf", "-ngld", "-3");
    }

    @Test
    public void theDraftTensorOverridesReachTheDraftOverrides() throws UsageException {
        EngineConfig c = engine("--draft", "d.gguf", "-otd", "exps=cpu,blk\\.0\\..*=gpu");
        assertEquals(2, c.draftTensorOverrides().size());
        assertTrue(c.draftTensorOverrides().get(0).host());
        assertFalse(c.draftTensorOverrides().get(1).host());
        assertRefused("--override-tensor-draft", "--draft", "d.gguf", "--override-tensor-draft", "exps");
        assertRefused("-otd", "--draft", "d.gguf", "-otd", "exps=vram");
    }

    @Test
    public void theDraftExpertPlacementReachesTheDraftConfig() throws UsageException {
        EngineConfig c = engine("--draft", "d.gguf", "-cmoed", "-ncmoed", "2");
        assertTrue(c.draftCpuMoe());
        assertEquals(2, c.draftNCpuMoe());
        assertRefused("--n-cpu-moe-draft", "--draft", "d.gguf", "--n-cpu-moe-draft", "-1");
        assertRefused("-ncmoed", "--draft", "d.gguf", "-ncmoed", "x");
    }

    @Test
    public void thePlacementSettingsNeedADraftModel() {
        for (String[] flagAndValue : new String[][] {
                {"-ngld", "4"}, {"-otd", "exps=cpu"}, {"-cmoed", null}, {"-ncmoed", "1"},
                {"-devd", "none"}, {"-devd", "0"}}) {
            String flag = flagAndValue[0];
            UsageException e = flagAndValue[1] == null
                    ? assertThrows(flag + " without a draft model is refused",
                            UsageException.class, () -> engine(flag))
                    : assertThrows(flag + " without a draft model is refused", UsageException.class,
                            () -> engine(flag, flagAndValue[1]));
            assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_DRAFT));
        }
    }

    @Test
    public void theDraftThreadFlagsNeedNoDraftModel() throws UsageException {
        engine("-td", "4");
        assertEquals("the thread flags size the pool without a draft model, as -t does",
                "4", System.getProperty(CpuParallel.THREADS_PROPERTY));
    }

    @Test
    public void theDraftThreadFlagsSizeTheCpuThreadPool() throws UsageException {
        engine("--draft", "d.gguf", "-td", "4");
        assertEquals("4", System.getProperty(CpuParallel.THREADS_PROPERTY));
        System.clearProperty(CpuParallel.THREADS_PROPERTY);
        engine("--draft", "d.gguf", "--threads-batch-draft", "5");
        assertEquals("5", System.getProperty(CpuParallel.THREADS_PROPERTY));
        System.clearProperty(CpuParallel.THREADS_PROPERTY);
        engine("--draft", "d.gguf", "--threads-draft", "7");
        assertEquals("the batch threads follow the generation threads when unset", "7",
                System.getProperty(CpuParallel.THREADS_PROPERTY));
    }

    @Test
    public void disagreeingDraftThreadFlagsAreRefused() {
        assertRefused("-td", "--draft", "d.gguf", "-td", "4", "-tbd", "8");
        assertRefused("-td", "--draft", "d.gguf", "--threads-batch-draft", "8", "-td", "4");
    }

    @Test
    public void agreeingDraftThreadFlagsAreTaken() throws UsageException {
        engine("--draft", "d.gguf", "-td", "4", "-tbd", "4");
        assertEquals("4", System.getProperty(CpuParallel.THREADS_PROPERTY));
        System.clearProperty(CpuParallel.THREADS_PROPERTY);
        engine("--draft", "d.gguf", "-td", "3", "--spec-draft-threads-batch", "3");
        assertEquals("3", System.getProperty(CpuParallel.THREADS_PROPERTY));
    }

    @Test
    public void aDraftThreadFlagConflictsWithAnExplicitThreadsFlag() throws UsageException {
        assertRefused("-td", "-t", "6", "--draft", "d.gguf", "-td", "4");
        System.clearProperty(CpuParallel.THREADS_PROPERTY);
        assertRefused("-tbd", "-t", "6", "--draft", "d.gguf", "-tbd", "8");
        System.clearProperty(CpuParallel.THREADS_PROPERTY);
        engine("-t", "6", "--draft", "d.gguf", "-td", "6");
        assertEquals("6", System.getProperty(CpuParallel.THREADS_PROPERTY));
    }

    @Test
    public void anAmbientThreadsPropertyKeepsUntilAFlagDisagrees() throws UsageException {
        System.setProperty(CpuParallel.THREADS_PROPERTY, "9");
        engine("--draft", "d.gguf", "-td", "4");
        assertEquals("a flag overrides the ambient property as -t does", "4",
                System.getProperty(CpuParallel.THREADS_PROPERTY));
        System.setProperty(CpuParallel.THREADS_PROPERTY, "9");
        assertRefused("-td", "-t", "6", "--draft", "d.gguf", "-td", "4");
        System.setProperty(CpuParallel.THREADS_PROPERTY, "9");
        engine("--draft", "d.gguf", "-td", "6");
        assertEquals("a flag overrides the ambient property", "6", System.getProperty(CpuParallel.THREADS_PROPERTY));
    }
}
