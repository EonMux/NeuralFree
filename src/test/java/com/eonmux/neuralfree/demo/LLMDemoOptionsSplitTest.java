package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The multi-GPU flags of {@link LLMDemoOptions} in llama.cpp's spellings ({@link SplitFlags}, parsed through
 * {@link EngineLoadFlags}): {@code --split-mode} /
 * {@code -sm}, {@code --tensor-split} / {@code -ts}, {@code --device} / {@code -dev} and {@code --main-gpu} /
 * {@code -mg} set the engine fields of the split properties, a device list implies the layer split, vLLM's
 * context-parallel sizes ({@code -dcp}, {@code -pcp}, {@code --dcp-kv-cache-interleave-size}) run the context
 * split, and every combination that cannot run is refused naming the flag.
 */
public class LLMDemoOptionsSplitTest {

    private static final String[] PROPS = {EngineConfig.PROP_SPLIT_MODE, EngineConfig.PROP_GPU_DEVICES,
            EngineConfig.PROP_TENSOR_SPLIT, EngineConfig.PROP_MAIN_GPU, EngineConfig.PROP_BACKEND,
            EngineConfig.PROP_DEVICE, MemoryConfig.PROP_PARALLEL, MemoryConfig.PROP_KV_PAGED,
            MemoryConfig.PROP_KV_PAGE_SIZE};

    private final Map<String, String> saved = new HashMap<>();

    @Before
    public void clearProperties() {
        for (String key : PROPS) {
            saved.put(key, System.getProperty(key));
            System.clearProperty(key);
        }
    }

    @After
    public void restoreProperties() {
        for (String key : PROPS) {
            String v = saved.get(key);
            if (v == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, v);
            }
        }
    }

    private static EngineConfig engine(String... args) throws UsageException {
        return LLMDemoOptions.parse(args).engine;
    }

    private static String refusal(String... args) {
        return assertThrows(UsageException.class, () -> LLMDemoOptions.parse(args)).getMessage();
    }

    @Test
    public void llamaSpellingsSetTheSplitFields() throws UsageException {
        EngineConfig cfg = engine("m.gguf", "--split-mode", "layer", "--device", "GeForce,Intel",
                "--tensor-split", "3,1", "--main-gpu", "1");
        assertEquals(EngineConfig.SplitMode.LAYER, cfg.splitMode());
        assertEquals(Arrays.asList("GeForce", "Intel"), cfg.gpuDevices());
        assertArrayEquals(new double[]{3, 1}, cfg.tensorSplit(), 0.0);
        assertEquals(1, cfg.mainGpu());
        assertNull("a device list selects no single GPU", cfg.deviceFilter());
    }

    @Test
    public void shortAliasesMatchTheLongFlags() throws UsageException {
        EngineConfig cfg = engine("m.gguf", "-sm", "layer", "-dev", "0, 1", "-ts", "1/2", "-mg", "0");
        assertEquals(EngineConfig.SplitMode.LAYER, cfg.splitMode());
        assertEquals(Arrays.asList("0", "1"), cfg.gpuDevices());
        assertArrayEquals("llama.cpp splits -ts on ',' and '/'", new double[]{1, 2}, cfg.tensorSplit(), 0.0);
        assertEquals(0, cfg.mainGpu());
    }

    @Test
    public void aDeviceListSplitsByLayerWithoutASplitMode() throws UsageException {
        EngineConfig cfg = engine("m.gguf", "-dev", "0,1");
        assertTrue("llama.cpp's default split mode is layer", cfg.split());
        assertEquals(Arrays.asList("0", "1"), cfg.gpuDevices());
    }

    @Test
    public void oneDeviceSelectsTheGpuOfAOneDeviceLoad() throws UsageException {
        EngineConfig cfg = engine("m.gguf", "--device", "GeForce");
        assertEquals(EngineConfig.SplitMode.NONE, cfg.splitMode());
        assertEquals("GeForce", cfg.deviceFilter());
        assertEquals(Arrays.asList("GeForce"), cfg.gpuDevices());
        assertEquals("the -device alias stays", "Intel", engine("m.gguf", "-device", "Intel").deviceFilter());
    }

    @Test
    public void deviceNoneRunsTheCpuEngine() throws UsageException {
        EngineConfig cfg = engine("m.gguf", "-dev", "none");
        assertEquals(EngineConfig.BackendMode.CPU, cfg.backend());
        assertTrue(cfg.gpuDevices().isEmpty());
        String mixed = refusal("m.gguf", "-dev", "none,0");
        assertTrue(mixed, mixed.startsWith("-dev none,0"));
        assertTrue(mixed, mixed.contains("cannot be listed with devices"));
        String layer = refusal("m.gguf", "-dev", "none", "-sm", "layer");
        assertTrue(layer, layer.contains(EngineConfig.PROP_SPLIT_MODE + "=layer"));
        assertTrue(layer, layer.contains(EngineConfig.PROP_BACKEND + "=cpu"));
    }

    @Test
    public void flagsOverrideTheProperties() throws UsageException {
        System.setProperty(EngineConfig.PROP_SPLIT_MODE, "layer");
        System.setProperty(EngineConfig.PROP_TENSOR_SPLIT, "1,1");
        EngineConfig cfg = engine("m.gguf", "-ts", "2,1");
        assertTrue(cfg.split());
        assertArrayEquals(new double[]{2, 1}, cfg.tensorSplit(), 0.0);
        assertEquals(EngineConfig.SplitMode.NONE, engine("m.gguf", "-sm", "none").splitMode());
    }

    @Test
    public void rowTensorAndContextModesParseAndAnUnknownModeIsRefusedByName() throws UsageException {
        assertEquals(EngineConfig.SplitMode.ROW, engine("m.gguf", "-sm", "row").splitMode());
        assertEquals(EngineConfig.SplitMode.TENSOR, engine("m.gguf", "-sm", "tensor").splitMode());
        assertEquals(EngineConfig.SplitMode.CONTEXT, engine("m.gguf", "-sm", "context").splitMode());
        String msg = refusal("m.gguf", "--split-mode", "pipeline");
        assertTrue(msg, msg.startsWith("--split-mode pipeline"));
        assertTrue(msg, msg.contains("none, layer, row, tensor or context"));
    }

    @Test
    public void badSharesAreRefusedNamingTheFlag() {
        assertTrue(refusal("m.gguf", "-ts", "3,0").contains("share of 0"));
        String negative = refusal("m.gguf", "-ts", "3,-1");
        assertTrue(negative, negative.startsWith("-ts 3,-1"));
        assertTrue(refusal("m.gguf", "--tensor-split", "3,x").contains("'x' is not a number"));
        assertTrue(refusal("m.gguf", "--tensor-split", "NaN").contains("finite value above 0"));
    }

    @Test
    public void aTensorSplitWithoutALayerSplitIsRefused() {
        assertTrue(refusal("m.gguf", "-ts", "3,1").contains("--tensor-split"));
        assertTrue(refusal("m.gguf", "-sm", "none", "-dev", "0,1", "-ts", "1,1").contains("--tensor-split"));
    }

    @Test
    public void theMainGpuPicksTheOneGpuOfSplitModeNone() throws UsageException {
        EngineConfig listed = engine("m.gguf", "-sm", "none", "-dev", "GeForce,Intel", "-mg", "1");
        assertEquals(EngineConfig.SplitMode.NONE, listed.splitMode());
        assertTrue(listed.picksMainGpu());
        assertEquals(Arrays.asList("GeForce", "Intel"), listed.gpuDevices());
        assertEquals(1, listed.mainGpu());

        EngineConfig firstOfList = engine("m.gguf", "-sm", "none", "-dev", "0,1");
        assertTrue(firstOfList.picksMainGpu());
        assertEquals(0, firstOfList.mainGpu());

        EngineConfig enumerated = engine("m.gguf", "-mg", "1");
        assertEquals(EngineConfig.SplitMode.NONE, enumerated.splitMode());
        assertTrue(enumerated.picksMainGpu());

        EngineConfig index = engine("m.gguf", "-dev", "1");
        assertTrue("a single device index picks its GPU in enumeration order", index.picksMainGpu());
        assertNull("an index is no device-name substring", index.deviceFilter());
        assertEquals(Arrays.asList("1"), index.gpuDevices());

        assertTrue(!engine("m.gguf", "-dev", "GeForce").picksMainGpu());
        assertTrue(!engine("m.gguf").picksMainGpu());
        assertTrue(!engine("m.gguf", "-dev", "0,1", "-mg", "1").picksMainGpu());

        String beyond = refusal("m.gguf", "-sm", "none", "-dev", "0,1", "-mg", "2");
        assertTrue(beyond, beyond.contains("--main-gpu 2 is beyond the 2 GPU(s) of --device"));
        String single = refusal("m.gguf", "-dev", "0", "-mg", "1");
        assertTrue(single, single.contains("--main-gpu 1 is beyond the 1 GPU(s) of --device"));
    }

    @Test
    public void aLayerSplitOverOneListedIndexRunsOnThatGpu() throws UsageException {
        EngineConfig one = engine("m.gguf", "-sm", "layer", "-dev", "1");
        assertTrue(one.split());
        assertTrue("a layer split over one device index runs on that GPU, as split mode none does",
                one.picksMainGpu());
        assertEquals(Arrays.asList("1"), one.gpuDevices());
        assertNull(one.deviceFilter());

        EngineConfig named = engine("m.gguf", "-sm", "layer", "-dev", "Intel");
        assertTrue("a name substring picks its GPU across the APIs", !named.picksMainGpu());
        assertEquals("Intel", named.deviceFilter());
        assertTrue(!engine("m.gguf", "-sm", "layer").picksMainGpu());
        assertTrue(!engine("m.gguf", "-sm", "layer", "-dev", "0,1").picksMainGpu());
    }

    @Test
    public void inconsistentListsAreRefused() {
        String shares = refusal("m.gguf", "-dev", "0,1", "-ts", "1,1,1");
        assertTrue(shares, shares.contains("--tensor-split gives 3 share(s)"));
        String main = refusal("m.gguf", "-dev", "0,1", "-mg", "2");
        assertTrue(main, main.contains("--main-gpu 2 is beyond"));
        assertTrue(refusal("m.gguf", "-dev", "0,,1").contains("empty device entry"));
        assertTrue(refusal("m.gguf", "-mg", "-1").startsWith("-mg -1 must be a device index"));
        assertTrue(refusal("m.gguf", "--main-gpu", "x").startsWith("--main-gpu x is not a device index"));
        assertTrue(refusal("m.gguf", "--cpu", "-sm", "layer").contains(EngineConfig.PROP_BACKEND + "=cpu"));
    }

    @Test
    public void vllmContextSizesRunTheContextSplitOverThatManyDevices() throws UsageException {
        EngineConfig decode = engine("m.gguf", "-dcp", "2");
        assertEquals(EngineConfig.SplitMode.CONTEXT, decode.splitMode());
        assertEquals(Arrays.asList("0", "1"), decode.gpuDevices());

        EngineConfig listed = engine("m.gguf", "--decode-context-parallel-size", "3", "-dev", "A,B,C");
        assertEquals(EngineConfig.SplitMode.CONTEXT, listed.splitMode());
        assertEquals("a --device list of the right length is kept", Arrays.asList("A", "B", "C"),
                listed.gpuDevices());

        EngineConfig prefill = engine("m.gguf", "-pcp", "2");
        assertEquals(EngineConfig.SplitMode.CONTEXT, prefill.splitMode());
        assertEquals(2, prefill.prefillContextParallelSize());
        assertEquals(Arrays.asList("0", "1"), prefill.gpuDevices());

        EngineConfig both = engine("m.gguf", "-dcp", "2", "-pcp", "2");
        assertEquals(EngineConfig.SplitMode.CONTEXT, both.splitMode());
        assertEquals("the prefill query split runs over the context split's own ranks", 2,
                both.prefillContextParallelSize());

        assertEquals("vLLM's default context sizes are no flags at all", EngineConfig.SplitMode.NONE,
                engine("m.gguf", "-dcp", "1").splitMode());
        assertEquals(1, engine("m.gguf", "-pcp", "1").prefillContextParallelSize());
    }

    @Test
    public void contradictoryContextFlagsAreRefused() {
        String unequal = refusal("m.gguf", "-dcp", "2", "-pcp", "3");
        assertTrue(unequal, unequal.contains("--prefill-context-parallel-size")
                && unequal.contains("--decode-context-parallel-size") && unequal.contains("must equal 2"));

        String besideLayer = refusal("m.gguf", "-dcp", "2", "-sm", "layer");
        assertTrue(besideLayer, besideLayer.contains("--decode-context-parallel-size")
                && besideLayer.contains("--split-mode=layer") && besideLayer.contains("dcp-within-tp"));

        String besideRow = refusal("m.gguf", "-pcp", "2", "-sm", "row");
        assertTrue(besideRow, besideRow.contains("--prefill-context-parallel-size") && besideRow.contains("row"));

        String mismatch = refusal("m.gguf", "-dcp", "2", "-dev", "0");
        assertTrue(mismatch, mismatch.contains("--decode-context-parallel-size") && mismatch.contains("2 rank(s)"));
        String rpc = refusal("m.gguf", "-pcp", "2", "--rpc", "box:50052@k");
        assertTrue("remote devices refuse from the configuration's build, naming the property and the mode",
                rpc.contains(EngineConfig.PROP_RPC) && rpc.contains("context"));
        assertTrue(refusal("m.gguf", "-dcp", "2", "-dev", "none").contains(EngineConfig.PROP_BACKEND + "=cpu"));
        assertTrue(refusal("m.gguf", "-dcp", "x").contains("-dcp"));
        assertTrue(refusal("m.gguf", "-pcp", "0").contains("-pcp"));
    }

    @Test
    public void thePrefillContextParallelSizeNeedsTheContiguousMemory() {
        String paged = refusal("m.gguf", "-np", "2", "-pcp", "2");
        assertTrue(paged, paged.contains("--prefill-context-parallel-size") && paged.contains("paged"));
        System.setProperty(MemoryConfig.PROP_KV_PAGED, "true");
        assertTrue(refusal("m.gguf", "-pcp", "2").contains("paged"));
    }

    @Test
    public void theDcpKvInterleaveTakesThePageSizeOfAPagedContextSplit() throws UsageException {
        String besideLayer = refusal("m.gguf", "-sm", "layer", "--dcp-kv-cache-interleave-size", "16");
        assertTrue(besideLayer, besideLayer.contains("--dcp-kv-cache-interleave-size")
                && besideLayer.contains("layer"));

        String contiguous = refusal("m.gguf", "-dcp", "2", "--dcp-kv-cache-interleave-size", "16");
        assertTrue(contiguous, contiguous.contains("--dcp-kv-cache-interleave-size")
                && contiguous.contains("contiguous") && contiguous.contains(MemoryConfig.PROP_KV_PAGED));

        String wrongSize = refusal("m.gguf", "-np", "2", "-dcp", "2", "--dcp-kv-cache-interleave-size", "8");
        assertTrue(wrongSize, wrongSize.contains("--dcp-kv-cache-interleave-size") && wrongSize.contains("page size 16")
                && wrongSize.contains(MemoryConfig.PROP_KV_PAGE_SIZE));

        EngineConfig page = engine("m.gguf", "-np", "2", "-dcp", "2", "--dcp-kv-cache-interleave-size", "16");
        assertEquals(EngineConfig.SplitMode.CONTEXT, page.splitMode());
        assertTrue(page.memory().paged());
        assertEquals(16, page.memory().kvPageSize());
        assertEquals("the interleave configures nothing further", 1, page.prefillContextParallelSize());

        System.setProperty(MemoryConfig.PROP_KV_PAGE_SIZE, "8");
        assertEquals(8, engine("m.gguf", "-np", "2", "-dcp", "2", "--dcp-kv-cache-interleave-size", "8")
                .memory().kvPageSize());
        assertTrue(refusal("m.gguf", "-np", "2", "-dcp", "2", "--dcp-kv-cache-interleave-size", "16")
                .contains("page size 8"));
    }

    @Test
    public void theHelpListsTheFlagsInTheBackendGroup() {
        String usage = String.join("\n", LLMDemoOptions.usage());
        String backend = usage.substring(usage.indexOf("Backend:"), usage.indexOf("Memory:"));
        for (String flag : new String[]{"--split-mode, -sm", "--tensor-split, -ts", "--device, -dev",
                "--main-gpu, -mg", "--decode-context-parallel-size, -dcp",
                "--prefill-context-parallel-size, -pcp", "--dcp-kv-cache-interleave-size"}) {
            assertTrue(flag, backend.contains(flag));
        }
    }
}
