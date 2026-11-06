package com.eonmux.neuralfree.demo.tools;

import org.junit.Test;

import java.nio.file.Paths;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The command-line boundary of {@code nftool slice}: the repository expression and its revision forms, the default
 * output directory, every flag reaching the options, and the refusals — a quant tag (the tool reads safetensors
 * weights), a second repository, an unknown flag, a flag without its value, a malformed integer or size, an
 * out-of-range value and two revisions. Parsing never touches the network.
 */
public class SliceCommandTest {

    private static SliceCommand.Parsed parse(String... args) throws Exception {
        return SliceCommand.parse(args);
    }

    private static SliceCommand.UsageException usage(String... args) {
        return assertThrows(SliceCommand.UsageException.class, () -> parse(args));
    }

    @Test
    public void parsesTheRepositoryAndTheDefaults() throws Exception {
        SliceCommand.Parsed parsed = parse("Qwen/Qwen3-0.6B");
        assertEquals("Qwen/Qwen3-0.6B", parsed.ref().repository());
        assertNull(parsed.ref().revision());
        assertEquals(2, parsed.options().numLayers());
        assertNull(parsed.options().keepLayers());
        assertEquals(8, parsed.options().numExperts());
        assertEquals(0, parsed.options().vocabSize());
        assertEquals(1L << 30, parsed.options().maxBytes());
        assertEquals(8, parsed.options().jobs());
        assertEquals(Paths.get("tmp", "hf_slices", "Qwen--Qwen3-0.6B"), parsed.options().output());
        assertTrue(!parsed.options().inspect() && !parsed.options().dryRun() && !parsed.options().overwrite());
    }

    @Test
    public void parsesEveryOption() throws Exception {
        SliceCommand.Parsed parsed = parse("user/model", "-o", "out", "--num-layers", "3", "--num-experts", "4",
                "--vocab-size", "8", "--max-size", "2M", "--jobs", "2", "--inspect", "--dry-run", "--overwrite",
                "--hf-token", "hf_abcdefghijklmnopqrstuvwxyz0123456789");
        assertEquals(Paths.get("out"), parsed.options().output());
        assertEquals(3, parsed.options().numLayers());
        assertEquals(4, parsed.options().numExperts());
        assertEquals(8, parsed.options().vocabSize());
        assertEquals(2L << 20, parsed.options().maxBytes());
        assertEquals(2, parsed.options().jobs());
        assertTrue(parsed.options().inspect());
        assertTrue(parsed.options().dryRun());
        assertTrue(parsed.options().overwrite());
        assertEquals("hf_abcdefghijklmnopqrstuvwxyz0123456789", parsed.token());
    }

    @Test
    public void keepLayersAreSortedAndDeduplicated() throws Exception {
        SliceCommand.Parsed parsed = parse("user/model", "--keep-layers", "3, 1,3",
                "--num-layers", "9", "--dry-run");
        assertEquals(Arrays.asList(1, 3), parsed.options().keepLayers());
    }

    @Test
    public void parsesARevisionFromTheRepositoryOrTheFlag() throws Exception {
        assertEquals("main", parse("user/model@main").ref().revision());
        assertEquals("main", parse("user/model", "--revision", "main").ref().revision());
        SliceCommand.UsageException e = usage("user/model@main", "--revision", "dev");
        assertTrue(e.getMessage(), e.getMessage().contains("--revision"));
    }

    @Test
    public void refusesAQuantTagBecauseTheToolReadsSafetensorsWeights() {
        SliceCommand.UsageException e = usage("user/model:Q4_K_M");
        assertTrue(e.getMessage(), e.getMessage().contains("Q4_K_M"));
        assertTrue(e.getMessage(), e.getMessage().contains("safetensors"));
    }

    @Test
    public void refusesMalformedCommandLines() {
        assertTrue(usage().getMessage().contains("needs a repository"));
        assertTrue(usage("user/model", "user/other").getMessage().contains("too many repository arguments"));
        assertTrue(usage("user/model", "--nope").getMessage().contains("unknown option --nope"));
        assertTrue(usage("user/model", "--num-layers").getMessage().contains("--num-layers needs a value"));
        assertTrue(usage("user/model", "--num-layers", "two").getMessage().contains("not an integer"));
        assertTrue(usage("user/model", "--num-layers", "-1").getMessage().contains("num-layers must be >= 0"));
        assertTrue(usage("user/model", "--num-experts", "0").getMessage().contains("num-experts must be >= 1"));
        assertTrue(usage("user/model", "--jobs", "0").getMessage().contains("jobs must be >= 1"));
        assertTrue(usage("user/model", "--keep-layers", "0,,2").getMessage().contains("--keep-layers"));
        assertTrue(usage("user/model", "--keep-layers", "0,x").getMessage().contains("not an integer"));
        assertTrue(usage("user/model", "--max-size", "big").getMessage().contains("is not a size"));
        assertTrue(usage("user/model", "--max-size", "0").getMessage().contains("max-size must be > 0"));
        assertTrue(usage("user").getMessage().contains("invalid HF repo"));
    }

    @Test
    public void helpPrintsTheUsageAndRunsNothing() throws Exception {
        assertNull(parse("--help"));
        assertNull(parse("user/model", "-h"));
    }

    @Test
    public void parsesByteSizes() throws Exception {
        assertEquals(10L, SliceCommand.size("10"));
        assertEquals(10L, SliceCommand.size("10B"));
        assertEquals(1536L, SliceCommand.size("1.5K"));
        assertEquals(512L << 20, SliceCommand.size("512M"));
        assertEquals(1L << 30, SliceCommand.size("1G"));
        assertEquals(2L << 40, SliceCommand.size("2T"));
        assertThrows(SliceCommand.UsageException.class, () -> SliceCommand.size("1.2.3G"));
        assertThrows(SliceCommand.UsageException.class, () -> SliceCommand.size(""));
    }
}
