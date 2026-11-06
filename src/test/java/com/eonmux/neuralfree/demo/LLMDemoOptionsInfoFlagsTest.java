package com.eonmux.neuralfree.demo;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The information flags of {@link LLMDemoOptions} (llama.cpp's common {@code --version},
 * {@code --list-devices} and {@code -cl}/{@code --cache-list}): each parses on its own, each
 * combines with a leading model path without consuming a value, and the usage documents all three.
 */
public class LLMDemoOptionsInfoFlagsTest {

    @Test
    public void theVersionFlagParsesAlone() throws Exception {
        assertTrue(LLMDemoOptions.parse(new String[]{"--version"}).version);
        assertFalse(LLMDemoOptions.parse(new String[]{"--model", "m.gguf"}).version);
    }

    @Test
    public void theListDevicesFlagParsesAlone() throws Exception {
        assertTrue(LLMDemoOptions.parse(new String[]{"--list-devices"}).listDevices);
        assertFalse(LLMDemoOptions.parse(new String[]{"--model", "m.gguf"}).listDevices);
    }

    @Test
    public void theCacheListFlagParsesInBothSpellings() throws Exception {
        assertTrue(LLMDemoOptions.parse(new String[]{"-cl"}).cacheList);
        assertTrue(LLMDemoOptions.parse(new String[]{"--cache-list"}).cacheList);
        assertFalse(LLMDemoOptions.parse(new String[]{"--model", "m.gguf"}).cacheList);
    }

    @Test
    public void theFlagsCombineWithALeadingModelPath() throws Exception {
        LLMDemoOptions opts = LLMDemoOptions.parse(new String[]{"m.gguf", "--version", "--list-devices",
                "--cache-list"});
        assertEquals("m.gguf", opts.modelPath);
        assertTrue(opts.version);
        assertTrue(opts.listDevices);
        assertTrue(opts.cacheList);
    }

    @Test
    public void theFlagsTakeNoValue() throws Exception {
        // a flag that consumed a value would turn the next option into a stray argument; these stay boolean
        LLMDemoOptions opts = LLMDemoOptions.parse(new String[]{"--version", "--list-devices",
                "-cl"});
        assertTrue(opts.version);
        assertTrue(opts.listDevices);
        assertTrue(opts.cacheList);
        assertNull(opts.modelPath);
    }

    @Test
    public void theUsageDocumentsTheInformationFlags() {
        StringBuilder text = new StringBuilder();
        for (String line : LLMDemoOptions.usage()) {
            text.append(line).append('\n');
        }
        String usage = text.toString();
        assertTrue(usage, usage.contains("--version"));
        assertTrue(usage, usage.contains("--list-devices"));
        assertTrue(usage, usage.contains("--cache-list"));
        assertTrue(usage, usage.contains("-cl"));
    }
}
