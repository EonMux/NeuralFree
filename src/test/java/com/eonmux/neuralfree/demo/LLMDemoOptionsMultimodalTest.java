package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.MediaFile;
import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.mm.MmprojOptions;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The multimodal flags of {@link LLMDemoOptions}, spelled as llama.cpp common/arg.cpp spells them
 * ({@code --mmproj}, {@code -mmu}, the {@code --mmproj-auto} switch family, {@code --mmproj-offload},
 * {@code -mmdev}, the mtmd media lists and the image token budget), the {@link MmprojOptions} they build, and the
 * combinations refused by name.
 */
public class LLMDemoOptionsMultimodalTest {

    private static LLMDemoOptions parse(String... args) throws UsageException {
        return LLMDemoOptions.parse(args);
    }

    private static String refusal(String... args) {
        return assertThrows(UsageException.class, () -> parse(args)).getMessage();
    }

    @Test
    public void theProjectorFlagsParse() throws UsageException {
        LLMDemoOptions o = parse("m.gguf", "--mmproj", "p.gguf", "-mmu", "https://h/mmproj.gguf",
                "--mmproj-offload", "--image-min-tokens", "64", "--image-max-tokens", "512");
        assertEquals("p.gguf", o.mmproj);
        assertEquals("https://h/mmproj.gguf", o.mmprojUrl);
        assertTrue(o.mmprojOffload);
        assertEquals(64, o.imageMinTokens);
        assertEquals(512, o.imageMaxTokens);
        assertFalse(o.noMmproj);
        assertTrue(o.multimodalInputGiven());
        assertFalse("--mmproj alone is no multimodal input", parse("m.gguf", "--mmproj", "p.gguf")
                .multimodalInputGiven());
    }

    @Test
    public void theAutoSwitchFollowsItsLastFlag() throws UsageException {
        assertFalse("the search is on without a flag", parse("m.gguf", "--mmproj-auto").noMmproj);
        assertTrue(parse("m.gguf", "--no-mmproj").noMmproj);
        assertTrue(parse("m.gguf", "--no-mmproj-auto").noMmproj);
        assertFalse(parse("m.gguf", "--no-mmproj", "--mmproj-auto").noMmproj);
        assertTrue(parse("m.gguf", "--mmproj-auto", "--no-mmproj").noMmproj);
    }

    @Test
    public void theDeviceSelectorBuildsTheEncoderOptions() throws UsageException {
        assertEquals(MmprojOptions.defaults(), parse("m.gguf", "-p", "x", "--image", "a.png").encoderOptions());
        MmprojOptions off = parse("m.gguf", "--no-mmproj-offload").encoderOptions();
        assertFalse(off.offload());
        assertNull(off.device());
        MmprojOptions cpu = parse("m.gguf", "-mmdev", "none").encoderOptions();
        assertNull("none is the CPU encoder without a device selector", cpu.device());
        assertFalse(cpu.offload());
        MmprojOptions gpu = parse("m.gguf", "--mmproj-device", "OpenCL:0").encoderOptions();
        assertEquals("OpenCL:0", gpu.device());
        assertTrue(gpu.offload());
        assertFalse("an explicit offload wins over the device's",
                parse("m.gguf", "--mmproj-device", "OpenCL:0", "--no-mmproj-offload").encoderOptions().offload());
    }

    @Test
    public void theMediaListsKeepTheirFlagKind() throws UsageException {
        LLMDemoOptions o = parse("m.gguf", "-p", "look", "--image", "a.png,b.png", "--audio", "c.wav",
                "--video", "d.mp4");
        assertEquals(List.of(new MediaFile("image", "a.png"), new MediaFile("image", "b.png"),
                new MediaFile("audio", "c.wav"), new MediaFile("video", "d.mp4")), o.media);
        assertTrue(o.multimodalInputGiven());
        assertTrue(refusal("m.gguf", "-p", "x", "--image", "a.png,,b.png").contains("empty file"));
    }

    @Test
    public void theImageBudgetRefusesAnInvertedRange() throws UsageException {
        assertTrue(refusal("m.gguf", "-p", "x", "--image", "a.png", "--image-min-tokens", "512",
                "--image-max-tokens", "64").contains("--image-min-tokens 512 exceeds --image-max-tokens 64"));
        LLMDemoOptions o = parse("m.gguf", "-p", "x", "--image", "a.png");
        assertEquals(0, o.imageBudget().minTokens());
        assertEquals(0, o.imageBudget().maxTokens());
        assertTrue(o.imageBudget().isEmpty());
        assertEquals(64, parse("m.gguf", "-p", "x", "--image", "a.png", "--image-min-tokens", "64")
                .imageBudget().minTokens());
    }

    @Test
    public void theAudioModeKeepsMmprojForItsSpeechGenerator() throws UsageException {
        LLMDemoOptions o = parse("m.gguf", "--mode", "audio", "--mmproj", "gen.gguf", "-p", "hello");
        assertEquals("gen.gguf", o.mmproj);
        assertFalse("the speech generator is no multimodal input", o.multimodalInputGiven());
        assertTrue(refusal("m.gguf", "--mode", "audio", "--mmproj-url", "https://h/p.gguf")
                .contains("--mode audio speaks through the --mmproj speech generator"));
        assertTrue(refusal("m.gguf", "--mode", "audio", "--image", "a.png")
                .contains("configure a text model's multimodal"));
        assertTrue(refusal("m.gguf", "--mode", "audio", "--mmproj-device", "OpenCL:0")
                .contains("--mode audio speaks"));
    }

    @Test
    public void theNonGenerateModesRefuseTheWholeSet() {
        assertTrue(refusal("m.gguf", "--mode", "embed", "--mmproj", "p.gguf").contains("--mode embed reads text"));
        assertTrue(refusal("m.gguf", "--mode", "diffusion", "--mmproj-url", "https://h/p.gguf")
                .contains("belong to the"));
        assertTrue(refusal("m.gguf", "--mode", "classify", "--mmproj-device", "OpenCL:0")
                .contains("belong to the"));
        assertTrue(refusal("m.gguf", "--mode", "rerank", "--no-mmproj").contains("belong to the"));
        assertTrue(refusal("m.gguf", "--mode", "embed", "--image-min-tokens", "8").contains("belong to the"));
    }

    @Test
    public void mediaNeedsTheChatTemplateAndThePrompt() throws UsageException {
        assertTrue(refusal("m.gguf", "-p", "x", "--image", "a.png", "--raw")
                .contains("--raw and -no-cnv complete plain text"));
        assertTrue(refusal("m.gguf", "-p", "x", "--image", "a.png", "-no-cnv").contains("complete plain text"));
        assertTrue(refusal("m.gguf", "--video", "v.mp4").contains("attach to the -p prompt's turn"));
        assertEquals("--non-interactive reads the prompt, so the files attach to it", 1,
                parse("m.gguf", "--non-interactive", "--image", "a.png").media.size());
    }

    @Test
    public void mediaRefusesTheCompletionRepl() throws UsageException {
        assertTrue(refusal("m.gguf", "-p", "x", "--image", "a.png", "-i").contains("completes plain text"));
        assertTrue(refusal("m.gguf", "-p", "x", "--mmproj", "p.gguf", "--in-prefix", ">")
                .contains("completes plain text"));
        assertFalse("the audio mode's speech generator is no REPL media",
                parse("m.gguf", "--mode", "audio", "--mmproj", "gen.gguf", "-p", "speak").mmproj.isEmpty());
    }

    @Test
    public void mediaRefusesTheTextGeneratorsItDoesNotRide() throws UsageException {
        assertTrue(refusal("m.gguf", "-p", "x", "--image", "a.png", "--draft", "d.gguf")
                .contains("speculative decoding rides the text generator"));
        assertTrue(refusal("m.gguf", "-p", "x", "--image", "a.png", "--prompt-cache", "c.bin")
                .contains("--prompt-cache rides the text generator"));
        assertFalse("--no-mmproj leaves the text generators alone",
                parse("m.gguf", "-p", "x", "--no-mmproj", "--prompt-cache", "c.bin") == null);
    }

    @Test
    public void theProjectorUrlMustBeHttpOrHttps() {
        assertTrue(refusal("m.gguf", "--mmproj-url", "ftp://h/p.gguf").contains("must be an http or https URL"));
        assertTrue(refusal("m.gguf", "-mmu", "/local/p.gguf").contains("-mmu"));
        assertTrue(refusal("--server-base", "http://h:1", "-mmu", "https://h/p.gguf")
                .contains("--server-base and -mmu"));
        assertTrue(refusal("--server-base", "http://h:1", "--mmproj", "p.gguf").contains("--mmproj"));
        assertTrue(refusal("--server-base", "http://h:1", "--mmproj-auto").contains("--mmproj-auto"));
    }

    @Test
    public void theDeviceValueIsCheckedAtParse() {
        assertTrue(refusal("m.gguf", "--mmproj-device", "OpenCL:0,cuda:1").contains("--mmproj-device"));
    }
}
