package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.image.ImageDecoder;
import com.eonmux.neuralfree.gguf.image.ImagePreprocessors;
import com.eonmux.neuralfree.gguf.image.ImageProcessorParams;
import com.eonmux.neuralfree.gguf.image.ImageSize;
import com.eonmux.neuralfree.gguf.image.ImageTile;
import com.eonmux.neuralfree.gguf.image.ImageTokenBudget;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import org.junit.Assume;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link MediaContext}: images and audio become marker text and embedding chunks the way llama.cpp's
 * {@code mtmd_tokenizer} lays them out, the rows equal a direct encode of the family's preprocessed tile, M-RoPE
 * models take the 2D token grid, Gemma 3 rows decode without the causal mask, a yasa2 image is its 64 pooled rows,
 * a Qwen2.5-Omni file serves both media under the types its legacy name resolves to, a Gemma 3n file serves its
 * audio (188 rows between the audio tags) and refuses images naming its gemma3nv tower, repeated media come from the
 * digest cache, an offloaded vision encoder gives the CPU rows, and a projector of another width or an undecodable
 * medium is refused.
 */
public class MediaContextTest {

    private static final int WIDTH = 16;

    /** A text encoder recording every segment it tokenizes, one token per character. */
    private static final class RecordingText implements MediaContext.TextEncoder {
        final List<String> seen = new ArrayList<>();

        @Override
        public int[] encode(String text) {
            seen.add(text);
            int[] out = new int[text.length()];
            for (int i = 0; i < out.length; i++) {
                out[i] = text.charAt(i);
            }
            return out;
        }
    }

    static byte[] png(int width, int height) throws IOException {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                img.setRGB(x, y, ((x * 31) & 0xFF) << 16 | ((y * 17) & 0xFF) << 8 | ((x + y) * 11 & 0xFF));
            }
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(img, "png", out);
        return out.toByteArray();
    }

    /** A 16 kHz mono PCM16 sine WAV of {@code seconds}. */
    static byte[] wav(int seconds) {
        int rate = 16000;
        int frames = seconds * rate;
        int dataBytes = frames * 2;
        ByteBuffer bb = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        bb.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt(36 + dataBytes);
        bb.put("WAVE".getBytes(StandardCharsets.US_ASCII));
        bb.put("fmt ".getBytes(StandardCharsets.US_ASCII)).putInt(16);
        bb.putShort((short) 1).putShort((short) 1).putInt(rate).putInt(rate * 2);
        bb.putShort((short) 2).putShort((short) 16);
        bb.put("data".getBytes(StandardCharsets.US_ASCII)).putInt(dataBytes);
        for (int i = 0; i < frames; i++) {
            bb.putShort((short) (Math.sin(2.0 * Math.PI * 440 * i / rate) * 12000));
        }
        return bb.array();
    }

    private static MediaContext open(Path file, boolean mrope) throws IOException {
        return open(file, mrope, MediaContext.DEFAULT_CACHE_BYTES);
    }

    private static MediaContext open(Path file, boolean mrope, long cacheBytes) throws IOException {
        return MediaContext.open(file.toString(), WIDTH, mrope, ImageTokenBudget.none(), t -> false,
                MmprojOptions.cpu(), cacheBytes);
    }

    @Test
    public void aLlavaImageIsOneChunkOfTheEncodedTileBetweenTheTextSegments() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        byte[] png = png(12, 8);
        try (MediaContext ctx = open(file, false)) {
            assertTrue(ctx.supportsVision());
            assertFalse(ctx.supportsAudio());
            assertEquals(WIDTH, ctx.width());
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder().appendText(text.encode("ab"));
            ctx.appendImage(b, png, 0, text);
            b.appendText(text.encode("c"));
            MediaPrompt prompt = b.build();

            assertEquals(3, prompt.chunks().size());
            MediaChunk image = prompt.chunks().get(1);
            assertFalse(image.isText());
            assertFalse(image.nonCausal());
            assertEquals(MediaRopePositions.normal(2, image.nTokens()), image.positions());
            assertEquals(2 + image.nTokens() + 1, prompt.nTokens());

            float[][] expected = directEncode(file, png);
            assertEquals(expected.length, image.nTokens());
            for (int r = 0; r < expected.length; r++) {
                assertArrayEquals(expected[r], image.embeddings()[r], 0f);
            }
        }
    }

    @Test
    public void anOffloadedVisionEncoderGivesTheCpuRows() throws IOException {
        Assume.assumeTrue("no OpenCL device", GpuTestUtil.isOpenCLAvailable());
        Path file = MmprojTestFiles.llava(WIDTH);
        byte[] png = png(12, 8);
        try (MediaContext cpu = open(file, false);
             MediaContext gpu = MediaContext.open(file.toString(), WIDTH, false, ImageTokenBudget.none(),
                     t -> false, MmprojOptions.defaults(), MediaContext.DEFAULT_CACHE_BYTES)) {
            assertEquals(VisionEncoder.CPU_DEVICE, cpu.visionDevice());
            Assume.assumeFalse("no GPU device runs the LLM kernels",
                    VisionEncoder.CPU_DEVICE.equals(gpu.visionDevice()));
            MediaPrompt.Builder want = MediaPrompt.builder();
            cpu.appendImage(want, png, 0, new RecordingText());
            MediaPrompt.Builder got = MediaPrompt.builder();
            gpu.appendImage(got, png, 0, new RecordingText());
            float[][] wantRows = want.build().chunks().get(0).embeddings();
            float[][] gotRows = got.build().chunks().get(0).embeddings();
            assertEquals(wantRows.length, gotRows.length);
            for (int r = 0; r < wantRows.length; r++) {
                assertArrayEquals("row " + r, wantRows[r], gotRows[r], 1e-3f);
            }
        }
    }

    /** The rows of the projector's own pipeline: decode, the family's preprocessing, normalize, encode. */
    private static float[][] directEncode(Path file, byte[] png) throws IOException {
        try (MmprojModel mm = MmprojModel.open(file.toString());
             CpuVisionEncoder enc = mm.openCpuVisionEncoder()) {
            ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(), ImageTokenBudget.none());
            List<ImageTile> tiles = ImagePreprocessors.forParams(params).prepare(ImageDecoder.decode(png))
                    .encodedTiles();
            assertEquals(1, tiles.size());
            ImageTile tile = tiles.get(0);
            return enc.encode(ClipImage.of(tile.width(), tile.height(),
                    tile.normalized(params.mean(), params.std())));
        }
    }

    @Test
    public void gemma3ImagesTakeTheirMarkersAndDecodeWithoutTheCausalMask() throws IOException {
        try (MediaContext ctx = open(MmprojTestFiles.gemma3(WIDTH), false)) {
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png(16, 16), 0, text);
            MediaPrompt prompt = b.build();

            assertEquals(List.of(ctx.markers().imageBegin(), ctx.markers().imageEnd()), text.seen);
            assertTrue(ctx.markers().imageBegin().contains("<start_of_image>"));
            assertTrue(ctx.markers().imageEnd().contains("<end_of_image>"));
            assertEquals(3, prompt.chunks().size());
            MediaChunk image = prompt.chunks().get(1);
            assertTrue(image.nonCausal());
            assertEquals(4, image.nTokens());
            assertEquals(ctx.markers().imageBegin().length(), image.positions().pos0());
        }
    }

    @Test
    public void mropeModelsLayAnUnslicedImageOnItsTokenGrid() throws IOException {
        Path file = MmprojTestFiles.qwen2vl(WIDTH);
        byte[] png = png(32, 16);
        try (MediaContext ctx = open(file, true)) {
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png, 0, text);
            MediaPrompt prompt = b.build();

            assertEquals(List.of("<|vision_start|>", "<|vision_end|>"), text.seen);
            MediaChunk image = prompt.chunks().get(1);
            MediaRopePositions pos = image.positions();
            assertEquals(MediaRopePositions.Kind.IMAGE_2D, pos.kind());
            ImageSize grid = tokenGrid(file, png);
            assertEquals(grid.width(), pos.nx());
            assertEquals(grid.height(), pos.ny());
            assertTrue(pos.nx() > pos.ny());
            assertEquals(image.nTokens(), pos.nx() * pos.ny());
            // a 2D grid advances the text position by its larger side
            assertEquals("<|vision_start|>".length() + pos.nPositions() + "<|vision_end|>".length(),
                    prompt.nPositions());
        }
    }

    /** The merged token grid the family's preprocessing gives the image. */
    private static ImageSize tokenGrid(Path file, byte[] png) throws IOException {
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(), ImageTokenBudget.none());
            ImageTile tile = ImagePreprocessors.forParams(params).prepare(ImageDecoder.decode(png))
                    .encodedTiles().get(0);
            return params.tokenGridFor(tile.size());
        }
    }

    @Test
    public void audioTakesItsMarkersAndOneChunkPerEncoderWindow() throws IOException {
        try (MediaContext ctx = open(MmprojTestFiles.qwen2aAudio(WIDTH), false)) {
            assertTrue(ctx.supportsAudio());
            assertFalse(ctx.supportsVision());
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendAudio(b, wav(2), "clip.wav", text);
            MediaPrompt prompt = b.build();

            assertEquals(List.of("<|audio_bos|>", "<|audio_eos|>"), text.seen);
            assertEquals(3, prompt.chunks().size());
            MediaChunk audio = prompt.chunks().get(1);
            assertEquals(750, audio.nTokens());
            assertEquals(WIDTH, audio.embeddings()[0].length);
            assertEquals(MediaRopePositions.Kind.NORMAL, audio.positions().kind());

            MediaException.assertRefused(() -> ctx.appendImage(MediaPrompt.builder(), png(8, 8), 0, text),
                    "no vision encoder");
        }
    }

    @Test
    public void aYasa2ImageIsItsEightByEightPoolBetweenTheImageTags() throws IOException {
        Path file = Yasa2Fixture.full().gguf().writeTempFile();
        // stretched to the 64-pixel tile whatever its aspect
        byte[] png = png(40, 24);
        try (MediaContext ctx = open(file, false)) {
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png, 0, text);
            MediaPrompt prompt = b.build();

            assertEquals(List.of("<image>", "</image>"), text.seen);
            assertEquals(3, prompt.chunks().size());
            MediaChunk image = prompt.chunks().get(1);
            assertEquals(64, image.nTokens());
            float[][] expected = directEncode(file, png);
            assertEquals(expected.length, image.nTokens());
            for (int r = 0; r < expected.length; r++) {
                assertArrayEquals(expected[r], image.embeddings()[r], 0f);
            }
        }
    }

    @Test
    public void aQwen25OmniFileServesImagesAndAudioUnderTheirResolvedTypes() throws IOException {
        Path file = MmprojTestFiles.qwen25Omni(WIDTH);
        byte[] png = png(16, 8);
        try (MediaContext ctx = open(file, true)) {
            assertTrue(ctx.supportsVision());
            assertTrue(ctx.supportsAudio());
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png, 0, text);
            ctx.appendAudio(b, wav(1), "clip.wav", text);
            MediaPrompt prompt = b.build();

            assertEquals(List.of("<|vision_start|>", "<|vision_end|>", "<|audio_bos|>", "<|audio_eos|>"),
                    text.seen);
            // the image's end tag and the audio's begin tag share one text chunk
            assertEquals(5, prompt.chunks().size());
            MediaChunk image = prompt.chunks().get(1);
            assertEquals(MediaRopePositions.Kind.IMAGE_2D, image.positions().kind());
            float[][] expected = directEncode(file, png);
            assertEquals(expected.length, image.nTokens());
            for (int r = 0; r < expected.length; r++) {
                assertArrayEquals(expected[r], image.embeddings()[r], 0f);
            }
            MediaChunk audio = prompt.chunks().get(3);
            assertEquals(750, audio.nTokens());
            assertEquals(WIDTH, audio.embeddings()[0].length);
            assertEquals(MediaRopePositions.Kind.AUDIO_1D, audio.positions().kind());
        }
    }

    @Test
    public void aGemma3nFileServesAudioAndRefusesImagesNamingItsVisionTower() throws IOException {
        Path file = Gemma3nAudioFixture.write(Gemma3nAudioFixture.withVisionMeta(Gemma3nAudioFixture.meta()),
                Gemma3nAudioFixture.tensors(31));
        try (MediaContext ctx = MediaContext.open(file.toString(), Gemma3nAudioFixture.TEXT, false,
                ImageTokenBudget.none(), t -> false, MmprojOptions.cpu(), MediaContext.DEFAULT_CACHE_BYTES)) {
            assertTrue(ctx.supportsAudio());
            assertFalse(ctx.supportsVision());
            RecordingText text = new RecordingText();
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendAudio(b, wav(2), "clip.wav", text);
            MediaPrompt prompt = b.build();

            assertEquals(List.of("<start_of_audio>", "<end_of_audio>"), text.seen);
            assertEquals(3, prompt.chunks().size());
            MediaChunk audio = prompt.chunks().get(1);
            assertEquals(Gemma3nAudioModel.TOKENS_PER_WINDOW, audio.nTokens());
            assertEquals(Gemma3nAudioFixture.TEXT, audio.embeddings()[0].length);
            assertEquals(MediaRopePositions.Kind.NORMAL, audio.positions().kind());

            MediaException.assertRefused(() -> ctx.appendImage(MediaPrompt.builder(), png(8, 8), 0, text),
                    "the projector's vision encoder does not run (projector type 'gemma3nv'");
        }
    }

    @Test
    public void mropeAudioTakesOnePositionPerRowInEverySection() throws IOException {
        try (MediaContext ctx = open(MmprojTestFiles.qwen2aAudio(WIDTH), true)) {
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendAudio(b, wav(1), "clip.wav", new RecordingText());
            MediaChunk audio = b.build().chunks().get(1);
            assertEquals(MediaRopePositions.Kind.AUDIO_1D, audio.positions().kind());
            assertEquals(audio.nTokens(), audio.positions().nPositions());
        }
    }

    @Test
    public void repeatedMediaYieldTheSameChunks() throws IOException {
        byte[] png = png(8, 8);
        try (MediaContext ctx = open(MmprojTestFiles.llava(WIDTH), false)) {
            MediaPrompt.Builder first = MediaPrompt.builder();
            ctx.appendImage(first, png, 0, new RecordingText());
            MediaPrompt.Builder second = MediaPrompt.builder();
            ctx.appendImage(second, png.clone(), 0, new RecordingText());
            assertEquals(first.build().chunks(), second.build().chunks());
        }
    }

    @Test
    public void mediaChunksCarryTheContentDigestAsTheirId() throws IOException {
        byte[] png = png(8, 8);
        try (MediaContext ctx = open(MmprojTestFiles.llava(WIDTH), false)) {
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendImage(b, png, 0, new RecordingText());
            MediaChunk image = firstMedia(b.build());
            assertTrue(image.id(), image.id().matches("image:content:[0-9a-f]{64}#0"));
        }
        try (MediaContext ctx = open(MmprojTestFiles.qwen2aAudio(WIDTH), false)) {
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendAudio(b, wav(1), "clip.wav", new RecordingText());
            MediaChunk audio = firstMedia(b.build());
            assertTrue(audio.id(), audio.id().matches("audio:content:[0-9a-f]{64}#0"));
        }
    }

    /** The prompt's first media chunk. */
    private static MediaChunk firstMedia(MediaPrompt prompt) {
        for (MediaChunk c : prompt.chunks()) {
            if (!c.isText()) {
                return c;
            }
        }
        throw new AssertionError("the prompt holds no media chunk: " + prompt);
    }

    @Test
    public void everyChunkOfOneMediumNamesItsOwnRows() throws IOException {
        try (MediaContext ctx = open(MmprojTestFiles.qwen2aAudio(WIDTH), false)) {
            MediaPrompt.Builder b = MediaPrompt.builder();
            ctx.appendAudio(b, wav(31), "clip.wav", new RecordingText());
            MediaPrompt prompt = b.build();
            assertEquals("a clip past one 30 s window encodes as two chunks", 2, prompt.nMediaChunks());
            MediaChunk first = prompt.chunks().get(1);
            MediaChunk second = prompt.chunks().get(2);
            String key = first.id().substring(0, first.id().indexOf('#'));
            assertEquals(key + "#0", first.id());
            assertEquals("the windows share the clip's digest but not an identity", key + "#1", second.id());
            assertFalse(java.util.Arrays.deepEquals(first.embeddings(), second.embeddings()));
        }
    }

    @Test
    public void aProjectorOfAnotherWidthIsRefusedNamingBothWidths() throws IOException {
        Path file = MmprojTestFiles.llava(WIDTH);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MediaContext.open(file.toString(), 24, false, ImageTokenBudget.none(), t -> false,
                        MmprojOptions.cpu(), MediaContext.DEFAULT_CACHE_BYTES));
        assertTrue(e.getMessage(), e.getMessage().contains("n_embd = 24"));
        assertTrue(e.getMessage(), e.getMessage().contains("n_embd = " + WIDTH));
    }

    @Test
    public void undecodableMediaAreClientErrors() throws IOException {
        try (MediaContext ctx = open(MmprojTestFiles.llava(WIDTH), false)) {
            MediaException.assertRefused(() -> ctx.appendImage(MediaPrompt.builder(),
                    "not an image".getBytes(StandardCharsets.US_ASCII), 0, new RecordingText()), "cannot be decoded");
        }
        try (MediaContext ctx = open(MmprojTestFiles.qwen2aAudio(WIDTH), false)) {
            MediaException.assertRefused(() -> ctx.appendAudio(MediaPrompt.builder(),
                    "RIFF0000WAVEjunk".getBytes(StandardCharsets.US_ASCII), "x.wav", new RecordingText()),
                    "cannot be decoded");
        }
    }

    @Test
    public void aClosedContextRefusesToEncode() throws IOException {
        MediaContext ctx = open(MmprojTestFiles.llava(WIDTH), false);
        ctx.close();
        ctx.close();
        assertThrows(IllegalStateException.class,
                () -> ctx.appendImage(MediaPrompt.builder(), png(8, 8), 0, new RecordingText()));
    }

    /** Assertions on refused media. */
    private static final class MediaException {
        private MediaException() {
        }

        static void assertRefused(org.junit.function.ThrowingRunnable call, String fragment) {
            MediaInputException e = assertThrows(MediaInputException.class, call);
            assertTrue(e.getMessage(), e.getMessage().contains(fragment));
        }
    }
}
