package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The datasets added for the reference's remaining samplers: their record shapes, their seeded
 * order, their output lengths and the content parts they send.
 */
public class ExtendedDatasetsTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A tokenizer that counts one token per character and decodes one marker per token. */
    private static final BenchTokenizer DECODING = new BenchTokenizer() {
        @Override
        public int[] tokenize(String text) {
            return new int[text.length()];
        }

        @Override
        public String detokenize(int[] tokens) {
            StringBuilder text = new StringBuilder();
            for (int i = 0; i < tokens.length; i++) {
                text.append("t ");
            }
            return text.toString().trim();
        }
    };

    private Path write(String name, String content) throws IOException {
        Path file = tmp.newFile(name).toPath();
        Files.write(file, content.getBytes(StandardCharsets.UTF_8));
        return file;
    }

    private static ServeBenchOptions.Builder options(String dataset, Path path) {
        return ServeBenchOptions.builder().datasetName(dataset).datasetPath(path.toString())
                .seed(0).disableShuffle(true).requestIdPrefix("r-");
    }

    @Test
    public void specBenchReadsItsTurnsAndFiltersTheCategory() throws IOException {
        Path file = write("spec.jsonl",
                "{\"turns\":[\"first\"],\"category\":\"translation\"}\n"
                        + "{\"turns\":[\"second\"],\"category\":\"summarization\"}\n"
                        + "{\"turns\":[\"third\"],\"category\":\"translation\"}\n");
        List<BenchSample> samples = new SpecBenchDataset(options("spec_bench", file).numPrompts(2)
                .specBenchCategory("translation").build(), DECODING).sample();
        assertEquals(2, samples.size());
        assertEquals("first", samples.get(0).text());
        assertEquals(5, samples.get(0).promptLen());
        assertEquals(256, samples.get(0).outputLen());
        assertEquals("third", samples.get(1).text());
    }

    @Test
    public void speedBenchReadsItsSubsetFile() throws IOException {
        Path dir = tmp.newFolder("speed").toPath();
        Files.write(dir.resolve("qualitative.jsonl"), ("{\"messages\":[{\"content\":\"hello\"}],"
                + "\"category\":\"math\"}\n{\"messages\":[{\"content\":\"world!\"}],"
                + "\"category\":\"code\"}\n").getBytes(StandardCharsets.UTF_8));
        List<BenchSample> samples = new SpeedBenchDataset(options("speed_bench", dir).numPrompts(2)
                .build(), DECODING).sample();
        assertEquals(2, samples.size());
        assertEquals("hello", samples.get(0).text());
        assertEquals(4096, samples.get(0).outputLen());
        assertEquals("world!", samples.get(1).text());
    }

    @Test
    public void blazeditAppliesTheTemplateAndTheDistanceFilter() throws IOException {
        Path file = write("edit.jsonl",
                "{\"code\":\"a=1\",\"change_request\":\"rename\",\"norm_distance\":0.1}\n"
                        + "{\"code\":\"b=2\",\"change_request\":\"drop\",\"norm_distance\":0.9}\n");
        List<BenchSample> samples = new BlazeditDataset(options("blazedit", file).numPrompts(1)
                .blazeditDistance(0.0, 0.5).build(), DECODING).sample();
        assertEquals("the 0.9 record is filtered", 1, samples.size());
        assertTrue(samples.get(0).text().contains("Original file:\n```python\na=1\n```"));
        assertTrue(samples.get(0).text().contains("Change request:\nrename"));
        assertEquals(4000, samples.get(0).outputLen());
    }

    @Test
    public void hfReadsAConversationDirectory() throws IOException {
        Path dir = tmp.newFolder("hf").toPath();
        Files.write(dir.resolve("data.jsonl"), ("{\"conversations\":[{\"value\":\"hello\"},"
                + "{\"value\":\"world!\"}]}\n{\"conversations\":[{\"value\":\"solo\"}]}\n")
                .getBytes(StandardCharsets.UTF_8));
        List<BenchSample> samples = new HfDataset(options("hf", dir).numPrompts(1).build(), DECODING)
                .sample();
        assertEquals(1, samples.size());
        assertEquals("hello", samples.get(0).text());
        assertEquals(5, samples.get(0).promptLen());
        assertEquals(6, samples.get(0).outputLen());
    }

    @Test
    public void hfReadsAPromptDirectory() throws IOException {
        Path dir = tmp.newFolder("hf2").toPath();
        Files.write(dir.resolve("data.csv"), "prompt,output_tokens\nabcd,9\n".getBytes(StandardCharsets.UTF_8));
        List<BenchSample> samples = new HfDataset(options("hf", dir).numPrompts(1).build(), DECODING)
                .sample();
        assertEquals(1, samples.size());
        assertEquals("abcd", samples.get(0).text());
        assertEquals(9, samples.get(0).outputLen());
    }

    @Test
    public void timedTraceExpandsItsChunksAndKeepsTheTimestamp() throws IOException {
        Path file = write("trace.jsonl",
                "{\"timestamp\":1.5,\"input_length\":4,\"output_length\":3,\"hash_ids\":[10,11]}\n");
        ServeBenchOptions opts = options("timed_trace", file).numPrompts(1)
                .timedTraceChunkHashSize(2).selfTimed(true).build();
        List<BenchSample> samples = new TimedTraceDataset(opts, DECODING, 100, new int[]{0, 1}).sample();
        assertEquals(1, samples.size());
        assertEquals(4, samples.get(0).tokens().length);
        assertEquals(3, samples.get(0).outputLen());
        assertEquals(1.5, samples.get(0).timestamp(), 0.0);
        for (int token : samples.get(0).tokens()) {
            assertTrue(token >= 2 && token < 100);
        }
    }

    @Test
    public void theTimedTraceLabelsSelectTheirFields() throws IOException {
        Path file = write("trace2.jsonl", "{\"ts\":2,\"in\":2,\"out\":1,\"hashes\":[7]}\n");
        ServeBenchOptions opts = options("timed_trace", file).numPrompts(1)
                .timedTraceLabels("ts", "in", "out", "hashes").build();
        List<BenchSample> samples = new TimedTraceDataset(opts, DECODING, 10, new int[]{}).sample();
        assertEquals(2, samples.get(0).tokens().length);
        assertEquals(2.0, samples.get(0).timestamp(), 0.0);
    }

    @Test
    public void customImageBuildsChatContentParts() throws IOException {
        Path dir = tmp.newFolder("images").toPath();
        Path image = dir.resolve("a.png");
        Files.write(image, "pixels".getBytes(StandardCharsets.UTF_8));
        Path file = dir.resolve("images.jsonl");
        Files.write(file, ("{\"prompt\":\"look\",\"image_files\":[\"a.png\"]}\n"
                + "{\"content\":[{\"type\":\"text\",\"text\":\"see \"},"
                + "{\"type\":\"image\",\"image\":\"" + image + "\"}]}\n")
                .getBytes(StandardCharsets.UTF_8));
        List<BenchSample> samples = new CustomImageDataset(options("custom_image", file).numPrompts(2)
                .build(), DECODING).sample();
        assertEquals(2, samples.size());
        assertEquals(BenchSample.Kind.CHAT, samples.get(0).kind());
        assertEquals("look", CustomImageDataset.textOf(samples.get(0).chatContent()));
        assertEquals(2, samples.get(0).chatContent().size());
        assertEquals("image_url", ((Map<?, ?>) samples.get(0).chatContent().get(1)).get("type"));
        String url = (String) ((Map<?, ?>) ((Map<?, ?>) samples.get(0).chatContent().get(1))
                .get("image_url")).get("url");
        assertTrue(url, url.startsWith("data:image/png;base64,"));
        assertEquals("see ", CustomImageDataset.textOf(samples.get(1).chatContent()));
    }

    @Test
    public void customAudioReadsItsFileAndRecordedLength() throws IOException {
        Path dir = tmp.newFolder("audio").toPath();
        Files.write(dir.resolve("clip.wav"), wav(0.1));
        Path file = dir.resolve("audio.jsonl");
        Files.write(file, "{\"prompt\":\"hi\",\"audio\":\"clip.wav\",\"output_tokens\":7}\n"
                .getBytes(StandardCharsets.UTF_8));
        ServeBenchOptions opts = options("custom_audio", file).numPrompts(1).customOutputLen(-1).build();
        List<BenchSample> samples = new CustomAudioDataset(opts, DECODING).sample();
        assertEquals(1, samples.size());
        assertEquals(BenchSample.Kind.AUDIO, samples.get(0).kind());
        assertEquals(0.1, samples.get(0).audioDurationSec(), 0.0001);
        assertEquals(7, samples.get(0).outputLen());
        assertEquals(2, samples.get(0).promptLen());
    }

    @Test
    public void asrSkipsAudioOutsideItsDurationRange() throws IOException {
        Path dir = tmp.newFolder("asr").toPath();
        Files.write(dir.resolve("short.wav"), wav(0.1));
        Files.write(dir.resolve("long.wav"), wav(0.5));
        Files.write(dir.resolve("data.jsonl"), ("{\"audio\":\"short.wav\",\"text\":\"a\"}\n"
                + "{\"audio\":\"long.wav\",\"text\":\"b\"}\n").getBytes(StandardCharsets.UTF_8));
        ServeBenchOptions opts = options("asr", dir).numPrompts(1)
                .asrAudioLen(0.0, 0.2).build();
        List<BenchSample> samples = new AsrDataset(opts, DECODING).sample();
        assertEquals(1, samples.size());
        assertEquals(1024, samples.get(0).outputLen());
    }

    @Test
    public void randomRerankBatchesItsDocuments() {
        ServeBenchOptions opts = ServeBenchOptions.builder().datasetName("random-rerank")
                .numPrompts(4).seed(1).randomInputLen(10).randomBatchSize(1)
                .requestIdPrefix("rr-").build();
        List<BenchSample> samples = new RandomRerankDataset(opts, DECODING, 100, new int[]{0}).sample();
        assertEquals(4, samples.size());
        for (BenchSample sample : samples) {
            assertEquals(BenchSample.Kind.RERANK, sample.kind());
            assertEquals(2, sample.rerankTexts().size());
            assertFalse(sample.rerankTexts().get(0).isEmpty());
            assertEquals("rr-", sample.requestId().substring(0, 3));
        }
    }

    @Test
    public void randomMmAttachesOneSyntheticImage() {
        Map<String, Double> buckets = new LinkedHashMap<>();
        buckets.put("2,2,1", 1.0);
        ServeBenchOptions opts = ServeBenchOptions.builder().datasetName("random-mm")
                .numPrompts(2).seed(1).randomInputLen(4).randomOutputLen(2)
                .randomMmBucketConfig(buckets).requestIdPrefix("mm-").build();
        List<BenchSample> samples = new RandomMultiModalDataset(opts, DECODING, 100, new int[]{0}).sample();
        assertEquals(2, samples.size());
        assertEquals(BenchSample.Kind.CHAT, samples.get(0).kind());
        assertEquals(2, samples.get(0).chatContent().size());
        assertEquals("image_url", ((Map<?, ?>) samples.get(0).chatContent().get(1)).get("type"));
        assertEquals(4, samples.get(0).promptLen());
        assertEquals(2, samples.get(0).outputLen());
    }

    @Test
    public void randomMmRefusesVideoBucketsByName() {
        Map<String, Double> buckets = new LinkedHashMap<>();
        buckets.put("2,2,4", 1.0);
        ServeBenchOptions opts = ServeBenchOptions.builder().datasetName("random-mm").numPrompts(1)
                .randomMmBucketConfig(buckets).build();
        try {
            new RandomMultiModalDataset(opts, DECODING, 10, new int[]{}).sample();
            fail("video buckets need an MP4 encoder");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("video"));
        }
    }

    @Test
    public void theLocalTokenizerLoadsAGgufAndDecodes() throws IOException {
        Path model = com.eonmux.neuralfree.gguf.testutil.SyntheticModels.llama().writeTempFile();
        LocalBenchTokenizer tokenizer = LocalBenchTokenizer.open(model.toString());
        assertTrue(tokenizer.vocabSize() > 0);
        int[] tokens = tokenizer.tokenize("hello");
        assertTrue(tokens.length > 0);
        assertNotNull(tokenizer.detokenize(tokens));
        assertTrue(tokenizer.specialIds().length > 0);
    }

    @Test
    public void theLocalTokenizerLoadsAHuggingFaceDirectory() throws IOException {
        Path dir = com.eonmux.neuralfree.gguf.testutil.SyntheticHf.llama(1, 64, 4, 2, 320)
                .write(tmp.newFolder("hfmodel").toPath());
        LocalBenchTokenizer tokenizer = LocalBenchTokenizer.open(dir.toString());
        assertTrue(tokenizer.vocabSize() > 0);
        assertTrue(tokenizer.tokenize("ab").length > 0);
    }

    /** A PCM WAVE file of {@code seconds} seconds at 8 kHz, mono, 16 bits. */
    private static byte[] wav(double seconds) {
        int sampleRate = 8000;
        int dataSize = (int) Math.round(seconds * sampleRate) * 2;
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(44 + dataSize)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.put("RIFF".getBytes(StandardCharsets.US_ASCII));
        buffer.putInt(36 + dataSize);
        buffer.put("WAVE".getBytes(StandardCharsets.US_ASCII));
        buffer.put("fmt ".getBytes(StandardCharsets.US_ASCII));
        buffer.putInt(16);
        buffer.putShort((short) 1);
        buffer.putShort((short) 1);
        buffer.putInt(sampleRate);
        buffer.putInt(sampleRate * 2);
        buffer.putShort((short) 2);
        buffer.putShort((short) 16);
        buffer.put("data".getBytes(StandardCharsets.US_ASCII));
        buffer.putInt(dataSize);
        buffer.put(new byte[dataSize]);
        return buffer.array();
    }
}
