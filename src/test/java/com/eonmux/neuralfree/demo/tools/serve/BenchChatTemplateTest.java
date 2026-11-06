package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The client-side chat template: the local tokenizer renders the model's template (ChatML for the
 * synthetic llama), the datasets the reference renders client-side send the rendered prompt, and
 * {@code --skip-chat-template} sends the raw one.
 */
public class BenchChatTemplateTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static LocalBenchTokenizer tokenizer() throws IOException {
        Path model = com.eonmux.neuralfree.gguf.testutil.SyntheticModels.llama().writeTempFile();
        return LocalBenchTokenizer.open(model.toString());
    }

    @Test
    public void theLocalTokenizerRendersTheModelsTemplate() throws IOException {
        LocalBenchTokenizer tokenizer = tokenizer();
        BenchChatTemplate template = tokenizer.chatTemplate();
        assertNotNull(template);
        String rendered = template.render("hello", Collections.singletonMap("enable_thinking", false));
        assertTrue(rendered, rendered.contains("hello"));
        assertTrue("the chat template wraps the prompt in its turn markers: " + rendered,
                rendered.contains("<|im_start|>") || rendered.contains("user"));
        assertFalse("the raw prompt is no longer the whole text", rendered.equals("hello"));
    }

    @Test
    public void theCustomDatasetRendersUnlessSkipped() throws IOException {
        LocalBenchTokenizer tokenizer = tokenizer();
        Path file = tmp.newFile("custom.jsonl").toPath();
        Files.write(file, "{\"prompt\":\"hello world\"}\n".getBytes(StandardCharsets.UTF_8));
        ServeBenchOptions rendered = ServeBenchOptions.builder().datasetName("custom")
                .datasetPath(file.toString()).numPrompts(1).customOutputLen(4).seed(1)
                .requestIdPrefix("r-").build();
        BenchSample renderedSample = new CustomDataset(rendered, tokenizer).sample().get(0);
        assertTrue(renderedSample.text(), renderedSample.text().contains("hello world"));
        assertTrue("the prompt is wrapped by the template",
                renderedSample.text().length() > "hello world".length());
        assertTrue(renderedSample.promptLen() > 0);

        ServeBenchOptions raw = ServeBenchOptions.builder().datasetName("custom")
                .datasetPath(file.toString()).numPrompts(1).customOutputLen(4).seed(1)
                .skipChatTemplate(true).requestIdPrefix("r-").build();
        assertEquals("hello world", new CustomDataset(raw, tokenizer).sample().get(0).text());
    }

    @Test
    public void theHfPromptRowRendersThroughTheTemplate() throws IOException {
        LocalBenchTokenizer tokenizer = tokenizer();
        Path file = tmp.newFile("hf.jsonl").toPath();
        Files.write(file, "{\"prompt\":\"write code\",\"output_tokens\":8}\n"
                .getBytes(StandardCharsets.UTF_8));
        ServeBenchOptions opts = ServeBenchOptions.builder().datasetName("hf")
                .datasetPath(file.toString()).numPrompts(1).seed(1).requestIdPrefix("h-").build();
        BenchSample sample = new HfDataset(opts, tokenizer).sample().get(0);
        assertTrue(sample.text(), sample.text().contains("write code"));
        assertTrue("HumanEval-like prompt rows apply the template",
                sample.text().length() > "write code".length());
        assertEquals(8, sample.outputLen());
    }

    @Test
    public void theHfConversationStaysRawAsTheReferenceKeepsIt() throws IOException {
        LocalBenchTokenizer tokenizer = tokenizer();
        Path file = tmp.newFile("hf-conv.jsonl").toPath();
        Files.write(file, ("{\"conversations\":[{\"from\":\"human\",\"value\":\"hello\"},"
                + "{\"from\":\"gpt\",\"value\":\"world!\"}]}\n").getBytes(StandardCharsets.UTF_8));
        ServeBenchOptions opts = ServeBenchOptions.builder().datasetName("hf")
                .datasetPath(file.toString()).numPrompts(1).seed(1).requestIdPrefix("h-").build();
        List<BenchSample> samples = new HfDataset(opts, tokenizer).sample();
        assertEquals("hello", samples.get(0).text());
    }
}
