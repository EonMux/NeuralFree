package com.eonmux.neuralfree.demo.tools.eval;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The math suites' row shaping: the gold answer's normalization, the prompt templates, and the
 * refusal of a row that lacks the field its suite requires.
 */
public class EvalDatasetsTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void gsm8kReadsTheQuestionAndTheAnswerAfterTheMarker() throws Exception {
        File file = jsonl("gsm8k.jsonl",
                "{\"question\":\"Ann has 2 apples and buys 3.\",\"answer\":\"2 + 3 = 5\\n#### 42\"}\n"
                        + "{\"question\":\"A plain one.\",\"answer\":\"#### 7\"}\n");
        EvalDataset dataset = EvalDatasets.load("gsm8k", file.getAbsolutePath());
        assertEquals(2, dataset.size());
        EvalQuestion first = dataset.question(0);
        assertEquals("Ann has 2 apples and buys 3.", first.text());
        assertEquals("42", first.expected());
        assertEquals("Ann has 2 apples and buys 3.\nPlease reason step by step, and put your final"
                + " numeric answer within \\boxed{} without any extra characters.\n", first.prompt());
        assertEquals("7", dataset.question(1).expected());
    }

    @Test
    public void gsm8kUsesAProblemColumnAndAGoldColumnWhenTheRowHasThem() throws Exception {
        File file = jsonl("gsm8k-problem.jsonl",
                "{\"problem\":\"How many?\",\"answer\":\"ignored\",\"gold\":\"012\"}\n");
        EvalQuestion question = EvalDatasets.load("gsm8k", file.getAbsolutePath()).question(0);
        assertEquals("How many?", question.text());
        assertEquals("012", question.expected());
    }

    @Test
    public void aimeNormalizesItsAnswerAndRendersTheBoxedPrompt() throws Exception {
        File file = jsonl("aime.jsonl",
                "{\"problem\":\"Find x.\",\"answer\":\"042\"}\n"
                        + "{\"problem\":\"Find y.\",\"answer\":17}\n");
        EvalDataset dataset = EvalDatasets.load("aime", file.getAbsolutePath());
        assertEquals("42", dataset.question(0).expected());
        assertEquals("17", dataset.question(1).expected());
        assertEquals("Solve the following math problem step by step. Put your answer inside"
                + " \\boxed{}.\n\nFind x.\n\nRemember to put your answer inside \\boxed{}.\n",
                dataset.question(0).prompt());
    }

    @Test
    public void aime2025ReadsItsQuestionColumn() throws Exception {
        File file = jsonl("aime2025.jsonl", "{\"question\":\"Part one?\",\"answer\":\"9\"}\n");
        EvalQuestion question = EvalDatasets.load("aime2025", file.getAbsolutePath()).question(0);
        assertEquals("Part one?", question.text());
        assertEquals("9", question.expected());
    }

    @Test
    public void aime2026KeepsItsAnswerAsWrittenAndReadsTheProblem() throws Exception {
        File file = jsonl("aime2026.jsonl", "{\"problem\":\"New?\",\"answer\":\"007\"}\n");
        EvalQuestion question = EvalDatasets.load("aime2026", file.getAbsolutePath()).question(0);
        assertEquals("New?", question.text());
        assertEquals("007", question.expected());
    }

    @Test
    public void aMissingRequiredFieldIsRefusedByName() throws Exception {
        File file = jsonl("broken.jsonl", "{\"answer\":\"42\"}\n");
        try {
            EvalDatasets.load("gsm8k", file.getAbsolutePath());
            fail("a gsm8k row without a question must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("'question'"));
            assertTrue(expected.getMessage(), expected.getMessage().contains("row 1"));
        }
    }

    @Test
    public void aMissingAnswerIsRefusedByName() throws Exception {
        File file = jsonl("broken-aime.jsonl", "{\"problem\":\"No answer\"}\n");
        try {
            EvalDatasets.load("aime", file.getAbsolutePath());
            fail("an AIME row without an answer must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("'answer'"));
        }
    }

    @Test
    public void anUnknownSuiteIsRefused() {
        try {
            EvalDatasets.load("humaneval", null);
            fail("an unsupported suite must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("--dataset"));
        }
    }

    @Test
    public void anEmptyDatasetIsRefused() throws Exception {
        File file = jsonl("empty.jsonl", "\n");
        try {
            EvalDatasets.load("gsm8k", file.getAbsolutePath());
            fail("an empty dataset must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("no record"));
        }
    }

    @Test
    public void aLocalDatasetDirectoryIsReadInPathOrder() throws Exception {
        File dir = tmp.newFolder("tree");
        File first = new File(dir, "part-a.jsonl");
        File second = new File(dir, "part-b.jsonl");
        Files.write(first.toPath(), "{\"question\":\"one\",\"answer\":\"#### 1\"}\n"
                .getBytes(StandardCharsets.UTF_8));
        Files.write(second.toPath(), "{\"question\":\"two\",\"answer\":\"#### 2\"}\n"
                .getBytes(StandardCharsets.UTF_8));
        EvalDataset dataset = EvalDatasets.load("gsm8k", dir.getAbsolutePath());
        assertEquals(2, dataset.size());
        assertEquals("one", dataset.question(0).text());
        assertEquals("two", dataset.question(1).text());
    }

    @Test
    public void numbersAreNormalizedFromTheStartOnly() {
        assertEquals("7", EvalDatasets.normalizeNumber("007"));
        assertEquals("42", EvalDatasets.normalizeNumber("42abc"));
        assertNull(EvalDatasets.normalizeNumber(" 42"));
        assertNull(EvalDatasets.normalizeNumber("abc"));
        assertEquals("notanumber", EvalDatasets.normalizeOrSame("notanumber"));
    }

    private File jsonl(String name, String content) throws Exception {
        File file = tmp.newFile(name);
        Files.write(file.toPath(), content.getBytes(StandardCharsets.UTF_8));
        return file;
    }
}
