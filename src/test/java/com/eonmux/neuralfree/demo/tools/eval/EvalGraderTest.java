package com.eonmux.neuralfree.demo.tools.eval;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The three graders: the regex extraction rules per suite, the last-ten-lines truncation, the LLM
 * grader's extraction request, and the CLI grader's exit status.
 */
public class EvalGraderTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void gsm8kTakesTheLastNumberOfTheReply() {
        EvalGrader grader = EvalGrader.regex("gsm8k");
        EvalGrader.Grade grade = grader.grade("42", "She had 3, then 7.\nThe answer is 42.", "");
        assertTrue(grade.correct());
        assertEquals("42", grade.answer());
    }

    @Test
    public void aimePrefersTheLastBoxedAnswer() {
        EvalGrader grader = EvalGrader.regex("aime");
        EvalGrader.Grade grade = grader.grade("123", "First \\boxed{1}, then \\boxed{123}, and 7.", "");
        assertTrue(grade.correct());
        assertEquals("123", grade.answer());
    }

    @Test
    public void aimeFallsBackToTheLastNumber() {
        EvalGrader grader = EvalGrader.regex("aime");
        EvalGrader.Grade grade = grader.grade("456", "The value is 12 and finally 456.", "");
        assertTrue(grade.correct());
        assertEquals("456", grade.answer());
    }

    @Test
    public void aime2026UsesThePatternInsteadOfTheBoxedPriority() {
        EvalGrader grader = EvalGrader.regex("aime2026");
        EvalGrader.Grade grade = grader.grade("9", "First \\boxed{5}, then 9.", "");
        assertTrue(grade.correct());
        assertEquals("9", grade.answer());
    }

    @Test
    public void aReplyWithoutANumberIsIncorrectWithNoAnswer() {
        EvalGrader.Grade grade = EvalGrader.regex("gsm8k").grade("42", "I cannot tell.", "");
        assertFalse(grade.correct());
        assertNull(grade.answer());
    }

    @Test
    public void gradingKeepsOnlyTheLastTenLines() {
        StringBuilder response = new StringBuilder();
        for (int i = 0; i < 20; i++) {
            response.append("line ").append(i).append('\n');
        }
        response.append("the answer is 42");
        String graded = EvalGrader.truncate(response.toString());
        assertEquals(10, graded.split("\n", -1).length);
        assertTrue(graded.endsWith("the answer is 42"));
        assertTrue(graded.startsWith("line 11"));
        assertFalse(graded.contains("line 10\n"));
    }

    @Test
    public void theLlmGraderAsksForExtractionAndComparesCaseInsensitively() throws Exception {
        try (FakeEvalServer server = new FakeEvalServer()) {
            server.graderContent = "  FORTY-TWO  ";
            EvalServer endpoint = new EvalServer(server.baseUrl(), 1, "grader", null);
            EvalClient client = new EvalClient(endpoint, "grader-model", 0, 0.0, null);
            EvalGrader grader = EvalGrader.llm("aime", client, "grader-model");
            EvalGrader.Grade grade = grader.grade("forty-two", "The reply.", "the prompt");
            assertTrue(grade.correct());
            assertEquals("FORTY-TWO", grade.answer());

            List<Map<String, Object>> bodies = server.bodies();
            assertEquals(1, bodies.size());
            assertEquals("grader-model", bodies.get(0).get("model"));
            assertEquals(0, ((Number) bodies.get(0).get("temperature")).intValue());
            List<?> messages = (List<?>) bodies.get(0).get("messages");
            assertEquals(2, messages.size());
            assertEquals("system", ((Map<?, ?>) messages.get(0)).get("role"));
            assertTrue(String.valueOf(((Map<?, ?>) messages.get(0)).get("content"))
                    .contains("answer extraction system"));
            assertEquals("user", ((Map<?, ?>) messages.get(1)).get("role"));
            assertTrue(String.valueOf(((Map<?, ?>) messages.get(1)).get("content"))
                    .contains("The reply."));
        }
    }

    @Test
    public void anLlmGraderFailureScoresTheCaseWrong() throws Exception {
        try (FakeEvalServer server = new FakeEvalServer()) {
            server.status = 500;
            EvalServer endpoint = new EvalServer(server.baseUrl(), 1, "grader", null);
            EvalGrader grader = EvalGrader.llm("gsm8k", new EvalClient(endpoint, "m", 0, 0.0, null),
                    "m");
            EvalGrader.Grade grade = grader.grade("42", "42", "");
            assertFalse(grade.correct());
            assertNull(grade.answer());
        }
    }

    @Test
    public void theCliGraderUsesTheScriptsExitStatus() throws Exception {
        File script = tmp.newFile("grade.sh");
        Files.write(script.toPath(), ("#!/bin/sh\n[ \"$2\" = \"$4\" ]\n")
                .getBytes(StandardCharsets.UTF_8));
        Files.setPosixFilePermissions(script.toPath(), PosixFilePermissions.fromString("rwxr-xr-x"));
        EvalGrader grader = EvalGrader.cli("gsm8k", script.getAbsolutePath());
        assertTrue(grader.grade("42", "42", "").correct());
        assertFalse(grader.grade("42", "41", "").correct());
    }

    @Test
    public void theLlmGraderNeedsAModelAndTheCliGraderAScript() {
        try {
            EvalGrader.llm("gsm8k", null, "");
            fail("an LLM grader without a model must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("--grader-model"));
        }
        try {
            EvalGrader.cli("gsm8k", null);
            fail("a CLI grader without a script must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("--grader-script"));
        }
    }
}
