package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool perplexity --winogrande} over a running engine: the CSV loader of perplexity.cpp (quote
 * state machine, malformed rows logged and skipped, the unterminated last line dropped), the partial
 * evaluation of the sentence tail after the choice, and the per-task lines with the running accuracy.
 */
public class WinoGrandeTest {

    private static EngineConfig config(int nCtx) {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(nCtx).ubatch(8)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    private static final String ROWS = ""
            + "0,Sarah was a much better surgeon than Maria so _ always got the easier cases.,Sarah,Maria,2\n"
            + "1,The trophy doesn't fit in the suitcase because _ is too large.,trophy,suitcase,1\n"
            + "2,John moved the couch from the garage to the backyard to create space. The _ is small.,garage,backyard,1\n";

    @Test
    public void theScoreRunsOverTheCsvTasks() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(128))) {
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            WinoGrande.score(m, 0, ROWS, out, err);
            String report = out.toString();
            assertTrue(err.toString(), err.toString().contains("loaded 3 tasks"));
            String[] lines = report.trim().split("\n");
            assertEquals("one line per task", 3, lines.length);
            String[] fields = lines[0].trim().split("\t");
            assertEquals("the task number leads", "1", fields[0]);
            assertTrue("the running accuracy in percent: " + lines[0], fields[1].matches("\\d+\\.\\d{4}"));
            String[] rest = fields[2].trim().split("\\s+");
            assertEquals("the two side scores, the choice and the gold answer", 4, rest.length);
            assertTrue("the chosen answer: " + lines[0], rest[2].matches("[12]"));
            assertTrue("the gold answer: " + lines[0], rest[3].matches("[12]"));
        }
    }

    @Test
    public void theTaskLimitDrawsTheReferenceTasks() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(128))) {
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            WinoGrande.score(m, 2, ROWS, out, err);
            assertEquals(2, out.toString().trim().split("\n").length);
            assertTrue(err.toString(), err.toString().contains("selecting 2 random tasks"));
        }
    }

    @Test
    public void malformedRowsAreLoggedAndSkipped() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(128))) {
            String csv = ""
                    + "0,no commas here\n"                                                    // too few commas
                    + "1,The sentence has no blank slot.,a,b,1\n"                              // no _
                    + "2,The _ slot is here.,a,b,not a number\n"                               // bad answer
                    + "3,The _ slot is here.,a,b,7\n"                                          // answer of 3
                    + "4,The _ slot is here.,a,b,1\n";                                         // usable
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            WinoGrande.score(m, 0, csv, out, err);
            String[] lines = out.toString().trim().split("\n");
            assertEquals("only the usable row is scored", 1, lines.length);
            assertEquals("the usable row is the last one, task 4", "1", lines[0].trim().split("\t")[0]);
            assertTrue(err.toString(), err.toString().contains("failed to find comma separators"));
            assertTrue(err.toString(), err.toString().contains("no _ in"));
            assertEquals(2, err.toString().split("failed to parse answer", -1).length - 1);
        }
    }

    @Test
    public void quotedSentenceFieldsKeepTheirCommas() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(128))) {
            String csv = "0,\"Ian volunteered to eat worms, because _ is brave.\",Ian,Donald,1\n";
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            WinoGrande.score(m, 0, csv, out, err);
            assertTrue("the quoted row parses despite its comma: " + err,
                    !err.toString().contains("failed") && !err.toString().contains("no _ in"));
            assertEquals(1, out.toString().trim().split("\n").length);
        }
    }

    @Test
    public void anUnterminatedLastLineIsDroppedLikeTheReference() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(128))) {
            StringBuilder out = new StringBuilder();
            WinoGrande.score(m, 0, ROWS + "9,orphan row without a newline.,a,b,1", out, new StringBuilder());
            assertEquals("the CSV loop's eof check drops the orphan", 3,
                    out.toString().trim().split("\n").length);
        }
    }

    @Test
    public void anEmptyDatafileIsRefused() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(128))) {
            IOException e = assertThrows(IOException.class,
                    () -> WinoGrande.score(m, 0, "", new StringBuilder(), new StringBuilder()));
            assertTrue(e.getMessage(), e.getMessage().contains("no usable tasks"));
        }
    }
}
