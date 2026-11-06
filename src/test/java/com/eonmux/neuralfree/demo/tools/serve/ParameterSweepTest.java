package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The sweep's parameter records: both file shapes, the name derivation, the flag replacement of
 * {@code applyToCmd} (booleans toggled, objects as JSON) and the uniqueness the reference demands.
 */
public class ParameterSweepTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path write(String name, String content) throws IOException {
        Path file = tmp.newFile(name).toPath();
        Files.write(file, content.getBytes(StandardCharsets.UTF_8));
        return file;
    }

    private static ParameterSweep.Item item(Path file, int index) throws IOException {
        return ParameterSweep.readJson(file).items().get(index);
    }

    @Test
    public void aListFileCarriesItsRecordsInOrder() throws IOException {
        Path file = write("bench.json", "[{\"num-prompts\":4},{\"num-prompts\":8}]");
        List<ParameterSweep.Item> items = ParameterSweep.readJson(file).items();
        assertEquals(2, items.size());
        assertEquals("num-prompts=4", items.get(0).name());
        assertEquals(4, ((Number) items.get(0).valueOf("num-prompts")).intValue());
    }

    @Test
    public void anObjectFileNamesEveryRecord() throws IOException {
        Path file = write("bench.json", "{\"small\":{\"num-prompts\":4},\"large\":{\"num-prompts\":8}}");
        List<ParameterSweep.Item> items = ParameterSweep.readJson(file).items();
        assertEquals("small", items.get(0).name());
        assertEquals("large", items.get(1).name());
        assertEquals(4, ((Number) items.get(0).valueOf("num-prompts")).intValue());
        assertEquals(8, ((Number) items.get(1).valueOf("num-prompts")).intValue());
    }

    @Test
    public void duplicateNamesAreRefused() throws IOException {
        // a JSON object cannot repeat a key, so the duplicates carry an explicit _benchmark_name
        Path list = write("bench2.json", "[{\"_benchmark_name\":\"same\",\"num-prompts\":4},"
                + "{\"_benchmark_name\":\"same\",\"num-prompts\":8}]");
        try {
            ParameterSweep.readJson(list);
            fail("duplicate benchmark names must be refused");
        } catch (IOException expected) {
            assertTrue(expected.getMessage().contains("duplicate"));
        }
    }

    @Test
    public void applyToCmdReplacesExistingValuesAndAppendsNewFlags() {
        ParameterSweep.Item record = new ParameterSweep.Item(new java.util.LinkedHashMap<>());
        java.util.Map<String, Object> params = new java.util.LinkedHashMap<>();
        params.put("num-prompts", 8);
        params.put("label", "run");
        params.put("save-result", true);
        params.put("disable-shuffle", false);
        java.util.Map<String, Object> ratio = new java.util.LinkedHashMap<>();
        ratio.put("input", 0.1);
        ratio.put("output", 0.2);
        params.put("random-range-ratio", ratio);
        ParameterSweep.Item with = new ParameterSweep.Item(params);
        List<String> cmd = with.applyToCmd(Arrays.asList(
                "bench-serve", "--num-prompts", "4", "--label", "old", "--save-result"));
        assertEquals("bench-serve", cmd.get(0));
        assertTrue(cmd.contains("--num-prompts"));
        assertEquals("8", cmd.get(cmd.indexOf("--num-prompts") + 1));
        assertEquals("run", cmd.get(cmd.indexOf("--label") + 1));
        assertTrue("the boolean stays a bare flag", cmd.contains("--save-result"));
        assertTrue("a false boolean negates its flag", cmd.contains("--no-disable-shuffle"));
        assertEquals("{\"input\": 0.1, \"output\": 0.2}",
                cmd.get(cmd.indexOf("--random-range-ratio") + 1));
        assertFalse("the old label is replaced", cmd.contains("old"));
        assertNull(record.valueOf("anything"));
    }

    @Test
    public void anItemUnderstandsBothSpellings() throws IOException {
        Path file = write("bench.json", "[{\"_benchmark_name\":\"n\",\"num_prompts\":5}]");
        ParameterSweep.Item item = item(file, 0);
        assertEquals("n", item.name());
        assertTrue(item.hasParam("num-prompts"));
        assertTrue(item.hasParam("num_prompts"));
        assertEquals(5, ((Number) item.valueOf("num-prompts")).intValue());
        assertEquals(5, ((Number) item.valueOf("num_prompts")).intValue());
    }

    @Test
    public void aWrongShapeIsRefused() throws IOException {
        Path notJson = write("bad.json", "not json");
        try {
            ParameterSweep.readJson(notJson);
            fail("the sweep file must be JSON");
        } catch (IOException expected) {
            assertTrue(expected.getMessage().contains("not JSON"));
        }
        Path wrong = write("wrong.json", "42");
        try {
            ParameterSweep.readJson(wrong);
            fail("the sweep file must be a list or an object");
        } catch (IOException expected) {
            assertTrue(expected.getMessage().contains("list of objects"));
        }
        Path badEntry = write("bad-entry.json", "[42]");
        try {
            ParameterSweep.readJson(badEntry);
            fail("every record must be an object");
        } catch (IOException expected) {
            assertTrue(expected.getMessage().contains("must be an object"));
        }
    }

    @Test
    public void theEmptySweepIsOneBareRecord() {
        List<ParameterSweep.Item> items = ParameterSweep.none().items();
        assertEquals(1, items.size());
        assertEquals("", items.get(0).name());
        assertEquals(Arrays.asList("bench-serve"), items.get(0).applyToCmd(Arrays.asList("bench-serve")));
    }
}
