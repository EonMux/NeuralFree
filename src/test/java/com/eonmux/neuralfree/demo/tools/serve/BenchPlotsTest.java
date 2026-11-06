package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The self-contained plots: the timeline's TTFT and ITL segments with their threshold colours, the
 * dataset statistics' four panels, the clock labels, and the result-file suffix the runner derives.
 */
public class BenchPlotsTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static BenchRequestOutput output(double start, double ttft, double latency, double... itls) {
        BenchRequestOutput out = new BenchRequestOutput();
        out.success = true;
        out.startTime = start;
        out.ttft = ttft;
        out.latency = latency;
        out.promptLen = 7;
        out.outputTokens = itls.length + 1;
        for (double itl : itls) {
            out.itls.add(itl);
        }
        return out;
    }

    @Test
    public void theTimelineCarriesEverySegmentColourAndStaysSelfContained() throws IOException {
        Path file = tmp.newFile("run.timeline.html").toPath();
        List<BenchRequestOutput> outputs = Arrays.asList(
                output(0.1, 0.05, 0.20, 0.01, 0.04, 0.10),
                output(0.2, 0.02, 0.05, 0.01));
        BenchPlots.timeline(outputs, file, Arrays.asList(25.0, 50.0));
        String html = Files.readString(file);
        assertTrue(html, html.contains("<!DOCTYPE html>"));
        assertTrue(html, html.contains("<svg"));
        assertTrue(html, html.contains("Request Timeline"));
        assertTrue(html, html.contains("TTFT"));
        assertTrue(html, html.contains("ITL &lt; 25ms") || html.contains("ITL < 25ms"));
        assertTrue("a fast ITL is green", html.contains("#109618"));
        assertTrue("a middle ITL is orange", html.contains("#FF7F0E"));
        assertTrue("a slow ITL is red", html.contains("#D62728"));
        assertTrue("no external asset is referenced", !html.contains("src=") && !html.contains("href="));
    }

    @Test
    public void theDatasetStatsSvgCarriesTheFourPanels() throws IOException {
        Path file = tmp.newFile("run.dataset_stats.svg").toPath();
        BenchPlots.datasetStats(Arrays.asList(output(0.0, 0.1, 0.2, 0.05),
                output(0.1, 0.1, 0.2, 0.06)), file);
        String svg = Files.readString(file);
        assertTrue(svg, svg.startsWith("<svg"));
        assertTrue(svg, svg.contains("Prompt Tokens Distribution"));
        assertTrue(svg, svg.contains("Output Tokens Distribution"));
        assertTrue(svg, svg.contains("Total Tokens Distribution"));
        assertTrue(svg, svg.contains("Tokens per Request (Stacked)"));
    }

    @Test
    public void theClockLabelsHoursMinutesSecondsAndMillis() {
        assertEquals("00:00:00.000", BenchPlots.clock(0.0));
        assertEquals("00:00:01.500", BenchPlots.clock(1.5));
        assertEquals("01:02:03.004", BenchPlots.clock(3723.004));
    }

    @Test
    public void thePlotSuffixReplacesTheResultExtension() {
        Path result = Path.of("/tmp/openai-1qps-model-20260101-000000.json");
        assertEquals(Path.of("/tmp/openai-1qps-model-20260101-000000.timeline.html"),
                ServeBenchRunner.withSuffix(result, ".timeline.html"));
        assertEquals(Path.of("/tmp/openai-1qps-model-20260101-000000.dataset_stats.svg"),
                ServeBenchRunner.withSuffix(result, ".dataset_stats.svg"));
        assertEquals(Path.of("/tmp/run.timeline.html"),
                ServeBenchRunner.withSuffix(Path.of("/tmp/run"), ".timeline.html"));
    }
}
