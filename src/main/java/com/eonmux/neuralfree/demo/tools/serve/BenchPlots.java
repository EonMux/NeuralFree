package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The plots the reference writes beside a benchmark result, as self-contained files with no
 * plotting library: the request timeline ({@code <result>.timeline.html}, plotly's timeline in a
 * plain inline SVG) with one row per request and the TTFT and ITL segments coloured by the two ITL
 * thresholds, and the dataset statistics ({@code <result>.dataset_stats.svg}, matplotlib's four
 * panels: prompt, output and total token histograms and a stacked per-request bar chart).
 */
final class BenchPlots {

    private static final int WIDTH = 1200;
    private static final int LEFT = 130;
    private static final int RIGHT = 40;
    private static final int TOP = 80;
    private static final int BOTTOM = 60;

    private BenchPlots() {
    }

    /**
     * Writes the timeline of {@code outputs} at {@code file}.
     *
     * @param thresholdsMs the two ITL thresholds in milliseconds, lower first
     * @throws IOException when the file cannot be written
     */
    static void timeline(List<BenchRequestOutput> outputs, Path file, List<Double> thresholdsMs)
            throws IOException {
        double low = thresholdsMs.get(0) / 1000.0;
        double high = thresholdsMs.get(1) / 1000.0;
        List<TimelineRequest> requests = new ArrayList<>();
        double t0 = Double.POSITIVE_INFINITY;
        for (BenchRequestOutput output : outputs) {
            if (output.success) {
                t0 = Math.min(t0, output.startTime);
            }
        }
        if (t0 == Double.POSITIVE_INFINITY) {
            t0 = 0.0;
        }
        double maxEnd = 0.0;
        for (BenchRequestOutput output : outputs) {
            if (!output.success) {
                continue;
            }
            double start = output.startTime - t0;
            maxEnd = Math.max(maxEnd, start + Math.max(output.latency, output.ttft));
            requests.add(new TimelineRequest(start, output.ttft, new ArrayList<>(output.itls)));
        }
        if (requests.isEmpty()) {
            Files.createDirectories(file.toAbsolutePath().getParent());
            Files.write(file, ("<!DOCTYPE html><html><body><p>No timeline data to plot</p></body></html>")
                    .getBytes(StandardCharsets.UTF_8));
            return;
        }
        int rowHeight = 12;
        int height = TOP + BOTTOM + requests.size() * rowHeight;
        double scale = WIDTH - LEFT - RIGHT;
        double span = Math.max(maxEnd, 1e-9);
        String lowLabel = "ITL < " + String.format(Locale.ROOT, "%.0f", thresholdsMs.get(0)) + "ms";
        String midLabel = String.format(Locale.ROOT, "%.0f", thresholdsMs.get(0)) + "ms ≤ ITL < "
                + String.format(Locale.ROOT, "%.0f", thresholdsMs.get(1)) + "ms";
        String highLabel = "ITL ≥ " + String.format(Locale.ROOT, "%.0f", thresholdsMs.get(1)) + "ms";

        StringBuilder svg = new StringBuilder(64 * 1024);
        svg.append("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"").append(WIDTH)
                .append("\" height=\"").append(height).append("\" viewBox=\"0 0 ").append(WIDTH)
                .append(' ').append(height).append("\" font-family=\"sans-serif\" font-size=\"11\">\n");
        svg.append("<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n");
        svg.append(text(LEFT, 28, "Request Timeline", 16, "#222"));
        legend(svg, LEFT, 48, "#636EFA", "TTFT");
        legend(svg, LEFT + 90, 48, "#109618", lowLabel);
        legend(svg, LEFT + 250, 48, "#FF7F0E", midLabel);
        legend(svg, LEFT + 450, 48, "#D62728", highLabel);
        // time axis
        int ticks = 6;
        for (int i = 0; i <= ticks; i++) {
            double value = span * i / ticks;
            double x = LEFT + scale * i / ticks;
            svg.append(line(x, TOP - 6, x, height - BOTTOM, "#e5e5e5"));
            svg.append(text(x - 30, height - BOTTOM + 18, clock(value), 10, "#555"));
        }
        svg.append(text(LEFT, height - BOTTOM + 38, "Time", 11, "#555"));
        for (int row = 0; row < requests.size(); row++) {
            TimelineRequest request = requests.get(row);
            double y = TOP + row * rowHeight;
            svg.append(text(6, y + rowHeight - 3, "Req " + row, 10, "#333"));
            double x = LEFT + scale * request.start / span;
            double w = Math.max(0.5, scale * request.ttft / span);
            svg.append(rect(x, y, w, rowHeight - 1, "#636EFA"));
            double cursor = request.start + request.ttft;
            for (double itl : request.itls) {
                double segment = LEFT + scale * cursor / span;
                double width = Math.max(0.5, scale * itl / span);
                String colour = itl < low ? "#109618" : itl < high ? "#FF7F0E" : "#D62728";
                svg.append(rect(segment, y, width, rowHeight - 1, colour));
                cursor += itl;
            }
        }
        svg.append("</svg>\n");
        String html = "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                + "<title>Request Timeline</title>\n<style>body{margin:0;padding:16px;"
                + "background:#fff;color:#222;font-family:sans-serif}svg{max-width:100%;height:auto}"
                + "</style>\n</head>\n<body>\n" + svg + "</body>\n</html>\n";
        Files.createDirectories(file.toAbsolutePath().getParent());
        Files.write(file, html.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Writes the dataset statistics of {@code outputs} at {@code file}.
     *
     * @throws IOException when the file cannot be written
     */
    static void datasetStats(List<BenchRequestOutput> outputs, Path file) throws IOException {
        List<Integer> prompts = new ArrayList<>();
        List<Integer> generated = new ArrayList<>();
        List<Integer> totals = new ArrayList<>();
        for (BenchRequestOutput output : outputs) {
            int prompt = output.promptLen;
            int completion = output.success ? output.outputLenOrChunkCount() : 0;
            prompts.add(prompt);
            generated.add(completion);
            totals.add(prompt + completion);
        }
        int width = 1200;
        int height = 900;
        StringBuilder svg = new StringBuilder(64 * 1024);
        svg.append("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"").append(width)
                .append("\" height=\"").append(height).append("\" viewBox=\"0 0 ").append(width)
                .append(' ').append(height).append("\" font-family=\"sans-serif\" font-size=\"11\">\n");
        svg.append("<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n");
        histogram(svg, 40, 30, 540, 380, prompts, "Prompt Tokens", "Prompt Tokens Distribution",
                "#4682B4");
        histogram(svg, 620, 30, 540, 380, generated, "Output Tokens", "Output Tokens Distribution",
                "#FF7F50");
        histogram(svg, 40, 470, 540, 380, totals, "Total Tokens (Prompt + Output)",
                "Total Tokens Distribution", "#3CB371");
        stackedBars(svg, 620, 470, 540, 380, prompts, generated);
        svg.append("</svg>\n");
        Files.createDirectories(file.toAbsolutePath().getParent());
        Files.write(file, svg.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static void histogram(StringBuilder svg, int x, int y, int width, int height,
                                  List<Integer> values, String axis, String title, String colour) {
        svg.append(text(x, y + 16, title, 13, "#222"));
        int plotX = x + 50;
        int plotY = y + 30;
        int plotWidth = width - 70;
        int plotHeight = height - 70;
        svg.append(line(plotX, plotY, plotX, plotY + plotHeight, "#999"));
        svg.append(line(plotX, plotY + plotHeight, plotX + plotWidth, plotY + plotHeight, "#999"));
        if (values.isEmpty()) {
            return;
        }
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int value : values) {
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        int bins = 30;
        int span = Math.max(1, max - min + 1);
        int[] counts = new int[bins];
        for (int value : values) {
            int bin = (int) ((long) (value - min) * bins / span);
            counts[Math.min(bins - 1, Math.max(0, bin))]++;
        }
        int peak = 1;
        for (int count : counts) {
            peak = Math.max(peak, count);
        }
        double barWidth = (double) plotWidth / bins;
        for (int i = 0; i < bins; i++) {
            double barHeight = (double) plotHeight * counts[i] / peak;
            svg.append(rect(plotX + i * barWidth, plotY + plotHeight - barHeight,
                    Math.max(0.5, barWidth - 1), barHeight, colour));
        }
        svg.append(text(plotX, plotY + plotHeight + 18, String.valueOf(min), 10, "#555"));
        svg.append(text(plotX + plotWidth - 30, plotY + plotHeight + 18, String.valueOf(max), 10, "#555"));
        svg.append(text(plotX + plotWidth / 2 - axis.length() * 3, plotY + plotHeight + 34, axis,
                10, "#555"));
        svg.append(text(plotX - 40, plotY + 10, String.valueOf(peak), 10, "#555"));
    }

    private static void stackedBars(StringBuilder svg, int x, int y, int width, int height,
                                    List<Integer> prompts, List<Integer> generated) {
        svg.append(text(x, y + 16, "Tokens per Request (Stacked)", 13, "#222"));
        int plotX = x + 50;
        int plotY = y + 30;
        int plotWidth = width - 70;
        int plotHeight = height - 70;
        svg.append(line(plotX, plotY, plotX, plotY + plotHeight, "#999"));
        svg.append(line(plotX, plotY + plotHeight, plotX + plotWidth, plotY + plotHeight, "#999"));
        if (prompts.isEmpty()) {
            return;
        }
        int peak = 1;
        for (int i = 0; i < prompts.size(); i++) {
            peak = Math.max(peak, prompts.get(i) + generated.get(i));
        }
        double barWidth = (double) plotWidth / prompts.size();
        for (int i = 0; i < prompts.size(); i++) {
            double promptHeight = (double) plotHeight * prompts.get(i) / peak;
            double outputHeight = (double) plotHeight * generated.get(i) / peak;
            double left = plotX + i * barWidth;
            svg.append(rect(left, plotY + plotHeight - promptHeight, Math.max(0.5, barWidth - 0.5),
                    promptHeight, "#4682B4"));
            svg.append(rect(left, plotY + plotHeight - promptHeight - outputHeight,
                    Math.max(0.5, barWidth - 0.5), outputHeight, "#FF7F50"));
        }
        svg.append(rect(plotX, plotY + 4, 10, 10, "#4682B4"));
        svg.append(text(plotX + 16, plotY + 13, "Prompt Tokens", 10, "#333"));
        svg.append(rect(plotX + 120, plotY + 4, 10, 10, "#FF7F50"));
        svg.append(text(plotX + 136, plotY + 13, "Output Tokens", 10, "#333"));
        svg.append(text(plotX, plotY + plotHeight + 18, "0", 10, "#555"));
        svg.append(text(plotX + plotWidth - 30, plotY + plotHeight + 18,
                String.valueOf(prompts.size() - 1), 10, "#555"));
        svg.append(text(plotX + plotWidth / 2 - 30, plotY + plotHeight + 34, "Request ID", 10, "#555"));
        svg.append(text(plotX - 40, plotY + 10, String.valueOf(peak), 10, "#555"));
    }

    private static void legend(StringBuilder svg, int x, int y, String colour, String label) {
        svg.append(rect(x, y - 9, 10, 10, colour));
        svg.append(text(x + 15, y, label, 11, "#333"));
    }

    private static String rect(double x, double y, double width, double height, String fill) {
        return "<rect x=\"" + f(x) + "\" y=\"" + f(y) + "\" width=\"" + f(Math.max(0.0, width))
                + "\" height=\"" + f(Math.max(0.0, height)) + "\" fill=\"" + fill + "\"/>\n";
    }

    private static String line(double x1, double y1, double x2, double y2, String stroke) {
        return "<line x1=\"" + f(x1) + "\" y1=\"" + f(y1) + "\" x2=\"" + f(x2) + "\" y2=\"" + f(y2)
                + "\" stroke=\"" + stroke + "\" stroke-width=\"1\"/>\n";
    }

    private static String text(double x, double y, String value, int size, String fill) {
        return "<text x=\"" + f(x) + "\" y=\"" + f(y) + "\" font-size=\"" + size + "\" fill=\"" + fill
                + "\">" + escape(value) + "</text>\n";
    }

    private static String escape(String value) {
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
    }

    private static String f(double value) {
        return String.format(Locale.ROOT, "%.1f", value);
    }

    /** {@code HH:MM:SS.mmm}, the reference's timeline label. */
    static String clock(double seconds) {
        long totalMillis = Math.round(seconds * 1000.0);
        long hours = totalMillis / 3_600_000L;
        long minutes = totalMillis / 60_000L % 60;
        long secs = totalMillis / 1000L % 60;
        long millis = totalMillis % 1000L;
        return String.format(Locale.ROOT, "%02d:%02d:%02d.%03d", hours, minutes, secs, millis);
    }

    /** One request's timeline: its normalized start, TTFT and ITLs. */
    private static final class TimelineRequest {

        final double start;
        final double ttft;
        final List<Double> itls;

        TimelineRequest(double start, double ttft, List<Double> itls) {
            this.start = start;
            this.ttft = ttft;
            this.itls = itls;
        }
    }
}
