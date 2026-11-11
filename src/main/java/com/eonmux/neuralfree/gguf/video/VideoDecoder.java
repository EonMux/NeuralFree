package com.eonmux.neuralfree.gguf.video;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.image.ImageSize;
import com.eonmux.neuralfree.gguf.image.RasterImage;
import com.eonmux.neuralfree.gguf.media.FfmpegInput;
import com.eonmux.neuralfree.gguf.media.MediaInputException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

/**
 * Decoder of video bytes to sampled RGB frames and, for a family that hears it, the video's soundtrack, through
 * one external {@code ffprobe} and one {@code ffmpeg} process per stream, the decode path of llama.cpp's
 * {@code mtmd-helper} (tools/mtmd/mtmd-helper.cpp): the probe reports the stream's size, rate, frame count,
 * duration and start time; the decode streams raw RGB24 frames, of which the sampling plan's source indices are
 * kept. On a constant-rate stream that starts with its container ffmpeg's {@code select} filter picks the plan's
 * frames, so only they are converted, scaled and piped; another stream (a variable rate, or a video that starts
 * after its soundtrack, whose first frame the reference's decode repeats over the lead) pipes every frame up to
 * the plan's last and the reader keeps the plan's. When the vision preprocessor recommends a frame size, the decode scales the frames to it with the
 * reference's bicubic RGB filter ({@code mtmd_helper_video::start_ffmpeg}), so the kept frames never hold the
 * source's full resolution. The soundtrack of an LFM3-Omni video decodes to mono float32 at the audio tower's rate
 * and splits into the temporal groups' spans. The binaries are taken from the configured directory (llama.cpp's
 * {@code --video-ffmpeg-dir}) or the PATH, and are refused by name when missing. Immutable and stateless.
 */
public final class VideoDecoder {

    /** The bytes one decode keeps at most: an hour of 1920x1080 RGB frames. */
    private static final long MAX_FRAME_BYTES = 60L * 60 * 1920 * 1080 * 3;
    /** The wall-clock budget of one ffmpeg or ffprobe run. */
    private static final int PROCESS_TIMEOUT_SECONDS = 120;
    /**
     * The video scale filter of llama.cpp's mtmd helper ({@code mtmd_helper_video::start_ffmpeg}): RGB first, then
     * bicubic with the accurate rounding and full chroma interpolation that land closest to Pillow's resize.
     */
    private static final String SCALE_FILTER =
            "format=rgb24,scale=%d:%d:flags=bicubic+accurate_rnd+full_chroma_int:param0=0:param1=0.5";
    /** The share of the maximum heap the kept frames of one decode may take at the decoded size. */
    private static final int KEPT_HEAP_DIVISOR = 4;
    /** The longest frame-select expression a decode passes on its command line (Windows caps one at 32767). */
    static final int MAX_SELECT_CHARS = 24 * 1024;
    /** The share of the maximum heap one video's soundtrack samples may take. */
    private static final int SOUNDTRACK_HEAP_DIVISOR = 4;

    private final String ffmpeg;
    private final String ffprobe;

    /** A decoder over the binaries on the PATH. */
    public VideoDecoder() {
        this(null);
    }

    /**
     * A decoder over the binaries in {@code dir} (llama.cpp's {@code --video-ffmpeg-dir}), or on the PATH for a
     * null directory.
     */
    public VideoDecoder(Path binDir) {
        this.ffmpeg = binary(binDir, "ffmpeg");
        this.ffprobe = binary(binDir, "ffprobe");
    }

    private static String binary(Path dir, String name) {
        return dir == null ? name : dir.resolve(name).toString();
    }

    /**
     * Whether the bytes start with the magic of a video container this decoder recognizes, the check that tells an
     * untyped media buffer's video from its audio and images: Matroska and WebM (EBML), AVI (RIFF + AVI ), FLV,
     * MPEG-TS (two 188-byte packet sync bytes) and MP4/MOV whose {@code ftyp} major brand is not an audio-only
     * one ({@code M4A }, {@code M4B }, {@code m4a}, {@code alac}, {@code acla}).
     */
    public static boolean isVideo(byte[] bytes) {
        if (bytes == null) {
            return false;
        }
        if (startsWithAt(bytes, 0, new byte[] {(byte) 0x1A, (byte) 0x45, (byte) 0xDF, (byte) 0xA3})) {
            return true; // EBML: Matroska or WebM
        }
        if (bytes.length >= 12 && startsWithAt(bytes, 0, "RIFF") && startsWithAt(bytes, 8, "AVI ")) {
            return true;
        }
        if (startsWithAt(bytes, 0, "FLV")) {
            return true;
        }
        if (bytes.length >= 189 && (bytes[0] & 0xFF) == 0x47 && (bytes[188] & 0xFF) == 0x47) {
            return true; // MPEG-TS packets
        }
        if (bytes.length >= 12 && startsWithAt(bytes, 4, "ftyp")) {
            return !audioBrand(bytes);
        }
        return false;
    }

    /** The major brand at bytes 8..12, when it names an audio-only container. */
    private static boolean audioBrand(byte[] bytes) {
        String brand = new String(bytes, 8, 4, StandardCharsets.US_ASCII);
        return brand.equals("M4A ") || brand.equals("M4B ") || brand.equals("m4a") || brand.equals("alac")
                || brand.equals("acla");
    }

    private static boolean startsWithAt(byte[] b, int off, String s) {
        return b.length >= off + s.length() && startsWithAt(b, off, s.getBytes(StandardCharsets.US_ASCII));
    }

    private static boolean startsWithAt(byte[] b, int off, byte[] magic) {
        if (b.length < off + magic.length) {
            return false;
        }
        for (int i = 0; i < magic.length; i++) {
            if (b[off + i] != magic[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * The stream metadata of {@code bytes}: one ffprobe run over a temporary file (the input has to be seekable
     * for the container headers, as llama.cpp's {@code cache:pipe:0} makes it). A stream whose container reports
     * no frame rate keeps rate 0 ({@link VideoMeta#hasFrameRate()}); the projector family's
     * {@link VideoConfig} assumes a rate or refuses the source when it plans.
     *
     * @param field the request field the bytes came from, named in every refusal
     * @throws MediaInputException when ffprobe is missing, the bytes hold no video stream, or the probe fails
     */
    public VideoMeta probe(byte[] bytes, String field) {
        Objects.requireNonNull(bytes, "bytes");
        if (bytes.length == 0) {
            throw new MediaInputException(field + " is empty: video bytes are required");
        }
        Path file = null;
        try {
            file = tempFile(bytes);
            Map<String, String> fields = ffprobe(file, field);
            if (!fields.containsKey("width") && !fields.containsKey("height")) {
                throw new MediaInputException(field + " holds no video stream to decode");
            }
            return VideoMeta.ofProbe(fields);
        } finally {
            deleteQuietly(file);
        }
    }

    /**
     * The sampled frames of {@code bytes}: one ffmpeg run over a temporary file streaming raw RGB24 frames, of
     * which {@code plan}'s source indices are kept; the run stops after the plan's last source frame.
     *
     * @param field the request field the bytes came from, named in every refusal
     * @throws MediaInputException when ffmpeg is missing, the decode fails, or the video ends before the plan's
     *                             last frame
     */
    public DecodedVideo decode(byte[] bytes, VideoSamplePlan plan, String field) {
        return decode(bytes, plan, null, 0, field);
    }

    /**
     * The sampled frames of {@code bytes} and the soundtrack slices of its temporal groups.
     *
     * @param frameSize          the size the preprocessor recommends for the frames, or null for the source size
     * @param soundtrackRate     the audio tower's sample rate, or a non-positive value when the family hears no
     *                           soundtrack; the slices are the soundtrack from the stream's start for the video's
     *                           span, one per temporal group
     * @param field              the request field the bytes came from, named in every refusal
     * @throws MediaInputException when ffmpeg is missing, the decode fails, the video ends before the plan's last
     *                             frame, or the kept frames or the soundtrack exceed their heap share
     */
    public DecodedVideo decode(byte[] bytes, VideoSamplePlan plan, ImageSize frameSize, int soundtrackRate,
                               String field) {
        Objects.requireNonNull(bytes, "bytes");
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(field, "field");
        VideoMeta meta = plan.source();
        ImageSize size = decodedSize(meta, frameSize);
        int last = plan.indices()[plan.indices().length - 1];
        checkKeptBytes((long) plan.indices().length * size.width() * size.height() * 3,
                Runtime.getRuntime().maxMemory() / KEPT_HEAP_DIVISOR, size, field);
        Path file = null;
        try {
            file = tempFile(bytes);
            List<RasterImage> frames = readFrames(file, plan, size, last, field);
            List<AudioClip> soundtrack = soundtrackRate > 0
                    ? readSoundtrack(file, plan, soundtrackRate, field)
                    : Collections.emptyList();
            return new DecodedVideo(meta, plan, frames, soundtrack);
        } finally {
            deleteQuietly(file);
        }
    }

    /** The size the decode emits frames at: the recommendation, or the probed stream's own size. */
    private static ImageSize decodedSize(VideoMeta meta, ImageSize frameSize) {
        if (frameSize == null) {
            return new ImageSize(meta.width(), meta.height());
        }
        if (frameSize.width() <= 0 || frameSize.height() <= 0) {
            throw new IllegalArgumentException("the decoded frame size must be positive, got " + frameSize);
        }
        return frameSize;
    }

    /**
     * Refuses a decode whose kept frames, at the decoded size, would take more than {@code limit} bytes.
     *
     * @throws MediaInputException naming {@code field}, the needed bytes and the limit
     */
    static void checkKeptBytes(long kept, long limit, ImageSize size, String field) {
        if (kept > limit) {
            throw new MediaInputException(field + ": the sampled frames need " + (kept >> 20) + " MiB at "
                    + size.width() + "x" + size.height() + ", more than the " + (limit >> 20) + " MiB limit (a"
                    + " quarter of the heap); lower num_frames or fps, or shorten the video");
        }
    }

    /**
     * Refuses a soundtrack whose samples would take more than {@code limit} bytes.
     *
     * @throws MediaInputException naming {@code field}, the needed bytes and the limit
     */
    /**
     * The bytes the soundtrack read buffers: the samples before the video's start time and the video's span, both
     * derived from container metadata, saturating instead of overflowing so oversized values are refused.
     */
    static long bufferedBytes(long start, long length) {
        long samples = start > Long.MAX_VALUE - length ? Long.MAX_VALUE : start + length;
        return samples > Long.MAX_VALUE / Float.BYTES ? Long.MAX_VALUE : samples * Float.BYTES;
    }

    static void checkSoundtrackBytes(long kept, long limit, String field) {
        if (kept > limit) {
            throw new MediaInputException(field + ": the soundtrack needs " + (kept >> 20) + " MiB, more than the "
                    + (limit >> 20) + " MiB limit (a quarter of the heap); shorten the video");
        }
    }

    private static Path tempFile(byte[] bytes) throws MediaInputException {
        try {
            Path file = Files.createTempFile("neuralfree-video-", "");
            Files.write(file, bytes);
            return file;
        } catch (IOException e) {
            throw new MediaInputException("the video cannot be staged for the ffmpeg decode: " + e.getMessage(), e);
        }
    }

    private static void deleteQuietly(Path file) {
        if (file != null) {
            try {
                Files.deleteIfExists(file);
            } catch (IOException ignored) {
                // a leftover temporary file is harmless
            }
        }
    }

    /** The probe fields ffprobe printed for the first video stream of {@code file}. */
    private Map<String, String> ffprobe(Path file, String field) {
        List<String> command = new ArrayList<>(List.of(ffprobe, "-v", "quiet"));
        command.addAll(FfmpegInput.ffprobeFile(FfmpegInput.Media.VIDEO, file));
        command.addAll(List.of("-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,start_time:format=start_time",
                "-select_streams", "v:0", "-of", "default"));
        Process process = start(command, field);
        try {
            String out;
            try (InputStream in = process.getInputStream()) {
                out = readAll(in, field, 64 * 1024);
            } catch (IOException e) {
                throw new MediaInputException(field + ": reading the ffprobe output failed: " + e.getMessage(), e);
            }
            try {
                if (!process.waitFor(PROCESS_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                    throw new MediaInputException(field + ": ffprobe timed out after " + PROCESS_TIMEOUT_SECONDS
                            + " s");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new MediaInputException(field + ": the ffprobe run was interrupted", e);
            }
            if (process.exitValue() != 0) {
                throw new MediaInputException(field + ": ffprobe exit code " + process.exitValue() + " rejected"
                        + " the video");
            }
            return parseProbe(out);
        } finally {
            process.destroyForcibly();
        }
    }

    /**
     * The {@code key=value} lines of an ffprobe default output: the stream's keys as printed, the container's
     * ({@code [FORMAT]} section) prefixed {@code format.}.
     */
    static Map<String, String> parseProbe(String out) {
        Map<String, String> fields = new HashMap<>();
        String prefix = "";
        for (String raw : out.split("\n")) {
            String line = raw.trim();
            if (line.equals("[FORMAT]")) {
                prefix = "format.";
                continue;
            }
            if (line.startsWith("[")) {
                prefix = "";
                continue;
            }
            int eq = line.indexOf('=');
            if (eq > 0) {
                fields.put(prefix + line.substring(0, eq).trim(), line.substring(eq + 1).trim());
            }
        }
        return fields;
    }

    /**
     * Starts one ffmpeg or ffprobe run. Its stderr is discarded: the frames travel on stdout, and an undrained
     * stderr pipe would block the process on a verbose or malformed input.
     */
    private Process start(List<String> command, String field) {
        ProcessBuilder builder = new ProcessBuilder(command)
                .redirectError(ProcessBuilder.Redirect.DISCARD);
        try {
            return builder.start();
        } catch (IOException e) {
            throw new MediaInputException(field + ": the video decoder needs the '" + command.get(0)
                    + "' binary" + (command.get(0).indexOf('/') >= 0 ? " (missing from its directory)"
                    : " on the PATH (none was found)") + ": " + e.getMessage(), e);
        }
    }

    /**
     * The plan's frames of one decode run over the staged file: an ffmpeg run streaming raw RGB24 frames at
     * {@code size}, scaled with the reference's filter when the size differs from the stream's. An index-aligned
     * stream ({@link VideoMeta#indexAligned()}) whose plan skips frames selects the plan's frames in the filter graph
     * ({@link #selectFilter}) and renumbers their timestamps consecutively, so the rate-keeping output neither
     * duplicates nor drops a frame and every frame read is kept; any other stream streams every frame up to the
     * plan's last.
     */
    private List<RasterImage> readFrames(Path file, VideoSamplePlan plan, ImageSize size, int last, String field) {
        VideoMeta meta = plan.source();
        int[] indices = plan.indices();
        String select = meta.indexAligned() && indices.length < last + 1 ? selectFilter(indices) : null;
        boolean scaled = size.width() != meta.width() || size.height() != meta.height();
        List<String> filters = new ArrayList<>();
        if (select != null) {
            filters.add(select);
        }
        if (scaled) {
            filters.add(String.format(Locale.ROOT, SCALE_FILTER, size.width(), size.height()));
        }
        List<String> command = new ArrayList<>(List.of(ffmpeg));
        command.addAll(FfmpegInput.ffmpegFile(FfmpegInput.Media.VIDEO, file));
        command.addAll(List.of("-map", "0:v:0"));
        if (!filters.isEmpty()) {
            command.addAll(List.of("-vf", String.join(",", filters)));
        }
        int frames = select != null ? indices.length : last + 1;
        command.addAll(List.of("-frames:v", String.valueOf(frames), "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-loglevel", "error", "pipe:1"));
        Process process = start(command, field);
        try {
            return readFrames(process, plan, size, select != null, field);
        } finally {
            process.destroyForcibly();
        }
    }

    /**
     * The filters that keep only the decoded frames {@code indices} (ascending) and number them consecutively at the
     * stream's rate: {@code select} over the frame index {@code n}, a run of consecutive indices as one
     * {@code between} term, then {@code setpts} so the kept frames sit one frame apart. Null when the expression
     * would pass {@link #MAX_SELECT_CHARS}; the decode then streams every frame.
     */
    static String selectFilter(int[] indices) {
        StringBuilder expr = new StringBuilder("select='");
        int i = 0;
        while (i < indices.length) {
            int j = i;
            while (j + 1 < indices.length && indices[j + 1] == indices[j] + 1) {
                j++;
            }
            if (i > 0) {
                expr.append('+');
            }
            if (j == i) {
                expr.append("eq(n,").append(indices[i]).append(')');
            } else {
                expr.append("between(n,").append(indices[i]).append(',').append(indices[j]).append(')');
            }
            if (expr.length() > MAX_SELECT_CHARS) {
                return null;
            }
            i = j + 1;
        }
        return expr.append("',setpts=N/FRAME_RATE/TB").toString();
    }

    /**
     * Reads the raw frames of a running decode until the plan's last one: every frame when the decode
     * {@code selected} the plan's frames itself, else the frames at the plan's indices.
     */
    private List<RasterImage> readFrames(Process process, VideoSamplePlan plan, ImageSize size, boolean selected,
                                         String field) {
        int frameBytes = Math.multiplyExact(size.width(), Math.multiplyExact(size.height(), 3));
        int[] indices = plan.indices();
        List<RasterImage> frames = new ArrayList<>(indices.length);
        int next = 0;
        long total = 0;
        byte[] frame = new byte[frameBytes];
        try (OutputStream stdin = process.getOutputStream(); InputStream stdout = process.getInputStream()) {
            stdin.close(); // the bytes are staged in the input file; nothing flows through the pipe
            int source = 0;
            int at = 0;
            while (next < indices.length) {
                int n = stdout.read(frame, at, frameBytes - at);
                if (n < 0) {
                    throw new MediaInputException(field + ": the video ended after " + (selected ? next : source)
                            + (selected ? " sampled" : "") + " frame(s), before the sampled frame " + indices[next]);
                }
                at += n;
                total += n;
                if (total > MAX_FRAME_BYTES) {
                    throw new MediaInputException(field + ": the decoded frames exceed " + MAX_FRAME_BYTES
                            + " bytes");
                }
                if (at == frameBytes) {
                    if (selected || indices[next] == source) {
                        frames.add(new RasterImage(size.width(), size.height(), frame));
                        next++;
                    }
                    source++;
                    at = 0;
                }
            }
        } catch (IOException e) {
            throw new MediaInputException(field + ": reading the decoded frames failed: " + e.getMessage(), e);
        }
        try {
            if (!process.waitFor(PROCESS_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                throw new MediaInputException(field + ": the ffmpeg decode timed out after "
                        + PROCESS_TIMEOUT_SECONDS + " s");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new MediaInputException(field + ": the ffmpeg decode was interrupted", e);
        }
        if (process.exitValue() != 0) {
            throw new MediaInputException(field + ": ffmpeg exit code " + process.exitValue() + " rejected the"
                    + " video");
        }
        return frames;
    }

    /**
     * The soundtrack of the staged file as one clip per temporal group, the reference's slice of a mono float32
     * decode at {@code rate} ({@code mtmd_helper_video::load_soundtrack} and {@code soundtrack_slice}): the track
     * spans the source video's frames from its start time, and a group takes the samples from its first frame's
     * time to the next group's, the last group taking the rest. A video without an audio track (or whose track
     * starts after the video's span) has no soundtrack, an empty list.
     */
    private List<AudioClip> readSoundtrack(Path file, VideoSamplePlan plan, int rate, String field) {
        VideoMeta meta = plan.source();
        long start = Math.max(0L, (long) Math.rint(meta.startTime() * rate));
        long length = Math.max(0L, (long) Math.rint((double) meta.nFrames() / meta.fps() * rate));
        if (length == 0) {
            return Collections.emptyList();
        }
        checkSoundtrackBytes(bufferedBytes(start, length),
                Runtime.getRuntime().maxMemory() / SOUNDTRACK_HEAP_DIVISOR, field);
        List<String> command = new ArrayList<>(List.of(ffmpeg));
        command.addAll(FfmpegInput.ffmpegFile(FfmpegInput.Media.VIDEO, file));
        command.addAll(List.of("-map", "0:a:0", "-vn", "-ac", "1", "-ar",
                String.valueOf(rate), "-t", String.format(Locale.ROOT, "%.6f", (double) (start + length) / rate),
                "-f", "f32le", "-loglevel", "error", "pipe:1"));
        Process process = start(command, field);
        float[] track;
        try {
            track = readTrack(process, start, length, rate, field);
        } finally {
            process.destroyForcibly();
        }
        if (track.length == 0) {
            return Collections.emptyList();
        }
        int groups = plan.groups();
        int merge = plan.temporalMerge();
        int[] indices = plan.indices();
        List<AudioClip> slices = new ArrayList<>(groups);
        for (int g = 0; g < groups; g++) {
            int begin = g == 0 ? 0 : groupStart(g, indices, merge, meta, rate, track.length);
            int end = g == groups - 1 ? track.length
                    : groupStart(g + 1, indices, merge, meta, rate, track.length);
            if (end < begin) {
                end = begin;
            }
            slices.add(new AudioClip(Arrays.copyOfRange(track, begin, end), rate));
        }
        return slices;
    }

    /** The sample index the temporal group {@code group} starts at, clamped to the track. */
    private static int groupStart(int group, int[] indices, int merge, VideoMeta meta, int rate, int trackLength) {
        double seconds = indices[group * merge] / meta.fps();
        return (int) Math.min(trackLength, Math.max(0L, (long) Math.rint(seconds * rate)));
    }

    /**
     * The samples of the soundtrack from {@code start} for {@code length} frames of a running decode. The run is
     * stopped once the span is read; a track shorter than {@code start} yields no samples.
     */
    private float[] readTrack(Process process, long start, long length, int rate, String field) {
        long want = (start + length) * Float.BYTES;
        ByteArrayOutputStream pcm = new ByteArrayOutputStream((int) Math.min(want, 1 << 20));
        try (OutputStream stdin = process.getOutputStream(); InputStream stdout = process.getInputStream()) {
            stdin.close(); // the bytes are staged in the input file; nothing flows through the pipe
            byte[] buf = new byte[8192];
            long got = 0;
            while (got < want) {
                int n = stdout.read(buf, 0, (int) Math.min(buf.length, want - got));
                if (n < 0) {
                    break;
                }
                pcm.write(buf, 0, n);
                got += n;
            }
            // the reference reads the whole track; drain the rest so the run finishes without blocking
            while (stdout.read(buf) >= 0) {
                // samples past the video's span are dropped
            }
        } catch (IOException e) {
            throw new MediaInputException(field + ": reading the soundtrack failed: " + e.getMessage(), e);
        }
        try {
            // the run may still be streaming past the span; it is ended either way
            if (!process.waitFor(PROCESS_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                throw new MediaInputException(field + ": the soundtrack decode timed out after "
                        + PROCESS_TIMEOUT_SECONDS + " s");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new MediaInputException(field + ": the soundtrack decode was interrupted", e);
        }
        byte[] bytes = pcm.toByteArray();
        int samples = bytes.length / Float.BYTES;
        if (start >= samples) {
            return new float[0];
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] track = new float[(int) Math.min(samples, start + length) - (int) start];
        buffer.position((int) start * Float.BYTES);
        buffer.asFloatBuffer().get(track);
        return track;
    }

    private static String readAll(InputStream in, String field, int limit) throws IOException {
        StringBuilder out = new StringBuilder();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) > 0) {
            if (out.length() + n > limit) {
                throw new MediaInputException(field + ": the ffprobe output exceeds " + limit + " bytes");
            }
            out.append(new String(buf, 0, n, StandardCharsets.UTF_8));
        }
        return out.toString();
    }
}
