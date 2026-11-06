package com.eonmux.neuralfree.gguf.hub.slice;

import com.eonmux.neuralfree.gguf.hub.HubClient;
import com.eonmux.neuralfree.gguf.hub.HubRef;
import com.eonmux.neuralfree.io.safetensors.SafetensorsHeader;
import com.eonmux.neuralfree.utils.json.Json;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Writes a {@link SlicePlan} to a directory: one {@code model.safetensors} holding every kept tensor back to back,
 * each fetched as an HTTP range of its source shard and written at its output offset, then the auxiliary repository
 * files the reference skill copies, the patched {@code config.json} and {@code .slice_meta.json}. The weights land
 * in {@code model.safetensors.tmp} and are renamed into place, so a failed run never leaves a half-written shard
 * under the name a loader opens. The directory's own files are the only ones written; a repository path that would
 * leave it is refused.
 */
final class SliceWriter {

    /** Files larger than this are not copied as auxiliary data. */
    static final long AUX_MAX_BYTES = 50L * 1024 * 1024;
    /** The single shard a slice writes. */
    static final String SHARD_FILE = "model.safetensors";
    /** The temporary shard a run writes before renaming it. */
    static final String SHARD_TEMP = SHARD_FILE + ".tmp";
    /** The metadata file a slice writes, the reference skill's name. */
    static final String META_FILE = ".slice_meta.json";
    /** The repository index a slice replaces with its single shard. */
    static final String INDEX_FILE = "model.safetensors.index.json";
    /** The repository config a slice patches. */
    static final String CONFIG_FILE = "config.json";

    private static final Json.Style COMPACT = new Json.Style(false, null, ",", ":", false);
    private static final Json.Style INDENTED = new Json.Style(true, "  ", ",", ": ", false);
    private static final Set<String> AUX_BLOCKED_EXTENSIONS = extensions("bin", "pth", "pt", "ckpt", "h5",
            "msgpack", "onnx", "tflite", "mlmodel", "gguf", "nemo");
    private static final Set<String> AUX_BLOCKED_DIRECTORIES =
            extensions("onnx", "openvino", "coreml", "original", "metal");
    private static final Set<String> AUX_BLOCKED_MEDIA = extensions("png", "jpg", "jpeg", "gif", "webp", "avif",
            "svg", "mp4", "webm", "mov", "wav", "mp3", "flac", "ogg", "pdf");

    /** What one write produced: the directory, and the auxiliary files the size rule skipped. */
    static final class Result {

        private final Path directory;
        private final List<String> skippedAux;

        private Result(Path directory, List<String> skippedAux) {
            this.directory = directory;
            this.skippedAux = Collections.unmodifiableList(new ArrayList<>(skippedAux));
        }

        Path directory() {
            return directory;
        }

        List<String> skippedAux() {
            return skippedAux;
        }
    }

    private SliceWriter() {
    }

    /**
     * Writes {@code plan} into {@code output}.
     *
     * @throws IOException when the directory cannot be prepared or a file cannot be written, or when it is not
     *                     empty and {@code options.overwrite()} is false
     */
    static Result write(Path output, HubRef ref, String commit, HubClient client,
                        List<HubClient.RepoFile> files, SlicePlan plan, List<String> configPatches,
                        SliceOptions options) throws IOException, InterruptedException {
        Path directory = output.toAbsolutePath().normalize();
        prepare(directory, options.overwrite());
        Map<String, HubClient.RepoFile> byPath = new LinkedHashMap<>();
        for (HubClient.RepoFile file : files) {
            byPath.put(file.path(), file);
        }
        Path shard = directory.resolve(SHARD_FILE);
        Path temp = directory.resolve(SHARD_TEMP);
        try {
            writeWeights(temp, ref, commit, client, byPath, plan, options.jobs());
            move(temp, shard);
        } catch (InterruptedException e) {
            Files.deleteIfExists(temp);
            throw e;
        } catch (IOException | RuntimeException e) {
            Files.deleteIfExists(temp);
            throw e;
        }
        List<String> skipped = copyAux(directory, ref, commit, client, files, plan);
        if (plan.config().config() != null) {
            Files.write(directory.resolve(CONFIG_FILE),
                    (Json.toString(plan.config().config(), INDENTED) + "\n").getBytes(StandardCharsets.UTF_8));
        }
        writeMeta(directory, ref, commit, plan, configPatches, skipped);
        return new Result(directory, skipped);
    }

    /** Checks the output path and creates it, refusing a non-empty directory unless it may be replaced. */
    private static void prepare(Path directory, boolean overwrite) throws IOException {
        if (Files.exists(directory) && !Files.isDirectory(directory)) {
            throw new IOException("the output path '" + directory + "' is not a directory");
        }
        if (Files.isDirectory(directory) && !overwrite) {
            try (DirectoryStream<Path> entries = Files.newDirectoryStream(directory)) {
                if (entries.iterator().hasNext()) {
                    throw new IOException("the output directory '" + directory
                            + "' is not empty; pass --overwrite to replace its files");
                }
            }
        }
        Files.createDirectories(directory);
    }

    /** Lays out the header, extends the file to its final size and fetches every tensor into its offset. */
    private static void writeWeights(Path temp, HubRef ref, String commit, HubClient client,
                                     Map<String, HubClient.RepoFile> files, SlicePlan plan, int jobs)
            throws IOException, InterruptedException {
        Map<String, Object> header = new LinkedHashMap<>();
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("format", "pt");
        header.put("__metadata__", metadata);
        long total = 0L;
        for (TensorSlice tensor : plan.tensors()) {
            if (header.containsKey(tensor.outputName())) {
                throw new IllegalStateException("the slice would write the tensor '"
                        + tensor.outputName() + "' twice");
            }
            Map<String, Object> info = new LinkedHashMap<>();
            info.put("dtype", tensor.dtype().headerName());
            info.put("shape", boxed(tensor.outputShape()));
            info.put("data_offsets", Arrays.asList(total, total + tensor.byteLength()));
            header.put(tensor.outputName(), info);
            total = Math.addExact(total, tensor.byteLength());
        }
        byte[] json = Json.toString(header, COMPACT).getBytes(StandardCharsets.UTF_8);
        long headerLength = json.length + pad(json.length + 8);
        byte[] padded = Arrays.copyOf(json, (int) headerLength);
        Arrays.fill(padded, json.length, padded.length, (byte) ' ');
        try (FileChannel channel = FileChannel.open(temp, StandardOpenOption.CREATE,
                StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer length = ByteBuffer.allocate(SafetensorsHeader.LENGTH_FIELD_BYTES)
                    .order(ByteOrder.LITTLE_ENDIAN);
            length.putLong(0, headerLength);
            write(channel, length, 0L);
            write(channel, ByteBuffer.wrap(padded), SafetensorsHeader.LENGTH_FIELD_BYTES);
            long dataStart = SafetensorsHeader.LENGTH_FIELD_BYTES + headerLength;
            if (total > 0L && channel.size() < dataStart + total) {
                channel.position(dataStart + total - 1L);
                channel.write(ByteBuffer.wrap(new byte[]{0}));
            }
            if (channel.size() != dataStart + total) {
                throw new IOException("the shard '" + temp + "' is " + channel.size() + " bytes, not the planned "
                        + (dataStart + total));
            }
            fetch(channel, dataStart, ref, commit, client, files, plan, jobs);
            channel.force(true);
        }
    }

    /** Fetches the plan's tensors concurrently into their offsets of {@code channel}. */
    private static void fetch(FileChannel channel, long dataStart, HubRef ref, String commit, HubClient client,
                              Map<String, HubClient.RepoFile> files, SlicePlan plan, int jobs)
            throws IOException, InterruptedException {
        ExecutorService pool = Executors.newFixedThreadPool(jobs);
        try {
            List<Future<?>> futures = new ArrayList<>(plan.tensors().size());
            long offset = 0L;
            for (TensorSlice tensor : plan.tensors()) {
                HubClient.RepoFile source = files.get(tensor.sourceFile());
                if (source == null) {
                    throw new IllegalStateException("the source shard '" + tensor.sourceFile()
                            + "' is not in the repository listing");
                }
                long base = dataStart + offset;
                offset += tensor.byteLength();
                if (tensor.byteLength() == 0L) {
                    continue;
                }
                futures.add(pool.submit(() -> {
                    client.readRange(ref, commit, source, tensor.sourceBegin(),
                            tensor.sourceBegin() + tensor.byteLength() - 1L,
                            (rangeOffset, bytes, position, length)
                                    -> writeAt(channel, base + rangeOffset, bytes, position, length));
                    return null;
                }));
            }
            for (Future<?> future : futures) {
                SliceFutures.await(future);
            }
        } finally {
            pool.shutdownNow();
        }
    }

    /** Copies the auxiliary repository files the reference skill copies, returning the ones over the size rule. */
    private static List<String> copyAux(Path directory, HubRef ref, String commit, HubClient client,
                                        List<HubClient.RepoFile> files, SlicePlan plan) throws IOException,
            InterruptedException {
        List<String> skipped = new ArrayList<>();
        for (HubClient.RepoFile file : files) {
            String path = file.path();
            if (plan.weightFiles().contains(path) || path.equals(CONFIG_FILE) || path.equals(INDEX_FILE)
                    || path.endsWith(".safetensors") || auxBlacklisted(path)) {
                continue;
            }
            long size = file.size() >= 0L ? file.size() : client.contentLength(ref, commit, file);
            if (size > AUX_MAX_BYTES) {
                skipped.add(path + " (" + SlicePlan.human(size) + ")");
                continue;
            }
            Path destination = directory.resolve(path).normalize();
            if (!destination.startsWith(directory) || destination.equals(directory)) {
                throw new IOException("the repository path '" + path + "' leaves the output directory");
            }
            Path parent = destination.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.write(destination, client.readAll(ref, commit, file, AUX_MAX_BYTES));
        }
        return skipped;
    }

    /** Writes {@code .slice_meta.json}: what was read, what was kept and how the config changed. */
    private static void writeMeta(Path directory, HubRef ref, String commit, SlicePlan plan,
                                  List<String> configPatches, List<String> skipped) throws IOException {
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("repo", ref.repository());
        meta.put("revision", commit);
        meta.put("tensors", (long) plan.tensors().size());
        meta.put("total_tensors", (long) plan.totalTensors());
        meta.put("size", plan.totalBytes());
        meta.put("config_patches", configPatches);
        meta.put("skipped_aux", skipped);
        Files.write(directory.resolve(META_FILE),
                (Json.toString(meta, INDENTED) + "\n").getBytes(StandardCharsets.UTF_8));
    }

    /** Whether an auxiliary repository file is a weight in another format or a media or demo asset. */
    static boolean auxBlacklisted(String path) {
        String lower = path.toLowerCase(Locale.ROOT);
        String name = lower.substring(lower.lastIndexOf('/') + 1);
        if (name.equals(".gitattributes") || lower.contains(".mlpackage/")) {
            return true;
        }
        int dot = name.lastIndexOf('.');
        if (dot >= 0 && AUX_BLOCKED_EXTENSIONS.contains(name.substring(dot + 1))) {
            return true;
        }
        if (dot >= 0 && AUX_BLOCKED_MEDIA.contains(name.substring(dot + 1))) {
            return true;
        }
        for (String segment : lower.split("/")) {
            if (AUX_BLOCKED_DIRECTORIES.contains(segment)) {
                return true;
            }
        }
        return false;
    }

    /** Padding that brings {@code length + 8} to a multiple of 8. */
    private static int pad(int length) {
        return (8 - ((length + 8) % 8)) % 8;
    }

    private static List<Long> boxed(long[] values) {
        List<Long> out = new ArrayList<>(values.length);
        for (long value : values) {
            out.add(value);
        }
        return out;
    }

    private static void write(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
        long at = position;
        while (buffer.hasRemaining()) {
            at += channel.write(buffer, at);
        }
    }

    private static void writeAt(FileChannel channel, long position, byte[] bytes, int offset, int length)
            throws IOException {
        write(channel, ByteBuffer.wrap(bytes, offset, length), position);
    }

    /** Renames {@code temp} onto {@code target}, atomically where the file system supports it. */
    private static void move(Path temp, Path target) throws IOException {
        try {
            Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static Set<String> extensions(String... values) {
        Set<String> set = new HashSet<>();
        Collections.addAll(set, values);
        return Collections.unmodifiableSet(set);
    }
}
