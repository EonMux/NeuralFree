package com.eonmux.neuralfree.gguf.hub.slice;

import com.eonmux.neuralfree.gguf.hub.HubClient;
import com.eonmux.neuralfree.gguf.hub.HubRef;
import com.eonmux.neuralfree.io.safetensors.SafetensorsHeader;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Builds a small test slice of a Hugging Face safetensors model without downloading its weights: the repository's
 * shard headers are read with HTTP range requests, {@link SlicePlanner} decides which tensors to keep, and
 * {@link SliceWriter} fetches exactly those byte spans, renumbers them and writes the slice with a patched
 * {@code config.json}. The reference is the {@code slice-model} skill; its heuristics and their limits are its own,
 * so {@code --inspect} should be read before a slice is trusted, and a model whose layer family is not recognized
 * carries a warning.
 *
 * <p>The run is online only ({@code HF_HUB_OFFLINE} is refused), reads with the token its
 * {@link HubClient} carries, and leaves the hub cache untouched: nothing but the range reads leaves the process.
 * The result names the tensors kept and dropped, the size, the config changes and the directory written.</p>
 */
public final class ModelSlicer {

    /** The largest {@code config.json} or index the slicer reads, the untrusted JSON byte bound. */
    static final long JSON_MAX_BYTES = 16L * 1024 * 1024;
    private static final String INDEX_FILE = SliceWriter.INDEX_FILE;
    private static final String CONFIG_FILE = SliceWriter.CONFIG_FILE;

    private ModelSlicer() {
    }

    /**
     * Plans and, unless the options ask for nothing to be written, writes the slice of {@code ref}.
     *
     * @throws IOException          when the Hub cannot be read, the repository has no safetensors weights, the
     *                              slice exceeds {@link SliceOptions#maxBytes()}, or a file cannot be written
     * @throws InterruptedException when the calling thread is interrupted
     */
    public static SliceReport run(HubClient client, HubRef ref, SliceOptions options)
            throws IOException, InterruptedException {
        if (HubClient.offline()) {
            throw new IOException("the model slice tool needs the Hub; HF_HUB_OFFLINE is set");
        }
        String commit = client.resolveCommit(ref);
        List<HubClient.RepoFile> files = client.listFiles(ref, commit);
        if (files.isEmpty()) {
            throw new IOException("the repository '" + ref.repository() + "' at " + commit + " lists no files");
        }
        Map<String, HubClient.RepoFile> byPath = new LinkedHashMap<>();
        for (HubClient.RepoFile file : files) {
            byPath.put(file.path(), file);
        }
        Object config = readJson(client, ref, commit, byPath.get(CONFIG_FILE), CONFIG_FILE);
        List<String> weightFiles = weightFiles(client, ref, commit, byPath);
        Map<String, SafetensorsHeader> headers = readHeaders(client, ref, commit, byPath, weightFiles,
                options.jobs());
        SlicePlan plan = SlicePlanner.plan(weightFiles, headers, config, options);
        List<String> warnings = warnings(plan, options);
        if (plan.totalBytes() > options.maxBytes()) {
            throw new IOException("the slice is " + SlicePlan.human(plan.totalBytes()) + ", over the "
                    + SlicePlan.human(options.maxBytes()) + " max-size; raise --max-size or keep fewer tensors");
        }
        if (options.writesNothing()) {
            return new SliceReport(ref.repository(), commit, plan.tensors().size(), plan.totalTensors(),
                    plan.totalBytes(), null, options.inspect() ? plan.inspect() : null, options.dryRun(),
                    new ArrayList<>(), new ArrayList<>(), warnings);
        }
        List<String> patches = plan.patchConfig(newVocab(plan, options));
        SliceWriter.Result written = SliceWriter.write(options.output(), ref, commit, client, files, plan,
                patches, options);
        return new SliceReport(ref.repository(), commit, plan.tensors().size(), plan.totalTensors(),
                plan.totalBytes(), written.directory(), null, false, patches, written.skippedAux(), warnings);
    }

    /** The vocab size the config is cut to, or null when the option is off or no declared size is larger. */
    private static Integer newVocab(SlicePlan plan, SliceOptions options) {
        if (options.vocabSize() <= 0) {
            return null;
        }
        for (Long vocab : plan.config().vocabSizes()) {
            if (vocab.longValue() > options.vocabSize()) {
                return Integer.valueOf(options.vocabSize());
            }
        }
        return null;
    }

    /** The warnings a plan deserves: an unrecognized layer family and a vocabulary the tokenizer will not match. */
    private static List<String> warnings(SlicePlan plan, SliceOptions options) {
        List<String> warnings = new ArrayList<>();
        if (!plan.hasLayerFamily()) {
            warnings.add("no layer family was detected; run with --inspect and check the config's layer keys");
        }
        if (options.vocabSize() > 0) {
            warnings.add("--vocab-size does not slice the tokenizer, so a converted model's vocabulary will not"
                    + " match its tokenizer");
        }
        return warnings;
    }

    /**
     * The shards the slice reads, as the reference skill picks them: the index's {@code weight_map} when the
     * repository has one, else {@code model.safetensors}, else the first root-level safetensors file (preferring
     * {@code consolidated.safetensors}).
     */
    private static List<String> weightFiles(HubClient client, HubRef ref, String commit,
                                            Map<String, HubClient.RepoFile> byPath)
            throws IOException, InterruptedException {
        if (byPath.containsKey(INDEX_FILE)) {
            Object parsed = readJson(client, ref, commit, byPath.get(INDEX_FILE), INDEX_FILE);
            if (!(parsed instanceof Map)) {
                throw new IOException("'" + INDEX_FILE + "' is not a JSON object");
            }
            Object weightMap = ((Map<?, ?>) parsed).get("weight_map");
            if (!(weightMap instanceof Map)) {
                throw new IOException("'" + INDEX_FILE + "' carries no weight_map object");
            }
            Set<String> shards = new TreeSet<>();
            for (Map.Entry<?, ?> e : ((Map<?, ?>) weightMap).entrySet()) {
                if (!(e.getValue() instanceof String)) {
                    throw new IOException("'" + INDEX_FILE + "' maps '" + e.getKey() + "' to a non-string path");
                }
                shards.add((String) e.getValue());
            }
            if (shards.isEmpty()) {
                throw new IOException("'" + INDEX_FILE + "' names no shards");
            }
            for (String shard : shards) {
                if (!byPath.containsKey(shard)) {
                    throw new IOException("'" + INDEX_FILE + "' names the shard '" + shard
                            + "', which the repository listing does not carry");
                }
            }
            return new ArrayList<>(shards);
        }
        if (byPath.containsKey("model.safetensors")) {
            return single("model.safetensors");
        }
        List<String> root = new ArrayList<>();
        for (String path : byPath.keySet()) {
            if (path.endsWith(".safetensors") && path.indexOf('/') < 0) {
                root.add(path);
            }
        }
        if (root.isEmpty()) {
            throw new IOException("the repository '" + ref.repository() + "' carries no .safetensors weights");
        }
        Collections.sort(root);
        if (root.contains("consolidated.safetensors")) {
            return single("consolidated.safetensors");
        }
        return single(root.get(0));
    }

    private static List<String> single(String path) {
        List<String> out = new ArrayList<>();
        out.add(path);
        return out;
    }

    /** Reads and validates every shard header, concurrently. */
    private static Map<String, SafetensorsHeader> readHeaders(HubClient client, HubRef ref, String commit,
                                                              Map<String, HubClient.RepoFile> byPath,
                                                              List<String> weightFiles, int jobs)
            throws IOException, InterruptedException {
        Map<String, SafetensorsHeader> headers = new LinkedHashMap<>();
        ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, Math.min(jobs, weightFiles.size())));
        try {
            Map<String, Future<SafetensorsHeader>> futures = new LinkedHashMap<>();
            for (String path : weightFiles) {
                HubClient.RepoFile file = byPath.get(path);
                futures.put(path, pool.submit(() -> readHeader(client, ref, commit, file)));
            }
            for (Map.Entry<String, Future<SafetensorsHeader>> e : futures.entrySet()) {
                headers.put(e.getKey(), SliceFutures.await(e.getValue()));
            }
        } finally {
            pool.shutdownNow();
        }
        return headers;
    }

    /** Reads one shard's header with two range requests: its length field, then the announced JSON bytes. */
    private static SafetensorsHeader readHeader(HubClient client, HubRef ref, String commit,
                                                HubClient.RepoFile file) throws IOException, InterruptedException {
        long size = file.size() >= 0L ? file.size() : client.contentLength(ref, commit, file);
        byte[] lengthField = client.readRange(ref, commit, file, 0,
                SafetensorsHeader.LENGTH_FIELD_BYTES - 1);
        long n = ByteBuffer.wrap(lengthField).order(ByteOrder.LITTLE_ENDIAN).getLong(0);
        if (n < SafetensorsHeader.MIN_HEADER_BYTES || n > SafetensorsHeader.MAX_HEADER_BYTES
                || n > size - SafetensorsHeader.LENGTH_FIELD_BYTES) {
            throw new IOException("Invalid safetensors file '" + file.path() + "': header length " + n
                    + " is outside what the file's " + size + " bytes hold");
        }
        byte[] json = client.readRange(ref, commit, file, SafetensorsHeader.LENGTH_FIELD_BYTES,
                SafetensorsHeader.LENGTH_FIELD_BYTES + n - 1L);
        byte[] prefix = new byte[SafetensorsHeader.LENGTH_FIELD_BYTES + (int) n];
        System.arraycopy(lengthField, 0, prefix, 0, SafetensorsHeader.LENGTH_FIELD_BYTES);
        System.arraycopy(json, 0, prefix, SafetensorsHeader.LENGTH_FIELD_BYTES, (int) n);
        return SafetensorsHeader.parse(prefix, size, file.path());
    }

    /** One repository JSON file, or null when the repository does not carry it. */
    private static Object readJson(HubClient client, HubRef ref, String commit, HubClient.RepoFile file,
                                   String name) throws IOException, InterruptedException {
        if (file == null) {
            return null;
        }
        byte[] bytes = client.readAll(ref, commit, file, JSON_MAX_BYTES);
        try {
            return Json.parse(new String(bytes, StandardCharsets.UTF_8), Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new IOException("'" + name + "' is no valid JSON within the untrusted limits", e);
        }
    }
}
