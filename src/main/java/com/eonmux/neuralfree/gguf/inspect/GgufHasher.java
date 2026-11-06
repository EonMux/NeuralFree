package com.eonmux.neuralfree.gguf.inspect;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;

import java.io.IOException;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Hashes the tensor data of a GGUF model the way llama.cpp's {@code gguf-hash} does: one digest per tensor
 * ("layer") and one over every tensor's bytes in tensor-table order (the "model" digest). The hashes cover the
 * tensor bytes only, never the metadata, so a model whose key/value store changed still matches its manifest.
 *
 * <p>Algorithms are xxHash64 ({@link Xxh64}), SHA-1 and SHA-256 through the JVM's {@link MessageDigest}, and a
 * UUIDv5 of the model tensor bytes in llama.cpp's namespace ({@link UuidV5}), printed model-wide only: the
 * reference's UUID identifies a whole model and is not a per-tensor hash. Per-tensor lines are emitted in the
 * reference's order (xxh64, sha1, sha256); model lines follow them (xxh64, sha1, sha256, uuid). Each line names
 * its target as {@code <file>:<tensor>} for a tensor and {@code <file>} for the model, the form the reference's
 * manifest uses.</p>
 *
 * <p>Tensor bytes are read through {@link GGUFFileParser#readTensorData(GGUFTensorInfo, long, byte[], int, int)}
 * in {@value #CHUNK_BYTES}-byte chunks, so no tensor is ever held whole in the heap. A split model is hashed over
 * the tensors of every shard, in split order.</p>
 */
public final class GgufHasher {

    /** Bytes read per tensor read; a multiple of every block size. */
    static final int CHUNK_BYTES = 1 << 20;

    private GgufHasher() {
    }

    /** A hash algorithm of the tool and the name its output lines carry. */
    public enum Algorithm {
        XXH64("xxh64"),
        SHA1("sha1"),
        SHA256("sha256"),
        UUID("uuid");

        private final String text;

        Algorithm(String text) {
            this.text = text;
        }

        /** The name the reference prints for the algorithm. */
        public String text() {
            return text;
        }

        /** The algorithm whose printed name is {@code text}, or null. */
        public static Algorithm of(String text) {
            for (Algorithm a : values()) {
                if (a.text.equals(text)) {
                    return a;
                }
            }
            return null;
        }
    }

    /** One printed hash line: the algorithm, its digest text and the target it belongs to. */
    public static final class Line {

        private final Algorithm algorithm;
        private final String hash;
        private final String target;

        Line(Algorithm algorithm, String hash, String target) {
            this.algorithm = algorithm;
            this.hash = hash;
            this.target = target;
        }

        public Algorithm algorithm() {
            return algorithm;
        }

        /** The digest text: hex for xxh64, sha1 and sha256; a UUID for uuid. */
        public String hash() {
            return hash;
        }

        /** {@code <file>:<tensor>} for a tensor line, {@code <file>} for a model line. */
        public String target() {
            return target;
        }
    }

    /** The lines of one run, in the order the tool prints them. */
    public static final class Result {

        private final List<Line> lines;

        Result(List<Line> lines) {
            this.lines = Collections.unmodifiableList(new ArrayList<>(lines));
        }

        /** Every line, tensor lines first. */
        public List<Line> lines() {
            return lines;
        }
    }

    /**
     * Hashes {@code src} with {@code algorithms}; {@code noLayer} suppresses the per-tensor lines (the model
     * digests are still computed from every tensor).
     *
     * @throws IllegalArgumentException when {@code algorithms} is empty
     * @throws IOException              when the tensor bytes cannot be read
     */
    public static Result hash(GGUFFileParser src, Set<Algorithm> algorithms, boolean noLayer) throws IOException {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(algorithms, "algorithms");
        if (algorithms.isEmpty()) {
            throw new IllegalArgumentException("at least one hash algorithm is needed");
        }
        Set<Algorithm> set = EnumSet.copyOf(algorithms);
        boolean perTensorXxh = set.contains(Algorithm.XXH64) && !noLayer;
        boolean perTensorSha1 = set.contains(Algorithm.SHA1) && !noLayer;
        boolean perTensorSha256 = set.contains(Algorithm.SHA256) && !noLayer;
        boolean modelXxh = set.contains(Algorithm.XXH64);
        boolean modelSha1 = set.contains(Algorithm.SHA1);
        boolean modelSha256 = set.contains(Algorithm.SHA256);
        boolean modelUuid = set.contains(Algorithm.UUID);

        String file = src.getFilePath();
        Xxh64 modelXxhState = modelXxh ? new Xxh64() : null;
        MessageDigest modelSha1State = modelSha1 ? UuidV5.sha1() : null;
        MessageDigest modelSha256State = modelSha256 ? UuidV5.sha256() : null;
        MessageDigest uuidState = modelUuid ? UuidV5.sha1() : null;
        if (uuidState != null) {
            uuidState.update(UuidV5.namespaceBytes(UuidV5.NAMESPACE_LLAMA_CPP));
        }

        List<Line> lines = new ArrayList<>();
        byte[] chunk = new byte[CHUNK_BYTES];
        for (GGUFTensorInfo tensor : src.getTensors()) {
            String target = file + ":" + tensor.name();
            Xxh64 layerXxh = perTensorXxh ? new Xxh64() : null;
            MessageDigest layerSha1 = perTensorSha1 ? UuidV5.sha1() : null;
            MessageDigest layerSha256 = perTensorSha256 ? UuidV5.sha256() : null;
            long bytes = tensor.sizeInBytes();
            long done = 0;
            while (done < bytes) {
                int n = (int) Math.min(chunk.length, bytes - done);
                src.readTensorData(tensor, done, chunk, 0, n);
                if (layerXxh != null) {
                    layerXxh.update(chunk, 0, n);
                }
                if (layerSha1 != null) {
                    layerSha1.update(chunk, 0, n);
                }
                if (layerSha256 != null) {
                    layerSha256.update(chunk, 0, n);
                }
                if (modelXxhState != null) {
                    modelXxhState.update(chunk, 0, n);
                }
                if (modelSha1State != null) {
                    modelSha1State.update(chunk, 0, n);
                }
                if (modelSha256State != null) {
                    modelSha256State.update(chunk, 0, n);
                }
                if (uuidState != null) {
                    uuidState.update(chunk, 0, n);
                }
                done += n;
            }
            if (layerXxh != null) {
                lines.add(new Line(Algorithm.XXH64, Xxh64.hex(layerXxh.digest()), target));
            }
            if (layerSha1 != null) {
                lines.add(new Line(Algorithm.SHA1, UuidV5.hex(layerSha1.digest()), target));
            }
            if (layerSha256 != null) {
                lines.add(new Line(Algorithm.SHA256, UuidV5.hex(layerSha256.digest()), target));
            }
        }
        if (modelXxh) {
            lines.add(new Line(Algorithm.XXH64, Xxh64.hex(modelXxhState.digest()), file));
        }
        if (modelSha1) {
            lines.add(new Line(Algorithm.SHA1, UuidV5.hex(modelSha1State.digest()), file));
        }
        if (modelSha256) {
            lines.add(new Line(Algorithm.SHA256, UuidV5.hex(modelSha256State.digest()), file));
        }
        if (modelUuid) {
            lines.add(new Line(Algorithm.UUID, UuidV5.fromDigest(uuidState.digest()).toString(), file));
        }
        return new Result(lines);
    }
}
