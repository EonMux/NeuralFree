package com.eonmux.neuralfree.gguf;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The destination {@link GgufWriter} streams an export's tensor bytes into: one GGUF file, or a rolling
 * set of split shard files named and keyed the way {@link GGUFFileParser#loadShards()} reads them
 * (llama.cpp {@code llama_split_path}/{@code llama_split_prefix}, src/llama.cpp:570-596). One sink is
 * created per export by {@link GgufWriter}, which alone assigns the tensors to files, composes every
 * shard's key/value set and writes the bytes; the sink owns the files: it opens each output from its
 * {@code .tmp} twin, hands back the {@link GgufEncoder} the file is written with, forces its bytes to
 * disk and renames it atomically, so a failing export never replaces a previous file. {@link #close()}
 * runs either way and deletes every temporary file of the export that never reached its final name.
 *
 * <p>Files are written strictly in order: {@code begin(i)} is invalid until {@code end(i - 1)} renamed
 * the previous one. Not thread-safe; {@link GgufWriter} is the only thread.</p>
 */
public interface GgufTensorSink extends AutoCloseable {

    /** The number of output files: one for a single-file export, the shard count for the split. */
    int files();

    /**
     * The final path of output file {@code index}: the given target for a single file, the llama.cpp
     * shard name {@code <prefix>-<shard, %05d>-of-<shardCount, %05d>.gguf} for a split.
     */
    Path file(int index);

    /**
     * Opens output file {@code index} of the export from its {@code .tmp} twin and hands back the
     * encoder the caller writes the file's header, key/values, tensor infos and tensor data with. The
     * file reaches its final name only when {@link #end} closes it.
     */
    GgufEncoder begin(int index) throws IOException;

    /**
     * The file channel {@link #begin} opened for output file {@code index}; {@link GgufWriter} passes it
     * to {@link GgufEncoder#writeHeader}, whose signature keeps the file's channel explicit. Null is
     * refused with an error unless {@link #begin} ran; never call it after {@link #end}.
     */
    FileChannel channel(int index);

    /** Forces output file {@code index} to disk, closes its channel and renames it to {@link #file}. */
    void end(int index) throws IOException;

    /**
     * Closes the sink: the temporary twin of every output file that never reached its final name is
     * deleted, leaving any previous output of the export untouched.
     *
     * @throws IOException when a temporary file cannot be closed or deleted
     */
    @Override
    void close() throws IOException;

    /**
     * A sink whose sole output file is {@code out}, written to {@code out + ".tmp"} and renamed
     * atomically, replacing an existing {@code out} only when {@code overwrite} allows it.
     */
    static GgufTensorSink singleFile(Path out, boolean overwrite) {
        return new Sink(List.of(new GGUFFile(out, overwrite)));
    }

    /**
     * A sink whose output files are the shard names derived from {@code out}'s split prefix
     * ({@link #shardPrefix}): the shards share {@code out}'s directory, which must exist.
     */
    static GgufTensorSink shards(Path out, int shardCount, boolean overwrite) {
        String prefix = shardPrefix(out);
        List<GGUFFile> files = new ArrayList<>(shardCount);
        for (int i = 0; i < shardCount; i++) {
            String name = String.format(Locale.ROOT, "%s-%05d-of-%05d.gguf", prefix, i + 1, shardCount);
            files.add(new GGUFFile(out.resolveSibling(name), overwrite));
        }
        return new Sink(files);
    }

    /**
     * The split prefix of {@code out}: the file name with a trailing {@code -00001-of-00001.gguf} (a
     * llama.cpp split name; {@code llama_split_prefix} strips exactly that form) or a trailing
     * {@code .gguf} removed, and the name as it is when it carries neither suffix. Never empty: a name
     * that is exactly one of the suffixes is rejected, as llama.cpp rejects a {@code llama_split_prefix}
     * call that would return empty.
     */
    static String shardPrefix(Path out) {
        String name = out.getFileName().toString();
        String split = String.format(Locale.ROOT, "-%05d-of-%05d.gguf", 1, 1);
        String suffix = name.endsWith(split) ? split
                : name.endsWith(".gguf") ? ".gguf"
                : "";
        String prefix = name.substring(0, name.length() - suffix.length());
        if (prefix.isEmpty()) {
            throw new IllegalArgumentException("split prefix of '" + name + "' is empty; name the export "
                    + "'<prefix>" + split + "'");
        }
        return prefix;
    }

    /** The file machinery both sinks share: {@code .tmp} twin, encoder, forced close, atomic rename. */
    final class GGUFFile {

        private final Path target;
        private final Path tmp;
        private final boolean overwrite;
        /** open() to rename(): the channel of the file's temporary twin; null otherwise. */
        private FileChannel channel;
        /** {@code true} once {@link #rename} moved the twin over {@link #target}. */
        private boolean renamed;

        GGUFFile(Path target, boolean overwrite) {
            this.target = target;
            this.overwrite = overwrite;
            this.tmp = target.resolveSibling(target.getFileName().toString() + ".tmp");
        }

        Path target() {
            return target;
        }

        /** {@code true} once {@link #rename} moved the twin over {@link #target}. */
        boolean renamed() {
            return renamed;
        }

        FileChannel currentChannel() {
            return channel;
        }

        /** Opens {@code tmp} and returns the encoder the file is written with. */
        GgufEncoder open() throws IOException {
            if (channel != null) {
                throw new IllegalStateException(target + " is already open");
            }
            channel = FileChannel.open(tmp, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                    StandardOpenOption.TRUNCATE_EXISTING);
            return new GgufEncoder(channel, tmp.toString());
        }

        /** Forces, closes and atomically renames the file over its target. */
        void rename() throws IOException {
            if (channel == null) {
                throw new IllegalStateException(target + "'s encoder was never opened");
            }
            try {
                try {
                    channel.force(true);
                } finally {
                    channel.close();
                }
                if (overwrite) {
                    try {
                        Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE,
                                StandardCopyOption.REPLACE_EXISTING);
                    } catch (AtomicMoveNotSupportedException e) {
                        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
                    }
                } else {
                    // A plain move, not ATOMIC_MOVE: the atomic move flag replaces an existing target on
                    // rename(2) platforms, which would break the overwrite refusal this sink promises.
                    Files.move(tmp, target);
                }
                renamed = true;
            } finally {
                channel = null;
            }
        }

        /** Closes a half-written file's channel and deletes its temporary twin. */
        void discard() throws IOException {
            renamed = false;
            try {
                if (channel != null) {
                    channel.close();
                    channel = null;
                }
            } finally {
                Files.deleteIfExists(tmp);
            }
        }
    }

    /** The one sink implementation: every output file of the export, written in order. */
    final class Sink implements GgufTensorSink {

        private final List<GGUFFile> files;
        private int open;
        private boolean closed;

        private Sink(List<GGUFFile> files) {
            this.files = files;
        }

        @Override
        public int files() {
            return files.size();
        }

        @Override
        public Path file(int index) {
            return files.get(index).target();
        }

        @Override
        public GgufEncoder begin(int index) throws IOException {
            if (index != open) {
                throw new IllegalStateException("output file " + index + " taken before file " + open
                        + " was closed; the files of one export are written in order");
            }
            open++;
            return files.get(index).open();
        }

        @Override
        public FileChannel channel(int index) {
            FileChannel channel = files.get(index).currentChannel();
            if (channel == null) {
                throw new IllegalStateException("file " + index + " was never opened or is already closed");
            }
            return channel;
        }

        @Override
        public void end(int index) throws IOException {
            files.get(index).rename();
        }

        @Override
        public void close() throws IOException {
            if (closed) {
                return;
            }
            closed = true;
            IOException failure = null;
            for (GGUFFile f : files) {
                try {
                    if (f.renamed()) {
                        continue;
                    }
                    f.discard();
                } catch (IOException e) {
                    if (failure == null) {
                        failure = e;
                    }
                }
            }
            if (failure != null) {
                throw failure;
            }
        }
    }
}
