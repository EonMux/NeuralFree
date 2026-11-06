package com.eonmux.neuralfree.demo.tools;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * The base-logits file of llama.cpp's perplexity tool ({@code --save-all-logits} /
 * {@code --kl-divergence-base}, tools/perplexity/perplexity.cpp): the magic {@code _logits_}, the context
 * size, the vocabulary size and chunk count, the evaluation tokens, then per chunk the stored rows of
 * {@link PerplexityMath#quantLogSoftmaxRow} (little-endian throughout). The layout is llama.cpp's byte for
 * byte, so each tool reads the other's file and compares their KLD.
 */
final class LogitsFile {

    /** The first 8 bytes of the file ({@code "_logits_"}). */
    private static final byte[] MAGIC = "_logits_".getBytes(StandardCharsets.US_ASCII);

    private LogitsFile() {
    }

    /** Appends the header and the stored rows of a perplexity run to a file. */
    static final class Out implements AutoCloseable {

        private final OutputStream out;

        /** Creates (replacing) the file and writes the magic and the context size. */
        Out(Path path, int nCtx) throws IOException {
            this.out = new BufferedOutputStream(Files.newOutputStream(path));
            try {
                out.write(MAGIC);
                writeLeInt(out, nCtx);
            } catch (IOException e) {
                close();
                throw e;
            }
        }

        /** Writes the vocabulary size, the chunk count and the evaluation tokens of the whole run. */
        void writeRun(int nVocab, int nChunk, int[] tokens, int length) throws IOException {
            writeLeInt(out, nVocab);
            writeLeInt(out, nChunk);
            for (int i = 0; i < length; i++) {
                writeLeInt(out, tokens[i]);
            }
        }

        /** Appends {@code count} stored rows of {@code rowLength} entries from the buffer. */
        void writeRows(short[] rows, int count, int rowLength) throws IOException {
            int end = count * rowLength;
            for (int i = 0; i < end; i++) {
                writeLeShort(out, rows[i] & 0xFFFF);
            }
        }

        @Override
        public void close() throws IOException {
            out.close();
        }
    }

    /** A base file opened for a KL-divergence run: the header, the tokens and the rows in chunk order. */
    static final class In implements AutoCloseable {

        private final InputStream in;
        private final Path path;
        private final int nCtx;
        private final int nVocab;
        private final int nChunk;
        private final int[] tokens;

        /** Opens the file and reads the header, refusing anything that is not a base-logits file. */
        In(Path path) throws IOException {
            this.path = path;
            this.in = new BufferedInputStream(Files.newInputStream(path));
            try {
                byte[] magic = new byte[MAGIC.length];
                if (in.read(magic) != magic.length || !Arrays.equals(magic, MAGIC)) {
                    throw new IOException(path + " does not look like a file containing log-probabilities"
                            + " (missing the _logits_ header)");
                }
                nCtx = readLeInt("the context size");
                nVocab = readLeInt("the vocabulary size");
                nChunk = readLeInt("the chunk count");
                long tokensLeft = (long) nCtx * nChunk;
                if (nCtx <= 0 || nVocab <= 0 || nChunk <= 0 || tokensLeft > Integer.MAX_VALUE - 8L) {
                    throw new IOException(path + " holds an impossible header: context " + nCtx + ", vocab "
                            + nVocab + ", chunks " + nChunk);
                }
                tokens = new int[(int) tokensLeft];
                for (int i = 0; i < tokens.length; i++) {
                    tokens[i] = readLeInt("the evaluation tokens");
                }
            } catch (IOException e) {
                close();
                throw e;
            }
        }

        int nCtx() {
            return nCtx;
        }

        int nVocab() {
            return nVocab;
        }

        int nChunk() {
            return nChunk;
        }

        int[] tokens() {
            return tokens;
        }

        /**
         * Reads the next chunk's stored rows into the buffer (a chunk writes one row per evaluated
         * position, each {@link PerplexityMath#rowLength} entries).
         *
         * @throws IOException naming the file when the rows are missing or truncated
         */
        void readRows(short[] rows) throws IOException {
            for (int i = 0; i < rows.length; i++) {
                int v;
                try {
                    v = readLeShort();
                } catch (EOFException e) {
                    throw new IOException(path + " ends inside the stored rows", e);
                }
                rows[i] = (short) v;
            }
        }

        private int readLeInt(String what) throws IOException {
            int b0 = in.read();
            int b1 = in.read();
            int b2 = in.read();
            int b3 = in.read();
            if ((b0 | b1 | b2 | b3) < 0) {
                throw new IOException(path + " ends inside " + what);
            }
            return (b3 << 24) | (b2 << 16) | (b1 << 8) | b0;
        }

        private int readLeShort() throws IOException {
            int b0 = in.read();
            int b1 = in.read();
            if ((b0 | b1) < 0) {
                throw new EOFException();
            }
            return (b1 << 8) | b0;
        }

        @Override
        public void close() throws IOException {
            in.close();
        }
    }

    private static void writeLeInt(OutputStream out, int v) throws IOException {
        out.write(v & 0xFF);
        out.write((v >>> 8) & 0xFF);
        out.write((v >>> 16) & 0xFF);
        out.write((v >>> 24) & 0xFF);
    }

    private static void writeLeShort(OutputStream out, int v) throws IOException {
        out.write(v & 0xFF);
        out.write((v >>> 8) & 0xFF);
    }
}
