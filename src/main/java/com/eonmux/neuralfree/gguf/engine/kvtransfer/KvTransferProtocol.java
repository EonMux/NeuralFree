package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * The transfer wire format of the KV connectors and the shared-storage payload file: a magic, a version, a frame
 * type and the frame's fields as big-endian primitives.
 *
 * <p>A consumer sends a {@code GET} naming the transfer id (and the optional shared token); the producer answers a
 * {@code PAYLOAD} — the transfer id, its engine id, the prompt tokens and the chain bytes — or an {@code ERROR} with
 * the reason. The same payload frame is what {@link FileKvConnector} writes to its storage file.</p>
 */
final class KvTransferProtocol {

    /** "NFKV". */
    static final int MAGIC = 0x4E464B56;
    static final int VERSION = 1;

    static final int GET = 1;
    static final int PAYLOAD = 2;
    static final int ERROR = 3;

    private KvTransferProtocol() {
    }

    /** The magic, version and frame type every frame opens with. */
    static void header(DataOutputStream out, int type) throws IOException {
        out.writeInt(MAGIC);
        out.writeInt(VERSION);
        out.writeInt(type);
    }

    /** Reads and checks the magic, version and frame type; returns the type. */
    static int header(DataInputStream in, String source) throws IOException {
        int magic = in.readInt();
        if (magic != MAGIC) {
            throw new IOException(source + " is not a NeuralFree KV transfer frame");
        }
        int version = in.readInt();
        if (version != VERSION) {
            throw new IOException(source + " holds KV transfer format " + version + ", this build reads " + VERSION);
        }
        return in.readInt();
    }

    /** Writes a {@code GET} frame. */
    static void writeGet(DataOutputStream out, String transferId, String token) throws IOException {
        header(out, GET);
        out.writeUTF(transferId);
        out.writeBoolean(token != null);
        if (token != null) {
            out.writeUTF(token);
        }
        out.flush();
    }

    /** Reads a {@code GET} frame's fields. */
    static String[] readGet(DataInputStream in) throws IOException {
        String transferId = in.readUTF();
        String token = in.readBoolean() ? in.readUTF() : null;
        return new String[] {transferId, token};
    }

    /** Writes a {@code PAYLOAD} frame. */
    static void writePayload(DataOutputStream out, KvTransferPayload payload) throws IOException {
        header(out, PAYLOAD);
        out.writeUTF(payload.transferId());
        out.writeUTF(payload.engineId());
        out.writeInt(payload.promptTokens().length);
        for (int token : payload.promptTokens()) {
            out.writeInt(token);
        }
        out.writeLong(payload.chain().length);
        out.write(payload.chain());
        out.flush();
    }

    /**
     * Reads a {@code PAYLOAD} frame.
     *
     * @param maxChainBytes the largest chain this peer accepts; a larger declared length is refused before any
     *                      allocation
     * @throws IOException for a malformed frame, a negative token count or a chain over the bound
     */
    static KvTransferPayload readPayload(DataInputStream in, long maxChainBytes, String source) throws IOException {
        String transferId = in.readUTF();
        String engineId = in.readUTF();
        int tokens = in.readInt();
        if (tokens < 1 || tokens > 1 << 24) {
            throw new IOException(source + " declares " + tokens + " prompt tokens");
        }
        int[] promptTokens = new int[tokens];
        for (int i = 0; i < tokens; i++) {
            promptTokens[i] = in.readInt();
        }
        long chainBytes = in.readLong();
        if (chainBytes < 1 || chainBytes > maxChainBytes || chainBytes > Integer.MAX_VALUE) {
            throw new IOException(source + " holds a chain of " + chainBytes + " bytes, outside [1, "
                    + Math.min(maxChainBytes, Integer.MAX_VALUE) + "]");
        }
        byte[] chain = new byte[(int) chainBytes];
        in.readFully(chain);
        return new KvTransferPayload(transferId, engineId, promptTokens, chain);
    }

    /** Writes an {@code ERROR} frame and flushes it. */
    static void writeError(DataOutputStream out, String message) throws IOException {
        header(out, ERROR);
        out.writeUTF(message == null ? "KV transfer failed" : message);
        out.flush();
    }

    /** Reads an {@code ERROR} frame's message. */
    static String readError(DataInputStream in) throws IOException {
        return in.readUTF();
    }

    /** Wraps an {@link EOFException} with {@code source} so a truncated peer names itself. */
    static IOException truncated(EOFException e, String source) {
        return new IOException(source + " ended before its frame did", e);
    }

    /** Writes a whole payload as a stand-alone file body (the shared-storage format). */
    static void writeFile(OutputStream out, KvTransferPayload payload) throws IOException {
        DataOutputStream data = new DataOutputStream(out);
        writePayload(data, payload);
        data.flush();
    }

    /** Reads a stand-alone payload body written by {@link #writeFile}. */
    static KvTransferPayload readFile(InputStream in, long maxChainBytes, String source) throws IOException {
        DataInputStream data = new DataInputStream(in);
        int type = header(data, source);
        if (type != PAYLOAD) {
            if (type == ERROR) {
                throw new IOException(source + ": " + readError(data));
            }
            throw new IOException(source + " holds frame type " + type + " where a payload belongs");
        }
        return readPayload(data, maxChainBytes, source);
    }
}
