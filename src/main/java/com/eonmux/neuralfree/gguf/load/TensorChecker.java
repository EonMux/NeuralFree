package com.eonmux.neuralfree.gguf.load;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * The tensor value validation of llama.cpp {@code --check-tensors}: every tensor of a source is decoded to F32
 * with {@link CpuBlockDecoders} and rejected when any of its values is NaN or infinite. Tensors are checked in
 * parallel, one worker per tensor up to the processor count; the first failing tensor in the tensor table is
 * thrown with the later failures attached as suppressed, so a load stops with a message naming the tensor that
 * fails. A type with no CPU decoder is skipped here and refused where the load itself reads it.
 *
 * <p>The checks read the same bytes the model would load, through the source's regions in the process's
 * {@linkplain TensorReadMode#current() read mode}; the regions are closed before {@code check} returns. Thread
 * safety comes from the regions ({@link TensorRegions} implementations are thread-safe); the workers share no
 * mutable state.
 */
public final class TensorChecker {

    /** Bytes of tensor data one worker read / decode / check pass covers. */
    private static final int CHUNK_BYTES = 1 << 20;
    /** Decoded floats one pass may keep, so blocks with many elements keep the pass small. */
    private static final int CHUNK_ELEMENTS = 1 << 16;

    private TensorChecker() {
    }

    /**
     * Checks every tensor of {@code source} value by value.
     *
     * @throws IOException when a tensor's bytes cannot be read or a checked tensor holds NaN or an infinite value
     * @throws InterruptedException when the scan is interrupted
     */
    public static void check(ModelSource source) throws IOException, InterruptedException {
        Objects.requireNonNull(source, "source");
        List<GGUFTensorInfo> tensors = source.getTensors();
        if (tensors.isEmpty()) {
            return;
        }
        String path = source.getFilePath();
        TensorRegions regions = source.openRegions(TensorReadMode.current());
        List<IOException> failures = new ArrayList<>();
        ExecutorService pool = null;
        try {
            int threads = Math.min(Runtime.getRuntime().availableProcessors(), tensors.size());
            pool = Executors.newFixedThreadPool(threads);
            List<Callable<IOException>> checks = new ArrayList<>();
            for (final GGUFTensorInfo info : tensors) {
                checks.add(new Callable<IOException>() {
                    @Override
                    public IOException call() {
                        try {
                            checkTensor(regions, info, path);
                        } catch (IOException e) {
                            return e;
                        }
                        return null;
                    }
                });
            }
            List<Future<IOException>> futures = pool.invokeAll(checks);
            for (int i = 0; i < futures.size(); i++) {
                IOException e;
                try {
                    e = futures.get(i).get();
                } catch (ExecutionException x) {
                    Throwable c = x.getCause() == null ? x : x.getCause();
                    e = c instanceof IOException ? (IOException) c : new IOException(c.getMessage(), c);
                }
                if (e != null) {
                    e.addSuppressed(new Exception("tensor " + tensors.get(i).name()));
                    failures.add(e);
                }
            }
        } finally {
            if (pool != null) {
                pool.shutdown();
            }
            try {
                regions.close();
            } catch (IOException closeFailure) {
                for (IOException f : failures) {
                    closeFailure.addSuppressed(f);
                }
                failures = new ArrayList<>();
                failures.add(closeFailure);
            }
        }
        if (!failures.isEmpty()) {
            IOException first = failures.get(0);
            for (int i = 1; i < failures.size(); i++) {
                first.addSuppressed(failures.get(i));
            }
            throw first;
        }
    }

    /**
     * Checks one tensor chunk by chunk: reads {@code chunk} bytes, decodes them block-wise to F32 and rejects the
     * first NaN or infinity with the value's block index.
     */
    private static void checkTensor(TensorRegions regions, GGUFTensorInfo info, String path) throws IOException {
        GGMLType type = info.type();
        if (!CpuBlockDecoders.supports(type)) {
            return;
        }
        int typeSize = type.getTypeSize();
        int blockElements = type.getBlockSize();
        long size = info.sizeInBytes();
        long chunkBlocks = Math.max(1, Math.min(CHUNK_BYTES / typeSize, CHUNK_ELEMENTS / blockElements));
        long chunk = chunkBlocks * typeSize;
        float[] out = new float[(int) (chunkBlocks * blockElements)];
        long pos = 0;
        while (pos < size) {
            long len = Math.min(chunk, size - pos);
            if (pos % typeSize != 0 || len % typeSize != 0) {
                throw new IOException("cannot check tensor '" + info.name() + "' of " + path
                        + ": its bytes are not aligned to one " + type + " block at offset " + pos);
            }
            int n = (int) (len / typeSize);
            ByteBuffer row = regions.region(info, pos, len);
            CpuBlockDecoders.decode(type, row, 0, out, 0, n);
            long block0 = pos / typeSize;
            for (int b = 0; b < n; b++) {
                for (int i = 0; i < blockElements; i++) {
                    float v = out[b * blockElements + i];
                    if (v != v || v == Float.NEGATIVE_INFINITY || v == Float.POSITIVE_INFINITY) {
                        throw new IOException("Invalid model file '" + path + "': tensor '" + info.name()
                                + "' has non-finite data: " + (v != v ? "NaN" : v) + " in block "
                                + (block0 + b) + " element " + i);
                    }
                }
            }
            pos += len;
        }
    }
}
