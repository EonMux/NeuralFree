package com.eonmux.neuralfree.gguf.hf.store;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.FileTensorRegions;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The CPU tensor regions of a Hugging Face model source: stored tensors come from the checkpoint files through
 * {@link FileTensorRegions} (mapped or read per {@link TensorReadMode}); derived tensors come from the
 * {@link ProducedSegments.View} of the CPU reader, which produces a row-local tensor kept in memory into a new direct
 * buffer per region and hands out slices of the materialized copy otherwise. The reader may keep every buffer.
 * Thread-safe.
 */
public final class HfTensorRegions implements TensorRegions {

    private final String description;
    private final FileTensorRegions files;
    private final ProducedSegments.View produced;
    private final ProducedSegments table;
    private final AtomicBoolean closed = new AtomicBoolean();

    /**
     * Regions over {@code layout} for a CPU reader; {@code produced} must be a view of the layout's derived tensors
     * and is closed with the regions (also when construction fails).
     *
     * @throws IOException when a checkpoint file cannot be opened
     */
    public HfTensorRegions(VirtualLayout layout, ProducedSegments.View produced, TensorReadMode mode,
                           String description) throws IOException {
        Objects.requireNonNull(layout, "layout");
        this.produced = Objects.requireNonNull(produced, "produced");
        this.description = Objects.requireNonNull(description, "description");
        Objects.requireNonNull(mode, "mode");
        this.table = layout.produced();
        if (produced.segments() != table) {
            throw new IllegalArgumentException("the derived-tensor view belongs to another layout");
        }
        List<DataSegment> data = new ArrayList<>();
        for (DataSegment s : layout.segments()) {
            if (s.dataLength() > 0) {
                data.add(s);
            }
        }
        FileTensorRegions opened = null;
        try {
            opened = data.isEmpty() ? null : FileTensorRegions.open(data, mode, description);
        } catch (IOException | RuntimeException e) {
            try {
                produced.close();
            } catch (IOException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
        this.files = opened;
    }

    @Override
    public ByteBuffer region(GGUFTensorInfo info, long byteOff, long length) throws IOException {
        Objects.requireNonNull(info, "info");
        int index = table.indexOf(info.name());
        if (index >= 0 && table.offset(index) == info.offset()) {
            if (length < 0 || length > Integer.MAX_VALUE) {
                throw new IOException("tensor " + info.name() + ": region of " + length + " bytes of " + description
                        + " does not fit one buffer");
            }
            return produced.region(index, byteOff, (int) length);
        }
        if (files == null || info.offset() >= table.base()) {
            throw new IOException("tensor " + info.name() + " at virtual offset " + info.offset()
                    + " is neither stored in a file nor derived in " + description);
        }
        return files.region(info, byteOff, length);
    }

    @Override
    public String description() {
        return description;
    }

    /** Closes the derived-tensor view and the files; buffers handed out stay readable. Later calls do nothing. */
    @Override
    public void close() throws IOException {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        IOException first = null;
        try {
            produced.close();
        } catch (IOException e) {
            first = e;
        }
        if (files != null) {
            try {
                files.close();
            } catch (IOException e) {
                if (first == null) {
                    first = e;
                } else {
                    first.addSuppressed(e);
                }
            }
        }
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "HfTensorRegions{" + description + "}";
    }
}
