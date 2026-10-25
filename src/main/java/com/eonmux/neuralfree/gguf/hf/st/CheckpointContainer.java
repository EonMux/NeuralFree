package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.StTensor;

import java.io.IOException;
import java.util.List;

/**
 * The stored tensors of a Hugging Face checkpoint, whatever the container files are: every tensor's entry, a
 * {@link RowSource} over its bytes, the {@link DataSegment}s that place the files' data in one virtual address space
 * (tensors stored there are mapped in place), and side files of the model directory. Implementations are registered
 * in {@link CheckpointContainers}; they are immutable after opening and thread-safe until closed.
 */
public interface CheckpointContainer extends AutoCloseable {

    /** What was opened (path, file and tensor counts), for messages. */
    String description();

    /** Every tensor in the checkpoint's order (unmodifiable); names are unique. */
    List<StTensor> tensors();

    /** The entry of the tensor {@code name}, or null when the checkpoint has none. */
    StTensor tensor(String name);

    /**
     * A row source over the stored bytes of {@code name}.
     *
     * @throws IllegalArgumentException when the checkpoint has no such tensor
     */
    RowSource source(String name);

    /** The data segments of the checkpoint files, in increasing virtual order (unmodifiable). */
    List<DataSegment> segments();

    /**
     * The virtual offset of {@code name}'s bytes inside {@link #segments()}, or -1 when the tensor is not stored
     * there as is.
     *
     * @throws IllegalArgumentException when the checkpoint has no such tensor
     */
    long directVirtualOffset(String name);

    /**
     * Opens a safetensors side file of the model directory ({@code readout.safetensors}, {@code 2_Dense/...}). The
     * name comes from file contents or code and must be a relative path inside the model directory. The container
     * closes side files still open when it is closed.
     *
     * @throws java.io.FileNotFoundException when the file does not exist
     * @throws IOException                   when the name is not a path inside the model directory, or the file is
     *                                       too large or invalid
     */
    SafetensorsFile openSideFile(String relPath) throws IOException;

    /** Closes every file of the checkpoint and every side file it opened. */
    @Override
    void close() throws IOException;
}
