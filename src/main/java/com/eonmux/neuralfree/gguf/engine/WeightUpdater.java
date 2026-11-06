package com.eonmux.neuralfree.gguf.engine;

/**
 * Live weight updates into one open model (vLLM's in-process weight transfer engine): a trainer writes new tensor
 * contents over the weights a running engine serves, without reloading the model, so an RLHF-style loop trains in
 * one part of the process and inference picks the weights up between steps.
 *
 * <p>The lifecycle is the reference control plane: {@link #start} opens an update round, {@link #apply} writes one
 * tensor (it also works without a round, as vLLM's in-place engines do), {@link #finish} commits the round and may
 * stamp its version, and {@link #shutdown} releases the updater. The version is a free-form label the caller reads
 * back ({@code "default"} until first set), so a distributed trainer can confirm every replica serves the same
 * round. An updater created by {@link EngineModel#weightUpdater()} is not thread-safe: callers run it on the model's
 * own thread, between forward passes.
 *
 * <p>Every refusal names what failed: an unknown tensor, a shape or element type that does not match the model's,
 * or a tensor whose placement cannot take updates (streamed, host-placed or split tensors).
 */
public interface WeightUpdater extends AutoCloseable {

    /** The version an updater reports before the first {@link #finish} or {@link #setVersion}. */
    String DEFAULT_VERSION = "default";

    /** The backend name the in-process transfer registers under (vLLM's factory key). */
    String INPROCESS_BACKEND = "inprocess";

    /**
     * Begins an update round. Repeating it while one is open begins a new one, as a trainer retries do.
     *
     * @throws IllegalStateException when the updater is shut down
     */
    void start();

    /**
     * Writes {@code tensor} over the model's weights. The next forward pass the model runs serves the new
     * contents; cached context computed under earlier weights stays until the caller drops it (vLLM leaves
     * {@code reset_prefix_cache} to the RL loop too).
     *
     * @throws IllegalArgumentException      when the model has no tensor of that name, or the shape or element
     *                                       type does not match
     * @throws UnsupportedOperationException when the tensor's placement cannot take updates (streamed,
     *                                       host-placed, split or quantization-requiring), naming it
     * @throws IllegalStateException         when the updater is shut down
     */
    void apply(WeightUpdateTensor tensor);

    /**
     * Commits the current round: nothing is pending after it returns, and {@code version}, when given, becomes the
     * reported one ({@code finish_weight_update}'s optional {@code weight_version}). Null leaves it.
     *
     * @throws IllegalStateException when the updater is shut down
     */
    void finish(String version);

    /**
     * Sets the reported version without writing weights ({@code update_weight_version}).
     *
     * @throws IllegalStateException when the updater is shut down
     */
    void setVersion(String version);

    /** The committed weight version. */
    String version();

    /** True between a {@link #start} and its {@link #finish}. */
    boolean updating();

    /**
     * Releases the updater (vLLM's {@code shutdown_weight_transfer_engine}): further updates refuse naming the
     * shutdown. The model itself stays open. Idempotent.
     */
    void shutdown();

    /** {@link #shutdown()}, for try-with-resources. */
    @Override
    void close();
}
