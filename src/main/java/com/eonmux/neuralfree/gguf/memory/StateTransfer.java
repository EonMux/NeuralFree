package com.eonmux.neuralfree.gguf.memory;

import java.util.Collections;
import java.util.List;

/**
 * Host transfer of a {@link ModelMemory}'s stored state, the primitive under the session files, the slot files and
 * the host-RAM prompt cache (llama.cpp's {@code state_write_data} / {@code state_read_data} over any backend): the
 * raw rows of every owning KV cache layer, their side planes (the per-(token, KV head) scales of a per-token-head
 * cache), the live recurrent state tensors and the contiguous form's per-group position mirrors.
 *
 * <p>Rows cross in the cache's own dtype: {@link KvLayout#rowBytesK()} / {@link KvLayout#rowBytesV()} bytes per
 * row, little-endian, the layout the CPU caches and the device kernels share ({@link KvBlockCodec}), so a row read
 * from one backend writes unchanged into the other and quantized rows keep their exact bits. Reads return once the
 * bytes are on the host; writes are ordered before the next forward step of the memory. Used by the thread that
 * drives the memory.
 */
public interface StateTransfer {

    /** One owning KV cache layer of the memory (the views of shared-KV layers excluded). */
    interface Cache {

        /** The virtual layer that owns the cache. */
        int layer();

        /** The KV group addressing its rows. */
        int group();

        /** Row widths, dtypes and row count of the cache. */
        KvLayout layout();

        /**
         * Copies rows {@code [row0, row0 + rows)} of K ({@code v} false) or V into {@code dst[0, rows * rowBytes)}.
         *
         * @throws IllegalArgumentException  for V of a layout whose V aliases K, or a destination too small
         * @throws IndexOutOfBoundsException for rows outside the cache
         */
        void read(boolean v, int row0, int rows, byte[] dst);

        /**
         * Stores {@code src[0, rows * rowBytes)} as rows {@code [row0, row0 + rows)} of K or V.
         *
         * @throws IllegalArgumentException  for V of a layout whose V aliases K, or a source too small
         * @throws IndexOutOfBoundsException for rows outside the cache
         */
        void write(boolean v, int row0, int rows, byte[] src);

        /**
         * The side planes stored beside the K ({@code v} false) or V rows, in the dtype's
         * {@link KvDType#sidePlanes()} order: the per-(token, KV head) scale rows of a per-token-head cache, the
         * NVFP4 group scales, the TurboQuant norm / scale / zero rows and the DeepSeek MLA latent scales, empty for
         * every other type. A plane row is {@link KvDType.SidePlane#rowBytesFor(int, int)} bytes, sized by the row
         * width and the layout's head count.
         */
        default List<KvDType.SidePlane> sidePlanes(boolean v) {
            return Collections.emptyList();
        }

        /**
         * Copies rows {@code [row0, row0 + rows)} of side plane {@code plane} of K ({@code v} false) or V into
         * {@code dst[0, rows * planeRowBytes)}.
         *
         * @throws UnsupportedOperationException for a cache whose side planes do not cross the host transfer
         * @throws IndexOutOfBoundsException     for rows outside the cache
         */
        default void readSide(boolean v, int plane, int row0, int rows, byte[] dst) {
            throw new UnsupportedOperationException("side planes do not cross the host transfer of "
                    + getClass().getSimpleName());
        }

        /** Stores {@code src[0, rows * planeRowBytes)} as rows {@code [row0, row0 + rows)} of side plane {@code plane}. */
        default void writeSide(boolean v, int plane, int row0, int rows, byte[] src) {
            throw new UnsupportedOperationException("side planes do not cross the host transfer of "
                    + getClass().getSimpleName());
        }
    }

    /** The recurrent state of one layer: its F32 tensors by name. */
    interface State {

        /** The virtual layer that owns the state. */
        int layer();

        /** The planned tensors, in plan order. */
        List<LayerMemoryPlan.StateTensor> tensors();

        /**
         * A copy of the tensor's every float.
         *
         * @throws IllegalArgumentException for a name the layer does not plan
         */
        float[] read(String name);

        /**
         * Replaces the tensor's floats with {@code values}.
         *
         * @throws IllegalArgumentException for a name the layer does not plan or a length that differs from it
         */
        void write(String name, float[] values);
    }

    /** The owning KV cache layers, in allocation order. */
    List<Cache> caches();

    /** The recurrent-state layers, in allocation order. */
    List<State> states();

    /**
     * A copy of the contiguous form's position mirror of KV group {@code group}: the position each row stores,
     * -1 for an empty row.
     *
     * @throws IndexOutOfBoundsException for a group outside the plan
     */
    int[] positions(int group);

    /**
     * Installs {@code mirror} as the position record of KV group {@code group}, host and device alike, so
     * attention reads exactly the rows it names.
     *
     * @throws IllegalArgumentException  when the mirror's length differs from the group's rows
     * @throws IndexOutOfBoundsException for a group outside the plan
     */
    void setPositions(int group, int[] mirror);
}
