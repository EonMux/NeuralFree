package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.StateTransfer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The host transfer of a {@link DeviceModelMemory} ({@link StateTransfer}): K/V rows and recurrent state tensors move
 * between the device buffers and host arrays through the memory's queue, in order with its forward pass. A read
 * waits for every command enqueued before it and for its own bytes, then drains the queue so the backend recycles its
 * staging; a write is staged at once, so the caller may reuse its array.
 */
final class DeviceStateTransfer implements StateTransfer {

    private final ComputeQueue queue;
    private final DeviceStepInputs stepInputs;
    private final List<Cache> caches;
    private final List<State> states;

    DeviceStateTransfer(ComputeQueue queue, DeviceStepInputs stepInputs, List<DeviceKvLayer> kv,
                        List<DeviceRecurrentState> recurrent) {
        this.queue = Objects.requireNonNull(queue, "queue");
        this.stepInputs = Objects.requireNonNull(stepInputs, "stepInputs");
        List<Cache> c = new ArrayList<>(kv.size());
        for (DeviceKvLayer layer : kv) {
            c.add(new DeviceCache(layer));
        }
        List<State> s = new ArrayList<>(recurrent.size());
        for (DeviceRecurrentState state : recurrent) {
            s.add(new DeviceState(state));
        }
        this.caches = List.copyOf(c);
        this.states = List.copyOf(s);
    }

    @Override
    public List<Cache> caches() {
        return caches;
    }

    @Override
    public List<State> states() {
        return states;
    }

    @Override
    public int[] positions(int group) {
        return stepInputs.positions(group).snapshot();
    }

    @Override
    public void setPositions(int group, int[] mirror) {
        stepInputs.installPositions(group, Objects.requireNonNull(mirror, "mirror"));
    }

    /** Waits for {@code read}'s bytes on the host, releases the event and drains the queue. */
    private void complete(GpuEvent read) {
        try {
            read.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while reading device state", e);
        } finally {
            read.close();
        }
        queue.finish();
    }

    /** Closes a write's event; the backend staged the source bytes when it enqueued the write. */
    private static void enqueued(GpuEvent write) {
        if (write != null) {
            write.close();
        }
    }

    /** One owned device cache layer under the transfer. */
    private final class DeviceCache implements Cache {
        private final DeviceKvLayer layer;

        DeviceCache(DeviceKvLayer layer) {
            this.layer = layer;
        }

        @Override
        public int layer() {
            return layer.layer();
        }

        @Override
        public int group() {
            return layer.group();
        }

        @Override
        public KvLayout layout() {
            return layer.layout();
        }

        @Override
        public void read(boolean v, int row0, int rows, byte[] dst) {
            long rowBytes = check(v, row0, rows, dst);
            complete(queue.enqueueReadBuffer(buffer(v), row0 * rowBytes, rows * rowBytes, dst));
        }

        @Override
        public void write(boolean v, int row0, int rows, byte[] src) {
            long rowBytes = check(v, row0, rows, src);
            enqueued(queue.enqueueWriteBuffer(buffer(v), row0 * rowBytes, rows * rowBytes, src));
        }

        @Override
        public java.util.List<com.eonmux.neuralfree.gguf.memory.KvDType.SidePlane> sidePlanes(boolean v) {
            KvLayout l = layer.layout();
            if (v && l.vAliasesK()) {
                return java.util.Collections.emptyList();
            }
            return dtype(v).sidePlanes();
        }

        @Override
        public void readSide(boolean v, int plane, int row0, int rows, byte[] dst) {
            long planeBytes = checkSide(v, plane, row0, rows, dst.length);
            if (dtype(v).vendorFormat()) {
                long stride = layer.sideRowBytes();
                long base = sideRowBase(v);
                long offset = planeOffset(v, plane);
                byte[] packed = new byte[Math.toIntExact(rows * stride)];
                complete(queue.enqueueReadBuffer(sideBuffer(v), base + row0 * stride, packed.length, packed));
                for (int i = 0; i < rows; i++) {
                    System.arraycopy(packed, Math.toIntExact(i * stride + offset), dst,
                            Math.toIntExact(i * planeBytes), Math.toIntExact(planeBytes));
                }
                return;
            }
            long stride = 2 * planeBytes;
            byte[] interleaved = new byte[Math.toIntExact(rows * stride)];
            complete(queue.enqueueReadBuffer(sideBuffer(v), row0 * stride, interleaved.length, interleaved));
            int off = v ? (int) planeBytes : 0;
            for (int i = 0; i < rows; i++) {
                System.arraycopy(interleaved, Math.toIntExact(i * stride) + off, dst, Math.toIntExact(i * planeBytes),
                        Math.toIntExact(planeBytes));
            }
        }

        @Override
        public void writeSide(boolean v, int plane, int row0, int rows, byte[] src) {
            long planeBytes = checkSide(v, plane, row0, rows, src.length);
            if (dtype(v).vendorFormat()) {
                long stride = layer.sideRowBytes();
                long base = sideRowBase(v);
                long offset = planeOffset(v, plane);
                byte[] packed = new byte[Math.toIntExact(rows * stride)];
                complete(queue.enqueueReadBuffer(sideBuffer(v), base + row0 * stride, packed.length, packed));
                for (int i = 0; i < rows; i++) {
                    System.arraycopy(src, Math.toIntExact(i * planeBytes), packed,
                            Math.toIntExact(i * stride + offset), Math.toIntExact(planeBytes));
                }
                enqueued(queue.enqueueWriteBuffer(sideBuffer(v), base + row0 * stride, packed.length, packed));
                return;
            }
            long stride = 2 * planeBytes;
            byte[] interleaved = new byte[Math.toIntExact(rows * stride)];
            complete(queue.enqueueReadBuffer(sideBuffer(v), row0 * stride, interleaved.length, interleaved));
            int off = v ? (int) planeBytes : 0;
            for (int i = 0; i < rows; i++) {
                System.arraycopy(src, Math.toIntExact(i * planeBytes), interleaved, Math.toIntExact(i * stride) + off,
                        Math.toIntExact(planeBytes));
            }
            enqueued(queue.enqueueWriteBuffer(sideBuffer(v), row0 * stride, interleaved.length, interleaved));
        }

        private com.eonmux.neuralfree.gguf.memory.KvDType dtype(boolean v) {
            return v ? layer.layout().dtypeV() : layer.layout().dtypeK();
        }

        private int width(boolean v) {
            return v ? layer.layout().widthV() : layer.layout().widthK();
        }

        /** Byte offset of plane {@code plane} inside one packed side-plane row of a vendor cache. */
        private long planeOffset(boolean v, int plane) {
            java.util.List<com.eonmux.neuralfree.gguf.memory.KvDType.SidePlane> planes = dtype(v).sidePlanes();
            long offset = 0;
            for (int i = 0; i < plane; i++) {
                offset += planes.get(i).rowBytesFor(width(v), layer.layout().nHeadKv());
            }
            return offset;
        }

        /** Validates a side-plane row range and answers the plane's row size. */
        private long checkSide(boolean v, int plane, int row0, int rows, int hostBytes) {
            if (v && layer.layout().vAliasesK()) {
                throw new IllegalArgumentException("V of the latent cache of layer " + layer.layer()
                        + " is part of the K row");
            }
            java.util.List<com.eonmux.neuralfree.gguf.memory.KvDType.SidePlane> planes = dtype(v).sidePlanes();
            if (plane < 0 || plane >= planes.size()) {
                throw new IndexOutOfBoundsException("side plane " + plane + " outside the " + planes.size()
                        + " " + (v ? "V" : "K") + " side planes of layer " + layer.layer());
            }
            KvLayout l = layer.layout();
            if (rows < 0 || row0 < 0 || row0 > l.rows() - rows) {
                throw new IndexOutOfBoundsException("rows [" + row0 + ", " + ((long) row0 + rows)
                        + ") outside a cache of " + l.rows() + " rows");
            }
            long rowBytes = dtype(v).vendorFormat()
                    ? planes.get(plane).rowBytesFor(width(v), l.nHeadKv())
                    : (long) l.nHeadKv() * Float.BYTES;
            if ((long) rows * rowBytes > hostBytes) {
                throw new IllegalArgumentException(rows + " side rows of " + rowBytes + " bytes do not fit an array"
                        + " of " + hostBytes + " bytes");
            }
            return rowBytes;
        }

        /** Bytes of the side buffer ahead of the plane rows: the scales and centroid header plus the K planes. */
        private long sideRowBase(boolean v) {
            KvLayout l = layer.layout();
            return (long) DeviceKvFormats.sidePlaneBase(l.nHeadKv()) + DeviceKvFormats.sidePlaneOffset(l, v);
        }

        /** The device buffer holding the side planes of one side: the vendor side buffer or the row scales. */
        private GpuBuffer sideBuffer(boolean v) {
            GpuBuffer b = dtype(v).vendorFormat() ? layer.vendorSide() : layer.rowScales();
            if (b == null) {
                throw new UnsupportedOperationException("layer " + layer.layer() + " keeps no "
                        + (v ? "V" : "K") + " side planes");
            }
            return b;
        }

        private GpuBuffer buffer(boolean v) {
            return v ? layer.v() : layer.k();
        }

        /** Validates the row range and the host array, answering the row size of K or V. */
        private long check(boolean v, int row0, int rows, byte[] host) {
            Objects.requireNonNull(host, "host");
            KvLayout l = layer.layout();
            if (v && l.vAliasesK()) {
                throw new IllegalArgumentException("V of the latent cache of layer " + layer.layer()
                        + " is part of the K row");
            }
            if (rows < 0 || row0 < 0 || row0 > l.rows() - rows) {
                throw new IndexOutOfBoundsException("rows [" + row0 + ", " + ((long) row0 + rows)
                        + ") outside a cache of " + l.rows() + " rows");
            }
            long rowBytes = v ? l.rowBytesV() : l.rowBytesK();
            if (rows * rowBytes > host.length) {
                throw new IllegalArgumentException(rows + " rows of " + rowBytes + " bytes do not fit an array of "
                        + host.length + " bytes");
            }
            return rowBytes;
        }
    }

    /** One owned device recurrent state under the transfer. */
    private final class DeviceState implements State {
        private final DeviceRecurrentState state;

        DeviceState(DeviceRecurrentState state) {
            this.state = state;
        }

        @Override
        public int layer() {
            return state.layer();
        }

        @Override
        public List<LayerMemoryPlan.StateTensor> tensors() {
            return state.tensors();
        }

        @Override
        public float[] read(String name) {
            GpuBuffer b = state.tensor(name);
            float[] out = new float[floats(name)];
            complete(queue.enqueueReadBuffer(b, 0, (long) out.length * Float.BYTES, out));
            return out;
        }

        @Override
        public void write(String name, float[] values) {
            GpuBuffer b = state.tensor(name);
            int n = floats(name);
            if (values.length != n) {
                throw new IllegalArgumentException("state tensor " + name + " of layer " + state.layer() + " holds "
                        + n + " floats, got " + values.length);
            }
            enqueued(queue.enqueueWriteBuffer(b, 0, (long) n * Float.BYTES, values));
        }

        /** The planned float count of tensor {@code name} (its buffer may be padded to a multiple of 4 bytes). */
        private int floats(String name) {
            for (LayerMemoryPlan.StateTensor t : state.tensors()) {
                if (t.name().equals(name)) {
                    return Math.toIntExact(t.bytes() / Float.BYTES);
                }
            }
            throw new IllegalArgumentException("layer " + state.layer() + " has no state tensor '" + name + "'");
        }
    }
}
