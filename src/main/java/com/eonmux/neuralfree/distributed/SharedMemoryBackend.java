package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.tensor.Tensor;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;

/**
 * Collectives between the ranks of one host (processes, or backends of one process) through memory-mapped files in
 * {@code /dev/shm} (the temporary directory where {@code /dev/shm} does not exist). Every rank constructs its backend
 * with the same {@code shmPrefix}, unique to the job, and all ranks call the same collectives in the same order, as
 * with PyTorch's process groups.
 *
 * <p><b>Protocol.</b> Every rank passes tensors of the same element count to a collective. A segment file holds one
 * {@code arrive} and one {@code depart} round counter per rank, then one slot of {@code count} floats per rank. A
 * call is round {@code n} of its segment: the rank waits until every rank has departed round {@code n - 1} (so no
 * slot is overwritten while it is read), writes its slot, publishes {@code arrive = n} (release), waits until every
 * rank has arrived at round {@code n} (acquire), reads the slots it needs and publishes {@code depart = n}. The
 * counters only grow, so the segments need no reset between calls. Every collective first runs one round of the
 * control segment ({@code <prefix>_control}), in which each rank publishes its {@link CollectiveDescriptor} (kind,
 * reduce op or source rank, element count) and every rank checks that they agree; a barrier is that round alone. The
 * values then cross in one round of the segment of that kind and element count ({@code <prefix>_<kind>_<count>}).
 * Waiting spins briefly, then parks; a rank that waits longer than
 * {@value CollectiveDescriptor#TIMEOUT_MINUTES} minutes for another throws {@link IllegalStateException}.</p>
 *
 * <p><b>Results.</b> {@link #allReduce} reduces the ranks' values in rank order on every rank
 * ({@link CollectiveTensors#reduce}), so all ranks hold bit-identical results, and writes them into the caller's
 * FLOAT32 tensor in place (under {@code noGrad}); {@link #broadcast} writes the source rank's values into every other
 * rank's FLOAT32 tensor; {@link #allGather} replaces each entry of the output list with a new tensor (the entries it
 * replaces are not closed). Values cross through the host.</p>
 *
 * <p>Closing the backend closes and deletes its segment files; a segment from a job that ended without closing must
 * be removed before its prefix is used again, since its counters would be taken for the new job's.</p>
 *
 * <p><b>Access.</b> On file systems with POSIX permissions the segment files are created readable and writable by
 * their owner only, so every rank must run as the same user, and a segment file that other users can access (one
 * another user created first under the same name, for example) is refused. Choose a prefix unique to the job.</p>
 */
public final class SharedMemoryBackend implements DistributedBackend {
    private static final int SPINS_BEFORE_PARKING = 1_000;
    private static final long PARK_NANOS = 50_000;
    private static final VarHandle COUNTERS =
        MethodHandles.byteBufferViewVarHandle(long[].class, ByteOrder.nativeOrder());
    /** Permissions of a new segment file where the file system supports POSIX permissions: owner read and write. */
    private static final Set<PosixFilePermission> OWNER_ONLY = PosixFilePermissions.fromString("rw-------");

    private final int rank;
    private final int worldSize;
    private final String shmPrefix;
    private final AtomicBoolean closed;
    private final Map<String, Segment> segments;

    public SharedMemoryBackend(int rank, int worldSize, String shmPrefix) {
        if (worldSize < 1) {
            throw new IllegalArgumentException("World size must be at least 1");
        }
        if (rank < 0 || rank >= worldSize) {
            throw new IllegalArgumentException("Rank must be in range [0, " + (worldSize - 1) + "]");
        }
        if (shmPrefix == null || shmPrefix.isEmpty()) {
            throw new IllegalArgumentException("Shared memory prefix cannot be null or empty");
        }
        this.rank = rank;
        this.worldSize = worldSize;
        this.shmPrefix = shmPrefix;
        this.closed = new AtomicBoolean(false);
        this.segments = new LinkedHashMap<>();
    }

    /**
     * A backend whose prefix is derived from the current time: the ranks of one job can only find each other through
     * the same prefix, so this form suits a single rank; pass the job's prefix to
     * {@link #SharedMemoryBackend(int, int, String)} otherwise.
     */
    public SharedMemoryBackend(int rank, int worldSize) {
        this(rank, worldSize, "neuralfree_shm_" + System.currentTimeMillis());
    }

    @Override
    public int rank() {
        ensureOpen();
        return rank;
    }

    @Override
    public int worldSize() {
        ensureOpen();
        return worldSize;
    }

    /**
     * Reduces {@code tensor} across the ranks with {@code op} and writes the result into it in place.
     *
     * @throws IllegalArgumentException when {@code tensor} is not FLOAT32, or the ranks' calls differ (another
     *                                  collective, another op, or a tensor of another size), on every rank
     * @throws IllegalStateException    when the backend is closed or a rank does not arrive in time
     */
    @Override
    public synchronized void allReduce(Tensor tensor, ReduceOp op) {
        ensureOpen();
        Objects.requireNonNull(op, "ReduceOp cannot be null");
        CollectiveTensors.requireWritable(tensor);
        float[] data = CollectiveTensors.read(tensor);
        agree(CollectiveDescriptor.allReduce(op, data.length));
        float[][] slots = segment("allreduce", data.length).exchange(data, -1);
        CollectiveTensors.writeBack(tensor, CollectiveTensors.reduce(slots, op));
    }

    /**
     * Copies the values of {@code srcRank}'s tensor into {@code tensor} on every other rank, in place.
     *
     * @throws IllegalArgumentException when {@code srcRank} is out of range or {@code tensor} is not FLOAT32, or the
     *                                  ranks' calls differ (another collective, another source rank, or a tensor of
     *                                  another size), on every rank
     * @throws IllegalStateException    when the backend is closed or a rank does not arrive in time
     */
    @Override
    public synchronized void broadcast(Tensor tensor, int srcRank) {
        ensureOpen();
        if (srcRank < 0 || srcRank >= worldSize) {
            throw new IllegalArgumentException("srcRank must be in range [0, " + (worldSize - 1) + "]");
        }
        CollectiveTensors.requireWritable(tensor);
        boolean source = rank == srcRank;
        float[] data = source ? CollectiveTensors.read(tensor) : null;
        int count = Math.toIntExact(tensor.numElements());
        agree(CollectiveDescriptor.broadcast(srcRank, count));
        float[][] slots = segment("broadcast", count).exchange(data, srcRank);
        if (!source) {
            CollectiveTensors.writeBack(tensor, slots[srcRank]);
        }
    }

    /**
     * Sets entry {@code r} of {@code output} to a new tensor holding rank {@code r}'s {@code input}, with the shape,
     * backend and dtype of this rank's {@code input}; the entries replaced are not closed.
     *
     * @throws IllegalArgumentException when {@code output} does not have {@code worldSize} entries, or the ranks'
     *                                  calls differ (another collective or a tensor of another size), on every rank
     * @throws IllegalStateException    when the backend is closed or a rank does not arrive in time
     */
    @Override
    public synchronized void allGather(List<Tensor> output, Tensor input) {
        ensureOpen();
        Objects.requireNonNull(output, "Output list cannot be null");
        if (output.size() != worldSize) {
            throw new IllegalArgumentException("Output list must have size equal to worldSize");
        }
        float[] data = CollectiveTensors.read(input);
        agree(CollectiveDescriptor.allGather(data.length));
        float[][] slots = segment("allgather", data.length).exchange(data, -1);
        for (int r = 0; r < worldSize; r++) {
            output.set(r, CollectiveTensors.shapedLike(input, slots[r]));
        }
    }

    /**
     * Returns once every rank has called {@code barrier} the same number of times.
     *
     * @throws IllegalArgumentException when another rank called another collective, on every rank
     * @throws IllegalStateException    when the backend is closed or a rank does not arrive in time
     */
    @Override
    public synchronized void barrier() {
        ensureOpen();
        agree(CollectiveDescriptor.barrier());
    }

    @Override
    public boolean isInitialized() {
        return !closed.get();
    }

    /**
     * The control round of a collective: publishes {@code mine}, reads every rank's descriptor and checks that they
     * agree ({@link CollectiveDescriptor#requireAgreement}), before any value is exchanged.
     *
     * @throws IllegalArgumentException when the ranks' descriptors differ
     * @throws IllegalStateException    when a control slot does not hold a descriptor (a segment of another job)
     */
    private void agree(CollectiveDescriptor mine) {
        Segment control = namedSegment(shmPrefix.replace('/', '_') + "_control", CollectiveDescriptor.FLOATS);
        float[][] slots = control.exchange(mine.toFloats(), -1);
        CollectiveDescriptor[] perRank = new CollectiveDescriptor[worldSize];
        for (int r = 0; r < worldSize; r++) {
            try {
                perRank[r] = CollectiveDescriptor.fromFloats(slots[r]);
            } catch (IllegalArgumentException e) {
                throw new IllegalStateException("Rank " + r + "'s control slot of " + control.name + " is corrupt;"
                    + " remove the segments of an earlier job using prefix '" + shmPrefix + "'", e);
            }
        }
        CollectiveDescriptor.requireAgreement(perRank);
    }

    private Segment segment(String kind, int count) {
        return namedSegment(shmPrefix.replace('/', '_') + "_" + kind + "_" + count, count);
    }

    private Segment namedSegment(String name, int count) {
        Segment segment = segments.get(name);
        if (segment == null) {
            segment = new Segment(name, worldSize, count);
            segments.put(name, segment);
        }
        return segment;
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("SharedMemoryBackend has been closed");
        }
    }

    /**
     * Closes and deletes every segment file this backend opened; idempotent. Every segment is tried; the first failure
     * is thrown afterwards with later ones suppressed.
     */
    @Override
    public synchronized void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        UncheckedIOException failure = null;
        for (Segment segment : segments.values()) {
            try {
                segment.close();
            } catch (UncheckedIOException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        segments.clear();
        if (failure != null) {
            throw failure;
        }
    }

    /** One segment file: the round counters and the per-rank slots of one kind of collective and element count. */
    private final class Segment {
        private final String name;
        private final Path path;
        private final int count;
        private final FileChannel channel;
        private final MappedByteBuffer buffer;
        private final int dataOffset;
        private long round;

        Segment(String name, int ranks, int count) {
            this.name = name;
            this.count = count;
            long header = 16L * ranks;
            long size = header + 4L * count * ranks;
            if (size > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("A collective over " + count + " floats and " + ranks
                    + " ranks needs " + size + " bytes of shared memory, more than one segment can map");
            }
            this.dataOffset = (int) header;
            Path shm = Paths.get("/dev/shm");
            Path directory = Files.isDirectory(shm) ? shm : Paths.get(System.getProperty("java.io.tmpdir"));
            this.path = directory.resolve(name);
            FileChannel opened = null;
            try {
                boolean posix = directory.getFileSystem().supportedFileAttributeViews().contains("posix");
                Set<StandardOpenOption> options = EnumSet.of(StandardOpenOption.CREATE, StandardOpenOption.READ,
                    StandardOpenOption.WRITE);
                FileAttribute<?>[] attributes = posix
                    ? new FileAttribute<?>[] {PosixFilePermissions.asFileAttribute(OWNER_ONLY)}
                    : new FileAttribute<?>[0];
                opened = FileChannel.open(path, options, attributes);
                if (posix && !OWNER_ONLY.containsAll(Files.getPosixFilePermissions(path))) {
                    throw new IOException("other users can access the segment file "
                        + PosixFilePermissions.toString(Files.getPosixFilePermissions(path))
                        + "; remove it or choose another prefix");
                }
                this.buffer = opened.map(FileChannel.MapMode.READ_WRITE, 0, size);
                this.channel = opened;
            } catch (IOException e) {
                if (opened != null) {
                    try {
                        opened.close();
                    } catch (IOException closeFailure) {
                        e.addSuppressed(closeFailure);
                    }
                }
                throw new UncheckedIOException("Failed to map the shared memory segment " + path, e);
            }
        }

        /**
         * One round: waits for every rank to leave the previous round, writes {@code mine} to this rank's slot (none
         * when null), announces it, waits for every rank to arrive and reads the slots: every slot for
         * {@code only == -1}, slot {@code only} alone for {@code only >= 0} (the other entries are null). Returns one
         * entry per rank.
         */
        float[][] exchange(float[] mine, int only) {
            long n = ++round;
            awaitAll(true, n - 1);
            if (mine != null) {
                write(rank, mine);
            }
            COUNTERS.setRelease(buffer, arriveIndex(rank), n);
            awaitAll(false, n);
            float[][] slots = new float[worldSize][];
            for (int r = 0; r < worldSize; r++) {
                if (only == -1 || only == r) {
                    slots[r] = read(r);
                }
            }
            COUNTERS.setRelease(buffer, departIndex(rank), n);
            return slots;
        }

        /** Waits until every rank's depart (or arrive) counter is at least {@code target}. */
        private void awaitAll(boolean depart, long target) {
            if (target <= 0) {
                return;
            }
            long deadline = System.nanoTime() + CollectiveDescriptor.TIMEOUT_NANOS;
            for (int r = 0; r < worldSize; r++) {
                int index = depart ? departIndex(r) : arriveIndex(r);
                int spins = 0;
                while ((long) COUNTERS.getAcquire(buffer, index) < target) {
                    if (Thread.currentThread().isInterrupted()) {
                        throw new IllegalStateException("Interrupted while rank " + rank + " waited for rank " + r
                            + " in " + name);
                    }
                    if (System.nanoTime() - deadline > 0) {
                        throw new IllegalStateException("Rank " + rank + " waited more than "
                            + CollectiveDescriptor.TIMEOUT_MINUTES
                            + " minutes for rank " + r + " to " + (depart ? "leave" : "reach") + " round " + target
                            + " of " + name);
                    }
                    if (++spins < SPINS_BEFORE_PARKING) {
                        Thread.onSpinWait();
                    } else {
                        LockSupport.parkNanos(PARK_NANOS);
                    }
                }
            }
        }

        private int arriveIndex(int r) {
            return 8 * r;
        }

        private int departIndex(int r) {
            return 8 * (worldSize + r);
        }

        private void write(int slot, float[] values) {
            ByteBuffer view = buffer.duplicate().order(ByteOrder.nativeOrder());
            view.position(dataOffset + 4 * count * slot);
            view.asFloatBuffer().put(values, 0, count);
        }

        private float[] read(int slot) {
            ByteBuffer view = buffer.duplicate().order(ByteOrder.nativeOrder());
            view.position(dataOffset + 4 * count * slot);
            float[] values = new float[count];
            view.asFloatBuffer().get(values);
            return values;
        }

        /** Closes the file channel and deletes the file (the ranks still mapping it keep their mapping). */
        void close() {
            IOException failure = null;
            try {
                channel.close();
            } catch (IOException e) {
                failure = e;
            }
            try {
                Files.deleteIfExists(path);
            } catch (IOException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
            if (failure != null) {
                throw new UncheckedIOException("Failed to release the shared memory segment " + path, failure);
            }
        }
    }
}
