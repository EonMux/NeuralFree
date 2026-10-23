package com.eonmux.neuralfree.training;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.io.ModelSerializer;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.Tensor;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * Saves checkpoints of the trained model, and optionally of its optimizer, at the end of selected epochs, like the
 * Keras and Lightning {@code ModelCheckpoint} callbacks.
 *
 * <p><b>Which epochs.</b> An epoch's monitored value is its validation loss, or its training loss when no validation
 * ran. In mode ({@code saveBestOnly}) {@code "min"} or {@code "max"} an epoch is saved when its monitored value is
 * strictly lower, or higher, than the best so far (the first epoch with a value that is not NaN always is; NaN never
 * improves), and it becomes the best checkpoint; in mode {@code "none"} every epoch is saved and no best is tracked.
 * Independently, when the periodic interval {@code n} is positive every epoch {@code e} with
 * {@code (e + 1) % n == 0} is saved. {@code n} is {@code saveEveryNEpochs}, or, for a run whose trainer's
 * {@link TrainerConfig#checkpointEveryNEpochs()} is positive, that value ({@link #onTrainBegin}); a trainer refuses a
 * checkpoint whose own positive {@code saveEveryNEpochs} differs from it.</p>
 *
 * <p><b>In memory</b> (no directory). Each checkpoint is an unmodifiable map with {@code "epoch"},
 * {@code "trainLoss"}, {@code "valLoss"} (null when no validation ran), {@code "monitored"}, {@code "modelState"}
 * (the model's {@link Module#stateDict()}: every parameter and persistent buffer, such as BatchNorm running statistics,
 * as host {@code float[]} copies) and, when optimizer state is saved, {@code "optimizerState"} (the optimizer's
 * {@link Optimizer#stateDict()} with every state tensor replaced by a host {@code float[]} copy, and the step count and
 * hyper-parameters as boxed values). The copies do not change as training continues.</p>
 *
 * <p><b>On disk</b> (a directory is configured). Each checkpoint is written to
 * {@code <directory>/<prefix>-epoch<NNNN>.nfc} ({@link #checkpointFile(int)}; the epoch zero-padded to at least four
 * digits), replacing a file of that name. The file is a {@link ModelSerializer} checkpoint (model state with
 * persistent buffers, and the optimizer state when it is saved), which {@link ModelSerializer#load} and
 * {@link ModelSerializer#inspect} read, plus a {@value #METADATA_ENTRY} entry ({@link #readMetadata(Path)}) holding
 * the epoch, the losses, the monitored value, the mode, whether optimizer state is included and of which optimizer
 * class, and the model's parameter and buffer names. It is written atomically: to a temporary file in the same
 * directory, forced to the storage device, renamed over the target in one atomic step, and the directory is then
 * forced too on file systems with POSIX semantics (elsewhere the JDK cannot open a directory, and the rename's
 * durability rests on the file system). In memory only the map {@code "epoch"}, {@code "trainLoss"},
 * {@code "valLoss"}, {@code "monitored"} and {@code "path"} (the file's {@link Path}) is kept.</p>
 *
 * <p><b>How many.</b> With {@code keep} positive, after each save the oldest checkpoints of the current run are
 * dropped (on disk, their files deleted) until at most {@code keep} remain; the best checkpoint and the one just saved
 * are never dropped, so with periodic saves and {@code keep == 1} the best and the latest periodic one both stay.
 * {@code keep == 0} keeps every checkpoint. Files from earlier runs are neither counted nor deleted, and
 * {@link #onTrainBegin} starts a new run.</p>
 *
 * <p><b>Restoring.</b> {@link #restore(int)} and {@link #restoreBest()} load a checkpoint back into the trainer's
 * model and, when the checkpoint holds optimizer state, its optimizer, all or nothing. A checkpoint held in memory is
 * used directly; otherwise, with a directory, the epoch's file is read, also when an earlier run or process wrote
 * it.</p>
 *
 * <p>File system failures propagate as {@link UncheckedIOException} naming the file; they are never ignored, and a
 * failed write leaves no temporary file behind.</p>
 */
public final class ModelCheckpoint implements Callback {
    public static final String MODE_MIN = "min";
    public static final String MODE_MAX = "max";
    public static final String MODE_NONE = "none";
    public static final String DEFAULT_PREFIX = "checkpoint";
    public static final String FILE_EXTENSION = ".nfc";
    /** The zip entry of a checkpoint file that holds its {@link #readMetadata(Path) metadata}. */
    public static final String METADATA_ENTRY = "checkpoint.properties";

    private static final int METADATA_VERSION = 1;
    private static final String EPOCH = "epoch";
    private static final String TRAIN_LOSS = "trainLoss";
    private static final String VAL_LOSS = "valLoss";
    private static final String MONITORED = "monitored";
    private static final String MODE = "mode";
    private static final String HAS_OPTIMIZER_STATE = "hasOptimizerState";
    private static final String OPTIMIZER_TYPE = "optimizerType";
    private static final String PARAMETER_NAMES = "parameterNames";
    private static final String BUFFER_NAMES = "bufferNames";
    private static final String PATH = "path";
    private static final String MODEL_STATE = "modelState";
    private static final String OPTIMIZER_STATE = "optimizerState";

    private final String checkpointDir;
    private final Path directory;
    private final String prefix;
    private final String saveBestOnly;
    private final int saveEveryNEpochs;
    private final boolean saveOptimizerState;
    private final int keep;

    private Float bestLoss = null;
    private int bestEpoch = -1;
    private TrainingContext trainer;
    /** The periodic interval of the current run ({@link #periodicInterval}). */
    private int runSaveEveryNEpochs;
    /** The checkpoints of the current run in save order, keyed by epoch. */
    private final Map<Integer, Map<String, Object>> checkpoints = new LinkedHashMap<>();

    /** In-memory checkpoints of the best epoch by the minimum monitored value. */
    public ModelCheckpoint() {
        this(builder());
    }

    /** Checkpoints of the best epoch by the minimum monitored value, written to {@code checkpointDir} when non-null. */
    public ModelCheckpoint(String checkpointDir) {
        this(builder().directory(checkpointDir));
    }

    /**
     * Checkpoints written to {@code checkpointDir} (in memory when null), in mode {@code saveBestOnly}
     * ({@code "min"}, {@code "max"} or {@code "none"}), also every {@code saveEveryNEpochs}-th epoch when positive,
     * with the optimizer state when {@code saveOptimizerState}, keeping every checkpoint.
     */
    public ModelCheckpoint(String checkpointDir, String saveBestOnly, int saveEveryNEpochs,
                           boolean saveOptimizerState) {
        this(builder().directory(checkpointDir).saveBestOnly(saveBestOnly).saveEveryNEpochs(saveEveryNEpochs)
            .saveOptimizerState(saveOptimizerState));
    }

    private ModelCheckpoint(Builder builder) {
        if (!MODE_MIN.equals(builder.saveBestOnly) && !MODE_MAX.equals(builder.saveBestOnly)
                && !MODE_NONE.equals(builder.saveBestOnly)) {
            throw new IllegalArgumentException("Mode must be 'min', 'max' or 'none', got: " + builder.saveBestOnly);
        }
        if (builder.saveEveryNEpochs < 0) {
            throw new IllegalArgumentException("saveEveryNEpochs must be non-negative, got: "
                + builder.saveEveryNEpochs);
        }
        if (builder.keep < 0) {
            throw new IllegalArgumentException("keep must be non-negative (0 keeps every checkpoint), got: "
                + builder.keep);
        }
        requireValidPrefix(builder.prefix);
        this.checkpointDir = builder.checkpointDir;
        this.directory = builder.checkpointDir == null ? null : toPath(builder.checkpointDir);
        this.prefix = builder.prefix;
        this.saveBestOnly = builder.saveBestOnly;
        this.saveEveryNEpochs = builder.saveEveryNEpochs;
        this.saveOptimizerState = builder.saveOptimizerState;
        this.keep = builder.keep;
        this.runSaveEveryNEpochs = builder.saveEveryNEpochs;
    }

    private static Path toPath(String checkpointDir) {
        try {
            return Paths.get(checkpointDir);
        } catch (InvalidPathException e) {
            throw new IllegalArgumentException("Invalid checkpoint directory '" + checkpointDir + "': "
                + e.getMessage(), e);
        }
    }

    /** A file name prefix names one file inside the directory: non-empty, without separators, not "." or "..". */
    private static void requireValidPrefix(String prefix) {
        if (prefix == null || prefix.isEmpty() || prefix.equals(".") || prefix.equals("..")
                || prefix.indexOf('/') >= 0 || prefix.indexOf('\\') >= 0 || prefix.indexOf('\0') >= 0) {
            throw new IllegalArgumentException("Checkpoint file prefix must be a non-empty file name without path "
                + "separators, got: " + (prefix == null ? "null" : "'" + prefix + "'"));
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Starts a run with {@code trainer}: forgets the checkpoints and the best epoch of the previous run, and sets the
     * run's periodic interval to the trainer's {@link TrainerConfig#checkpointEveryNEpochs()} when that is positive,
     * else to {@link #saveEveryNEpochs()} (also for a null trainer).
     *
     * @throws IllegalArgumentException when this checkpoint's own positive {@code saveEveryNEpochs} differs from the
     *                                  trainer's positive {@code checkpointEveryNEpochs}
     */
    @Override
    public synchronized void onTrainBegin(TrainingContext trainer) {
        this.runSaveEveryNEpochs = periodicInterval(trainer == null ? 0 : trainer.config().checkpointEveryNEpochs());
        this.trainer = trainer;
        this.bestLoss = null;
        this.bestEpoch = -1;
        this.checkpoints.clear();
    }

    /**
     * Saves the epoch when the rules of this class select it, records it as the best checkpoint when its monitored
     * value improved, and then drops checkpoints beyond {@code keep}. A failed save leaves the best epoch and the kept
     * checkpoints unchanged, except that a file already renamed into place when forcing the directory fails stays
     * listed (not as the best).
     *
     * @throws IllegalStateException when called before {@link #onTrainBegin} and the epoch is to be saved
     * @throws UncheckedIOException  when writing the checkpoint file or deleting a dropped one fails
     */
    @Override
    public synchronized void onEpochEnd(int epoch, float trainLoss, Float valLoss) {
        if (epoch < 0) {
            throw new IllegalArgumentException("Epoch must be non-negative, got: " + epoch);
        }
        float monitored = valLoss != null ? valLoss : trainLoss;
        boolean improved = improves(monitored);
        boolean periodic = runSaveEveryNEpochs > 0 && (epoch + 1) % runSaveEveryNEpochs == 0;
        if (!improved && !periodic && !MODE_NONE.equals(saveBestOnly)) {
            return;
        }
        saveCheckpoint(epoch, trainLoss, valLoss, monitored);
        if (improved) {
            bestLoss = monitored;
            bestEpoch = epoch;
        }
        dropBeyondKeep(epoch);
    }

    /**
     * The periodic interval of a run whose trainer sets {@code trainerInterval}
     * ({@link TrainerConfig#checkpointEveryNEpochs()}): {@link #saveEveryNEpochs()} when {@code trainerInterval} is 0,
     * else {@code trainerInterval}.
     *
     * @throws IllegalArgumentException when both are positive and differ
     */
    int periodicInterval(int trainerInterval) {
        if (trainerInterval == 0) {
            return saveEveryNEpochs;
        }
        if (saveEveryNEpochs != 0 && saveEveryNEpochs != trainerInterval) {
            throw new IllegalArgumentException("ModelCheckpoint saves every " + saveEveryNEpochs + " epochs, which "
                + "conflicts with the trainer's checkpointEveryNEpochs " + trainerInterval + "; set one of them, or "
                + "both to the same value");
        }
        return trainerInterval;
    }

    /** Whether {@code monitored} improves on the best so far; NaN never does, as in Keras and Lightning. */
    private boolean improves(float monitored) {
        if (Float.isNaN(monitored)) {
            return false;
        }
        if (MODE_MIN.equals(saveBestOnly)) {
            return bestLoss == null || monitored < bestLoss;
        }
        if (MODE_MAX.equals(saveBestOnly)) {
            return bestLoss == null || monitored > bestLoss;
        }
        return false;
    }

    private void saveCheckpoint(int epoch, float trainLoss, Float valLoss, float monitored) {
        if (trainer == null) {
            throw new IllegalStateException("ModelCheckpoint received an epoch end before onTrainBegin");
        }
        Optimizer optimizer = saveOptimizerState ? trainer.optimizer() : null;
        Map<String, Object> checkpoint = new LinkedHashMap<>();
        checkpoint.put(EPOCH, epoch);
        checkpoint.put(TRAIN_LOSS, trainLoss);
        checkpoint.put(VAL_LOSS, valLoss);
        checkpoint.put(MONITORED, monitored);
        if (directory == null) {
            checkpoint.put(MODEL_STATE, Collections.unmodifiableMap(trainer.model().stateDict()));
            if (optimizer != null) {
                checkpoint.put(OPTIMIZER_STATE, Collections.unmodifiableMap(snapshot(optimizer)));
            }
        } else {
            Path file = checkpointFile(epoch);
            writeCheckpointFile(file, metadata(epoch, trainLoss, valLoss, monitored, optimizer), optimizer);
            checkpoint.put(PATH, file);
        }
        checkpoints.remove(epoch);
        checkpoints.put(epoch, Collections.unmodifiableMap(checkpoint));
        if (directory != null) {
            syncDirectory((Path) checkpoint.get(PATH));
        }
    }

    /**
     * Forces the directory entry of the renamed {@code file} to the device. The file is already in place and listed,
     * so a failure leaves it counted by {@code keep}; the failure propagates.
     */
    private void syncDirectory(Path file) {
        try {
            forceDirectory(directory);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot force checkpoint directory " + directory + " after writing " + file
                + ": " + e.getMessage(), e);
        }
    }

    /**
     * Drops the oldest checkpoints until at most {@code keep} remain, never the best one nor {@code justSaved}. A file
     * that cannot be deleted stays listed and the failure propagates.
     */
    private void dropBeyondKeep(int justSaved) {
        if (keep == 0) {
            return;
        }
        Iterator<Map.Entry<Integer, Map<String, Object>>> oldestFirst = checkpoints.entrySet().iterator();
        while (checkpoints.size() > keep && oldestFirst.hasNext()) {
            Map.Entry<Integer, Map<String, Object>> entry = oldestFirst.next();
            int epoch = entry.getKey();
            if (epoch == justSaved || epoch == bestEpoch) {
                continue;
            }
            Path file = (Path) entry.getValue().get(PATH);
            if (file != null) {
                try {
                    Files.deleteIfExists(file);
                } catch (IOException e) {
                    throw new UncheckedIOException("Cannot delete superseded checkpoint " + file + ": "
                        + e.getMessage(), e);
                }
            }
            oldestFirst.remove();
        }
    }

    /**
     * The optimizer's {@link Optimizer#stateDict()} with each state tensor read into a host {@code float[]},
     * keyed as the state dict keys them; the state dict's tensors are closed before this returns.
     */
    private static Map<String, Object> snapshot(Optimizer optimizer) {
        Map<String, Object> state = optimizer.stateDict();
        Map<String, Object> host = new LinkedHashMap<>();
        try {
            for (Map.Entry<String, Object> entry : state.entrySet()) {
                Object value = entry.getValue();
                host.put(entry.getKey(), value instanceof Tensor ? ((Tensor) value).toFloatArray() : value);
            }
        } catch (RuntimeException e) {
            try {
                Optimizer.closeStateDict(state);
            } catch (RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
        Optimizer.closeStateDict(state);
        return host;
    }

    /** The {@value #METADATA_ENTRY} contents of a checkpoint of the trainer's model. */
    private Properties metadata(int epoch, float trainLoss, Float valLoss, float monitored, Optimizer optimizer) {
        Module model = trainer.model();
        Properties metadata = new Properties();
        metadata.setProperty("format_version", Integer.toString(METADATA_VERSION));
        metadata.setProperty("epoch", Integer.toString(epoch));
        metadata.setProperty("train_loss", Float.toString(trainLoss));
        if (valLoss != null) {
            metadata.setProperty("val_loss", Float.toString(valLoss));
        }
        metadata.setProperty("monitored", Float.toString(monitored));
        metadata.setProperty("mode", saveBestOnly);
        metadata.setProperty("optimizer_state", Boolean.toString(optimizer != null));
        if (optimizer != null) {
            metadata.setProperty("optimizer_type", optimizer.getClass().getName());
        }
        putNames(metadata, "parameter", model.namedParameters().keySet());
        putNames(metadata, "buffer", model.namedBuffers().keySet());
        return metadata;
    }

    private static void putNames(Properties metadata, String kind, Collection<String> names) {
        metadata.setProperty(kind + ".count", Integer.toString(names.size()));
        int i = 0;
        for (String name : names) {
            metadata.setProperty(kind + "." + i++, name);
        }
    }

    /**
     * Writes {@code file} atomically: the {@link ModelSerializer} checkpoint goes to a staging file, which is copied
     * after the metadata entry into a temporary file that is forced to the device and renamed over {@code file}
     * ({@link #syncDirectory} then forces the directory). Both scratch files live next to {@code file} and are deleted
     * on success and failure.
     */
    private void writeCheckpointFile(Path file, Properties metadata, Optimizer optimizer) {
        try {
            Files.createDirectories(directory);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot create checkpoint directory " + directory + " for " + file + ": "
                + e.getMessage(), e);
        }
        String scratch = "." + file.getFileName() + "." + UUID.randomUUID();
        Path staged = directory.resolve(scratch + ".stage");
        Path temporary = directory.resolve(scratch + ".tmp");
        try {
            ModelSerializer.save(trainer.model(), optimizer, staged);
            writeWithMetadata(staged, temporary, metadata);
            Files.delete(staged);
            Files.move(temporary, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            UncheckedIOException failure = new UncheckedIOException("Cannot write checkpoint " + file + ": "
                + e.getMessage(), e);
            deleteScratch(failure, staged, temporary);
            throw failure;
        } catch (RuntimeException e) {
            deleteScratch(e, staged, temporary);
            throw e;
        }
    }

    private static void writeWithMetadata(Path staged, Path temporary, Properties metadata) throws IOException {
        try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.CREATE_NEW,
                StandardOpenOption.WRITE);
             ZipFile source = new ZipFile(staged.toFile());
             ZipOutputStream zip = new ZipOutputStream(new BufferedOutputStream(Channels.newOutputStream(channel)))) {
            zip.putNextEntry(new ZipEntry(METADATA_ENTRY));
            metadata.store(zip, "NeuralFree ModelCheckpoint");
            zip.closeEntry();
            Enumeration<? extends ZipEntry> entries = source.entries();
            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                zip.putNextEntry(new ZipEntry(entry.getName()));
                try (InputStream in = source.getInputStream(entry)) {
                    in.transferTo(zip);
                }
                zip.closeEntry();
            }
            zip.finish();
            zip.flush();
            channel.force(true);
        }
    }

    /** Forces {@code dir}'s entries to the device where the file system lets a directory be opened (POSIX). */
    private static void forceDirectory(Path dir) throws IOException {
        if (!dir.getFileSystem().supportedFileAttributeViews().contains("posix")) {
            return;
        }
        try (FileChannel channel = FileChannel.open(dir, StandardOpenOption.READ)) {
            channel.force(true);
        }
    }

    /** Deletes the scratch files of a failed write, attaching any deletion failure to {@code failure}. */
    private static void deleteScratch(Exception failure, Path... scratch) {
        for (Path path : scratch) {
            try {
                Files.deleteIfExists(path);
            } catch (IOException e) {
                failure.addSuppressed(e);
            }
        }
    }

    /**
     * Reads the {@value #METADATA_ENTRY} entry of a checkpoint file written by this class, as an unmodifiable map:
     * {@code "epoch"} (Integer), {@code "trainLoss"}, {@code "valLoss"} (null when no validation ran) and
     * {@code "monitored"} (Float), {@code "mode"} (String), {@code "hasOptimizerState"} (Boolean),
     * {@code "optimizerType"} (the optimizer's class name, null without optimizer state), {@code "parameterNames"}
     * and {@code "bufferNames"} (unmodifiable lists of the model's dotted names) and {@code "path"} ({@code file}).
     *
     * @throws IOException when the file cannot be read, is not a checkpoint of this class, was written by a newer
     *                     format, or holds malformed metadata; the message names the file
     */
    public static Map<String, Object> readMetadata(Path file) throws IOException {
        Objects.requireNonNull(file, "Checkpoint file cannot be null");
        Properties metadata = new Properties();
        try (ZipFile zip = new ZipFile(file.toFile())) {
            ZipEntry entry = zip.getEntry(METADATA_ENTRY);
            if (entry == null) {
                throw new IOException("no " + METADATA_ENTRY + " entry; it was not written by ModelCheckpoint");
            }
            try (InputStream in = zip.getInputStream(entry)) {
                metadata.load(in);
            }
            int version = parseInt(metadata, "format_version");
            if (version < 1 || version > METADATA_VERSION) {
                throw new IOException("unsupported metadata format version " + version + " (supported: 1 to "
                    + METADATA_VERSION + ")");
            }
            Map<String, Object> info = new LinkedHashMap<>();
            info.put(EPOCH, parseInt(metadata, "epoch"));
            info.put(TRAIN_LOSS, parseFloat(metadata, "train_loss"));
            info.put(VAL_LOSS, metadata.containsKey("val_loss") ? parseFloat(metadata, "val_loss") : null);
            info.put(MONITORED, parseFloat(metadata, "monitored"));
            info.put(MODE, required(metadata, "mode"));
            boolean hasOptimizerState = Boolean.parseBoolean(required(metadata, "optimizer_state"));
            info.put(HAS_OPTIMIZER_STATE, hasOptimizerState);
            info.put(OPTIMIZER_TYPE, hasOptimizerState ? required(metadata, "optimizer_type") : null);
            info.put(PARAMETER_NAMES, names(metadata, "parameter"));
            info.put(BUFFER_NAMES, names(metadata, "buffer"));
            info.put(PATH, file);
            return Collections.unmodifiableMap(info);
        } catch (IOException e) {
            throw new IOException("Cannot read checkpoint metadata from " + file + ": " + e.getMessage(), e);
        }
    }

    private static String required(Properties metadata, String key) throws IOException {
        String value = metadata.getProperty(key);
        if (value == null) {
            throw new IOException("missing metadata key '" + key + "'");
        }
        return value;
    }

    private static int parseInt(Properties metadata, String key) throws IOException {
        String value = required(metadata, key);
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IOException("metadata key '" + key + "' is not an integer: '" + value + "'", e);
        }
    }

    private static float parseFloat(Properties metadata, String key) throws IOException {
        String value = required(metadata, key);
        try {
            return Float.parseFloat(value);
        } catch (NumberFormatException e) {
            throw new IOException("metadata key '" + key + "' is not a number: '" + value + "'", e);
        }
    }

    private static List<String> names(Properties metadata, String kind) throws IOException {
        int count = parseInt(metadata, kind + ".count");
        if (count < 0) {
            throw new IOException("metadata key '" + kind + ".count' is negative: " + count);
        }
        if (count > metadata.size()) {
            throw new IOException("metadata key '" + kind + ".count' is " + count + " but the metadata holds only "
                + metadata.size() + " keys");
        }
        List<String> names = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            names.add(required(metadata, kind + "." + i));
        }
        return Collections.unmodifiableList(names);
    }

    /**
     * Loads the checkpoint of {@code epoch} back into the trainer's model, strictly (every parameter and persistent
     * buffer, {@link Module#loadStateDict(Map, boolean)}), and, when the checkpoint holds optimizer state and the
     * trainer has an optimizer, into the optimizer ({@link Optimizer#loadStateDict(Map)}). The checkpoint held in
     * memory for {@code epoch} is used when there is one; otherwise, with a directory, {@link #checkpointFile(int)} is
     * read ({@link ModelSerializer#load}), after checking that its metadata names this epoch, exactly the model's
     * parameters and buffers, and, when it holds optimizer state, the trainer's optimizer class. All or nothing: when
     * any part is refused, the model and the optimizer are put back as they were.
     *
     * @throws IllegalStateException    when no training began ({@link #onTrainBegin})
     * @throws IllegalArgumentException when {@code epoch} is negative, no checkpoint exists for it, or the checkpoint
     *                                  does not match the model (names, or for an in-memory checkpoint sizes) or the
     *                                  optimizer (its class, or for an in-memory checkpoint its buffers)
     * @throws UncheckedIOException     when the checkpoint file cannot be read or does not fit the model or the
     *                                  optimizer ({@link ModelSerializer#load} reports both as {@link IOException});
     *                                  the message names the file
     */
    public synchronized void restore(int epoch) {
        if (trainer == null) {
            throw new IllegalStateException("ModelCheckpoint has no trainer; restore is available after onTrainBegin");
        }
        if (epoch < 0) {
            throw new IllegalArgumentException("Epoch must be non-negative, got: " + epoch);
        }
        Map<String, Object> checkpoint = checkpoints.get(epoch);
        if (checkpoint != null && checkpoint.containsKey(MODEL_STATE)) {
            restoreFromMemory(checkpoint);
            return;
        }
        if (checkpoint == null && (directory == null || !Files.exists(checkpointFile(epoch)))) {
            throw new IllegalArgumentException("No checkpoint kept for epoch " + epoch + "; kept epochs: "
                + checkpoints.keySet() + (directory == null ? "" : "; no file " + checkpointFile(epoch)));
        }
        restoreFromFile(epoch, checkpoint != null ? (Path) checkpoint.get(PATH) : checkpointFile(epoch));
    }

    /**
     * Restores the best checkpoint of the current run ({@link #bestEpoch()}) as {@link #restore(int)} does.
     *
     * @throws IllegalStateException when there is no best checkpoint: mode {@code "none"}, or no epoch with a
     *                               monitored value other than NaN ended since {@link #onTrainBegin}
     */
    public synchronized void restoreBest() {
        if (bestEpoch < 0) {
            throw new IllegalStateException(MODE_NONE.equals(saveBestOnly)
                ? "ModelCheckpoint in mode 'none' tracks no best checkpoint"
                : "ModelCheckpoint has no best checkpoint; no epoch with a monitored value other than NaN ended"
                    + " since onTrainBegin");
        }
        restore(bestEpoch);
    }

    @SuppressWarnings("unchecked")
    private void restoreFromMemory(Map<String, Object> checkpoint) {
        Module model = trainer.model();
        Optimizer optimizer = trainer.optimizer();
        Map<String, Object> modelState = (Map<String, Object>) checkpoint.get(MODEL_STATE);
        Map<String, Object> optimizerState = (Map<String, Object>) checkpoint.get(OPTIMIZER_STATE);
        if (optimizerState == null || optimizer == null) {
            model.loadStateDict(modelState, true);
            return;
        }
        Map<String, Object> modelBefore = model.stateDict();
        model.loadStateDict(modelState, true);
        try {
            optimizer.loadStateDict(optimizerState);
        } catch (RuntimeException e) {
            rollBack(e, model, modelBefore, null, null);
            throw e;
        }
    }

    private void restoreFromFile(int epoch, Path file) {
        Map<String, Object> info;
        try {
            info = readMetadata(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        if ((Integer) info.get(EPOCH) != epoch) {
            throw new IllegalArgumentException("Checkpoint " + file + " records epoch " + info.get(EPOCH)
                + ", not epoch " + epoch);
        }
        Module model = trainer.model();
        requireSameStructure(file, info, model);
        Optimizer optimizer = Boolean.TRUE.equals(info.get(HAS_OPTIMIZER_STATE)) ? trainer.optimizer() : null;
        if (optimizer != null && !optimizer.getClass().getName().equals(info.get(OPTIMIZER_TYPE))) {
            throw new IllegalArgumentException("Checkpoint " + file + " holds the state of " + info.get(OPTIMIZER_TYPE)
                + ", not of the trainer's " + optimizer.getClass().getName());
        }
        ComputeBackend backend = stateBackend(model, file);

        Map<String, Object> modelBefore = model.stateDict();
        Map<String, Object> optimizerBefore = optimizer != null ? snapshot(optimizer) : null;
        try {
            ModelSerializer.load(file, model, optimizer, backend);
        } catch (IOException e) {
            UncheckedIOException failure = new UncheckedIOException("Cannot restore checkpoint " + file + ": "
                + e.getMessage(), e);
            rollBack(failure, model, modelBefore, optimizer, optimizerBefore);
            throw failure;
        } catch (RuntimeException e) {
            rollBack(e, model, modelBefore, optimizer, optimizerBefore);
            throw e;
        }
    }

    /** Puts the model and, when given, the optimizer back to the saved state, attaching failures to {@code cause}. */
    private static void rollBack(Exception cause, Module model, Map<String, Object> modelBefore,
                                 Optimizer optimizer, Map<String, Object> optimizerBefore) {
        try {
            model.loadStateDict(modelBefore, true);
        } catch (RuntimeException e) {
            cause.addSuppressed(e);
        }
        if (optimizer != null) {
            try {
                optimizer.loadStateDict(optimizerBefore);
            } catch (RuntimeException e) {
                cause.addSuppressed(e);
            }
        }
    }

    /** Requires the parameter and buffer names recorded in {@code info} to be exactly those of {@code model}. */
    @SuppressWarnings("unchecked")
    private static void requireSameStructure(Path file, Map<String, Object> info, Module model) {
        StringBuilder mismatch = new StringBuilder();
        describeMismatch(mismatch, "parameters", (List<String>) info.get(PARAMETER_NAMES),
            model.namedParameters().keySet());
        describeMismatch(mismatch, "buffers", (List<String>) info.get(BUFFER_NAMES), model.namedBuffers().keySet());
        if (mismatch.length() > 0) {
            throw new IllegalArgumentException("Checkpoint " + file + " does not match " + model.moduleName() + ":"
                + mismatch);
        }
    }

    private static void describeMismatch(StringBuilder into, String kind, List<String> saved, Set<String> current) {
        Set<String> missing = new LinkedHashSet<>(current);
        missing.removeAll(saved);
        Set<String> unexpected = new LinkedHashSet<>(saved);
        unexpected.removeAll(current);
        if (!missing.isEmpty()) {
            into.append(" missing ").append(kind).append(' ').append(missing).append(';');
        }
        if (!unexpected.isEmpty()) {
            into.append(" unexpected ").append(kind).append(' ').append(unexpected).append(';');
        }
    }

    /** The backend holding the model's state, on which {@link ModelSerializer#load} creates optimizer tensors. */
    private static ComputeBackend stateBackend(Module model, Path file) {
        ComputeBackend backend = model.backend();
        if (backend == null) {
            for (Tensor buffer : model.namedBuffers().values()) {
                return buffer.backend();
            }
            throw new IllegalStateException("Cannot restore checkpoint " + file + ": " + model.moduleName()
                + " has no parameters or buffers on a backend");
        }
        return backend;
    }

    /**
     * The file of {@code epoch}'s checkpoint: {@code <directory>/<prefix>-epoch<NNNN>.nfc}.
     *
     * @throws IllegalStateException    when no directory is configured
     * @throws IllegalArgumentException when {@code epoch} is negative
     */
    public Path checkpointFile(int epoch) {
        if (directory == null) {
            throw new IllegalStateException("ModelCheckpoint keeps its checkpoints in memory; no directory is set");
        }
        if (epoch < 0) {
            throw new IllegalArgumentException("Epoch must be non-negative, got: " + epoch);
        }
        return directory.resolve(String.format(Locale.ROOT, "%s-epoch%04d%s", prefix, epoch, FILE_EXTENSION));
    }

    /** The checkpoint kept for {@code epoch} in the current run (see the class description), or null. */
    public synchronized Map<String, Object> getCheckpoint(int epoch) {
        return checkpoints.get(epoch);
    }

    /** The best checkpoint of the current run, or null when there is none. */
    public synchronized Map<String, Object> getBestCheckpoint() {
        if (bestEpoch >= 0) {
            return checkpoints.get(bestEpoch);
        }
        return null;
    }

    /** The checkpoint directory as configured, or null when checkpoints stay in memory. */
    public String checkpointDir() {
        return checkpointDir;
    }

    public String prefix() {
        return prefix;
    }

    public String saveBestOnly() {
        return saveBestOnly;
    }

    /**
     * The periodic interval this checkpoint was built with (0 for none). A run whose trainer sets a positive
     * {@link TrainerConfig#checkpointEveryNEpochs()} uses that interval instead ({@link #onTrainBegin}).
     */
    public int saveEveryNEpochs() {
        return saveEveryNEpochs;
    }

    public boolean saveOptimizerState() {
        return saveOptimizerState;
    }

    /** The maximum number of checkpoints kept, or 0 when every checkpoint is kept. */
    public int keep() {
        return keep;
    }

    public synchronized Float bestLoss() {
        return bestLoss;
    }

    public synchronized int bestEpoch() {
        return bestEpoch;
    }

    public synchronized int checkpointCount() {
        return checkpoints.size();
    }

    @Override
    public String name() {
        return "ModelCheckpoint";
    }

    @Override
    public synchronized String toString() {
        return "ModelCheckpoint{" +
            "checkpointDir=" + (checkpointDir == null ? "null" : "'" + checkpointDir + '\'') +
            ", prefix='" + prefix + '\'' +
            ", saveBestOnly='" + saveBestOnly + '\'' +
            ", saveEveryNEpochs=" + saveEveryNEpochs +
            ", keep=" + keep +
            ", checkpointCount=" + checkpoints.size() +
            ", bestLoss=" + bestLoss +
            ", bestEpoch=" + bestEpoch +
            '}';
    }

    /**
     * Settings of a {@link ModelCheckpoint}; {@link #build()} validates them. Defaults: in memory, prefix
     * {@value ModelCheckpoint#DEFAULT_PREFIX}, mode {@code "min"}, no periodic saves, no optimizer state, every
     * checkpoint kept.
     */
    public static final class Builder {
        private String checkpointDir;
        private String prefix = DEFAULT_PREFIX;
        private String saveBestOnly = MODE_MIN;
        private int saveEveryNEpochs;
        private boolean saveOptimizerState;
        private int keep;

        private Builder() {
        }

        /** The directory checkpoint files are written to (created when missing); null keeps them in memory. */
        public Builder directory(String checkpointDir) {
            this.checkpointDir = checkpointDir;
            return this;
        }

        /** The directory checkpoint files are written to (created when missing); null keeps them in memory. */
        public Builder directory(Path checkpointDir) {
            this.checkpointDir = checkpointDir == null ? null : checkpointDir.toString();
            return this;
        }

        /** The file name prefix of checkpoint files: a file name without path separators. */
        public Builder prefix(String prefix) {
            this.prefix = prefix;
            return this;
        }

        /** {@code "min"} or {@code "max"} to save the improving epochs only, {@code "none"} to save every epoch. */
        public Builder saveBestOnly(String mode) {
            this.saveBestOnly = mode;
            return this;
        }

        /** Also saves every {@code n}-th epoch when positive; 0 disables periodic saves. */
        public Builder saveEveryNEpochs(int n) {
            this.saveEveryNEpochs = n;
            return this;
        }

        /** Includes the optimizer state in each checkpoint. */
        public Builder saveOptimizerState(boolean saveOptimizerState) {
            this.saveOptimizerState = saveOptimizerState;
            return this;
        }

        /** Keeps at most {@code keep} checkpoints (the best always among them); 0 keeps every checkpoint. */
        public Builder keep(int keep) {
            this.keep = keep;
            return this;
        }

        /**
         * @throws IllegalArgumentException when the mode is not {@code "min"}, {@code "max"} or {@code "none"},
         *                                  {@code saveEveryNEpochs} or {@code keep} is negative, the prefix is not a
         *                                  plain file name, or the directory is not a valid path
         */
        public ModelCheckpoint build() {
            return new ModelCheckpoint(this);
        }
    }
}
