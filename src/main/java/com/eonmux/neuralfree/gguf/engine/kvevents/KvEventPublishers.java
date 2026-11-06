package com.eonmux.neuralfree.gguf.engine.kvevents;

import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Modifier;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The KV cache event publishers the engine ships (vLLM ships {@code null} and {@code zmq}): {@link #none()}
 * publishes nothing, {@link #log()} logs every batch, {@link #zmq} publishes on vLLM's ZMQ transport,
 * {@link #listener} hands every batch to an in-process consumer, and {@link #named} resolves the
 * {@code neuralfree.kv.events} setting — {@code log}, {@code zmq}, vLLM's {@code KVEventsConfig} JSON
 * ({@link KvEventsConfig}), or the class name of any {@link KvEventPublisher}, so another transport plugs in from
 * the class path. {@link #ranked} stamps a replica's data-parallel rank on every batch.
 */
public final class KvEventPublishers {

    private static final Logger LOGGER = Logger.getLogger(KvEventPublishers.class.getName());

    /** The logger the {@link #log()} publisher writes to. */
    public static final String LOG_NAME = "com.eonmux.neuralfree.kvevents";

    private static final KvEventPublisher NONE = new KvEventPublisher() {
        @Override
        public void publish(KvEventBatch batch) {
        }

        @Override
        public boolean enabled() {
            return false;
        }

        @Override
        public String toString() {
            return "none";
        }
    };

    private KvEventPublishers() {
    }

    /** The publisher that publishes nothing; the engine then collects no events. */
    public static KvEventPublisher none() {
        return NONE;
    }

    /** Logs every batch at INFO on the {@link #LOG_NAME} logger. */
    public static KvEventPublisher log() {
        Logger out = Logger.getLogger(LOG_NAME);
        return new KvEventPublisher() {
            @Override
            public void publish(KvEventBatch batch) {
                out.info(batch.toString());
            }

            @Override
            public String toString() {
                return "log";
            }
        };
    }

    /** Hands every batch to {@code consumer} on the engine's thread. */
    public static KvEventPublisher listener(Consumer<KvEventBatch> consumer) {
        Objects.requireNonNull(consumer, "consumer");
        return new KvEventPublisher() {
            @Override
            public void publish(KvEventBatch batch) {
                consumer.accept(batch);
            }

            @Override
            public String toString() {
                return "listener";
            }
        };
    }

    /** {@code publisher} with every batch stamped with the data-parallel rank {@code rank}. */
    public static KvEventPublisher ranked(KvEventPublisher publisher, int rank) {
        Objects.requireNonNull(publisher, "publisher");
        if (rank < 0) {
            throw new IllegalArgumentException("data-parallel rank must be >= 0, got " + rank);
        }
        if (!publisher.enabled()) {
            return publisher;
        }
        return new KvEventPublisher() {
            @Override
            public void publish(KvEventBatch batch) {
                publisher.publish(batch.withRank(rank));
            }

            @Override
            public void close() {
                publisher.close();
            }

            @Override
            public String toString() {
                return publisher + "@rank" + rank;
            }
        };
    }

    /** The publisher {@code memory.kvEvents()} names ({@link #named}), rank 0. */
    public static KvEventPublisher forConfig(MemoryConfig memory) {
        return named(Objects.requireNonNull(memory, "memory").kvEvents());
    }

    /**
     * The publisher {@code memory.kvEvents()} names for the replica of data-parallel rank {@code dataParallelRank}:
     * a ZMQ publisher offsets its ports by the rank and stamps it on its batches, as vLLM's per-rank publishers do.
     */
    public static KvEventPublisher forConfig(MemoryConfig memory, int dataParallelRank) {
        return named(Objects.requireNonNull(memory, "memory").kvEvents(), dataParallelRank);
    }

    /**
     * The publisher {@code spec} names: null, {@code null} or {@code none} for {@link #none()}, {@code log} for
     * {@link #log()}, {@code zmq} for the ZMQ transport of vLLM's defaults ({@link #zmq}), a
     * {@code KVEventsConfig} JSON object ({@link KvEventsConfig}) for the publisher it selects, else a class name
     * implementing {@link KvEventPublisher} with a public no-argument constructor, instantiated fresh.
     *
     * @throws IllegalArgumentException when the class is missing, is no publisher, cannot be instantiated, or the
     *                                  JSON is not a usable {@code KVEventsConfig}
     * @throws IllegalStateException    when a ZMQ endpoint cannot be bound or connected
     */
    public static KvEventPublisher named(String spec) {
        return named(spec, 0);
    }

    /** The publisher {@link #named} resolves from {@code spec} for the replica of rank {@code dataParallelRank}. */
    public static KvEventPublisher named(String spec, int dataParallelRank) {
        String name = spec == null ? "" : spec.trim();
        if (name.isEmpty() || name.equalsIgnoreCase("null") || name.equalsIgnoreCase("none")) {
            return none();
        }
        if (name.equalsIgnoreCase("log")) {
            return log();
        }
        if (name.startsWith("{")) {
            KvEventsConfig config = KvEventsConfig.parse(name, MemoryConfig.PROP_KV_EVENTS);
            return resolved(config, dataParallelRank);
        }
        if (name.equalsIgnoreCase(KvEventsConfig.ZMQ)) {
            return zmq(ZmqEventPublisherConfig.defaults().withDataParallelRank(dataParallelRank));
        }
        return instantiate(name);
    }

    /** The publisher a parsed {@code KVEventsConfig} selects, a ZMQ one for rank {@code dataParallelRank}. */
    private static KvEventPublisher resolved(KvEventsConfig config, int dataParallelRank) {
        if (!config.enabled()) {
            return none();
        }
        if (config.zmq() != null) {
            return zmq(config.zmq().withDataParallelRank(dataParallelRank));
        }
        if (config.publisher().equalsIgnoreCase("log")) {
            return log();
        }
        return instantiate(config.publisher());
    }

    /**
     * The ZMQ transport {@code config} describes ({@link ZmqEventPublisher}), its sockets open on return.
     *
     * @throws IllegalStateException when an endpoint cannot be bound or connected
     */
    public static ZmqEventPublisher zmq(ZmqEventPublisherConfig config) {
        return new ZmqEventPublisher(config);
    }

    private static KvEventPublisher instantiate(String className) {
        String prop = MemoryConfig.PROP_KV_EVENTS;
        Class<?> type;
        try {
            ClassLoader loader = Thread.currentThread().getContextClassLoader();
            type = Class.forName(className, true, loader != null ? loader : KvEventPublishers.class.getClassLoader());
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException(prop + "=" + className + ": no such class (use zmq, log, none or the"
                    + " class name of a KvEventPublisher)", e);
        }
        if (!KvEventPublisher.class.isAssignableFrom(type) || Modifier.isAbstract(type.getModifiers())) {
            throw new IllegalArgumentException(prop + "=" + className + " is not a concrete KvEventPublisher");
        }
        try {
            Constructor<?> ctor = type.getConstructor();
            return (KvEventPublisher) ctor.newInstance();
        } catch (NoSuchMethodException | IllegalAccessException | InstantiationException e) {
            throw new IllegalArgumentException(prop + "=" + className + " needs a public no-argument constructor", e);
        } catch (InvocationTargetException e) {
            throw new IllegalArgumentException(prop + "=" + className + ": its constructor failed: "
                    + e.getCause(), e.getCause());
        }
    }

    /** Publishes {@code batch}, logging a publisher failure instead of passing it on (the batch is lost). */
    public static void publishSafely(KvEventPublisher publisher, KvEventBatch batch) {
        try {
            publisher.publish(batch);
        } catch (RuntimeException e) {
            LOGGER.log(Level.WARNING, "the KV cache event publisher " + publisher + " failed; the batch is lost", e);
        }
    }
}
