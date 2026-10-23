package com.eonmux.neuralfree.training;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.Logger;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.Property;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Records the messages logged under one class's logger while open: attaches an appender to that logger and detaches
 * it on {@link #close()}. The logger's level is set to INFO for the duration, so INFO messages are recorded whatever
 * the logging configuration, and restored on close.
 */
public final class LogCapture extends AbstractAppender implements AutoCloseable {
    private final Logger logger;
    private final Level previousLevel;
    private final List<LogEvent> events = new CopyOnWriteArrayList<>();

    /** Starts recording the logger named after {@code owner}. */
    public LogCapture(Class<?> owner) {
        super("capture-" + owner.getSimpleName() + "-" + UUID.randomUUID(), null, null, true, Property.EMPTY_ARRAY);
        this.logger = (Logger) LogManager.getLogger(owner.getName());
        this.previousLevel = logger.getLevel();
        start();
        logger.addAppender(this);
        logger.setLevel(Level.INFO);
    }

    @Override
    public void append(LogEvent event) {
        events.add(event.toImmutable());
    }

    /** The formatted messages recorded at {@code level}, in logging order. */
    public List<String> messages(Level level) {
        List<String> messages = new ArrayList<>();
        for (LogEvent event : events) {
            if (event.getLevel().equals(level)) {
                messages.add(event.getMessage().getFormattedMessage());
            }
        }
        return messages;
    }

    /** The formatted messages recorded at {@code level} that start with {@code prefix}, in logging order. */
    public List<String> messages(Level level, String prefix) {
        List<String> matching = new ArrayList<>();
        for (String message : messages(level)) {
            if (message.startsWith(prefix)) {
                matching.add(message);
            }
        }
        return matching;
    }

    @Override
    public void close() {
        logger.removeAppender(this);
        logger.setLevel(previousLevel);
        stop();
    }
}
