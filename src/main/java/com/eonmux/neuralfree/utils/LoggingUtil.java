package com.eonmux.neuralfree.utils;

import org.apache.logging.log4j.*;

import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Logger lookup by owner. The owner is either the {@link Class} whose name the logger takes (for static contexts,
 * {@code getLogger(Foo.class)}) or an instance whose runtime class names it ({@code getLogger(this)}).
 */
public class LoggingUtil {

	public static Map<String, Instant> lastLogTimes = new ConcurrentHashMap<>();

	public static Logger getLogger(Object a) {
		String s = loggerName(a);

		if (LogManager.getLogger(s).isDebugEnabled()) {
			lastLogTimes.put(s, Instant.now());
		}
		return LogManager.getLogger(s);
	}

	public static Logger getLogger(Object a, boolean getLastLogTime) {
		String s = loggerName(a);
		if (LogManager.getLogger(s).isDebugEnabled()) {
			if (getLastLogTime) {
				LogManager.getLogger(s)
						.debug("Last Log was "
								+ Math.max(0,
										(Instant.now().toEpochMilli() - lastLogTimes
												.getOrDefault(s, Instant.now()).toEpochMilli()))
								+ " milliseconds ago");
			}
			lastLogTimes.put(s, Instant.now());
		}
		return LogManager.getLogger(s);
	}

	/** The class name a logger owned by {@code owner} takes: the class itself, or the class of an instance. */
	private static String loggerName(Object owner) {
		if (owner == null) {
			throw new IllegalArgumentException("logger owner is null");
		}
		return owner instanceof Class ? ((Class<?>) owner).getName() : owner.getClass().getName();
	}
}
