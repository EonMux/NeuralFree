package com.eonmux.neuralfree.utils;

/**
 * Failure aggregation for {@code close()} paths that must release every resource even when some releases fail.
 * The first failure is the one reported; every later failure is attached to it as a suppressed exception.
 *
 * <p>Typical use:</p>
 * <pre>{@code
 * RuntimeException failure = null;
 * failure = Closeables.close(a, failure);
 * failure = Closeables.close(b, failure);
 * if (failure != null) throw failure;
 * }</pre>
 */
public final class Closeables {

    private Closeables() {
    }

    /**
     * Joins a new failure to the running one.
     *
     * @param first the failure reported so far, or null
     * @param next  the new failure (non-null)
     * @return {@code next} when {@code first} is null, otherwise {@code first} with {@code next} suppressed
     */
    public static RuntimeException chain(RuntimeException first, RuntimeException next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    /**
     * Closes {@code c} (nothing for null) and joins its failure to the running one. A checked exception that must
     * become the reported failure is wrapped in an {@link IllegalStateException} naming {@code c}; one joined to an
     * existing failure is suppressed as thrown.
     *
     * @param c     the resource to close, or null
     * @param first the failure reported so far, or null
     * @return the failure to report after this close, or null when none happened
     */
    public static RuntimeException close(AutoCloseable c, RuntimeException first) {
        if (c == null) {
            return first;
        }
        try {
            c.close();
            return first;
        } catch (RuntimeException e) {
            return chain(first, e);
        } catch (Exception e) {
            if (first == null) {
                return new IllegalStateException("closing " + c + " failed", e);
            }
            first.addSuppressed(e);
            return first;
        }
    }

    /**
     * Closes every element of {@code cs} in iteration order (null elements are skipped) and joins their failures
     * to the running one.
     *
     * @return the failure to report after these closes, or null when none happened
     */
    public static RuntimeException closeAll(Iterable<? extends AutoCloseable> cs, RuntimeException first) {
        RuntimeException failure = first;
        for (AutoCloseable c : cs) {
            failure = close(c, failure);
        }
        return failure;
    }
}
