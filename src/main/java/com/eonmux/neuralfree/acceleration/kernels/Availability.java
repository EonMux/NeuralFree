package com.eonmux.neuralfree.acceleration.kernels;

/**
 * Immutable answer to "can this be used here?", with the reason when it cannot.
 */
public final class Availability {

    private static final Availability OK = new Availability(true, null);

    private final boolean available;
    private final String reason;

    private Availability(boolean available, String reason) {
        this.available = available;
        this.reason = reason;
    }

    /** Available; {@link #reason()} is null. */
    public static Availability ok() {
        return OK;
    }

    /**
     * Not available, for the given human-readable reason.
     *
     * @throws IllegalArgumentException when the reason is null or blank
     */
    public static Availability no(String reason) {
        if (reason == null || reason.trim().isEmpty()) {
            throw new IllegalArgumentException("an unavailable result needs a reason");
        }
        return new Availability(false, reason);
    }

    public boolean available() {
        return available;
    }

    /** Why it is unavailable, or null when available. */
    public String reason() {
        return reason;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Availability)) {
            return false;
        }
        Availability that = (Availability) o;
        return available == that.available
                && (reason == null ? that.reason == null : reason.equals(that.reason));
    }

    @Override
    public int hashCode() {
        return (available ? 1 : 0) * 31 + (reason == null ? 0 : reason.hashCode());
    }

    @Override
    public String toString() {
        return available ? "available" : "unavailable: " + reason;
    }
}
