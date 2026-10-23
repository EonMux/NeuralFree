package com.eonmux.neuralfree.acceleration.vulkan;

import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * One reference on a {@link VkInstanceManager} paired with the physical device chosen from it: the single rule
 * for choosing ({@link #fastest()}) and adopting ({@link #adopt}) the Vulkan device that a {@link VkDevice} or a
 * compute backend opens.
 *
 * <p>A selection owns its instance reference. The object constructed from it takes the reference over and drops
 * it when it is released or closed; a constructor that fails after receiving the selection calls
 * {@link #release()} instead.</p>
 */
public final class VkDeviceSelection {

    private final VkInstanceManager manager;
    private final VkPhysicalDeviceInfo info;
    private final AtomicBoolean released = new AtomicBoolean();

    private VkDeviceSelection(VkInstanceManager manager, VkPhysicalDeviceInfo info) {
        this.manager = manager;
        this.info = info;
    }

    /**
     * The fastest eligible device of the shared instance ({@link VkDeviceRanking#fastest(List)}), with a new
     * reference on the instance.
     *
     * @throws NoBackendAvailableException when Vulkan is unavailable or no device is eligible; the message lists
     *                                     each device with the reason it cannot run the compute backend
     */
    public static VkDeviceSelection fastest() {
        VkInstanceManager manager = VkInstanceManager.acquire();
        try {
            VkPhysicalDeviceInfo best = VkDeviceRanking.fastest(manager.devices());
            if (best == null) {
                throw new NoBackendAvailableException(noEligibleDeviceMessage(manager.devices()));
            }
            return new VkDeviceSelection(manager, best);
        } catch (RuntimeException e) {
            manager.release();
            throw e;
        }
    }

    /**
     * {@code info} of {@code manager}, with a new reference on the manager.
     *
     * @param info one of {@code manager.devices()}
     * @throws IllegalArgumentException when an argument is null or {@code info} belongs to another instance
     * @throws IllegalStateException    when the manager has already been destroyed
     */
    public static VkDeviceSelection adopt(VkInstanceManager manager, VkPhysicalDeviceInfo info) {
        if (manager == null) {
            throw new IllegalArgumentException("instance manager must not be null");
        }
        if (info == null) {
            throw new IllegalArgumentException("physical device info must not be null");
        }
        boolean member = false;
        for (VkPhysicalDeviceInfo d : manager.devices()) {
            member |= d == info;
        }
        if (!member) {
            throw new IllegalArgumentException("Vulkan device '" + info.name()
                    + "' does not belong to this instance manager");
        }
        return new VkDeviceSelection(manager.retain(), info);
    }

    /** The instance whose reference this selection holds. */
    public VkInstanceManager manager() {
        return manager;
    }

    /** The chosen physical device, one of {@code manager().devices()}. */
    public VkPhysicalDeviceInfo info() {
        return info;
    }

    /**
     * Drops the instance reference, for a constructor that fails before taking the selection over. Idempotent, so
     * the reference is dropped at most once.
     */
    public void release() {
        if (released.compareAndSet(false, true)) {
            manager.release();
        }
    }

    /**
     * Why no device was chosen: "no Vulkan physical devices found" for an empty list, otherwise each device with
     * its {@link VkDeviceRanking#unsupportedReason(VkPhysicalDeviceInfo)}.
     */
    static String noEligibleDeviceMessage(List<VkPhysicalDeviceInfo> devices) {
        if (devices.isEmpty()) {
            return "no Vulkan physical devices found";
        }
        StringBuilder sb = new StringBuilder("no Vulkan device can run the compute backend:");
        for (VkPhysicalDeviceInfo d : devices) {
            String reason = VkDeviceRanking.unsupportedReason(d);
            sb.append(" '").append(d.name()).append("': ").append(reason == null ? "eligible" : reason).append(';');
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return "VkDeviceSelection{" + info.name() + '}';
    }
}
