package com.eonmux.neuralfree.acceleration.vulkan;

import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK;
import org.lwjgl.vulkan.VkApplicationInfo;
import org.lwjgl.vulkan.VkDebugUtilsMessengerCallbackDataEXT;
import org.lwjgl.vulkan.VkDebugUtilsMessengerCallbackEXT;
import org.lwjgl.vulkan.VkDebugUtilsMessengerCreateInfoEXT;
import org.lwjgl.vulkan.VkExtensionProperties;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkInstanceCreateInfo;
import org.lwjgl.vulkan.VkLayerProperties;
import org.lwjgl.vulkan.VkPhysicalDevice;

import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.lwjgl.vulkan.EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
import static org.lwjgl.vulkan.EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
import static org.lwjgl.vulkan.EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
import static org.lwjgl.vulkan.EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
import static org.lwjgl.vulkan.EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
import static org.lwjgl.vulkan.EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
import static org.lwjgl.vulkan.EXTDebugUtils.VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
import static org.lwjgl.vulkan.EXTDebugUtils.vkCreateDebugUtilsMessengerEXT;
import static org.lwjgl.vulkan.EXTDebugUtils.vkDestroyDebugUtilsMessengerEXT;
import static org.lwjgl.vulkan.KHRPortabilityEnumeration.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
import static org.lwjgl.vulkan.KHRPortabilityEnumeration.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
import static org.lwjgl.vulkan.VK10.VK_ERROR_INCOMPATIBLE_DRIVER;
import static org.lwjgl.vulkan.VK10.VK_FALSE;
import static org.lwjgl.vulkan.VK10.VK_MAKE_API_VERSION;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkCreateInstance;
import static org.lwjgl.vulkan.VK10.vkDestroyInstance;
import static org.lwjgl.vulkan.VK10.vkEnumerateInstanceExtensionProperties;
import static org.lwjgl.vulkan.VK10.vkEnumerateInstanceLayerProperties;
import static org.lwjgl.vulkan.VK10.vkEnumeratePhysicalDevices;
import static org.lwjgl.vulkan.VK12.VK_API_VERSION_1_2;
import static org.lwjgl.vulkan.VK13.VK_API_VERSION_1_3;

/**
 * Process-wide, reference-counted Vulkan instance (API 1.2 or newer) with
 * portability enumeration and optional validation, and the snapshot of its
 * physical devices.
 *
 * <p>{@link #acquire()} returns the shared manager and takes one reference;
 * {@link #retain()} takes another reference on a manager already held; each
 * reference is dropped by one {@link #release()}. The instance is destroyed when
 * the last reference goes, after which the manager is unusable and the next
 * {@link #acquire()} creates a fresh one. The physical-device snapshots in
 * {@link #devices()} stay valid only while a reference is held.</p>
 *
 * <p>Thread-safe: reference counting and creation are serialised on one lock.</p>
 */
public final class VkInstanceManager implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(VkInstanceManager.class);

    /** Set true to enable {@code VK_LAYER_KHRONOS_validation} and a debug messenger that logs to log4j. */
    public static final String VALIDATION_PROPERTY = "neuralfree.vulkan.validation";

    private static final String VALIDATION_LAYER = "VK_LAYER_KHRONOS_validation";
    private static final Object LOCK = new Object();
    private static VkInstanceManager shared;

    private final VkInstance instance;
    private final int apiVersion;
    private final boolean validation;
    private final long messenger;
    private final VkDebugUtilsMessengerCallbackEXT callback;
    private final List<VkPhysicalDeviceInfo> devices;
    private int references;

    private VkInstanceManager(VkInstance instance, int apiVersion, boolean validation, long messenger,
                              VkDebugUtilsMessengerCallbackEXT callback, List<VkPhysicalDeviceInfo> devices) {
        this.instance = instance;
        this.apiVersion = apiVersion;
        this.validation = validation;
        this.messenger = messenger;
        this.callback = callback;
        this.devices = Collections.unmodifiableList(new ArrayList<>(devices));
        this.references = 1;
    }

    /**
     * Acquires a reference to the shared instance, creating it on first use.
     *
     * @throws NoBackendAvailableException when the loader is missing
     *         or the instance version is below 1.2
     */
    public static VkInstanceManager acquire() {
        synchronized (LOCK) {
            if (shared != null) {
                shared.references++;
                return shared;
            }
            shared = create();
            return shared;
        }
    }

    /**
     * Takes one more reference on this manager, for a holder that releases it independently.
     *
     * @return this manager
     * @throws IllegalStateException when the instance has already been destroyed
     */
    public VkInstanceManager retain() {
        synchronized (LOCK) {
            requireAlive();
            references++;
            return this;
        }
    }

    /** Drops one reference; the instance is destroyed when the last reference is released. */
    public void release() {
        synchronized (LOCK) {
            if (references <= 0) {
                LOGGER.warn("Vulkan instance released more often than it was acquired; ignoring");
                return;
            }
            references--;
            if (references > 0) {
                return;
            }
            if (shared == this) {
                shared = null;
            }
            destroy();
        }
    }

    /** Same as {@link #release()}. */
    @Override
    public void close() {
        release();
    }

    /** @throws IllegalStateException when the instance has already been destroyed */
    public VkInstance instance() {
        synchronized (LOCK) {
            requireAlive();
            return instance;
        }
    }

    /** The instance API version in use: {@code min(vkEnumerateInstanceVersion(), 1.3)}. */
    public int apiVersion() {
        return apiVersion;
    }

    public boolean validationEnabled() {
        return validation;
    }

    /** Unmodifiable snapshot of every physical device, in enumeration order. */
    public List<VkPhysicalDeviceInfo> devices() {
        return devices;
    }

    private void requireAlive() {
        if (references <= 0) {
            throw new IllegalStateException("Vulkan instance has been destroyed");
        }
    }

    private void destroy() {
        if (messenger != VK_NULL_HANDLE) {
            vkDestroyDebugUtilsMessengerEXT(instance, messenger, null);
        }
        vkDestroyInstance(instance, null);
        if (callback != null) {
            callback.free();
        }
        LOGGER.debug("Vulkan instance destroyed");
    }

    // ---------------------------------------------------------------- creation

    private static VkInstanceManager create() {
        int supported = supportedInstanceVersion();
        if (Integer.compareUnsigned(supported, VK_API_VERSION_1_2) < 0) {
            throw new NoBackendAvailableException("Vulkan 1.2 is required, but the loader supports only "
                    + versionString(supported));
        }
        int api = Integer.compareUnsigned(supported, VK_API_VERSION_1_3) < 0
                ? VK_MAKE_API_VERSION(0, supported >>> 22, (supported >>> 12) & 0x3FF, 0)
                : VK_API_VERSION_1_3;

        Set<String> available = instanceExtensions(null);
        boolean portability = available.contains(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        boolean validation = Boolean.getBoolean(VALIDATION_PROPERTY) && validationLayerPresent();
        boolean debugUtils = validation && (available.contains(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
                || instanceExtensions(VALIDATION_LAYER).contains(VK_EXT_DEBUG_UTILS_EXTENSION_NAME));

        VkDebugUtilsMessengerCallbackEXT callback = debugUtils ? VkDebugUtilsMessengerCallbackEXT.create(
                VkInstanceManager::onDebugMessage) : null;
        VkInstance instance = null;
        long messenger = VK_NULL_HANDLE;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            List<String> extensions = new ArrayList<>();
            if (portability) {
                extensions.add(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            }
            if (debugUtils) {
                extensions.add(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }
            VkApplicationInfo app = VkApplicationInfo.calloc(stack).sType$Default()
                    .pApplicationName(stack.UTF8("NeuralFree"))
                    .applicationVersion(VK_MAKE_API_VERSION(0, 1, 0, 0))
                    .pEngineName(stack.UTF8("NeuralFree"))
                    .engineVersion(VK_MAKE_API_VERSION(0, 1, 0, 0))
                    .apiVersion(api);
            VkInstanceCreateInfo ci = VkInstanceCreateInfo.calloc(stack).sType$Default()
                    .flags(portability ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0)
                    .pApplicationInfo(app)
                    .ppEnabledExtensionNames(names(stack, extensions));
            if (validation) {
                ci.ppEnabledLayerNames(names(stack, Collections.singletonList(VALIDATION_LAYER)));
            }
            VkDebugUtilsMessengerCreateInfoEXT messengerInfo = null;
            if (callback != null) {
                messengerInfo = messengerCreateInfo(stack, callback);
                ci.pNext(messengerInfo.address());
            }

            PointerBuffer pInstance = stack.mallocPointer(1);
            int result = vkCreateInstance(ci, null, pInstance);
            if (result != VK_SUCCESS) {
                throw new NoBackendAvailableException(result == VK_ERROR_INCOMPATIBLE_DRIVER
                        ? "no Vulkan driver supports API " + versionString(api)
                        : "vkCreateInstance failed with " + result);
            }
            instance = new VkInstance(pInstance.get(0), ci);

            if (messengerInfo != null) {
                LongBuffer pMessenger = stack.mallocLong(1);
                result = vkCreateDebugUtilsMessengerEXT(instance, messengerInfo, null, pMessenger);
                if (result == VK_SUCCESS) {
                    messenger = pMessenger.get(0);
                } else {
                    LOGGER.warn("vkCreateDebugUtilsMessengerEXT failed with {}; validation messages are not logged",
                            result);
                }
            }
            List<VkPhysicalDeviceInfo> devices = enumerateDevices(instance, api, stack);
            LOGGER.info("Vulkan instance created: API {}, {} physical device(s), portability={}, validation={}",
                    versionString(api), devices.size(), portability, validation);
            return new VkInstanceManager(instance, api, validation, messenger, callback, devices);
        } catch (RuntimeException | Error e) {
            if (instance != null) {
                if (messenger != VK_NULL_HANDLE) {
                    vkDestroyDebugUtilsMessengerEXT(instance, messenger, null);
                }
                vkDestroyInstance(instance, null);
            }
            if (callback != null) {
                callback.free();
            }
            throw e;
        }
    }

    /** The loader's instance version; a missing or unloadable loader becomes {@link NoBackendAvailableException}. */
    private static int supportedInstanceVersion() {
        try {
            return VK.getInstanceVersionSupported();
        } catch (Throwable t) {
            throw new NoBackendAvailableException("Vulkan loader is not available: " + t, t);
        }
    }

    private static List<VkPhysicalDeviceInfo> enumerateDevices(VkInstance instance, int api, MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        int result = vkEnumeratePhysicalDevices(instance, count, null);
        if (result != VK_SUCCESS) {
            throw new NoBackendAvailableException("vkEnumeratePhysicalDevices failed with " + result);
        }
        int n = count.get(0);
        List<VkPhysicalDeviceInfo> devices = new ArrayList<>(n);
        if (n == 0) {
            return devices;
        }
        PointerBuffer handles = stack.mallocPointer(n);
        result = vkEnumeratePhysicalDevices(instance, count, handles);
        if (result != VK_SUCCESS) {
            throw new NoBackendAvailableException("vkEnumeratePhysicalDevices failed with " + result);
        }
        for (int i = 0; i < count.get(0); i++) {
            VkPhysicalDevice device = new VkPhysicalDevice(handles.get(i), instance);
            try {
                VkPhysicalDeviceInfo info = VkPhysicalDeviceInfo.query(device, api);
                LOGGER.debug("Vulkan device {}: {}", i, info);
                devices.add(info);
            } catch (RuntimeException e) {
                LOGGER.warn("Vulkan physical device {} skipped: its properties could not be read: {}", i, e.toString());
            }
        }
        return devices;
    }

    private static Set<String> instanceExtensions(String layer) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer count = stack.mallocInt(1);
            int result = vkEnumerateInstanceExtensionProperties(layer, count, null);
            Set<String> names = new LinkedHashSet<>();
            if (result != VK_SUCCESS || count.get(0) == 0) {
                return names;
            }
            VkExtensionProperties.Buffer props = VkExtensionProperties.malloc(count.get(0));
            try {
                if (vkEnumerateInstanceExtensionProperties(layer, count, props) == VK_SUCCESS) {
                    for (int i = 0; i < count.get(0); i++) {
                        names.add(props.get(i).extensionNameString());
                    }
                }
            } finally {
                props.free();
            }
            return names;
        }
    }

    private static boolean validationLayerPresent() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer count = stack.mallocInt(1);
            if (vkEnumerateInstanceLayerProperties(count, null) == VK_SUCCESS && count.get(0) > 0) {
                VkLayerProperties.Buffer layers = VkLayerProperties.malloc(count.get(0));
                try {
                    if (vkEnumerateInstanceLayerProperties(count, layers) == VK_SUCCESS) {
                        for (int i = 0; i < count.get(0); i++) {
                            if (VALIDATION_LAYER.equals(layers.get(i).layerNameString())) {
                                return true;
                            }
                        }
                    }
                } finally {
                    layers.free();
                }
            }
        }
        LOGGER.warn("-D{}=true but {} is not installed; continuing without validation", VALIDATION_PROPERTY,
                VALIDATION_LAYER);
        return false;
    }

    private static VkDebugUtilsMessengerCreateInfoEXT messengerCreateInfo(MemoryStack stack,
                                                                         VkDebugUtilsMessengerCallbackEXT callback) {
        return VkDebugUtilsMessengerCreateInfoEXT.calloc(stack).sType$Default()
                .messageSeverity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                .messageType(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
                .pfnUserCallback(callback);
    }

    private static int onDebugMessage(int severity, int types, long pCallbackData, long pUserData) {
        String message = pCallbackData == MemoryUtil.NULL ? "<no data>"
                : VkDebugUtilsMessengerCallbackDataEXT.create(pCallbackData).pMessageString();
        if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
            LOGGER.error("Vulkan validation: {}", message);
        } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
            LOGGER.warn("Vulkan validation: {}", message);
        } else {
            LOGGER.debug("Vulkan validation: {}", message);
        }
        return VK_FALSE;
    }

    private static PointerBuffer names(MemoryStack stack, List<String> names) {
        if (names.isEmpty()) {
            return null;
        }
        PointerBuffer buffer = stack.mallocPointer(names.size());
        for (String name : names) {
            buffer.put(stack.UTF8(name));
        }
        return buffer.flip();
    }

    private static String versionString(int version) {
        return (version >>> 22) + "." + ((version >>> 12) & 0x3FF);
    }
}
