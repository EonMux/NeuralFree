package com.eonmux.neuralfree.acceleration;

import java.util.UUID;

public interface HardwareDevice {
    void initialize();
    void release();
    UUID getUUID();
    String getDeviceName();
    void queryCapabilities();
}
