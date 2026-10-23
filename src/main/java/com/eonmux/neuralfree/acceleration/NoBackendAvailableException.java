package com.eonmux.neuralfree.acceleration;

public class NoBackendAvailableException extends RuntimeException {
    
    public NoBackendAvailableException(String message) {
        super(message);
    }
    
    public NoBackendAvailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
