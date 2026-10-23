package com.eonmux.neuralfree;

public class Config {
	
	public enum ACCELERATION_MODE {
		AUTO, CUDA, OPENCL, VULKAN
	}
	
	private static ACCELERATION_MODE accelerationMode = ACCELERATION_MODE.AUTO;
	
	public Config() {
		
	}
	
	public static ACCELERATION_MODE getAccelerationMode() {
		return accelerationMode;
	}
	
	public static void setAccelerationMode(ACCELERATION_MODE mode) {
		accelerationMode = mode;
	}
}
