package com.eonmux.neuralfree;

public class Config {
	
	/**
	 * The API the training framework runs its tensors on: {@code AUTO} or one of the built-in APIs
	 * ({@code CUDA}, {@code OPENCL}, {@code VULKAN}, {@code HIP}, {@code METAL}) or an accelerator API
	 * served by an installed backend provider ({@code SYCL}, {@code XPU}, {@code CANN},
	 * {@code HEXAGON}, {@code OPENVINO}, {@code WEBGPU}, {@code MUSA}, {@code ZDNN}, {@code ZENDNN}).
	 */
	public enum ACCELERATION_MODE {
		AUTO, CUDA, OPENCL, VULKAN, HIP, METAL, SYCL, XPU, CANN, HEXAGON, OPENVINO, WEBGPU, MUSA, ZDNN, ZENDNN
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
