# Builds the JNI shim of the HIP backend (com.eonmux.neuralfree.acceleration.hip.HipNative).
# Needs the ROCm toolchain (hipcc) and a JDK with the JNI headers; ROCm >= 5.5 for
# hipDeviceGetUuid. The result, libhipshim.so, is loaded by HipNative through
# System.loadLibrary("hipshim") or the absolute path -Dneuralfree.hip.shim.path.
#
#   make                build libhipshim.so
#   make JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

HIPCC ?= hipcc
JAVA_HOME ?= $(shell dirname $$(dirname $$(readlink -f $$(command -v java))))
JNI_INCLUDES := -I"$(JAVA_HOME)/include" -I"$(JAVA_HOME)/include/linux"

TARGET := libhipshim.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): hipshim.c
	$(HIPCC) -O2 -fPIC -shared -Wall -o $@ hipshim.c $(JNI_INCLUDES)

clean:
	rm -f $(TARGET)
