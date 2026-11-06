# Builds the JNI shim of the Metal backend
# (com.eonmux.neuralfree.acceleration.metal.MetalNative).
# Needs macOS with the Metal and Foundation frameworks and a JDK with the JNI headers.
# The result, libmetalshim.dylib, is loaded by MetalNative through
# System.loadLibrary("metalshim") or the absolute path -Dneuralfree.metal.shim.path.
#
#   make
#   make JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home

CLANG ?= clang
JAVA_HOME ?= $(shell /usr/libexec/java_home 2>/dev/null)
JNI_INCLUDES := -I"$(JAVA_HOME)/include" -I"$(JAVA_HOME)/include/darwin"

TARGET := libmetalshim.dylib

.PHONY: all clean check

all: $(TARGET)

$(TARGET): metalshim.m
	$(CLANG) -O2 -fPIC -dynamiclib -fno-objc-arc -Wall -Wextra -Wno-unused-parameter \
		-framework Metal -framework Foundation -o $@ $< $(JNI_INCLUDES)

check:
	@test -n "$(JAVA_HOME)" || (echo "JAVA_HOME not found; set it explicitly" >&2; exit 1)

clean:
	rm -f $(TARGET)
