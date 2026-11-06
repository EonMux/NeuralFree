/*
 * JNI shim between Apple's Metal framework and the Java Metal backend
 * (com.eonmux.neuralfree.acceleration.metal.MetalNative).
 *
 * Build with clang on macOS:  make -C native/metal
 * (see the Makefile next to this file).
 *
 * Contract, shared with MetalNative:
 *   - Metal objects (buffers, queues, libraries, functions, pipelines, shared events,
 *     command buffers and compute encoders) are jlong handles. Every handle this file
 *     returns is retained and released by the Java side through releaseObject (or
 *     endEncoding for an encoder). Handles are 0 only when a query has no answer.
 *   - Operations that create an object throw java.lang.IllegalStateException, carrying the
 *     Metal error's description, instead of returning 0.
 *   - Device queries use the shim's own attribute table (ATTR_* below), mirrored by the
 *     constants in MetalNative.
 *   - Host data crosses the boundary as arrays or direct buffers; nothing here keeps a
 *     pointer into JVM memory after a call returns.
 *
 * The file is compiled without ARC so the retain/release pairs that hand ownership to Java
 * are explicit.
 */
#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <TargetConditionals.h>

/* Device-attribute selector table, mirrored by MetalNative.ATTR_* constants. */
enum {
    ATTR_MAX_THREADS_PER_THREADGROUP = 0,
    ATTR_MAX_THREADGROUP_MEMORY = 1,
    ATTR_RECOMMENDED_MAX_WORKING_SET = 2,
    ATTR_HAS_UNIFIED_MEMORY = 3,
    ATTR_IS_LOW_POWER = 4,
    ATTR_IS_REMOVABLE = 5,
    ATTR_MAX_BUFFER_LENGTH = 6,
    ATTR_REGISTRY_ID = 7,
    ATTR_ARGUMENT_BUFFERS_TIER = 8
};

static pthread_mutex_t gDeviceLock = PTHREAD_MUTEX_INITIALIZER;
static NSArray *gDevices = nil;

/* All Metal devices of the machine, retained once per process. */
static NSArray *metalDevices(void) {
    pthread_mutex_lock(&gDeviceLock);
    if (gDevices == nil) {
        NSArray *devices = nil;
#if TARGET_OS_OSX
        if (@available(macOS 10.13, *)) {
            devices = MTLCopyAllDevices();
        }
#endif
        if (devices == nil) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            devices = device != nil
                    ? [[NSArray alloc] initWithObjects:device, nil]
                    : [[NSArray alloc] init];
            [device release];
        }
        gDevices = devices;
    }
    NSArray *result = gDevices;
    pthread_mutex_unlock(&gDeviceLock);
    return result;
}

static id<MTLDevice> deviceAt(jint index) {
    NSArray *devices = metalDevices();
    if (index < 0 || (NSUInteger) index >= [devices count]) {
        return nil;
    }
    return [devices objectAtIndex:(NSUInteger) index];
}

static void throwMetal(JNIEnv *env, NSString *message) {
    const char *text = [message UTF8String];
    jclass cls = (*env)->FindClass(env, "java/lang/IllegalStateException");
    if (cls != NULL) {
        (*env)->ThrowNew(env, cls, text == NULL ? "Metal operation failed" : text);
        (*env)->DeleteLocalRef(env, cls);
    }
}

static NSString *stringOf(JNIEnv *env, jstring value) {
    if (value == NULL) {
        return nil;
    }
    const char *chars = (*env)->GetStringUTFChars(env, value, NULL);
    NSString *text = chars == NULL ? nil : [NSString stringWithUTF8String:chars];
    if (chars != NULL) {
        (*env)->ReleaseStringUTFChars(env, value, chars);
    }
    return text;
}

static jstring stringToJava(JNIEnv *env, NSString *value) {
    if (value == nil) {
        return NULL;
    }
    const char *text = [value UTF8String];
    return text == NULL ? NULL : (*env)->NewStringUTF(env, text);
}

static inline id<MTLBuffer> bufferOf(jlong handle) {
    return (id<MTLBuffer>) (intptr_t) handle;
}

static inline id<MTLCommandQueue> queueOf(jlong handle) {
    return (id<MTLCommandQueue>) (intptr_t) handle;
}

static inline id<MTLCommandBuffer> commandBufferOf(jlong handle) {
    return (id<MTLCommandBuffer>) (intptr_t) handle;
}

static inline id<MTLComputeCommandEncoder> computeEncoderOf(jlong handle) {
    return (id<MTLComputeCommandEncoder>) (intptr_t) handle;
}

static inline id<MTLLibrary> libraryOf(jlong handle) {
    return (id<MTLLibrary>) (intptr_t) handle;
}

static inline id<MTLFunction> functionOf(jlong handle) {
    return (id<MTLFunction>) (intptr_t) handle;
}

static inline id<MTLComputePipelineState> pipelineOf(jlong handle) {
    return (id<MTLComputePipelineState>) (intptr_t) handle;
}

static inline id<MTLSharedEvent> eventOf(jlong handle) {
    return (id<MTLSharedEvent>) (intptr_t) handle;
}

/* ---------------------------------------------------------------- device enumeration */

JNIEXPORT jint JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_getDeviceCount(JNIEnv *env, jclass cls) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        return (jint) [metalDevices() count];
    }
}

JNIEXPORT jstring JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_getDeviceName(JNIEnv *env, jclass cls,
                                                                         jint index) {
    (void) cls;
    @autoreleasepool {
        id<MTLDevice> device = deviceAt(index);
        return device == nil ? NULL : stringToJava(env, [device name]);
    }
}

JNIEXPORT jstring JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_getDeviceFamily(JNIEnv *env, jclass cls,
                                                                           jint index) {
    (void) cls;
    @autoreleasepool {
        id<MTLDevice> device = deviceAt(index);
        if (device == nil) {
            return NULL;
        }
        NSMutableArray *families = [NSMutableArray array];
        if ([device respondsToSelector:@selector(supportsFamily:)]) {
            struct {
                MTLGPUFamily family;
                const char *name;
            } known[] = {
                    {MTLGPUFamilyApple1, "apple1"}, {MTLGPUFamilyApple2, "apple2"},
                    {MTLGPUFamilyApple3, "apple3"}, {MTLGPUFamilyApple4, "apple4"},
                    {MTLGPUFamilyApple5, "apple5"}, {MTLGPUFamilyApple6, "apple6"},
                    {MTLGPUFamilyApple7, "apple7"}, {MTLGPUFamilyMac2, "mac2"},
                    {MTLGPUFamilyCommon3, "common3"},
            };
            for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); i++) {
                if ([device supportsFamily:known[i].family]) {
                    [families addObject:[NSString stringWithUTF8String:known[i].name]];
                }
            }
        }
        return stringToJava(env, [families componentsJoinedByString:@","]);
    }
}

JNIEXPORT jint JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_deviceAttribute(JNIEnv *env, jclass cls,
                                                                           jint index, jint selector,
                                                                           jlongArray out) {
    (void) cls;
    @autoreleasepool {
        id<MTLDevice> device = deviceAt(index);
        if (device == nil) {
            return -1;
        }
        if (out == NULL || (*env)->GetArrayLength(env, out) < 1) {
            return -2;
        }
        jlong value = 0;
        switch (selector) {
            case ATTR_MAX_THREADS_PER_THREADGROUP: {
                MTLSize size = [device maxThreadsPerThreadgroup];
                value = (jlong) (size.width * size.height * size.depth);
                break;
            }
            case ATTR_MAX_THREADGROUP_MEMORY:
                value = (jlong) [device maxThreadgroupMemoryLength];
                break;
            case ATTR_RECOMMENDED_MAX_WORKING_SET:
                if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
                    value = (jlong) [device recommendedMaxWorkingSetSize];
                }
                break;
            case ATTR_HAS_UNIFIED_MEMORY:
                if ([device respondsToSelector:@selector(hasUnifiedMemory)]) {
                    value = [device hasUnifiedMemory] ? 1 : 0;
                }
                break;
            case ATTR_IS_LOW_POWER:
                value = [device isLowPower] ? 1 : 0;
                break;
            case ATTR_IS_REMOVABLE:
                if ([device respondsToSelector:@selector(isRemovable)]) {
                    value = [device isRemovable] ? 1 : 0;
                }
                break;
            case ATTR_MAX_BUFFER_LENGTH:
                if ([device respondsToSelector:@selector(maxBufferLength)]) {
                    value = (jlong) [device maxBufferLength];
                }
                break;
            case ATTR_REGISTRY_ID:
                value = (jlong) [device registryID];
                break;
            case ATTR_ARGUMENT_BUFFERS_TIER:
                if ([device respondsToSelector:@selector(argumentBuffersSupport)]) {
                    value = (jlong) [device argumentBuffersSupport];
                }
                break;
            default:
                return -3;
        }
        (*env)->SetLongArrayRegion(env, out, 0, 1, &value);
        return 0;
    }
}

JNIEXPORT jstring JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_driverVersion(JNIEnv *env, jclass cls) {
    (void) cls;
    @autoreleasepool {
        return stringToJava(env, [[NSProcessInfo processInfo] operatingSystemVersionString]);
    }
}

/* ---------------------------------------------------------------- memory and objects */

JNIEXPORT jlong JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_newBuffer(JNIEnv *env, jclass cls,
                                                                     jint index, jlong length) {
    (void) cls;
    @autoreleasepool {
        id<MTLDevice> device = deviceAt(index);
        if (device == nil) {
            throwMetal(env, @"no Metal device to allocate on");
            return 0;
        }
        if (length <= 0) {
            throwMetal(env, @"buffer length must be positive");
            return 0;
        }
        id<MTLBuffer> buffer = [device newBufferWithLength:(NSUInteger) length
                                                   options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throwMetal(env, [NSString stringWithFormat:@"Metal refused a %lld-byte buffer", (long long) length]);
            return 0;
        }
        return (jlong) (intptr_t) buffer;
    }
}

JNIEXPORT jobject JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_bufferHostView(JNIEnv *env, jclass cls,
                                                                          jlong handle, jlong length) {
    (void) cls;
    @autoreleasepool {
        id<MTLBuffer> buffer = bufferOf(handle);
        if (buffer == nil) {
            return NULL;
        }
        void *contents = [buffer contents];
        if (contents == NULL || length < 0) {
            return NULL;
        }
        return (*env)->NewDirectByteBuffer(env, contents, length);
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_retainObject(JNIEnv *env, jclass cls,
                                                                        jlong handle) {
    (void) env;
    (void) cls;
    if (handle != 0) {
        [(id) (intptr_t) handle retain];
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_releaseObject(JNIEnv *env, jclass cls,
                                                                         jlong handle) {
    (void) env;
    (void) cls;
    if (handle != 0) {
        [(id) (intptr_t) handle release];
    }
}

/* ---------------------------------------------------------------- queues and pipelines */

JNIEXPORT jlong JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_newCommandQueue(JNIEnv *env, jclass cls,
                                                                           jint index) {
    (void) cls;
    @autoreleasepool {
        id<MTLDevice> device = deviceAt(index);
        if (device == nil) {
            throwMetal(env, @"no Metal device for the command queue");
            return 0;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            throwMetal(env, @"Metal refused to create a command queue");
            return 0;
        }
        return (jlong) (intptr_t) queue;
    }
}

JNIEXPORT jlong JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_newLibrary(JNIEnv *env, jclass cls,
                                                                      jint index, jstring source,
                                                                      jstring options) {
    (void) cls;
    @autoreleasepool {
        id<MTLDevice> device = deviceAt(index);
        if (device == nil) {
            throwMetal(env, @"no Metal device to compile on");
            return 0;
        }
        NSString *text = stringOf(env, source);
        NSString *optionText = stringOf(env, options);
        if (text == nil) {
            throwMetal(env, @"Metal source is missing");
            return 0;
        }
        MTLCompileOptions *compile = [[MTLCompileOptions alloc] init];
        compile.fastMathEnabled = NO;
        NSMutableDictionary *macros = [NSMutableDictionary dictionary];
        for (NSString *token in [optionText componentsSeparatedByString:@" "]) {
            NSString *trimmed = [token stringByTrimmingCharactersInSet:
                    [NSCharacterSet whitespaceAndNewlineCharacterSet]];
            if ([trimmed length] == 0) {
                continue;
            }
            if (![trimmed hasPrefix:@"-D"] || [trimmed length] == 2) {
                throwMetal(env, [NSString stringWithFormat:@"unsupported Metal compile option '%@'", trimmed]);
                [compile release];
                return 0;
            }
            NSString *definition = [trimmed substringFromIndex:2];
            NSRange equals = [definition rangeOfString:@"="];
            NSString *name = equals.location == NSNotFound ? definition : [definition substringToIndex:equals.location];
            NSString *value = equals.location == NSNotFound ? @"1" : [definition substringFromIndex:equals.location + 1];
            [macros setObject:value forKey:name];
        }
        if ([macros count] > 0) {
            compile.preprocessorMacros = macros;
        }
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:text options:compile error:&error];
        [compile release];
        if (library == nil) {
            NSString *message = error == nil ? @"Metal source did not compile"
                                             : [error localizedDescription];
            throwMetal(env, [NSString stringWithFormat:@"Metal source did not compile: %@", message]);
            return 0;
        }
        return (jlong) (intptr_t) library;
    }
}

JNIEXPORT jlong JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_newFunction(JNIEnv *env, jclass cls,
                                                                       jlong libraryHandle, jstring name) {
    (void) cls;
    @autoreleasepool {
        id<MTLLibrary> library = libraryOf(libraryHandle);
        NSString *functionName = stringOf(env, name);
        if (library == nil || functionName == nil) {
            throwMetal(env, @"Metal library or function name is missing");
            return 0;
        }
        id<MTLFunction> function = [library newFunctionWithName:functionName];
        if (function == nil) {
            throwMetal(env, [NSString stringWithFormat:@"Metal library has no function '%@'", functionName]);
            return 0;
        }
        return (jlong) (intptr_t) function;
    }
}

JNIEXPORT jlong JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_newComputePipeline(JNIEnv *env, jclass cls,
                                                                              jint index,
                                                                              jlong functionHandle) {
    (void) cls;
    @autoreleasepool {
        id<MTLDevice> device = deviceAt(index);
        id<MTLFunction> function = functionOf(functionHandle);
        if (device == nil || function == nil) {
            throwMetal(env, @"no Metal device or function for the pipeline");
            return 0;
        }
        NSError *error = nil;
        id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            NSString *message = error == nil ? @"unknown error" : [error localizedDescription];
            throwMetal(env, [NSString stringWithFormat:@"Metal compute pipeline did not build: %@", message]);
            return 0;
        }
        return (jlong) (intptr_t) pipeline;
    }
}

JNIEXPORT jlong JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_newComputePipelineWithConstants(
        JNIEnv *env, jclass cls, jint index, jlong libraryHandle, jstring name,
        jintArray specIds, jintArray specKinds, jlongArray specValues) {
    (void) cls;
    @autoreleasepool {
        id<MTLDevice> device = deviceAt(index);
        id<MTLLibrary> library = libraryOf(libraryHandle);
        NSString *functionName = stringOf(env, name);
        if (device == nil || library == nil || functionName == nil) {
            throwMetal(env, @"no Metal device, library or function name for the pipeline");
            return 0;
        }
        jsize count = specIds == NULL ? 0 : (*env)->GetArrayLength(env, specIds);
        if ((specIds == NULL && count > 0) || specKinds == NULL || specValues == NULL
                || (*env)->GetArrayLength(env, specKinds) != count
                || (*env)->GetArrayLength(env, specValues) != count) {
            throwMetal(env, @"function-constant ids, kinds and values have different lengths");
            return 0;
        }
        MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
        if (count > 0) {
            jint *ids = (*env)->GetIntArrayElements(env, specIds, NULL);
            jint *kinds = (*env)->GetIntArrayElements(env, specKinds, NULL);
            jlong *values = (*env)->GetLongArrayElements(env, specValues, NULL);
            if (ids == NULL || kinds == NULL || values == NULL) {
                if (ids != NULL) (*env)->ReleaseIntArrayElements(env, specIds, ids, JNI_ABORT);
                if (kinds != NULL) (*env)->ReleaseIntArrayElements(env, specKinds, kinds, JNI_ABORT);
                if (values != NULL) (*env)->ReleaseLongArrayElements(env, specValues, values, JNI_ABORT);
                [constants release];
                throwMetal(env, @"out of memory reading the function constants");
                return 0;
            }
            for (jsize i = 0; i < count; i++) {
                uint32_t bits = (uint32_t) values[i];
                NSUInteger constantIndex = (NSUInteger) ids[i];
                switch (kinds[i]) {
                    case 0: {
                        uint32_t v = bits;
                        [constants setConstantValue:&v type:MTLDataTypeUInt atIndex:constantIndex];
                        break;
                    }
                    case 1: {
                        int32_t v = (int32_t) bits;
                        [constants setConstantValue:&v type:MTLDataTypeInt atIndex:constantIndex];
                        break;
                    }
                    case 2: {
                        float v;
                        memcpy(&v, &bits, sizeof(v));
                        [constants setConstantValue:&v type:MTLDataTypeFloat atIndex:constantIndex];
                        break;
                    }
                    default: {
                        (*env)->ReleaseIntArrayElements(env, specIds, ids, JNI_ABORT);
                        (*env)->ReleaseIntArrayElements(env, specKinds, kinds, JNI_ABORT);
                        (*env)->ReleaseLongArrayElements(env, specValues, values, JNI_ABORT);
                        [constants release];
                        throwMetal(env, [NSString stringWithFormat:@"unknown function-constant kind %d",
                                (int) kinds[i]]);
                        return 0;
                    }
                }
            }
            (*env)->ReleaseIntArrayElements(env, specIds, ids, JNI_ABORT);
            (*env)->ReleaseIntArrayElements(env, specKinds, kinds, JNI_ABORT);
            (*env)->ReleaseLongArrayElements(env, specValues, values, JNI_ABORT);
        }
        NSError *error = nil;
        id<MTLFunction> function = [library newFunctionWithName:functionName
                                               constantValues:constants
                                                        error:&error];
        [constants release];
        if (function == nil) {
            NSString *message = error == nil ? @"unknown error" : [error localizedDescription];
            throwMetal(env, [NSString stringWithFormat:@"Metal library has no function '%@' with those"
                    " function constants: %@", functionName, message]);
            return 0;
        }
        id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        if (pipeline == nil) {
            NSString *message = error == nil ? @"unknown error" : [error localizedDescription];
            throwMetal(env, [NSString stringWithFormat:@"Metal compute pipeline did not build: %@", message]);
            return 0;
        }
        return (jlong) (intptr_t) pipeline;
    }
}

JNIEXPORT jint JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_pipelineMaxThreads(JNIEnv *env, jclass cls,
                                                                              jlong handle) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = pipelineOf(handle);
        return pipeline == nil ? 0 : (jint) [pipeline maxTotalThreadsPerThreadgroup];
    }
}

/* ---------------------------------------------------------------- synchronisation */

JNIEXPORT jlong JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_newSharedEvent(JNIEnv *env, jclass cls,
                                                                          jint index) {
    (void) cls;
    @autoreleasepool {
        id<MTLDevice> device = deviceAt(index);
        if (device == nil) {
            throwMetal(env, @"no Metal device for the shared event");
            return 0;
        }
        id<MTLSharedEvent> event = [device newSharedEvent];
        if (event == nil) {
            throwMetal(env, @"Metal refused to create a shared event");
            return 0;
        }
        return (jlong) (intptr_t) event;
    }
}

JNIEXPORT jlong JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_eventSignaledValue(JNIEnv *env, jclass cls,
                                                                              jlong handle) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        id<MTLSharedEvent> event = eventOf(handle);
        return event == nil ? 0 : (jlong) [event signaledValue];
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_eventWait(JNIEnv *env, jclass cls,
                                                                     jlong handle, jlong value) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        id<MTLSharedEvent> event = eventOf(handle);
        if (event != nil) {
            [event waitUntilSignaledValue:(uint64_t) value timeoutMS:UINT64_MAX];
        }
    }
}

/* ---------------------------------------------------------------- command encoding */

JNIEXPORT jlong JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_newCommandBuffer(JNIEnv *env, jclass cls,
                                                                            jlong queueHandle) {
    (void) cls;
    @autoreleasepool {
        id<MTLCommandQueue> queue = queueOf(queueHandle);
        if (queue == nil) {
            throwMetal(env, @"no Metal command queue");
            return 0;
        }
        id<MTLCommandBuffer> buffer = [[queue commandBuffer] retain];
        if (buffer == nil) {
            throwMetal(env, @"Metal refused to create a command buffer");
            return 0;
        }
        return (jlong) (intptr_t) buffer;
    }
}

JNIEXPORT jlong JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_computeEncoder(JNIEnv *env, jclass cls,
                                                                          jlong commandBufferHandle) {
    (void) cls;
    @autoreleasepool {
        id<MTLCommandBuffer> buffer = commandBufferOf(commandBufferHandle);
        if (buffer == nil) {
            throwMetal(env, @"no Metal command buffer");
            return 0;
        }
        id<MTLComputeCommandEncoder> encoder = [[buffer computeCommandEncoder] retain];
        if (encoder == nil) {
            throwMetal(env, @"Metal refused to create a compute encoder");
            return 0;
        }
        return (jlong) (intptr_t) encoder;
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_endEncoding(JNIEnv *env, jclass cls,
                                                                       jlong encoderHandle) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = computeEncoderOf(encoderHandle);
        if (encoder != nil) {
            [encoder endEncoding];
            [encoder release];
        }
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_setPipeline(JNIEnv *env, jclass cls,
                                                                       jlong encoderHandle,
                                                                       jlong pipelineHandle) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = computeEncoderOf(encoderHandle);
        id<MTLComputePipelineState> pipeline = pipelineOf(pipelineHandle);
        if (encoder != nil && pipeline != nil) {
            [encoder setComputePipelineState:pipeline];
        }
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_setBuffer(JNIEnv *env, jclass cls,
                                                                     jlong encoderHandle,
                                                                     jlong bufferHandle, jlong offset,
                                                                     jint index) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = computeEncoderOf(encoderHandle);
        id<MTLBuffer> buffer = bufferOf(bufferHandle);
        if (encoder != nil && buffer != nil) {
            [encoder setBuffer:buffer offset:(NSUInteger) offset atIndex:(NSUInteger) index];
        }
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_setBytes(JNIEnv *env, jclass cls,
                                                                    jlong encoderHandle, jbyteArray data,
                                                                    jint length, jint index) {
    (void) cls;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = computeEncoderOf(encoderHandle);
        if (encoder == nil || data == NULL || length < 0) {
            return;
        }
        if ((*env)->GetArrayLength(env, data) < length) {
            throwMetal(env, @"scalar argument is shorter than its declared length");
            return;
        }
        jbyte *bytes = (*env)->GetByteArrayElements(env, data, NULL);
        if (bytes == NULL) {
            return;
        }
        [encoder setBytes:(const void *) bytes length:(NSUInteger) length atIndex:(NSUInteger) index];
        (*env)->ReleaseByteArrayElements(env, data, bytes, JNI_ABORT);
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_dispatchThreadgroups(JNIEnv *env, jclass cls,
                                                                                jlong encoderHandle,
                                                                                jlong groupsX, jlong groupsY,
                                                                                jlong groupsZ, jlong threadsX,
                                                                                jlong threadsY, jlong threadsZ) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = computeEncoderOf(encoderHandle);
        if (encoder != nil) {
            [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger) groupsX, (NSUInteger) groupsY,
                    (NSUInteger) groupsZ)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger) threadsX, (NSUInteger) threadsY,
                            (NSUInteger) threadsZ)];
        }
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_encodeWait(JNIEnv *env, jclass cls,
                                                                      jlong commandBufferHandle,
                                                                      jlong eventHandle, jlong value) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        id<MTLCommandBuffer> buffer = commandBufferOf(commandBufferHandle);
        id<MTLSharedEvent> event = eventOf(eventHandle);
        if (buffer != nil && event != nil) {
            [buffer encodeWaitForEvent:event value:(uint64_t) value];
        }
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_encodeSignal(JNIEnv *env, jclass cls,
                                                                        jlong commandBufferHandle,
                                                                        jlong eventHandle, jlong value) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        id<MTLCommandBuffer> buffer = commandBufferOf(commandBufferHandle);
        id<MTLSharedEvent> event = eventOf(eventHandle);
        if (buffer != nil && event != nil) {
            [buffer encodeSignalEvent:event value:(uint64_t) value];
        }
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_copyBuffer(JNIEnv *env, jclass cls,
                                                                      jlong commandBufferHandle,
                                                                      jlong srcHandle, jlong srcOffset,
                                                                      jlong dstHandle, jlong dstOffset,
                                                                      jlong size) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        id<MTLCommandBuffer> buffer = commandBufferOf(commandBufferHandle);
        id<MTLBuffer> src = bufferOf(srcHandle);
        id<MTLBuffer> dst = bufferOf(dstHandle);
        if (buffer == nil || src == nil || dst == nil) {
            return;
        }
        id<MTLBlitCommandEncoder> blit = [buffer blitCommandEncoder];
        if (blit == nil) {
            throwMetal(env, @"Metal refused to create a blit encoder");
            return;
        }
        [blit copyFromBuffer:src sourceOffset:(NSUInteger) srcOffset
                    toBuffer:dst destinationOffset:(NSUInteger) dstOffset
                        size:(NSUInteger) size];
        [blit endEncoding];
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_fillBuffer(JNIEnv *env, jclass cls,
                                                                      jlong commandBufferHandle,
                                                                      jlong bufferHandle, jlong offset,
                                                                      jlong size, jint value) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        id<MTLCommandBuffer> buffer = commandBufferOf(commandBufferHandle);
        id<MTLBuffer> target = bufferOf(bufferHandle);
        if (buffer == nil || target == nil) {
            return;
        }
        id<MTLBlitCommandEncoder> blit = [buffer blitCommandEncoder];
        if (blit == nil) {
            throwMetal(env, @"Metal refused to create a blit encoder");
            return;
        }
        [blit fillBuffer:target range:NSMakeRange((NSUInteger) offset, (NSUInteger) size)
                   value:(uint8_t) (value & 0xFF)];
        [blit endEncoding];
    }
}

JNIEXPORT void JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_commit(JNIEnv *env, jclass cls,
                                                                  jlong commandBufferHandle) {
    (void) env;
    (void) cls;
    @autoreleasepool {
        id<MTLCommandBuffer> buffer = commandBufferOf(commandBufferHandle);
        if (buffer != nil) {
            [buffer commit];
        }
    }
}

JNIEXPORT jstring JNICALL
Java_com_eonmux_neuralfree_acceleration_metal_MetalNative_commandBufferError(JNIEnv *env, jclass cls,
                                                                              jlong commandBufferHandle) {
    (void) cls;
    @autoreleasepool {
        id<MTLCommandBuffer> buffer = commandBufferOf(commandBufferHandle);
        if (buffer == nil || [buffer status] != MTLCommandBufferStatusError) {
            return NULL;
        }
        NSError *error = [buffer error];
        return stringToJava(env, error == nil ? @"unknown Metal command buffer error"
                                              : [error localizedDescription]);
    }
}
