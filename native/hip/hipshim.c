/*
 * JNI shim between the HIP runtime and the Java HIP backend
 * (com.eonmux.neuralfree.acceleration.hip.HipNative).
 *
 * Build with the ROCm toolchain:  make -C native/hip
 * (see the Makefile next to this file; needs ROCm >= 5.5 for hipDeviceGetUuid).
 *
 * Contract, shared with HipNative:
 *   - Every function that can fail returns the hipError_t status as jint; results
 *     travel through out-arrays. Nothing here throws or keeps JVM state.
 *   - Opaque handles (streams, events, modules, functions, device and host
 *     pointers) are jlong values.
 *   - launchKernel receives the kernel arguments as an Object[] snapshot, one
 *     element per declared parameter in order: a java.lang.Long holds a device
 *     pointer (8 bytes), Integer/Boolean a 4-byte int, Float a 4-byte float,
 *     Long a long, Double a double, and a primitive array is passed by
 *     reference for the duration of the launch. Anything else is
 *     hipErrorInvalidValue (the Java side validates first).
 *   - The device-attribute selector of deviceAttribute is this shim's own
 *     0-based table below, mirrored by the constants in HipNative.
 */
#include <jni.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Device-attribute selector table, mirrored by HipNative.ATTR_* constants. */
enum {
    ATTR_WARP_SIZE = 0,
    ATTR_MAX_THREADS_PER_BLOCK = 1,
    ATTR_MAX_GRID_DIM_X = 2,
    ATTR_MAX_GRID_DIM_Y = 3,
    ATTR_MAX_GRID_DIM_Z = 4,
    ATTR_MULTIPROCESSOR_COUNT = 5,
    ATTR_CLOCK_RATE = 6,
    ATTR_SHARED_PER_BLOCK = 7,
    ATTR_SHARED_PER_BLOCK_OPTIN = 8,
    ATTR_INTEGRATED = 9,
    ATTR_ASYNC_ENGINE_COUNT = 10
};

/* Kernel-function attribute, the hipFuncAttribute_t values the Java side names. */
#define SHIM_FUNC_MAX_THREADS_PER_BLOCK 0  /* hipFuncAttributeMaxThreadsPerBlock */
#define SHIM_FUNC_SHARED_SIZE_BYTES     1  /* hipFuncAttributeSharedSizeBytes    */

#define CLASS_NAME "com/eonmux/neuralfree/acceleration/hip/HipNative"

static jint status(hipError_t err) {
    return (jint) err;
}

static hipDeviceProp_t *deviceProperties(int device) {
    /* hipGetDeviceProperties allocates and caches per device inside the runtime. */
    hipDeviceProp_t *props = NULL;
    if (hipGetDeviceProperties(&props, device) != hipSuccess || props == NULL) {
        return NULL;
    }
    return props;
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_getDeviceCount(JNIEnv *env, jclass cls) {
    (void) env;
    (void) cls;
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess) {
        return -status(err);
    }
    return count;
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_setDevice(JNIEnv *env, jclass cls, jint device) {
    (void) env;
    (void) cls;
    return status(hipSetDevice((int) device));
}

JNIEXPORT jstring JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_getDeviceName(JNIEnv *env, jclass cls, jint device) {
    (void) cls;
    hipDeviceProp_t *props = deviceProperties((int) device);
    if (props == NULL) {
        return NULL;
    }
    props->name[sizeof(props->name) - 1] = '\0';
    return (*env)->NewStringUTF(env, props->name);
}

JNIEXPORT jstring JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_getGcnArchName(JNIEnv *env, jclass cls, jint device) {
    (void) cls;
    hipDeviceProp_t *props = deviceProperties((int) device);
    if (props == NULL) {
        return NULL;
    }
    props->gcnArchName[sizeof(props->gcnArchName) - 1] = '\0';
    return (*env)->NewStringUTF(env, props->gcnArchName);
}

JNIEXPORT jlong JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_deviceTotalMem(JNIEnv *env, jclass cls, jint device) {
    (void) env;
    (void) cls;
    size_t bytes = 0;
    if (hipDeviceTotalMem(&bytes, (int) device) != hipSuccess) {
        return -1;
    }
    return (jlong) bytes;
}

/*
 * Reads the device attribute the selector names into out[0]; the selector is the
 * shim's own table (see the enum above), not a hipDeviceAttribute_t value.
 */
JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_deviceAttribute(JNIEnv *env, jclass cls, jint device, jint selector, jintArray out) {
    (void) cls;
    hipDeviceProp_t *props = deviceProperties((int) device);
    if (props == NULL) {
        return (jint) hipErrorInvalidDevice;
    }
    jint value;
    switch (selector) {
        case ATTR_WARP_SIZE:
            value = (jint) props->warpSize;
            break;
        case ATTR_MAX_THREADS_PER_BLOCK:
            value = (jint) props->maxThreadsPerBlock;
            break;
        case ATTR_MAX_GRID_DIM_X:
            value = (jint) props->maxGridSize[0];
            break;
        case ATTR_MAX_GRID_DIM_Y:
            value = (jint) props->maxGridSize[1];
            break;
        case ATTR_MAX_GRID_DIM_Z:
            value = (jint) props->maxGridSize[2];
            break;
        case ATTR_MULTIPROCESSOR_COUNT:
            value = (jint) props->multiProcessorCount;
            break;
        case ATTR_CLOCK_RATE:
            value = (jint) props->clockRate;
            break;
        case ATTR_SHARED_PER_BLOCK:
            value = (jint) props->sharedMemPerBlock;
            break;
        case ATTR_SHARED_PER_BLOCK_OPTIN:
            value = (jint) props->sharedMemPerBlockOptin;
            break;
        case ATTR_INTEGRATED:
            value = (jint) props->integrated;
            break;
        case ATTR_ASYNC_ENGINE_COUNT:
            value = (jint) props->asyncEngineCount;
            break;
        default:
            return (jint) hipErrorInvalidValue;
    }
    (*env)->SetIntArrayRegion(env, out, 0, 1, &value);
    return (jint) hipSuccess;
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_getDriverVersion(JNIEnv *env, jclass cls, jintArray out) {
    (void) env;
    (void) cls;
    int version = 0;
    hipError_t err = hipDriverGetVersion(&version);
    if (err != hipSuccess) {
        return status(err);
    }
    jint v = (jint) version;
    (*env)->SetIntArrayRegion(env, out, 0, 1, &v);
    return (jint) hipSuccess;
}

/* Fills out (16 bytes) with the device UUID; hipErrorNotSupported when the driver has none. */
JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_getDeviceUuid(JNIEnv *env, jclass cls, jint device, jbyteArray out) {
    (void) cls;
    jsize len = (*env)->GetArrayLength(env, out);
    if (len < 16) {
        return (jint) hipErrorInvalidValue;
    }
    hipUUID uuid;
    memset(&uuid, 0, sizeof(uuid));
    hipError_t err = hipDeviceGetUuid(&uuid, (int) device);
    if (err != hipSuccess) {
        return status(err);
    }
    (*env)->SetByteArrayRegion(env, out, 0, 16, (const jbyte *) uuid.bytes);
    return (jint) hipSuccess;
}

/* Fills out (ASCII, NUL-terminated) with the PCI bus id. */
JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_getPciBusId(JNIEnv *env, jclass cls, jint device, jbyteArray out) {
    (void) cls;
    jsize len = (*env)->GetArrayLength(env, out);
    if (len < 1) {
        return (jint) hipErrorInvalidValue;
    }
    jbyte *buf = (*env)->GetByteArrayElements(env, out, NULL);
    if (buf == NULL) {
        return (jint) hipErrorOutOfMemory;
    }
    hipError_t err = hipDeviceGetPCIBusId((char *) buf, (int) len, (int) device);
    (*env)->ReleaseByteArrayElements(env, out, buf, 0);
    return status(err);
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_memAlloc(JNIEnv *env, jclass cls, jlong bytes, jlongArray out) {
    (void) env;
    (void) cls;
    void *ptr = NULL;
    hipError_t err = hipMalloc(&ptr, (size_t) bytes);
    if (err != hipSuccess) {
        return status(err);
    }
    jlong p = (jlong) (intptr_t) ptr;
    (*env)->SetLongArrayRegion(env, out, 0, 1, &p);
    return (jint) hipSuccess;
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_memFree(JNIEnv *env, jclass cls, jlong pointer) {
    (void) env;
    (void) cls;
    return status(hipFree((void *) (intptr_t) pointer));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_memGetInfo(JNIEnv *env, jclass cls, jlongArray freeOut, jlongArray totalOut) {
    (void) cls;
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    hipError_t err = hipMemGetInfo(&freeBytes, &totalBytes);
    if (err != hipSuccess) {
        return status(err);
    }
    jlong f = (jlong) freeBytes;
    jlong t = (jlong) totalBytes;
    (*env)->SetLongArrayRegion(env, freeOut, 0, 1, &f);
    (*env)->SetLongArrayRegion(env, totalOut, 0, 1, &t);
    return (jint) hipSuccess;
}

/* Page-locked host memory (portable); hipHostMalloc so transfers from it run on the async engines. */
JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_hostAlloc(JNIEnv *env, jclass cls, jlong bytes, jlongArray out) {
    (void) env;
    (void) cls;
    void *ptr = NULL;
    hipError_t err = hipHostMalloc(&ptr, (size_t) bytes, hipHostMallocPortable);
    if (err != hipSuccess) {
        return status(err);
    }
    jlong p = (jlong) (intptr_t) ptr;
    (*env)->SetLongArrayRegion(env, out, 0, 1, &p);
    return (jint) hipSuccess;
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_hostFree(JNIEnv *env, jclass cls, jlong pointer) {
    (void) env;
    (void) cls;
    return status(hipHostFree((void *) (intptr_t) pointer));
}

/* A direct ByteBuffer view of [pointer, pointer + bytes); NULL when the runtime refuses. */
JNIEXPORT jobject JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_pinnedView(JNIEnv *env, jclass cls, jlong pointer, jint bytes) {
    (void) cls;
    if (pointer == 0 || bytes <= 0) {
        return NULL;
    }
    return (*env)->NewDirectByteBuffer(env, (void *) (intptr_t) pointer, (jlong) bytes);
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_memcpyHtoDAsync(JNIEnv *env, jclass cls, jlong dst, jlong src, jlong bytes, jlong stream) {
    (void) env;
    (void) cls;
    return status(hipMemcpyAsync((void *) (intptr_t) dst, (const void *) (intptr_t) src,
                                 (size_t) bytes, hipMemcpyHostToDevice, (hipStream_t) (intptr_t) stream));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_memcpyDtoHAsync(JNIEnv *env, jclass cls, jlong dst, jlong src, jlong bytes, jlong stream) {
    (void) env;
    (void) cls;
    return status(hipMemcpyAsync((void *) (intptr_t) dst, (const void *) (intptr_t) src,
                                 (size_t) bytes, hipMemcpyDeviceToHost, (hipStream_t) (intptr_t) stream));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_memcpyDtoDAsync(JNIEnv *env, jclass cls, jlong dst, jlong src, jlong bytes, jlong stream) {
    (void) env;
    (void) cls;
    return status(hipMemcpyAsync((void *) (intptr_t) dst, (const void *) (intptr_t) src,
                                 (size_t) bytes, hipMemcpyDeviceToDevice, (hipStream_t) (intptr_t) stream));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_memsetD32Async(JNIEnv *env, jclass cls, jlong dst, jint value, jlong count, jlong stream) {
    (void) env;
    (void) cls;
    return status(hipMemsetD32Async((void *) (intptr_t) dst, (int) value, (size_t) count,
                                    (hipStream_t) (intptr_t) stream));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_streamCreate(JNIEnv *env, jclass cls, jlongArray out) {
    (void) env;
    (void) cls;
    hipStream_t stream = NULL;
    hipError_t err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (err != hipSuccess) {
        return status(err);
    }
    jlong s = (jlong) (intptr_t) stream;
    (*env)->SetLongArrayRegion(env, out, 0, 1, &s);
    return (jint) hipSuccess;
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_streamDestroy(JNIEnv *env, jclass cls, jlong stream) {
    (void) env;
    (void) cls;
    return status(hipStreamDestroy((hipStream_t) (intptr_t) stream));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_streamSynchronize(JNIEnv *env, jclass cls, jlong stream) {
    (void) env;
    (void) cls;
    return status(hipStreamSynchronize((hipStream_t) (intptr_t) stream));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_streamWaitEvent(JNIEnv *env, jclass cls, jlong stream, jlong event) {
    (void) env;
    (void) cls;
    return status(hipStreamWaitEvent((hipStream_t) (intptr_t) stream, (hipEvent_t) (intptr_t) event, 0));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_eventCreate(JNIEnv *env, jclass cls, jlongArray out) {
    (void) env;
    (void) cls;
    hipEvent_t event = NULL;
    hipError_t err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
    if (err != hipSuccess) {
        return status(err);
    }
    jlong e = (jlong) (intptr_t) event;
    (*env)->SetLongArrayRegion(env, out, 0, 1, &e);
    return (jint) hipSuccess;
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_eventDestroy(JNIEnv *env, jclass cls, jlong event) {
    (void) env;
    (void) cls;
    return status(hipEventDestroy((hipEvent_t) (intptr_t) event));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_eventRecord(JNIEnv *env, jclass cls, jlong event, jlong stream) {
    (void) env;
    (void) cls;
    return status(hipEventRecord((hipEvent_t) (intptr_t) event, (hipStream_t) (intptr_t) stream));
}

/* hipSuccess when the point is reached, hipErrorNotReady while it is not. */
JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_eventQuery(JNIEnv *env, jclass cls, jlong event) {
    (void) env;
    (void) cls;
    return status(hipEventQuery((hipEvent_t) (intptr_t) event));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_eventSynchronize(JNIEnv *env, jclass cls, jlong event) {
    (void) env;
    (void) cls;
    return status(hipEventSynchronize((hipEvent_t) (intptr_t) event));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_moduleLoadData(JNIEnv *env, jclass cls, jlongArray out, jbyteArray image) {
    (void) cls;
    jsize len = (*env)->GetArrayLength(env, image);
    if (len <= 0) {
        return (jint) hipErrorInvalidValue;
    }
    jbyte *bytes = (*env)->GetByteArrayElements(env, image, NULL);
    if (bytes == NULL) {
        return (jint) hipErrorOutOfMemory;
    }
    hipModule_t module = NULL;
    hipError_t err = hipModuleLoadData(&module, (const void *) bytes);
    (*env)->ReleaseByteArrayElements(env, image, bytes, JNI_ABORT);
    if (err != hipSuccess) {
        return status(err);
    }
    jlong m = (jlong) (intptr_t) module;
    (*env)->SetLongArrayRegion(env, out, 0, 1, &m);
    return (jint) hipSuccess;
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_moduleUnload(JNIEnv *env, jclass cls, jlong module) {
    (void) env;
    (void) cls;
    return status(hipModuleUnload((hipModule_t) (intptr_t) module));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_moduleGetFunction(JNIEnv *env, jclass cls, jlongArray out, jlong module, jstring name) {
    const char *symbol = (*env)->GetStringUTFChars(env, name, NULL);
    if (symbol == NULL) {
        return (jint) hipErrorOutOfMemory;
    }
    hipFunction_t function = NULL;
    hipError_t err = hipModuleGetFunction(&function, (hipModule_t) (intptr_t) module, symbol);
    (*env)->ReleaseStringUTFChars(env, name, symbol);
    if (err != hipSuccess) {
        return status(err);
    }
    jlong f = (jlong) (intptr_t) function;
    (*env)->SetLongArrayRegion(env, out, 0, 1, &f);
    return (jint) hipSuccess;
}

/* attribute is the hipFuncAttribute_t value (0 max threads per block, 1 static shared bytes). */
JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_funcGetAttribute(JNIEnv *env, jclass cls, jintArray out, jint attribute, jlong function) {
    (void) cls;
    int value = 0;
    hipError_t err = hipFuncGetAttribute(&value, (hipFuncAttribute_t) attribute, (hipFunction_t) (intptr_t) function);
    if (err != hipSuccess) {
        return status(err);
    }
    jint v = (jint) value;
    (*env)->SetIntArrayRegion(env, out, 0, 1, &v);
    return (jint) hipSuccess;
}

/* One parameter slot: the bytes one kernel parameter is read from, plus its critical array pin, if any. */
typedef struct {
    void *address;
    void *critical;
} ParamSlot;

/* Size of one parameter value in bytes, 0 when the type is not part of the contract. */
static jint paramSize(JNIEnv *env, jobject value) {
    if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "java/lang/Long"))) {
        return 8;
    }
    if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "java/lang/Integer"))
            || (*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "java/lang/Boolean"))) {
        return 4;
    }
    if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "java/lang/Float"))) {
        return 4;
    }
    if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "java/lang/Double"))) {
        return 8;
    }
    if ((*env)->GetArrayLength(env, (jarray) value) >= 0) {
        jarray array = (jarray) value;
        if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "[J"))) {
            return 8 * (jint) (*env)->GetArrayLength(env, array);
        }
        if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "[D"))) {
            return 8 * (jint) (*env)->GetArrayLength(env, array);
        }
        if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "[I"))
                || (*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "[F"))
                || (*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "[Z"))) {
            return 4 * (jint) (*env)->GetArrayLength(env, array);
        }
        if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "[S"))) {
            return 2 * (jint) (*env)->GetArrayLength(env, array);
        }
        if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "[B"))) {
            return (jint) (*env)->GetArrayLength(env, array);
        }
    }
    return 0;
}

/*
 * Fills one parameter slot. A boxed value is written into its own bytes of the slot
 * storage; a primitive array is pinned with GetPrimitiveArrayCritical and its base
 * address is the parameter, released right after hipModuleLaunchKernel returns.
 */
static jint fillParam(JNIEnv *env, jobject value, void *slot, ParamSlot *entry, jint size) {
    entry->critical = NULL;
    if (size == 0) {
        return (jint) hipErrorInvalidValue;
    }
    if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "java/lang/Long"))) {
        jlong v = (*env)->GetLongField(env, value, (*env)->GetFieldID(env, (*env)->FindClass(env, "java/lang/Long"), "value", "J"));
        memcpy(slot, &v, 8);
        entry->address = slot;
        return (jint) hipSuccess;
    }
    if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "java/lang/Integer"))) {
        jint v = (*env)->GetIntField(env, value, (*env)->GetFieldID(env, (*env)->FindClass(env, "java/lang/Integer"), "value", "I"));
        memcpy(slot, &v, 4);
        entry->address = slot;
        return (jint) hipSuccess;
    }
    if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "java/lang/Boolean"))) {
        jboolean v = (*env)->GetBooleanField(env, value, (*env)->GetFieldID(env, (*env)->FindClass(env, "java/lang/Boolean"), "value", "Z"));
        jint asInt = v ? 1 : 0;
        memcpy(slot, &asInt, 4);
        entry->address = slot;
        return (jint) hipSuccess;
    }
    if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "java/lang/Float"))) {
        jfloat v = (*env)->GetFloatField(env, value, (*env)->GetFieldID(env, (*env)->FindClass(env, "java/lang/Float"), "value", "F"));
        memcpy(slot, &v, 4);
        entry->address = slot;
        return (jint) hipSuccess;
    }
    if ((*env)->IsInstanceOf(env, value, (*env)->FindClass(env, "java/lang/Double"))) {
        jdouble v = (*env)->GetDoubleField(env, value, (*env)->GetFieldID(env, (*env)->FindClass(env, "java/lang/Double"), "value", "D"));
        memcpy(slot, &v, 8);
        entry->address = slot;
        return (jint) hipSuccess;
    }
    void *critical = (*env)->GetPrimitiveArrayCritical(env, (jarray) value, NULL);
    if (critical == NULL) {
        return (jint) hipErrorOutOfMemory;
    }
    entry->critical = critical;
    entry->address = critical;
    return (jint) hipSuccess;
}

/*
 * hipModuleLaunchKernel over the Java-side parameter snapshot. No parameter may be
 * null and every element must match the contract in the file comment.
 */
/* One parameter table: the addresses the runtime reads and the pinned arrays to release. */
typedef struct {
    jint count;
    jint *sizes;
    ParamSlot *entries;
    void **paramAddrs;
    void *slotStorage;
} KernelParams;

static void freeKernelParams(KernelParams *p) {
    free(p->sizes);
    free(p->entries);
    free(p->paramAddrs);
    free(p->slotStorage);
    p->sizes = NULL;
    p->entries = NULL;
    p->paramAddrs = NULL;
    p->slotStorage = NULL;
}

static void releaseKernelParams(JNIEnv *env, jobjectArray params, KernelParams *p);

/* Releases the pinned primitive arrays of the first count prepared slots. */
static void releasePinnedParams(JNIEnv *env, jobjectArray params, jint count, ParamSlot *entries) {
    if (entries == NULL) {
        return;
    }
    for (jint i = 0; i < count; i++) {
        if (entries[i].critical != NULL) {
            jobject value = (*env)->GetObjectArrayElement(env, params, i);
            if (value != NULL) {
                (*env)->ReleasePrimitiveArrayCritical(env, (jarray) value, entries[i].critical, JNI_ABORT);
                (*env)->DeleteLocalRef(env, value);
            }
            entries[i].critical = NULL;
        }
    }
}

/*
 * Builds the parameter table of params: one pass for the sizes, one for the fills, so the
 * slot storage is allocated once. Primitive arrays are pinned until releaseKernelParams.
 */
static jint prepareKernelParams(JNIEnv *env, jobjectArray params, KernelParams *p) {
    memset(p, 0, sizeof(*p));
    jsize count = (*env)->GetArrayLength(env, params);
    if (count < 0 || (jlong) count > 4096) {
        return (jint) hipErrorInvalidValue;
    }
    p->count = (jint) count;
    if (count == 0) {
        return (jint) hipSuccess;
    }
    p->sizes = (jint *) calloc((size_t) count, sizeof(jint));
    p->entries = (ParamSlot *) calloc((size_t) count, sizeof(ParamSlot));
    p->paramAddrs = (void **) calloc((size_t) count, sizeof(void *));
    if (p->sizes == NULL || p->entries == NULL || p->paramAddrs == NULL) {
        freeKernelParams(p);
        return (jint) hipErrorOutOfMemory;
    }
    jlong total = 0;
    for (jint i = 0; i < count; i++) {
        jobject value = (*env)->GetObjectArrayElement(env, params, i);
        if (value == NULL) {
            freeKernelParams(p);
            return (jint) hipErrorInvalidValue;
        }
        p->sizes[i] = paramSize(env, value);
        (*env)->DeleteLocalRef(env, value);
        if (p->sizes[i] <= 0) {
            freeKernelParams(p);
            return (jint) hipErrorInvalidValue;
        }
        total += p->sizes[i] + 8; /* 8 bytes of alignment slack per parameter */
    }
    p->slotStorage = malloc((size_t) total);
    if (p->slotStorage == NULL) {
        freeKernelParams(p);
        return (jint) hipErrorOutOfMemory;
    }
    intptr_t cursor = (intptr_t) p->slotStorage;
    for (jint i = 0; i < count; i++) {
        cursor = (cursor + 7) & ~(intptr_t) 7;
        jobject value = (*env)->GetObjectArrayElement(env, params, i);
        if (value == NULL) {
            /* entries of the parameters filled so far may hold pinned arrays */
            releaseKernelParams(env, params, p);
            return (jint) hipErrorInvalidValue;
        }
        jint fillFailure = fillParam(env, value, (void *) cursor, &p->entries[i], p->sizes[i]);
        (*env)->DeleteLocalRef(env, value);
        if (fillFailure != (jint) hipSuccess) {
            /* entries of the parameters filled so far may hold pinned arrays */
            releaseKernelParams(env, params, p);
            return fillFailure;
        }
        p->paramAddrs[i] = p->entries[i].address;
        cursor += p->sizes[i];
    }
    return (jint) hipSuccess;
}

/* Releases the pinned arrays of a prepared table and frees it. */
static void releaseKernelParams(JNIEnv *env, jobjectArray params, KernelParams *p) {
    releasePinnedParams(env, params, p->count, p->entries);
    freeKernelParams(p);
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_launchKernel(JNIEnv *env, jclass cls, jlong function,
                                                                                          jint gridDimX, jint gridDimY, jint gridDimZ,
                                                                                          jint blockDimX, jint blockDimY, jint blockDimZ,
                                                                                          jint sharedMemBytes, jlong stream, jobjectArray params) {
    (void) cls;
    KernelParams p;
    jint prep = prepareKernelParams(env, params, &p);
    if (prep != (jint) hipSuccess) {
        return prep;
    }
    hipError_t err;
    if (p.count == 0) {
        void *empty[1] = {NULL};
        err = hipModuleLaunchKernel((hipFunction_t) (intptr_t) function,
                                    (unsigned int) gridDimX, (unsigned int) gridDimY, (unsigned int) gridDimZ,
                                    (unsigned int) blockDimX, (unsigned int) blockDimY, (unsigned int) blockDimZ,
                                    (unsigned int) sharedMemBytes, (hipStream_t) (intptr_t) stream,
                                    (void **) empty, NULL);
    } else {
        err = hipModuleLaunchKernel((hipFunction_t) (intptr_t) function,
                                    (unsigned int) gridDimX, (unsigned int) gridDimY, (unsigned int) gridDimZ,
                                    (unsigned int) blockDimX, (unsigned int) blockDimY, (unsigned int) blockDimZ,
                                    (unsigned int) sharedMemBytes, (hipStream_t) (intptr_t) stream,
                                    p.paramAddrs, NULL);
    }
    releaseKernelParams(env, params, &p);
    return status(err);
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_streamBeginCapture(JNIEnv *env, jclass cls, jlong stream) {
    (void) env;
    (void) cls;
    return status(hipStreamBeginCapture((hipStream_t) (intptr_t) stream, hipStreamCaptureModeThreadLocal));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_streamEndCapture(JNIEnv *env, jclass cls, jlong stream, jlongArray out) {
    (void) cls;
    hipGraph_t graph = NULL;
    hipError_t err = hipStreamEndCapture((hipStream_t) (intptr_t) stream, &graph);
    if (err != hipSuccess) {
        return status(err);
    }
    jlong g = (jlong) (intptr_t) graph;
    (*env)->SetLongArrayRegion(env, out, 0, 1, &g);
    return (jint) hipSuccess;
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_graphInstantiate(JNIEnv *env, jclass cls, jlongArray out, jlong graph) {
    (void) cls;
    hipGraphExec_t exec = NULL;
    hipError_t err = hipGraphInstantiate(&exec, (hipGraph_t) (intptr_t) graph, NULL, NULL, 0);
    if (err != hipSuccess) {
        return status(err);
    }
    jlong e = (jlong) (intptr_t) exec;
    (*env)->SetLongArrayRegion(env, out, 0, 1, &e);
    return (jint) hipSuccess;
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_graphLaunch(JNIEnv *env, jclass cls, jlong exec, jlong stream) {
    (void) env;
    (void) cls;
    return status(hipGraphLaunch((hipGraphExec_t) (intptr_t) exec, (hipStream_t) (intptr_t) stream));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_graphDestroy(JNIEnv *env, jclass cls, jlong graph) {
    (void) env;
    (void) cls;
    return status(hipGraphDestroy((hipGraph_t) (intptr_t) graph));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_graphExecDestroy(JNIEnv *env, jclass cls, jlong exec) {
    (void) env;
    (void) cls;
    return status(hipGraphExecDestroy((hipGraphExec_t) (intptr_t) exec));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_graphNodeCount(JNIEnv *env, jclass cls, jlong graph, jintArray out) {
    (void) cls;
    size_t count = 0;
    hipError_t err = hipGraphGetNodes((hipGraph_t) (intptr_t) graph, NULL, &count);
    if (err != hipSuccess) {
        return status(err);
    }
    jint n = (jint) count;
    (*env)->SetIntArrayRegion(env, out, 0, 1, &n);
    return (jint) hipSuccess;
}

/*
 * Writes the nodes of the graph in dependency order. A stream capture produces a chain; the
 * order is recovered from the edges, because hipGraphGetNodes returns them in arbitrary order.
 */
JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_graphNodesInOrder(JNIEnv *env, jclass cls, jlong graphHandle, jlongArray out) {
    (void) cls;
    hipGraph_t graph = (hipGraph_t) (intptr_t) graphHandle;
    jsize count = (*env)->GetArrayLength(env, out);
    if (count <= 0) {
        return (jint) hipErrorInvalidValue;
    }
    hipGraphNode_t *nodes = (hipGraphNode_t *) calloc((size_t) count, sizeof(hipGraphNode_t));
    int *inDegree = (int *) calloc((size_t) count, sizeof(int));
    long *next = (long *) malloc((size_t) count * sizeof(long));
    jlong *order = (jlong *) malloc((size_t) count * sizeof(jlong));
    if (nodes == NULL || inDegree == NULL || next == NULL || order == NULL) {
        free(nodes); free(inDegree); free(next); free(order);
        return (jint) hipErrorOutOfMemory;
    }
    size_t found = (size_t) count;
    hipError_t err = hipGraphGetNodes(graph, nodes, &found);
    if (err != hipSuccess || found != (size_t) count) {
        free(nodes); free(inDegree); free(next); free(order);
        return err != hipSuccess ? status(err) : (jint) hipErrorInvalidValue;
    }
    for (jsize i = 0; i < count; i++) {
        next[i] = -1;
    }

    size_t edges = 0;
    err = hipGraphGetEdges(graph, NULL, NULL, &edges);
    if (err != hipSuccess) {
        free(nodes); free(inDegree); free(next); free(order);
        return status(err);
    }
    hipGraphNode_t *from = edges > 0 ? (hipGraphNode_t *) calloc(edges, sizeof(hipGraphNode_t)) : NULL;
    hipGraphNode_t *to = edges > 0 ? (hipGraphNode_t *) calloc(edges, sizeof(hipGraphNode_t)) : NULL;
    if (edges > 0 && (from == NULL || to == NULL)) {
        free(nodes); free(inDegree); free(next); free(order); free(from); free(to);
        return (jint) hipErrorOutOfMemory;
    }
    if (edges > 0) {
        err = hipGraphGetEdges(graph, from, to, &edges);
        if (err != hipSuccess) {
            free(nodes); free(inDegree); free(next); free(order); free(from); free(to);
            return status(err);
        }
        for (size_t e = 0; e < edges; e++) {
            jsize f = -1;
            jsize t = -1;
            for (jsize i = 0; i < count; i++) {
                if (nodes[i] == from[e]) f = i;
                if (nodes[i] == to[e]) t = i;
            }
            if (f < 0 || t < 0 || next[f] != -1) {
                /* a node outside the graph, a branch, or a duplicate edge: not one chain */
                free(nodes); free(inDegree); free(next); free(order); free(from); free(to);
                return (jint) hipErrorInvalidValue;
            }
            next[f] = (long) t;
            inDegree[t]++;
        }
    }
    free(from);
    free(to);

    jsize root = -1;
    for (jsize i = 0; i < count; i++) {
        if (inDegree[i] == 0) {
            if (root != -1) {
                free(nodes); free(inDegree); free(next); free(order);
                return (jint) hipErrorInvalidValue;
            }
            root = i;
        }
    }
    if (root < 0) {
        free(nodes); free(inDegree); free(next); free(order);
        return (jint) hipErrorInvalidValue;
    }
    long current = root;
    for (jsize i = 0; i < count; i++) {
        if (current < 0) {
            free(nodes); free(inDegree); free(next); free(order);
            return (jint) hipErrorInvalidValue;
        }
        order[i] = (jlong) (intptr_t) nodes[current];
        current = next[current];
    }
    if (current != -1) {
        free(nodes); free(inDegree); free(next); free(order);
        return (jint) hipErrorInvalidValue;
    }
    (*env)->SetLongArrayRegion(env, out, 0, count, order);
    free(nodes);
    free(inDegree);
    free(next);
    free(order);
    return (jint) hipSuccess;
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_graphNodeType(JNIEnv *env, jclass cls, jlong node, jintArray out) {
    (void) cls;
    hipGraphNodeType type;
    hipError_t err = hipGraphNodeGetType((hipGraphNode_t) (intptr_t) node, &type);
    if (err != hipSuccess) {
        return status(err);
    }
    jint value = (jint) type;
    (*env)->SetIntArrayRegion(env, out, 0, 1, &value);
    return (jint) hipSuccess;
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_graphExecKernelNodeSetParams(JNIEnv *env, jclass cls, jlong exec, jlong node,
                                                                                                         jlong function, jint gridDimX, jint gridDimY, jint gridDimZ,
                                                                                                         jint blockDimX, jint blockDimY, jint blockDimZ,
                                                                                                         jobjectArray params) {
    (void) cls;
    KernelParams p;
    jint prep = prepareKernelParams(env, params, &p);
    if (prep != (jint) hipSuccess) {
        return prep;
    }
    void *empty[1] = {NULL};
    hipKernelNodeParams nodeParams;
    memset(&nodeParams, 0, sizeof(nodeParams));
    nodeParams.func = (void *) (intptr_t) function;
    nodeParams.gridDim.x = (unsigned int) gridDimX;
    nodeParams.gridDim.y = (unsigned int) gridDimY;
    nodeParams.gridDim.z = (unsigned int) gridDimZ;
    nodeParams.blockDim.x = (unsigned int) blockDimX;
    nodeParams.blockDim.y = (unsigned int) blockDimY;
    nodeParams.blockDim.z = (unsigned int) blockDimZ;
    nodeParams.sharedMemBytes = 0;
    nodeParams.kernelParams = p.count == 0 ? (void **) empty : p.paramAddrs;
    nodeParams.extra = NULL;
    hipError_t err = hipGraphExecKernelNodeSetParams((hipGraphExec_t) (intptr_t) exec,
                                                     (hipGraphNode_t) (intptr_t) node, &nodeParams);
    releaseKernelParams(env, params, &p);
    return status(err);
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_graphExecMemcpyNodeSetParams(JNIEnv *env, jclass cls, jlong exec, jlong node,
                                                                                                          jlong dst, jlong src, jlong bytes) {
    (void) env;
    (void) cls;
    return status(hipGraphExecMemcpyNodeSetParams1D((hipGraphExec_t) (intptr_t) exec,
                                                    (hipGraphNode_t) (intptr_t) node,
                                                    (void *) (intptr_t) dst, (const void *) (intptr_t) src,
                                                    (size_t) bytes, hipMemcpyDeviceToDevice));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_graphExecMemsetNodeSetParams(JNIEnv *env, jclass cls, jlong exec, jlong node,
                                                                                                          jlong dst, jint value, jlong count) {
    (void) env;
    (void) cls;
    hipMemsetParams nodeParams;
    memset(&nodeParams, 0, sizeof(nodeParams));
    nodeParams.dst = (void *) (intptr_t) dst;
    nodeParams.elementSize = 4;
    nodeParams.height = 1;
    nodeParams.pitch = 0;
    nodeParams.value = (unsigned int) value;
    nodeParams.width = (size_t) count;
    return status(hipGraphExecMemsetNodeSetParams((hipGraphExec_t) (intptr_t) exec,
                                                  (hipGraphNode_t) (intptr_t) node, &nodeParams));
}

JNIEXPORT jint JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_synchronize(JNIEnv *env, jclass cls) {
    (void) env;
    (void) cls;
    return status(hipDeviceSynchronize());
}

/* The symbolic name of a hipError_t status, as hipGetErrorString reports it. */
JNIEXPORT jstring JNICALL Java_com_eonmux_neuralfree_acceleration_hip_HipNative_errorString(JNIEnv *env, jclass cls, jint code) {
    (void) cls;
    const char *name = hipGetErrorString((hipError_t) code);
    return (*env)->NewStringUTF(env, name == NULL ? "unknown HIP error" : name);
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    (void) vm;
    (void) reserved;
    return JNI_VERSION_1_8;
}
