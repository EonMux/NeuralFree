package com.eonmux.neuralfree.gguf.testutil;

/** Kind of a queue operation recorded by a {@link FakeComputeBackend}. */
public enum FakeOpKind { DISPATCH, WRITE, READ, COPY, FILL, MARKER, FLUSH, FINISH, REPLAY, LAUNCH }
