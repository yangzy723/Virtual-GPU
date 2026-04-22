#!/usr/bin/env python3
import ctypes
import sys

import torch

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())

torch.cuda.init()
print("cuda initialized")

# Handle type in cuBLAS is opaque pointer.
Handle = ctypes.c_void_p
h = Handle()

cublas = ctypes.CDLL("libcublas.so.12")
fn = cublas.cublasCreate_v2
fn.argtypes = [ctypes.POINTER(Handle)]
fn.restype = ctypes.c_int

rc = fn(ctypes.byref(h))
print("cublasCreate_v2 rc=", rc, "handle=", hex(h.value) if h.value else None)
if rc != 0:
    sys.exit(1)

cublasDestroy = cublas.cublasDestroy_v2
cublasDestroy.argtypes = [Handle]
cublasDestroy.restype = ctypes.c_int
rc2 = cublasDestroy(h)
print("cublasDestroy_v2 rc=", rc2)
