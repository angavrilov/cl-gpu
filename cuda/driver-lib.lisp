;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file contains CFFI bindings for the CUDA Driver API.
;;;

(in-package :cl-gpu)

(define-foreign-library libcuda
  (t (:default "libcuda")))

(use-foreign-library libcuda)

;;; Error handling

(macrolet ((mkerr (&rest errors)
             `(progn
                (defcenum cuda-error
                  ,@(mapcar (lambda (item)
                              `(,(first item) ,(second item)))
                            errors))
                (defun cuda-error-string (code)
                  (case code
                    ,@(mapcar (lambda (item)
                                `(,(first item) ,(third item)))
                              errors)
                    (t (format nil "UNKNOWN(~A)" code)))))))
  (mkerr (:SUCCESS                    0        "No errors")
         (:ERROR-INVALID-VALUE        1        "Invalid value")
         (:ERROR-OUT-OF-MEMORY        2        "Out of memory")
         (:ERROR-NOT-INITIALIZED      3        "Driver not initialized")
         (:ERROR-DEINITIALIZED        4        "Driver deinitialized")

         (:ERROR-NO-DEVICE            100      "No CUDA-capable device available")
         (:ERROR-INVALID-DEVICE       101      "Invalid device")

         (:ERROR-INVALID-IMAGE        200      "Invalid kernel image")
         (:ERROR-INVALID-CONTEXT      201      "Invalid context")
         (:ERROR-CONTEXT-ALREADY-CURRENT 202   "Context already current")
         (:ERROR-MAP-FAILED           205      "Map failed")
         (:ERROR-UNMAP-FAILED         206      "Unmap failed")
         (:ERROR-ARRAY-IS-MAPPED      207      "Array is mapped")
         (:ERROR-ALREADY-MAPPED       208      "Already mapped")
         (:ERROR-NO-BINARY-FOR-GPU    209      "No binary for GPU")
         (:ERROR-ALREADY-ACQUIRED     210      "Already acquired")
         (:ERROR-NOT-MAPPED           211      "Not mapped")

         (:ERROR-INVALID-SOURCE       300      "Invalid source")
         (:ERROR-FILE-NOT-FOUND       301      "File not found")

         (:ERROR-INVALID-HANDLE       400      "Invalid handle")

         (:ERROR-NOT-FOUND            500      "Not found")

         (:ERROR-NOT-READY            600      "CUDA not ready")

         (:ERROR-LAUNCH-FAILED        700      "Launch failed")
         (:ERROR-LAUNCH-OUT-OF-RESOURCES 701   "Launch exceeded resources")
         (:ERROR-LAUNCH-TIMEOUT       702      "Launch exceeded timeout")
         (:ERROR-LAUNCH-INCOMPATIBLE-TEXTURING 703 "Launch with incompatible texturing")

         (:ERROR-UNKNOWN              999       "Unknown error")))

;;; Driver initialization

(defcfun "cuInit" cuda-error
  (flags :int))

;;; Device count

(defcfun "cuDeviceGetCount" cuda-error
  (cnt-ptr (:pointer :int)))

(defcfun "cuDeviceGet" cuda-error
  (ptr (:pointer :int))
  (device-idx :int))

;;; Device capabilities

(defcenum cuda-device-attr
  (:MAX-THREADS-PER-BLOCK 1)
  :MAX-BLOCK-DIM-X :MAX-BLOCK-DIM-Y :MAX-BLOCK-DIM-Z
  :MAX-GRID-DIM-X :MAX-GRID-DIM-Y :MAX-GRID-DIM-Z
  :MAX-SHARED-MEMORY-PER-BLOCK :TOTAL-CONSTANT-MEMORY
  :WARP-SIZE :MAX-PITCH :MAX-REGISTERS-PER-BLOCK
  :CLOCK-RATE :TEXTURE-ALIGNMENT :GPU-OVERLAP
  :MULTIPROCESSOR-COUNT :KERNEL-EXEC-TIMEOUT
  :INTEGRATED :CAN-MAP-HOST-MEMORY :COMPUTE-MODE)

(defcfun "cuDeviceGetAttribute" cuda-error
  (ptr :pointer)
  (attr cuda-device-attr)
  (device :int))

(defcfun "cuDeviceComputeCapability" cuda-error
  (major (:pointer :int))
  (minor (:pointer :int))
  (device :int))

(defcfun "cuDeviceTotalMem" cuda-error
  (pmem (:pointer :unsigned-int))
  (device :int))

(defcfun "cuDeviceGetName" cuda-error
  (pbuf :pointer)
  (buf-size :int)
  (device :int))

;;; Cuda contexts

(defbitfield cuda-context-flags
  (:sched-spin 1)
  (:sched-yield 2)
  (:blocking-sync 4)
  (:map-host 8)
  (:lmem-resize-to-max 16))

(defctype cuda-context-handle :pointer)

(defcfun "cuCtxCreate" cuda-error
  (pctx (:pointer cuda-context-handle))
  (flags cuda-context-flags)
  (device :int))

(defcfun "cuCtxDestroy" cuda-error
  (ctx cuda-context-handle))

(defcfun "cuCtxSynchronize" cuda-error)

;;; Linear memory

(defctype cuda-device-ptr :unsigned-int)

;; allocation
(defcfun "cuMemAlloc" cuda-error
  (pptr (:pointer cuda-device-ptr))
  (bytes :unsigned-int))

(defcfun "cuMemAllocPitch" cuda-error
  (pptr (:pointer cuda-device-ptr))
  (ppitch (:pointer :unsigned-int))
  (byte-width :unsigned-int)
  (height :unsigned-int)
  (element-size :unsigned-int))

(defcfun "cuMemFree" cuda-error
  (ptr cuda-device-ptr))

;; transfer
(defcfun "cuMemcpyDtoD" cuda-error
  (dst cuda-device-ptr)
  (src cuda-device-ptr)
  (bytes :unsigned-int))

(defcfun "cuMemcpyDtoH" cuda-error
  (dst :pointer)
  (src cuda-device-ptr)
  (bytes :unsigned-int))

(defcfun "cuMemcpyHtoD" cuda-error
  (dst cuda-device-ptr)
  (src :pointer)
  (bytes :unsigned-int))

;; memset
(defcfun "cuMemsetD8" cuda-error
  (ptr   cuda-device-ptr)
  (item  :uint8)
  (count :unsigned-int))

(defcfun "cuMemsetD16" cuda-error
  (ptr   cuda-device-ptr)
  (item  :uint16)
  (count :unsigned-int))

(defcfun "cuMemsetD32" cuda-error
  (ptr   cuda-device-ptr)
  (item  :uint32)
  (count :unsigned-int))

(defcfun "cuMemsetD2D8" cuda-error
  (ptr    cuda-device-ptr)
  (pitch  :unsigned-int)
  (item   :uint8)
  (width  :unsigned-int)
  (height :unsigned-int))

(defcfun "cuMemsetD2D16" cuda-error
  (ptr    cuda-device-ptr)
  (pitch  :unsigned-int)
  (item   :uint16)
  (width  :unsigned-int)
  (height :unsigned-int))

(defcfun "cuMemsetD2D32" cuda-error
  (ptr    cuda-device-ptr)
  (pitch  :unsigned-int)
  (item   :uint32)
  (width  :unsigned-int)
  (height :unsigned-int))

;; 2d copy
(defctype cuda-array-handle :pointer)

(defcenum cuda-memory-type
  (:host 1)
  (:device 2)
  (:array 3))

(defcstruct cuda-memcpy-2d
  ;; Source
  (src-x-bytes :unsigned-int)
  (src-y       :unsigned-int)
  (src-type    cuda-memory-type)
  (src-host    :pointer)
  (src-device  cuda-device-ptr)
  (src-array   cuda-array-handle)
  (src-pitch   :unsigned-int)
  ;; Destination
  (dst-x-bytes :unsigned-int)
  (dst-y       :unsigned-int)
  (dst-type    cuda-memory-type)
  (dst-host    :pointer)
  (dst-device  cuda-device-ptr)
  (dst-array   cuda-array-handle)
  (dst-pitch   :unsigned-int)
  ;; General
  (width-bytes :unsigned-int)
  (height      :unsigned-int))

(defcfun "cuMemcpy2D" cuda-error
  (pspec (:pointer cuda-memcpy-2d)))

(defcfun "cuMemcpy2DUnaligned" cuda-error
  (pspec (:pointer cuda-memcpy-2d)))

;;; Host memory

(defbitfield cuda-host-blk-flags
  (:portable 1)
  (:mapped 2)
  (:write-combine 4))

(defcfun "cuMemHostAlloc" cuda-error
  (pptr (:pointer :pointer))
  (size :unsigned-long)
  (flags cuda-host-blk-flags))

(defcfun "cuMemFreeHost" cuda-error
  (ptr :pointer))

(defcfun "cuMemHostGetDevicePointer" cuda-error
  (pdptr (:pointer cuda-device-ptr))
  (ptr :pointer)
  (flags :unsigned-int))

;;; CUDA Modules

(defctype cuda-module-handle :pointer)
(defctype cuda-kernel-handle :pointer)

(defcfun "cuModuleGetGlobal" cuda-error
  (pptr (:pointer cuda-device-ptr))
  (psize (:pointer :unsigned-int))
  (module cuda-module-handle)
  (name :string))

(defcfun "cuModuleGetFunction" cuda-error
  (pfun (:pointer cuda-kernel-handle))
  (module cuda-module-handle)
  (name :string))

(defcfun "cuModuleUnload" cuda-error
  (module cuda-module-handle))

(defcenum cuda-jit-option
  (:max-registers 0)
  :threads-per-block :wall-time
  :info-log-buffer :info-log-buffer-size-bytes
  :error-log-buffer :error-log-buffer-size-bytes
  :optimization-level :target-from-cucontext :target
  :fallback-strategy)

(defcfun "cuModuleLoadDataEx" cuda-error
  (pmodule  (:pointer cuda-module-handle))
  (pimage   :pointer)
  (num-opts :unsigned-int)
  (options  (:pointer cuda-jit-option))
  (values   :pointer))

;;; Kernels

(defcenum cuda-function-attr
  (:max-threads-per-block 0)
  :shared-size-bytes :const-size-bytes
  :local-size-bytes :num-regs)

(defcfun "cuFuncGetAttribute" cuda-error
  (ptr  (:pointer :int))
  (attr cuda-function-attr)
  (func cuda-kernel-handle))

(defcfun "cuFuncSetBlockShape" cuda-error
  (handle cuda-kernel-handle)
  (x      :int)
  (y      :int)
  (z      :int))

(defcfun "cuLaunchGrid" cuda-error
  (handle cuda-kernel-handle)
  (x      :int)
  (y      :int))

(defcfun ("cuParamSetf" cuda-param-set-float) cuda-error
  (handle cuda-kernel-handle)
  (offset :int)
  (value  :float))

(defcfun ("cuParamSeti" cuda-param-set-uint32) cuda-error
  (handle cuda-kernel-handle)
  (offset :int)
  (value  :unsigned-int))

(defcfun ("cuParamSeti" cuda-param-set-int32) cuda-error
  (handle cuda-kernel-handle)
  (offset :int)
  (value  :int))

(defcfun "cuParamSetv" cuda-error
  (handle cuda-kernel-handle)
  (offset :int)
  (ptr    :pointer)
  (size   :unsigned-int))

(defcfun "cuParamSetSize" cuda-error
  (handle cuda-kernel-handle)
  (size   :unsigned-int))

