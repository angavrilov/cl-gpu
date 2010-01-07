;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by the authors.
;;;
;;; See LICENCE for details.

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
                    (t (format "UNKNOWN(~A)" code)))))))
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

(define-condition cuda-driver-error (error)
  ((method-name :initarg :method :reader method-name)
   (error-code :initarg :error :reader error-code))
  (:report (lambda (condition stream)
             (format stream "Cuda error: ~A while invoking ~A."
                     (cuda-error-string (error-code condition))
                     (method-name condition)))))

(defun cuda-raise-error (method rv)
  (unless (eql rv :success)
    (error 'cuda-driver-error :method method :error rv)))

(defmacro cuda-invoke (method &rest args)
  `(cuda-raise-error ',method (,method ,@args)))

;;; Driver initialization

(defvar *cuda-initialized* nil)

(defcfun "cuInit" cuda-error
  (flags :int))

(unless *cuda-initialized*
  (cuda-invoke cuInit 0)
  (setf *cuda-initialized* t))

;;; Device count

(defcfun "cuDeviceGetCount" cuda-error
  (cnt-ptr (:pointer :int)))

(def (function e) cuda-device-count ()
  (with-foreign-object (pcount :int)
    (cuda-invoke cuDeviceGetCount pcount)
    (let ((count (mem-ref pcount :int)))
      (when (and (> count 0)
                 (equal (cuda-device-version 0) '(9999 . 9999)))
        (setf count 0))
      count)))

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

(def (function e) cuda-device-attr (device attr)
  ;; The function hangs on negative values.
  (check-type device (unsigned-byte 31))
  (with-foreign-object (tmp :int)
    (cuda-invoke cuDeviceGetAttribute tmp attr device)
    (mem-ref tmp :int)))

(defcfun "cuDeviceComputeCapability" cuda-error
  (major (:pointer :int))
  (minor (:pointer :int))
  (device :int))

(def (function e) cuda-device-version (device)
  (check-type device (unsigned-byte 31))
  (with-foreign-objects ((pmajor :int)
                         (pminor :int))
    (cuda-invoke cuDeviceComputeCapability pmajor pminor device)
    (cons (mem-ref pmajor :int) (mem-ref pminor :int))))

(defcfun "cuDeviceTotalMem" cuda-error
  (pmem (:pointer :unsigned-int))
  (device :int))

(def (function e) cuda-device-total-mem (device)
  (check-type device (unsigned-byte 31))
  (with-foreign-object (tmp :unsigned-int)
    (cuda-invoke cuDeviceTotalMem tmp device)
    (mem-ref tmp :unsigned-int)))

(defcfun "cuDeviceGetName" cuda-error
  (pbuf :pointer)
  (buf-size :int)
  (device :int))

(def (function e) cuda-device-name (device)
  (check-type device (unsigned-byte 31))
  (with-foreign-pointer-as-string (name 256 :encoding :ascii)
    (cuda-invoke cuDeviceGetName name 256 device)))


