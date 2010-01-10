;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
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

(declaim (type simple-vector *cuda-devices*))

(defvar *cuda-devices* nil)

(defcfun "cuDeviceGetCount" cuda-error
  (cnt-ptr (:pointer :int)))

(defcfun "cuDeviceGet" cuda-error
  (ptr (:pointer :int))
  (device-idx :int))

(def function cuda-init-devices ()
  (with-foreign-object (tmp :int)
    (cuda-invoke cuDeviceGetCount tmp)
    (let ((count (mem-ref tmp :int)))
      (setf *cuda-devices*
            (coerce (loop for i from 0 to (1- count)
                       collect (progn
                                 (cuda-invoke cuDeviceGet tmp i)
                                 (mem-ref tmp :int)))
                    'vector))
      (when (and (> count 0)
                 (equal (cuda-device-version 0) '(9999 . 9999)))
        (setf *cuda-devices* #())))))

(declaim (inline cuda-device-count cuda-device-handle))

(def (function e) cuda-device-count ()
  "Retrieve the number of installed CUDA-capable devices."
  (unless *cuda-devices*
    (cuda-init-devices))
  (length *cuda-devices*))

(def function cuda-device-handle (device)
  (unless *cuda-devices*
    (cuda-init-devices))
  (svref *cuda-devices* device))

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
  "Retrieve information about a CUDA device."
  (with-foreign-object (tmp :int)
    (cuda-invoke cuDeviceGetAttribute tmp attr (cuda-device-handle device))
    (mem-ref tmp :int)))

(defcfun "cuDeviceComputeCapability" cuda-error
  (major (:pointer :int))
  (minor (:pointer :int))
  (device :int))

(def (function e) cuda-device-version (device)
  "Retrieve the CUDA device version."
  (with-foreign-objects ((pmajor :int)
                         (pminor :int))
    (cuda-invoke cuDeviceComputeCapability pmajor pminor (cuda-device-handle device))
    (cons (mem-ref pmajor :int) (mem-ref pminor :int))))

(defcfun "cuDeviceTotalMem" cuda-error
  (pmem (:pointer :unsigned-int))
  (device :int))

(def (function e) cuda-device-total-mem (device)
  "Retrieve the total amount of RAM in a CUDA device."
  (with-foreign-object (tmp :unsigned-int)
    (cuda-invoke cuDeviceTotalMem tmp (cuda-device-handle device))
    (mem-ref tmp :unsigned-int)))

(defcfun "cuDeviceGetName" cuda-error
  (pbuf :pointer)
  (buf-size :int)
  (device :int))

(def (function e) cuda-device-name (device)
  "Retrieve the name of a CUDA device."
  (with-foreign-pointer-as-string (name 256 :encoding :ascii)
    (cuda-invoke cuDeviceGetName name 256 (cuda-device-handle device))))

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

(defstruct cuda-context
  (device nil :read-only t)
  (handle nil)
  (thread nil)
  (blocks nil))

(defmethod print-object ((object cuda-context) stream)
  (print-unreadable-object (object stream :identity t)
    (format stream "CUDA Context @~A ~AKb"
            (cuda-context-device object)
            (ceiling (reduce #'+ (mapcar #'cuda-linear-extent
                                         (cuda-context-blocks object)))
                     1024))
    (unless (cuda-context-handle object)
      (format stream " (DEAD)"))))

(declaim (type (or cuda-context null) *cuda-context*)
         (type hash-table *cuda-thread-contexts*)
         (inline cuda-current-context))

(defparameter *cuda-context* nil
  "Current active CUDA context")

(defvar *cuda-context-lock* (make-lock "CUDA context"))

(defvar *cuda-context-list* nil
  "List of all allocated contexts")

(defvar *cuda-thread-contexts*
  (make-hash-table :test #'eq #+sbcl :synchronized #+sbcl t)
  "Table of thread-local context stacks")

(def (function e) cuda-current-context ()
  (or *cuda-context*
      ;; Assuming that this is safe:
      (first (gethash (current-thread) *cuda-thread-contexts*))))

(def (function e) cuda-valid-context-p (context)
  (and (cuda-context-p context)
       (not (null (cuda-context-handle context)))))

(def function cuda-ensure-context (context)
  (let ((current (cuda-current-context)))
    (unless (eq current context)
      (error "CUDA context ~A needed, ~A current." context current))
    current))

(def macro with-cuda-context ((context) &body body)
  `(let ((*cuda-context* (cuda-ensure-context ,context)))
     ,@body))

(def (function e) cuda-create-context (device &optional flags)
  (with-lock-held (*cuda-context-lock*)
    (with-foreign-object (phandle 'cuda-context-handle)
      (cuda-invoke cuCtxCreate phandle (ensure-list flags) (cuda-device-handle device))
      (let* ((thread (current-thread))
             (context (make-cuda-context :device device
                                         :handle (mem-ref phandle 'cuda-context-handle)
                                         :thread thread)))
        (push context *cuda-context-list*)
        (push context (gethash thread *cuda-thread-contexts*))
        context))))

(def (function e) cuda-destroy-context (context)
  (unless (cuda-context-handle context)
    (error "Context already destroyed."))
  (with-lock-held (*cuda-context-lock*)
    ;; Verify correctness
    (cuda-ensure-context context)
    (assert (eq (cuda-context-thread context) (current-thread)))
    ;; Destroy the context
    (cuda-invoke cuCtxDestroy (cuda-context-handle context))
    ;; Unlink the descriptor
    (setf (cuda-context-handle context) nil)
    (setf (cuda-context-thread context) nil)
    (deletef *cuda-context-list* context)
    (deletef (gethash (current-thread) *cuda-thread-contexts*) context)
    ;; Wipe the blocks
    (dolist (blk (cuda-context-blocks context))
      (setf (cuda-linear-handle blk) nil))
    nil))

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
(defctype cuda-array :pointer)

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
  (src-array   cuda-array)
  (src-pitch   :unsigned-int)
  ;; Destination
  (dst-x-bytes :unsigned-int)
  (dst-y       :unsigned-int)
  (dst-type    cuda-memory-type)
  (dst-host    :pointer)
  (dst-device  cuda-device-ptr)
  (dst-array   cuda-array)
  (dst-pitch   :unsigned-int)
  ;; General
  (width-bytes :unsigned-int)
  (height      :unsigned-int))

(defcfun "cuMemcpy2D" cuda-error
  (pspec (:pointer cuda-memcpy-2d)))

;; block descriptor

(defstruct cuda-linear
  (refcnt 1 :type fixnum)
  (context nil :type cuda-context :read-only t)
  (size 0 :type fixnum :read-only t)
  (extent 0 :type fixnum :read-only t)
  (width 0 :type fixnum :read-only t)
  (height 0 :type fixnum :read-only t)
  (pitch 0 :type fixnum :read-only t)
  (handle nil))

(declaim (inline cuda-linear-pitched-p cuda-linear-valid-p))

(def function cuda-linear-valid-p (blk)
  (and (cuda-linear-p blk)
       (not (null (cuda-linear-handle blk)))))

(def function cuda-linear-pitched-p (blk)
  (not (eql (cuda-linear-height blk) 1)))

(defmethod print-object ((object cuda-linear) stream)
  (print-unreadable-object (object stream :identity t)
    (format stream "CUDA Block ~AKb " (ceiling (cuda-linear-extent object) 1024))
    (if (cuda-linear-pitched-p object)
        (format stream "~A+~Ax~A" (cuda-linear-width object)
                (- (cuda-linear-pitch object) (cuda-linear-width object))
                (cuda-linear-height object))
        (format stream "~A" (cuda-linear-size object)))
    (unless (cuda-linear-handle object)
      (format stream " (DEAD)"))))

(def function cuda-alloc-linear (width height &key pitch-for)
  (assert (and (> width 0) (> height 0)) (width height)
          "Invalid linear dimensions: ~A x ~A" width height)
  (assert (cuda-valid-context-p (cuda-current-context)))
  (let ((context (cuda-current-context))
        (size (* width height)))
    (multiple-value-bind (handle pitch)
        (if (and pitch-for (> height 1))
            (with-foreign-objects ((phandle 'cuda-device-ptr)
                                   (ppitch :unsigned-int))
              (cuda-invoke cuMemAllocPitch phandle ppitch
                           width height pitch-for)
              (values (mem-ref phandle 'cuda-device-ptr)
                      (mem-ref ppitch :unsigned-int)))
            (with-foreign-object (phandle 'cuda-device-ptr)
              (cuda-invoke cuMemAlloc phandle size)
              (values (mem-ref phandle 'cuda-device-ptr)
                      width)))
      (when (= pitch width)
        (setf pitch size width size height 1))
      (let ((blk (make-cuda-linear :context context :handle handle
                                   :size size :extent (* height pitch)
                                   :width width :height height :pitch pitch)))
        (push blk (cuda-context-blocks context))
        blk))))

(def function cuda-free-linear (blk)
  (if (cuda-linear-handle blk)
      (let ((context (cuda-linear-context blk)))
        (with-cuda-context (context)
          (cuda-invoke cuMemFree (cuda-linear-handle blk))
          (setf (cuda-linear-handle blk) nil)
          (deletef (cuda-context-blocks context) blk)
          nil))
      (cerror "ignore" "CUDA block already destroyed.")))
