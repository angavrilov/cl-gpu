;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file contains CFFI bindings for the CUDA
;;; Driver API and some low-level utility wrappers.
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
  "CUDA Context handle wrapper"
  (device nil :read-only t)
  (handle nil)
  (thread nil)
  (blocks (make-weak-set) :read-only t)
  (modules nil)
  (module-hash #-ecl (make-weak-hash-table :test #'eq :weakness :key)
               #+ecl (make-hash-table :test #'eq)
               :read-only t)
  (reg-count 0)
  (warp-size 0)
  (shared-memory 0)
  (destroy-queue nil)
  (can-map? nil)
  (host-blocks (make-weak-set) :read-only t)
  (cleanup-queue nil))

(defstruct cuda-function
  "CUDA Kernel handle wrapper"
  (handle nil)
  (name nil :type string :read-only t)
  (context nil :type cuda-context :read-only t)
  (module nil :type cuda-module :read-only t)
  (num-regs 0)
  (local-bytes 0)
  (shared-bytes 0)
  (const-bytes 0)
  (max-block-size 0))

(defstruct cuda-module
  "CUDA Module handle wrapper"
  (handle nil)
  (context nil :type cuda-context :read-only t)
  (source nil)
  (vars nil)
  (texrefs nil)
  (functions nil))

(defstruct (cuda-linear (:include counted-block))
  "CUDA linear block wrapper"
  (context nil :type cuda-context :read-only t)
  (size 0 :type fixnum :read-only t)
  (extent 0 :type fixnum :read-only t)
  (width 0 :type fixnum :read-only t)
  (height 0 :type fixnum :read-only t)
  (pitch 0 :type fixnum :read-only t)
  (pitch-elt 0 :type fixnum :read-only t)
  (wipe-cb nil)
  ;; Reference lists; the weak ones must be shared
  ;; with the copy used by the finalizer.
  (references nil)
  (weak-references (make-weak-set) :read-only t)
  (referenced-by (make-weak-set) :read-only t))

(defstruct (cuda-static-blk (:include cuda-linear))
  (module nil :type cuda-module :read-only t))

(defstruct (cuda-host-blk (:include foreign-block))
  (context nil :type cuda-context :read-only t)
  (portable? nil)
  (write-combine? nil)
  (mappings nil)
  (weak-mappings nil :read-only t))

(defstruct (cuda-mapped-blk (:include cuda-linear))
  (root nil :type (or null cuda-host-blk)))

(defmethod print-object ((object cuda-context) stream)
  (print-unreadable-object (object stream :identity t)
    (format stream "CUDA Context @~A ~AKb"
            (cuda-context-device object)
            (ceiling (reduce #'+ (mapcar #'cuda-linear-extent
                                         (weak-set-snapshot
                                          (cuda-context-blocks object))))
                     1024))
    (unless (cuda-context-handle object)
      (format stream " (DEAD)"))))

(declaim (type (or null cuda-context) *cuda-context*)
         (type hash-table *cuda-thread-contexts*)
         (inline cuda-current-context cuda-ensure-context))

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
    (declare (type cuda-context current))
    (unless (eq current context)
      (error "CUDA context ~A needed, ~A current." context current))
    (when (cuda-context-cleanup-queue current)
      (cuda-context-cleanup-queued-items current))
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
                                         :thread thread
                                         :reg-count (cuda-device-attr
                                                     device :max-registers-per-block)
                                         :warp-size (cuda-device-attr
                                                     device :warp-size)
                                         :shared-memory (cuda-device-attr
                                                         device :max-shared-memory-per-block)
                                         :can-map? (and (member :map-host (ensure-list flags))
                                                        (/= 0 (cuda-device-attr
                                                               device :can-map-host-memory))))))
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
    (when (eq *cuda-context* context)
      (setf *cuda-context* nil))
    (setf (cuda-context-handle context) nil)
    (setf (cuda-context-thread context) nil)
    (deletef *cuda-context-list* context)
    (deletef (gethash (current-thread) *cuda-thread-contexts*) context)
    ;; Wipe the memory handles
    (setf (cuda-context-destroy-queue context) nil)
    (dolist (blk (weak-set-snapshot (cuda-context-blocks context)))
      (setf (cuda-linear-handle blk) nil))
    (dolist (blk (weak-set-snapshot (cuda-context-host-blocks context)))
      (%cuda-host-blk-invalidate blk))
    (dolist (module (cuda-context-modules context))
      (%cuda-module-invalidate module))
    (dolist (blk (weak-set-snapshot (cuda-context-host-blocks context)))
      (%cuda-host-blk-wipe-mappings blk :no-local t))
    nil))

(def function cuda-context-queue-finalizer (context object queue-item)
  (assert queue-item)
  (finalize object
            (lambda () (with-lock-held (*cuda-context-lock*)
                    (push queue-item (cuda-context-destroy-queue context))))))

(def function cuda-context-destroy-queued-items (context)
  (loop for object = (with-lock-held (*cuda-context-lock*)
                       (pop (cuda-context-destroy-queue context)))
     while object do
       (with-simple-restart (continue "Continue destroying queued objects")
         (deallocate object))))

(def function cuda-context-cleanup-queued-items (context)
  (loop for object = (with-lock-held (*cuda-context-lock*)
                       (pop (cuda-context-cleanup-queue context)))
     while object do
       (with-simple-restart (continue "Continue calling cleanup handlers")
         (funcall object))))

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

;; block descriptor

(declaim (inline cuda-linear-pitched-p cuda-linear-valid-p cuda-linear-adjust-offset
                 cuda-linear-ensure-handle))

(def function cuda-linear-ensure-handle (blk)
  (or (cuda-linear-handle blk)
      (error "Block has already been deallocated: ~S" blk)))

(def function cuda-linear-valid-p (blk)
  (and (cuda-linear-p blk)
       (not (null (cuda-linear-handle blk)))))

(def function cuda-linear-pitched-p (blk)
  (not (eql (cuda-linear-height blk) 1)))

(def function cuda-linear-adjust-offset (blk offset)
  (if (cuda-linear-pitched-p blk)
      (multiple-value-bind (div rem) (floor offset (cuda-linear-width blk))
        (+ (* div (cuda-linear-pitch blk)) rem))
      offset))

(defmethod print-object ((object cuda-linear) stream)
  (print-unreadable-object (object stream :identity t)
    (format stream "CUDA Block ~AKb &~A "
            (ceiling (cuda-linear-extent object) 1024)
            (buffer-refcnt object))
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
    (cuda-context-destroy-queued-items context)
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
                                   :pitch-elt (or pitch-for 0)
                                   :width width :height height :pitch pitch)))
        (cuda-context-queue-finalizer context blk (copy-cuda-linear blk))
        (weak-set-addf (cuda-context-blocks context) blk)
        blk))))

(def method buffer-refcnt ((blk cuda-static-blk))
  (if (cuda-linear-handle blk) t))

(def function cuda-linear-reference (blk target)
  (let ((crefs (cuda-linear-references blk))
        (wrefs (cuda-linear-weak-references blk))
        (trefs (cuda-linear-referenced-by target)))
    (unless (member target crefs :test #'eq)
      (weak-set-addf trefs blk)
      (weak-set-addf wrefs target)
      (push target (cuda-linear-references blk))
      (ref-buffer target))))

(def function cuda-linear-unreference (blk target)
  (let ((crefs (cuda-linear-references blk))
        (wrefs (cuda-linear-weak-references blk))
        (trefs (cuda-linear-referenced-by target)))
    (when (member target crefs :test #'eq)
      (weak-set-deletef trefs blk)
      (weak-set-deletef wrefs target)
      (deletef (cuda-linear-references blk) target :test #'eq)
      (deref-buffer target))))

(def function %cuda-linear-invalidate (blk)
  (setf (cuda-linear-handle blk) nil)
  (setf (cuda-linear-wipe-cb blk) nil))

(def function %cuda-linear-wipe-references (blk)
  ;; Wipe blocks that reference this one
  (dolist (tgt (weak-set-snapshot (cuda-linear-referenced-by blk)))
    (with-simple-restart (continue "Continue invoking wipe callbacks")
      (deletef (cuda-linear-references tgt) blk :test #'eq)
      (weak-set-deletef (cuda-linear-weak-references tgt) blk)
      (aif (ensure-weak-pointer-value (cuda-linear-wipe-cb tgt))
           (funcall it tgt blk)
           (awhen (cuda-linear-handle tgt)
             (cuda-invoke cuMemsetD8 it 0 (cuda-linear-extent tgt))
             (warn "Block deallocated while referenced, reference source wiped.")))))
  ;; Unreference blocks linked by this one
  (setf (cuda-linear-references blk)
        (weak-set-snapshot (cuda-linear-weak-references blk)))
  (dolist (tgt (cuda-linear-references blk))
    (with-simple-restart (continue "Continue invoking wipe callbacks")
      (cuda-linear-unreference blk tgt))))

(def method deallocate ((blk cuda-linear))
  (let ((context (cuda-linear-context blk)))
    (with-cuda-context (context)
      (cuda-invoke cuMemFree (cuda-linear-handle blk))
      (cancel-finalization blk)
      (%cuda-linear-invalidate blk)
      (weak-set-deletef (cuda-context-blocks context) blk)
      (%cuda-linear-wipe-references blk)
      nil)))

(def method deallocate ((obj cuda-static-blk))
  (error "Cannot deallocate blocks that belong to modules."))

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

(def function cuda-alloc-host (size &key flags)
  (assert (> size 0) (size)
          "Invalid host block size: ~A" size)
  (assert (cuda-valid-context-p (cuda-current-context)))
  (let ((context (cuda-current-context))
        (flags (ensure-list flags)))
    (cuda-context-destroy-queued-items context)
    (let* ((ptr
            (with-foreign-object (pptr :pointer)
              (cuda-invoke cuMemHostAlloc pptr size flags)
              (mem-ref pptr :pointer)))
           (blk
            (make-cuda-host-blk :handle ptr :size size
                                :context context
                                :portable? (member :portable flags)
                                :write-combine? (member :write-combine flags)
                                :weak-mappings
                                (if (and (cuda-context-can-map? context)
                                         (member :mapped flags))
                                    (list 'mappings)))))
      (cuda-context-queue-finalizer context blk (copy-cuda-host-blk blk))
      (with-lock-held (*cuda-context-lock*)
        (weak-set-addf (cuda-context-host-blocks context) blk))
      blk)))

(def function %cuda-host-blk-invalidate (blk)
  (setf (cuda-host-blk-handle blk) nil)
  (mapc #'%cuda-linear-invalidate (cuda-host-blk-mappings blk)))

(def function %cuda-host-blk-wipe-mappings (blk &key (context (cuda-host-blk-context blk)) no-local)
  (dolist (item (cdr (cuda-host-blk-weak-mappings blk)))
    (if (eq context (cuda-linear-context item))
        (unless no-local
          (%cuda-linear-wipe-references item))
        (with-lock-held (*cuda-context-lock*)
          (push (curry #'%cuda-linear-wipe-references item)
                (cuda-context-cleanup-queue (cuda-linear-context item)))))))

(def function cuda-host-blk-any-context (blk)
  (let ((context (cuda-host-blk-context blk)))
    (if (cuda-host-blk-portable? blk)
        (or (cuda-current-context) context)
        context)))

(def method deallocate ((blk cuda-host-blk))
  (let* ((context (cuda-host-blk-context blk))
         (wcontext (cuda-host-blk-any-context blk)))
    (with-cuda-context (wcontext)
      (cuda-invoke cuMemFreeHost (cuda-host-blk-handle blk))
      (cancel-finalization blk)
      (%cuda-host-blk-invalidate blk)
      (with-lock-held (*cuda-context-lock*)
        (weak-set-deletef (cuda-context-host-blocks context) blk))
      (%cuda-host-blk-wipe-mappings blk :context wcontext)
      nil)))

(def function cuda-map-host-blk (blk)
  (or (find (cuda-current-context) (cuda-host-blk-mappings blk)
            :test #'eq :key #'cuda-linear-context)
      (with-cuda-context ((cuda-host-blk-any-context blk))
        (unless (and (cuda-host-blk-weak-mappings blk)
                     (cuda-context-can-map? *cuda-context*))
          (error "Flag mismatch: cannot map ~S in context ~S" blk *cuda-context*))
        (with-foreign-objects ((paddr 'cuda-device-ptr))
          (cuda-invoke cuMemHostGetDevicePointer paddr (cuda-host-blk-handle blk) 0)
          (let* ((size (cuda-host-blk-size blk))
                 (mblk (make-cuda-mapped-blk :context *cuda-context* :root blk
                                             :size size :extent size
                                             :width size :pitch size :height 1
                                             :handle (mem-ref paddr 'cuda-device-ptr)))
                 (weak-mblk (copy-cuda-mapped-blk mblk)))
            (setf (cuda-mapped-blk-root weak-mblk) nil)
            (with-lock-held (*cuda-context-lock*)
              (push mblk (cuda-host-blk-mappings blk))
              (push weak-mblk (cdr (cuda-host-blk-weak-mappings blk))))
            (values mblk))))))

(delegate-buffer-refcnt (blk cuda-mapped-blk) (cuda-mapped-blk-root blk))

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


(def function cuda-module-ensure-handle (module)
  (or (cuda-module-handle module)
      (error "Module has already been unloaded: ~S" module)))

(def function cuda-module-valid-p (module)
  (and (cuda-module-p module)
       (not (null (cuda-module-handle module)))))

(defmethod print-object ((object cuda-module) stream)
  (print-unreadable-object (object stream :identity t)
    (format stream "CUDA Module: ~S ~S"
            (mapcar #'car (cuda-module-vars object))
            (mapcar #'cuda-function-name
                    (cuda-module-functions object)))
    (unless (cuda-module-handle object)
      (format stream " (DEAD)"))))

(def function %cuda-module-invalidate (module)
  (setf (cuda-module-handle module) nil)
  (dolist (var (cuda-module-vars module))
    (%cuda-linear-invalidate (cdr var)))
  (dolist (fun (cuda-module-functions module))
    (setf (cuda-function-handle fun) nil)))

(def function cuda-load-module-pointer (source image-ptr &key max-registers threads-per-block optimization-level)
  (assert (cuda-valid-context-p (cuda-current-context)))
  (with-foreign-objects ((popts 'cuda-jit-option 3)
                         (pvalues :pointer 3)
                         (pmax :int) (pthreads :int) (poptlev :int)
                         (pmodule 'cuda-module-handle))
    (let ((context (cuda-current-context))
          (opt-cnt 0))
      (cuda-context-destroy-queued-items context)
      (flet ((add-option (code type ptr value)
               (setf (mem-ref ptr type) value
                     (mem-aref pvalues :pointer opt-cnt) ptr
                     (mem-aref popts 'cuda-jit-option opt-cnt) code
                     opt-cnt (1+ opt-cnt))))
        (when max-registers
          (add-option :max-registers :int pmax max-registers))
        (when threads-per-block
          (add-option :threads-per-block :int pthreads threads-per-block))
        (when optimization-level
          (add-option :optimization-level :int poptlev optimization-level)))
      (cuda-invoke cuModuleLoadDataEx pmodule image-ptr opt-cnt popts pvalues)
      (aprog1 (make-cuda-module :handle (mem-ref pmodule 'cuda-module-handle)
                                :context context :source source)
        (push it (cuda-context-modules context))))))

(def function cuda-load-module (source &rest args)
  (etypecase source
    (string (with-foreign-string (ptr source)
              (apply #'cuda-load-module-pointer source ptr args)))
    (array (with-pointer-to-array (ptr source)
             (apply #'cuda-load-module-pointer source ptr args)))))

(def function cuda-unload-module (module)
  (if (cuda-module-handle module)
      (let ((context (cuda-module-context module)))
        (with-cuda-context (context)
          (cuda-invoke cuModuleUnload (cuda-module-handle module))
          (%cuda-module-invalidate module)
          (deletef (cuda-context-modules context) module)
          (dolist (var (cuda-module-vars module))
            (%cuda-linear-wipe-references (cdr var)))
          nil))
      (cerror "ignore" "CUDA module already destroyed.")))

(def method deallocate ((module cuda-module))
  (cuda-unload-module module))

(def function cuda-module-get-var (module name)
  "Returns a linear block that describes a module-global variable."
  (or (cdr (assoc name (cuda-module-vars module)
                  :test #'string-equal))
      (with-cuda-context ((cuda-module-context module))
        (let ((handle (cuda-module-ensure-handle module)))
          (with-foreign-objects ((paddr 'cuda-device-ptr)
                                 (psize :unsigned-int))
            (cuda-invoke cuModuleGetGlobal paddr psize handle name)
            (let* ((size (mem-ref psize :unsigned-int))
                   (blk (make-cuda-static-blk :context *cuda-context* :module module
                                              :size size :extent size
                                              :width size :pitch size :height 1
                                              :handle (mem-ref paddr 'cuda-device-ptr))))
              (push (cons name blk) (cuda-module-vars module))
              (values blk)))))))

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

(macrolet ((gen (type)
             (let ((name (symbolicate 'cuda-param-set- type)))
               `(defun ,name (handle offset value)
                  (with-foreign-object (pobj ,type)
                    (setf (mem-ref pobj ,type) value)
                    (cuda-invoke cuParamSetv handle offset pobj
                                 ,(foreign-type-size type)))))))
  (gen :int8) (gen :uint8)
  (gen :int16) (gen :uint16)
  (gen :double))

(def function cuda-param-setter-name (type)
  (macrolet ((gen (&rest types)
               `(case type
                  ,@(mapcar (lambda (type)
                             `(,type ',(symbolicate 'cuda-param-set- type)))
                           types))))
    (gen :int8 :uint8 :int16 :uint16 :int32 :uint32 :float :double)))

(defcfun "cuParamSetSize" cuda-error
  (handle cuda-kernel-handle)
  (size   :unsigned-int))

(def function cuda-func-attr (fhandle attr)
  "Retrieve information about a CUDA kernel handle."
  (with-foreign-object (tmp :int)
    (cuda-invoke cuFuncGetAttribute tmp attr fhandle)
    (mem-ref tmp :int)))

(def function cuda-module-get-function (module name)
  "Returns a function object for a kernel."
  (or (find name (cuda-module-functions module)
            :key #'cuda-function-name)
      (with-cuda-context ((cuda-module-context module))
        (let ((handle (cuda-module-ensure-handle module)))
          (with-foreign-object (pfun 'cuda-kernel-handle)
            (cuda-invoke cuModuleGetFunction pfun handle name)
            (let* ((handle (mem-ref pfun 'cuda-kernel-handle))
                   (fun (make-cuda-function
                         :context *cuda-context* :module module
                         :name name :handle handle
                         :num-regs (cuda-func-attr handle :num-regs)
                         :shared-bytes (cuda-func-attr handle :shared-size-bytes)
                         :local-bytes (cuda-func-attr handle :local-size-bytes)
                         :const-bytes (cuda-func-attr handle :const-size-bytes)
                         :max-block-size (cuda-func-attr handle :max-threads-per-block))))
              (push fun (cuda-module-functions module))
              (values fun)))))))

(def function cuda-function-ensure-handle (fun)
  (or (cuda-function-handle fun)
      (error "Module has already been unloaded: ~S" fun)))

(def function cuda-function-valid-p (fun)
  (and (cuda-function-p fun)
       (not (null (cuda-function-handle fun)))))

(defmethod print-object ((object cuda-function) stream)
  (print-unreadable-object (object stream :identity t)
    (format stream "CUDA Kernel: ~A ~S/~S/~S/~S:~S"
            (cuda-function-name object)
            (cuda-function-num-regs object)
            (cuda-function-shared-bytes object)
            (cuda-function-local-bytes object)
            (cuda-function-const-bytes object)
            (cuda-function-max-block-size object))
    (unless (cuda-function-handle object)
      (format stream " (DEAD)"))))
