;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file contains low-level lisp wrappers for the CUDA Driver API.
;;;

(in-package :cl-gpu)

(def (special-variable e) *cuda-debug* t
  "Enables a debugging-oriented operation mode for CUDA code.")

;;; Error handling

(define-condition cuda-driver-error (error)
  ((method-name :initarg :method :reader method-name)
   (error-code :initarg :error :reader error-code))
  (:report (lambda (condition stream)
             (format stream "Cuda error: ~A while invoking ~A."
                     (cuda-error-string (error-code condition))
                     (method-name condition)))))

(declaim (inline cuda-raise-error))

(defparameter *cuda-return-code* nil)

(defun cuda-report-errors (method rv)
  (let ((*cuda-return-code* rv))
    (cuda-scan-error-buffer)
    (error 'cuda-driver-error :method method :error rv)))

(defun cuda-raise-error (method rv)
  (unless (eql rv :success)
    (cuda-report-errors method rv)))

(defmacro cuda-invoke (method &rest args)
  `(cuda-raise-error ',method (,method ,@args)))

(def function report-cuda-r-retry (msg stream)
  (format stream "Recover the CUDA context and retry ~A" msg))

(defparameter *cuda-lock-recover* nil)

(defmacro with-cuda-recover ((retry-msg &key block-inner on-retry) &body code)
  (with-unique-names (recover-block retry-tag loop-tag)
    `(block ,recover-block
       (tagbody
          ,(if on-retry loop-tag retry-tag)
          (restart-bind ((recover-and-retry (lambda ()
                                              (cuda-recover)
                                              (go ,retry-tag))
                           :report-function (curry #'report-cuda-r-retry ,retry-msg)
                           :test-function (if *cuda-lock-recover*
                                              (constantly nil)
                                              #'cuda-may-recover?)))
            (return-from ,recover-block
              ,(if block-inner
                   `(let ((*cuda-lock-recover* t)) ,@code)
                   `(progn ,@code))))
          ,@(if on-retry (list retry-tag on-retry `(go ,loop-tag)))))))

;;; Driver initialization

(defvar *cuda-initialized* nil)

(unless *cuda-initialized*
  (cuda-invoke cuInit 0)
  (setf *cuda-initialized* t))

;;; Device count

(declaim (type simple-vector *cuda-devices*))

(defvar *cuda-devices* nil)

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

(def (function e) cuda-device-attr (device attr)
  "Retrieve information about a CUDA device."
  (with-foreign-object (tmp :int)
    (cuda-invoke cuDeviceGetAttribute tmp attr (cuda-device-handle device))
    (mem-ref tmp :int)))

(def (function e) cuda-device-version (device)
  "Retrieve the CUDA device version."
  (with-foreign-objects ((pmajor :int)
                         (pminor :int))
    (cuda-invoke cuDeviceComputeCapability pmajor pminor (cuda-device-handle device))
    (cons (mem-ref pmajor :int) (mem-ref pminor :int))))

(def (function e) cuda-device-total-mem (device)
  "Retrieve the total amount of RAM in a CUDA device."
  (with-foreign-object (tmp :unsigned-int)
    (cuda-invoke cuDeviceTotalMem tmp (cuda-device-handle device))
    (mem-ref tmp :unsigned-int)))

(def (function e) cuda-device-name (device)
  "Retrieve the name of a CUDA device."
  (with-foreign-pointer-as-string (name 256 :encoding :ascii)
    (cuda-invoke cuDeviceGetName name 256 (cuda-device-handle device))))

;;; Cuda contexts

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
  (cleanup-queue nil)
  (error-buffer nil)
  (next-error-group 0)
  (init-flags nil))

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

(defstruct (cuda-module (:include counted-block))
  "CUDA Module handle wrapper"
  (context nil :type cuda-context :read-only t)
  (source nil)
  (vars nil)
  (texrefs nil)
  (functions nil)
  (error-group nil)
  (error-table nil)
  (flags nil))

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
  (referenced-by (make-weak-set) :read-only t)
  ;; A copy made for the GC
  (shadow-copy nil)
  ;; A callback for post-reallocate data recovery.
  (recover-cb nil))

(defstruct (cuda-static-blk (:include cuda-linear))
  (module nil :type cuda-module :read-only t)
  (name nil :type string :read-only t))

(defstruct (cuda-host-blk (:include foreign-block))
  (context nil :type cuda-context :read-only t)
  (portable? nil)
  (write-combine? nil)
  (mappings nil)
  (weak-mappings nil :read-only t)
  (shadow-copy nil))

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

(def function %cuda-create-context-handle (device flags)
  (with-foreign-object (phandle 'cuda-context-handle)
    (cuda-invoke cuCtxCreate phandle flags (cuda-device-handle device))
    (mem-ref phandle 'cuda-context-handle)))

(def (function e) cuda-create-context (device &optional flags)
  (with-lock-held (*cuda-context-lock*)
    (let* ((flags (nconc (copy-list (ensure-list flags))
                         (if *cuda-debug* (list :map-host))))
           (thread (current-thread))
           (context (make-cuda-context :device device
                                       :handle (%cuda-create-context-handle device flags)
                                       :thread thread
                                       :init-flags flags
                                       :reg-count (cuda-device-attr
                                                   device :max-registers-per-block)
                                       :warp-size (cuda-device-attr
                                                   device :warp-size)
                                       :shared-memory (cuda-device-attr
                                                       device :max-shared-memory-per-block)
                                       :can-map? (and (member :map-host flags)
                                                      (/= 0 (cuda-device-attr
                                                             device :can-map-host-memory))))))
      (push context *cuda-context-list*)
      (push context (gethash thread *cuda-thread-contexts*))
      (when (cuda-context-can-map? context)
        (setf (cuda-context-error-buffer context) (cuda-alloc-error-buffer)))
      context)))

(defparameter *cuda-context-wipe* nil)

(def (function e) cuda-destroy-context (context)
  (unless (cuda-context-handle context)
    (error "Context already destroyed."))
  (with-lock-held (*cuda-context-lock*)
    ;; Verify correctness
    (with-cuda-context (context)
      (assert (eq (cuda-context-thread context) (current-thread)))
      ;; Destroy the context
      (cuda-invoke cuCtxDestroy (cuda-context-handle context))
      ;; Unlink the descriptor
      (setf (cuda-context-handle context) nil)
      (setf (cuda-context-thread context) nil)
      (deletef *cuda-context-list* context)
      (deletef (gethash (current-thread) *cuda-thread-contexts*) context)
      ;; Wipe the memory handles
      (with-deferred-actions (*cuda-context-wipe*)
        (dolist (obj (union (nconc (weak-set-snapshot (cuda-context-blocks context))
                                   (weak-set-snapshot (cuda-context-host-blocks context))
                                   (copy-list (cuda-context-modules context)))
                            (cuda-context-destroy-queue context)))
          (invalidate obj))
        (setf (cuda-context-destroy-queue context) nil
              (cuda-context-cleanup-queue context) nil)))
    (when (eq *cuda-context* context)
      (setf *cuda-context* nil))
    nil))

(defparameter *cuda-context-reallocate* nil)

(def function cuda-may-recover? (&rest stuff)
  (declare (ignore stuff))
  (and *cuda-return-code*
       (cuda-current-context)
       (null *cuda-context-reallocate*)))

(def method cuda-reallocate ((context cuda-context))
  (unless (cuda-context-handle context)
    (error "Context already destroyed."))
  (warn "Context reallocation resets non-debug device memory buffers to zero.")
  (with-lock-held (*cuda-context-lock*)
    ;; Defer reference invalidation actions until the context is OK.
    (with-deferred-actions (*cuda-context-wipe*)
      (with-cuda-context (context)
        ;; Likewise defer reinitialization actions.
        (with-deferred-actions (*cuda-context-reallocate*)
          ;; Invalidate objects in the destroy queue.
          (dolist (obj (cuda-context-destroy-queue context))
            (with-simple-restart (continue "Continue rebuilding the context.")
              (invalidate obj)))
          (setf (cuda-context-destroy-queue context) nil)
          ;; Save the contents of host blocks and queue reinitialization.
          (dolist (hbuf (weak-set-snapshot (cuda-context-host-blocks context)))
            (let* ((size (foreign-block-size hbuf))
                   (new-item (alloc-foreign-block :int8 size))
                   (new-handle (foreign-block-handle new-item)))
              (memcpy new-handle (foreign-block-handle hbuf) size)
              (defer-action (*cuda-context-reallocate*)
                ;; The hbuf handle changes before this point.
                (memcpy (foreign-block-handle hbuf) new-handle size)
                (deref-buffer new-item))))
          ;; Re-create the context
          (cuda-invoke cuCtxDestroy (cuda-context-handle context))
          (setf (cuda-context-handle context)
                (%cuda-create-context-handle (cuda-context-device context)
                                             (cuda-context-init-flags context)))
          ;; Reallocate the objects
          (dolist (obj (nconc (weak-set-snapshot (cuda-context-blocks context))
                              (weak-set-snapshot (cuda-context-host-blocks context))
                              (copy-list (cuda-context-modules context))))
            (cuda-reallocate obj)))))))

(def (function e) cuda-context-synchronize ()
  (cuda-invoke cuCtxSynchronize))

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

(def function cuda-context-add-cleanup (context item)
  (with-lock-held (*cuda-context-lock*)
    (push item (cuda-context-cleanup-queue context))))

(def function cuda-context-cleanup-queued-items (context)
  (unless *cuda-context-wipe*
    (loop for object = (with-lock-held (*cuda-context-lock*)
                         (pop (cuda-context-cleanup-queue context)))
       while object do
       (with-simple-restart (continue "Continue calling cleanup handlers")
         (funcall object)))))

;;; Linear memory

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

(def function %cuda-alloc-linear-handle (width height pitch-for)
  (if (and pitch-for (> height 1))
      (with-foreign-objects ((phandle 'cuda-device-ptr)
                             (ppitch :unsigned-int))
        (cuda-invoke cuMemAllocPitch phandle ppitch
                     width height pitch-for)
        (values (mem-ref phandle 'cuda-device-ptr)
                (mem-ref ppitch :unsigned-int)))
      (with-foreign-object (phandle 'cuda-device-ptr)
        (cuda-invoke cuMemAlloc phandle (* width height))
        (values (mem-ref phandle 'cuda-device-ptr)
                width))))

(def function cuda-alloc-linear (width height &key pitch-for)
  (assert (and (> width 0) (> height 0)) (width height)
          "Invalid linear dimensions: ~A x ~A" width height)
  (assert (cuda-valid-context-p (cuda-current-context)))
  (let ((context (cuda-current-context))
        (size (* width height)))
    (cuda-context-destroy-queued-items context)
    (multiple-value-bind (handle pitch)
        (with-cuda-recover ("allocating a linear block")
          (%cuda-alloc-linear-handle width height pitch-for))
      (when (= pitch width)
        (setf pitch size width size height 1))
      (let ((blk (make-cuda-linear :context context :handle handle
                                   :size size :extent (* height pitch)
                                   :pitch-elt (or pitch-for 0)
                                   :width width :height height :pitch pitch)))
        (setf (cuda-linear-shadow-copy blk) (copy-cuda-linear blk))
        (cuda-context-queue-finalizer context blk (cuda-linear-shadow-copy blk))
        (weak-set-addf (cuda-context-blocks context) blk)
        blk))))

(def function %cuda-linear-recover-or-wipe (field-fun blk ref &key silent)
  (awhen (ensure-weak-pointer-value (funcall field-fun blk))
    (with-simple-restart (continue "Simply fill the block with zeroes")
      (return-from %cuda-linear-recover-or-wipe
        (funcall it blk ref))))
  (awhen (cuda-linear-handle blk)
    (cuda-invoke cuMemsetD8 it 0 (cuda-linear-extent blk))
    (unless (or silent *cuda-context-reallocate*)
      (warn "Block wiped due to reference deletion or context reinitialization."))))

(def method cuda-reallocate ((blk cuda-linear))
  (setf (cuda-linear-handle blk)
        (%cuda-alloc-linear-handle (cuda-linear-extent blk) 1 nil)))

(def method cuda-reallocate :after ((blk cuda-linear))
  (awhen (cuda-linear-shadow-copy blk)
    (setf (cuda-linear-handle it) (cuda-linear-handle blk)))
  (defer-action (*cuda-context-reallocate*)
    (%cuda-linear-recover-or-wipe #'cuda-linear-recover-cb blk nil)))

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

(def function %cuda-linear-wipe-reference (blk target)
  (deletef (cuda-linear-references blk) target :test #'eq)
  (weak-set-deletef (cuda-linear-weak-references blk) target)
  (weak-set-deletef (cuda-linear-referenced-by target) blk)
  (%cuda-linear-recover-or-wipe #'cuda-linear-wipe-cb blk target))

(def function %cuda-linear-process-back-references (blk handler)
  (dolist (tgt (weak-set-snapshot (cuda-linear-referenced-by blk)))
    (with-simple-restart (continue "Continue fixing inter-block references")
      (when (member blk (cuda-linear-references tgt) :test #'eq)
        (funcall handler tgt blk)))))

(def function %cuda-linear-wipe-references (blk)
  ;; Unreference blocks linked by this one
  (setf (cuda-linear-references blk)
        (weak-set-snapshot (cuda-linear-weak-references blk)))
  (dolist (tgt (cuda-linear-references blk))
    (with-simple-restart (continue "Continue removing inter-block references")
      (cuda-linear-unreference blk tgt))))

(def method invalidate progn ((blk cuda-linear))
  (setf (cuda-linear-wipe-cb blk) nil)
  (setf (cuda-linear-recover-cb blk) nil)
  (defer-action (*cuda-context-wipe*)
    (%cuda-linear-process-back-references blk #'%cuda-linear-wipe-reference))
  (aif (cuda-linear-shadow-copy blk)
       (invalidate it)
       (defer-action (*cuda-context-wipe*)
         (%cuda-linear-wipe-references blk))))

(def method deallocate :around ((blk cuda-linear))
  (with-cuda-context ((cuda-linear-context blk))
    (call-next-method)))

(def method deallocate ((blk cuda-linear))
  (with-cuda-recover ("freeing a linear block")
    (cuda-invoke cuMemFree (cuda-linear-handle blk)))
  (weak-set-deletef (cuda-context-blocks *cuda-context*) blk))

(def method deallocate ((obj cuda-static-blk))
  (error "Cannot deallocate blocks that belong to modules."))

;;; Host memory

(def function %cuda-alloc-host-handle (size flags)
  (with-foreign-object (pptr :pointer)
    (cuda-invoke cuMemHostAlloc pptr size flags)
    (mem-ref pptr :pointer)))

(def function cuda-alloc-host (size &key flags)
  (assert (> size 0) (size)
          "Invalid host block size: ~A" size)
  (assert (cuda-valid-context-p (cuda-current-context)))
  (let ((context (cuda-current-context))
        (flags (ensure-list flags)))
    (cuda-context-destroy-queued-items context)
    (let* ((ptr (with-cuda-recover ("allocating a host block")
                  (%cuda-alloc-host-handle size flags)))
           (blk
            (make-cuda-host-blk :handle ptr :size size
                                :context context
                                :portable? (member :portable flags)
                                :write-combine? (member :write-combine flags)
                                :weak-mappings
                                (if (and (cuda-context-can-map? context)
                                         (member :mapped flags))
                                    (list 'mappings)))))
      (setf (cuda-host-blk-shadow-copy blk) (copy-cuda-host-blk blk))
      (cuda-context-queue-finalizer context blk (cuda-host-blk-shadow-copy blk))
      (with-lock-held (*cuda-context-lock*)
        (weak-set-addf (cuda-context-host-blocks context) blk))
      blk)))

(def method cuda-reallocate ((blk cuda-host-blk))
  (setf (cuda-host-blk-handle blk)
        (%cuda-alloc-host-handle
         (cuda-host-blk-size blk)
         (append (if (cuda-host-blk-portable? blk) '(:portable))
                 (if (cuda-host-blk-write-combine? blk) '(:write-combine))
                 (if (cuda-host-blk-weak-mappings blk) '(:mapped)))))
  (awhen (cuda-host-blk-shadow-copy blk)
    (setf (cuda-host-blk-handle it) (cuda-host-blk-handle blk)))
  (mapc #'cuda-reallocate (cuda-host-blk-mappings blk)))

(def method invalidate progn ((blk cuda-host-blk))
  (awhen (cuda-host-blk-shadow-copy blk)
    (setf (cuda-host-blk-handle it) nil))
  (mapc #'invalidate (if (cuda-host-blk-shadow-copy blk)
                         (cuda-host-blk-mappings blk)
                         (cdr (cuda-host-blk-weak-mappings blk)))))

(def function cuda-host-blk-any-context (blk)
  (let ((context (cuda-host-blk-context blk)))
    (if (cuda-host-blk-portable? blk)
        (or (cuda-current-context) context)
        context)))

(def method deallocate :around ((blk cuda-host-blk))
  (with-cuda-context ((cuda-host-blk-any-context blk))
    (call-next-method)))

(def method deallocate ((blk cuda-host-blk))
  (with-cuda-recover ("freeing a host block")
    (cuda-invoke cuMemFreeHost (cuda-host-blk-handle blk)))
  (with-lock-held (*cuda-context-lock*)
    (weak-set-deletef (cuda-context-host-blocks (cuda-host-blk-context blk)) blk)))

(def function %cuda-map-host-blk-handle (blk)
  (with-foreign-objects ((paddr 'cuda-device-ptr))
    (cuda-invoke cuMemHostGetDevicePointer paddr (cuda-host-blk-handle blk) 0)
    (mem-ref paddr 'cuda-device-ptr)))

(def function cuda-can-map-host-blk? (blk)
  (and (cuda-host-blk-weak-mappings blk)
       (cuda-context-can-map? (cuda-current-context))))

(def function cuda-map-host-blk (blk)
  (or (find (cuda-current-context) (cuda-host-blk-mappings blk)
            :test #'eq :key #'cuda-linear-context)
      (with-cuda-context ((cuda-host-blk-any-context blk))
        (unless (cuda-can-map-host-blk? blk)
          (error "Flag mismatch: cannot map ~S in context ~S" blk *cuda-context*))
        (let* ((size (cuda-host-blk-size blk))
               (handle (with-cuda-recover ("mapping a host block")
                         (%cuda-map-host-blk-handle blk)))
               (mblk (make-cuda-mapped-blk :context *cuda-context* :root blk
                                           :size size :extent size
                                           :width size :pitch size :height 1
                                           :handle handle
                                           ;; This prevents the reallocate method for
                                           ;; linear blocks from wiping the memory.
                                           :recover-cb (constantly nil)))
               (weak-mblk (copy-cuda-mapped-blk mblk)))
          (setf (cuda-linear-shadow-copy mblk) weak-mblk)
          (setf (cuda-mapped-blk-root weak-mblk) (cuda-host-blk-shadow-copy blk))
          (with-lock-held (*cuda-context-lock*)
            (push mblk (cuda-host-blk-mappings blk))
            (push weak-mblk (cdr (cuda-host-blk-weak-mappings blk))))
          (values mblk)))))

(delegate-buffer-refcnt (blk cuda-mapped-blk) (cuda-mapped-blk-root blk))

(macrolet ((defer (method)
             `(def method ,method :around ((blk cuda-mapped-blk))
                (if (eq (cuda-linear-context blk) (cuda-current-context))
                    (call-next-method)
                    (progn
                      (setf (cuda-linear-handle blk) nil)
                      (defer-action (*cuda-context-wipe*)
                        (cuda-context-add-cleanup (cuda-linear-context blk)
                                                  (curry #',method blk))))))))
  (defer invalidate)
  (defer cuda-reallocate))

(def method cuda-reallocate ((blk cuda-mapped-blk))
  (let ((root (cuda-mapped-blk-root blk)))
    (setf (cuda-linear-handle blk) (%cuda-map-host-blk-handle root))
    (unless (eq (cuda-linear-context blk) (cuda-host-blk-context root))
      (%cuda-linear-process-back-references
       blk (curry #'%cuda-linear-recover-or-wipe #'cuda-linear-recover-cb)))))

;;; CUDA Modules

(def function cuda-module-ensure-handle (module)
  (or (cuda-module-handle module)
      (error "Module has already been unloaded: ~S" module)))

(def function cuda-module-valid-p (module)
  (and (cuda-module-p module)
       (not (null (cuda-module-handle module)))))

(defmethod print-object ((object cuda-module) stream)
  (print-unreadable-object (object stream :identity t)
    (format stream "CUDA Module: ~S ~S"
            (mapcar #'cuda-static-blk-name (cuda-module-vars object))
            (mapcar #'cuda-function-name
                    (cuda-module-functions object)))
    (unless (cuda-module-handle object)
      (format stream " (DEAD)"))))

(def method invalidate progn ((fun cuda-function))
  (setf (cuda-function-handle fun) nil))

(def method invalidate progn ((module cuda-module))
  (deletef (cuda-context-modules (cuda-module-context module)) module)
  (with-deferred-actions (*cuda-context-wipe*)
    (mapc #'invalidate (cuda-module-vars module))
    (mapc #'invalidate (cuda-module-functions module))))

(def function %cuda-module-init-errors (context module)
  (let ((var (ignore-errors (cuda-module-get-var module "GPU_ERR_BUF")))
        (buf (cuda-context-error-buffer context)))
    (when var
      (when buf
        (setf (cuda-linear-recover-cb var)
              (lambda (var ref)
                (declare (ignore ref))
                (with-foreign-object (info :uint32 2)
                  (let ((group (incf (cuda-context-next-error-group context))))
                    (setf (cuda-module-error-group module) group)
                    (setf (mem-aref info :uint32 0) (ash group 8)))
                  (setf (mem-aref info :uint32 1)
                        (cuda-linear-ensure-handle (cuda-map-host-blk buf)))
                  (cuda-invoke cuMemcpyHtoD (cuda-linear-handle var) info 8)))))
      (%cuda-linear-recover-or-wipe #'cuda-linear-recover-cb var nil :silent t))))

(def function %cuda-load-module-handle (source &key max-registers threads-per-block optimization-level)
  (flet ((do-load (image-ptr)
           (with-foreign-objects ((popts 'cuda-jit-option 3)
                                  (pvalues :pointer 3)
                                  (pmax :int) (pthreads :int) (poptlev :int)
                                  (pmodule 'cuda-module-handle))
             (let ((opt-cnt 0))
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
               (values (mem-ref pmodule 'cuda-module-handle))))))
    (etypecase source
      (string (with-foreign-string (ptr source)
                (do-load ptr)))
      (array (with-pointer-to-array (ptr source)
               (do-load ptr))))))

(def function cuda-load-module (source &rest args)
  (assert (cuda-valid-context-p (cuda-current-context)))
  (let* ((context (cuda-current-context)))
    (cuda-context-destroy-queued-items context)
    (aprog1 (make-cuda-module :handle (with-cuda-recover ("loading a module")
                                        (apply #'%cuda-load-module-handle source args))
                              :context context :source source :flags args)
      (push it (cuda-context-modules context))
      (%cuda-module-init-errors context it))))

(def method cuda-reallocate ((obj cuda-module))
  (setf (cuda-module-handle obj)
        (apply #'%cuda-load-module-handle
               (cuda-module-source obj)
               (cuda-module-flags obj)))
  (mapc #'cuda-reallocate (cuda-module-vars obj))
  (mapc #'cuda-reallocate (cuda-module-functions obj)))

(def method deallocate :around ((module cuda-module))
  (with-cuda-context ((cuda-module-context module))
    (call-next-method)))

(def method deallocate ((module cuda-module))
  (with-cuda-recover ("unloading a module")
    (cuda-invoke cuModuleUnload (cuda-module-handle module))))

(def function %cuda-module-get-var-handle (module name)
  (let ((handle (cuda-module-ensure-handle module)))
    (with-foreign-objects ((paddr 'cuda-device-ptr)
                           (psize :unsigned-int))
      (cuda-invoke cuModuleGetGlobal paddr psize handle name)
      (values (mem-ref paddr 'cuda-device-ptr)
              (mem-ref psize :unsigned-int)))))

(def function cuda-module-get-var (module name)
  "Returns a linear block that describes a module-global variable."
  (or (find name (cuda-module-vars module)
            :key #'cuda-static-blk-name :test #'string-equal)
      (with-cuda-context ((cuda-module-context module))
        (bind (((:values handle size) (%cuda-module-get-var-handle module name))
               (blk (make-cuda-static-blk :context *cuda-context* :module module :name name
                                          :size size :extent size
                                          :width size :pitch size :height 1
                                          :handle handle)))
          (push blk (cuda-module-vars module))
          (values blk)))))

(def method buffer-refcnt ((blk cuda-static-blk))
  (if (cuda-linear-handle blk) t))

(def method cuda-reallocate ((blk cuda-static-blk))
  (setf (cuda-linear-handle blk)
        (%cuda-module-get-var-handle (cuda-static-blk-module blk)
                                     (cuda-static-blk-name blk))))

;;; Kernels

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

(defun cuda-param-set-any (type fhandle offset value)
  (flet ((thunk (ptr offset size dir)
           (assert (null dir))
           (cuda-invoke cuParamSetv fhandle offset ptr size)))
    (declare (dynamic-extent #'thunk))
    (setf (native-type-ref type #'thunk offset) value)))

(macrolet ((gen (&rest types)
             `(def generic gen-cuda-param-setter-call (type fhandle offset value)
                (:method (type fhandle offset value)
                  `(cuda-param-set-any ,type ,fhandle ,offset ,value))
                ,@(mapcar (lambda (type)
                            `(:method ((type ,(if (consp type)
                                                  (second type)
                                                  (symbolicate '#:gpu- type '#:-type)))
                                       fhandle offset value)
                               (list ',(symbolicate 'cuda-param-set- (ensure-car type))
                                     fhandle offset value)))
                          types))))
  (gen :int8 :uint8 :int16 :uint16 :int32 :uint32
       (:float gpu-single-float-type)
       (:double gpu-double-float-type)))

(def function cuda-func-attr (fhandle attr)
  "Retrieve information about a CUDA kernel handle."
  (with-foreign-object (tmp :int)
    (cuda-invoke cuFuncGetAttribute tmp attr fhandle)
    (mem-ref tmp :int)))

(def function %cuda-module-get-function-handle (module name)
  (let ((handle (cuda-module-ensure-handle module)))
    (with-foreign-object (pfun 'cuda-kernel-handle)
      (cuda-invoke cuModuleGetFunction pfun handle name)
      (mem-ref pfun 'cuda-kernel-handle))))

(def function cuda-module-get-function (module name)
  "Returns a function object for a kernel."
  (or (find name (cuda-module-functions module)
            :key #'cuda-function-name)
      (with-cuda-context ((cuda-module-context module))
        (let* ((handle (%cuda-module-get-function-handle module name))
               (fun (make-cuda-function
                     :context *cuda-context* :module module
                     :name name :handle handle
                     :num-regs (cuda-func-attr handle :num-regs)
                     :shared-bytes (cuda-func-attr handle :shared-size-bytes)
                     :local-bytes (cuda-func-attr handle :local-size-bytes)
                     :const-bytes (cuda-func-attr handle :const-size-bytes)
                     :max-block-size (cuda-func-attr handle :max-threads-per-block))))
          (push fun (cuda-module-functions module))
          (values fun)))))

(def method cuda-reallocate ((obj cuda-function))
  (setf (cuda-function-handle obj)
        (%cuda-module-get-function-handle (cuda-function-module obj)
                                          (cuda-function-name obj))))

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

;;; Error reporting

(defconstant +cuda-error-magic+ #x7F8CDAE7) ; This value is an SNAN
(defconstant +cuda-error-buf-size+ (/ 65536 4))

(def function cuda-alloc-error-buffer ()
  (aprog1
      (cuda-alloc-host (* +cuda-error-buf-size+ 4) :flags :mapped)
    (memset (cuda-host-blk-handle it) 0 (cuda-host-blk-size it))))

(def function cuda-scan-error-buffer ()
  (let* ((context (cuda-current-context))
         (buffer (if context (cuda-context-error-buffer context)))
         (int-size (foreign-type-size :uint32)))
    ;; If the context has an error buffer
    (when buffer
      (let* ((ptr (cuda-host-blk-handle buffer))
             (count (/ (cuda-host-blk-size buffer) int-size))
             (used (min (mem-aref ptr :uint32 0) (1- count))))
        (unwind-protect
             ;; Scan the buffer for magic markers
             (loop for i = 1 then (1+ i)
                while (< i used)
                when (= (mem-aref ptr :uint32 i) +cuda-error-magic+)
                do (let* ((hdr (mem-aref ptr :uint32 (1+ i))) ; 4 byte int: GGGS
                          (group (ash hdr -8))
                          (size (logand hdr 255))
                          (module (find group (cuda-context-modules context)
                                        :key #'cuda-module-error-group)))
                     (when (and module (<= (+ i size) used) (>= size 2))
                       ;; Report a seemingly valid error entry
                       (with-simple-restart (next-error "Proceed to the next error reported by GPU code.")
                         (let* ((eid (mem-aref ptr :uint32 (+ i 2)))
                                (einfo (assoc eid (cuda-module-error-table module))))
                           (if einfo
                               (let ((arg-data (loop
                                                  for j = 3 then (+ j (/ (align-offset (native-type-byte-size type) 4) 4))
                                                  and type in (second einfo)
                                                  for offset = (* int-size (+ i j))
                                                  while (<= j size)
                                                  collect (native-type-ref type ptr offset))))
                                 ;; Evaluate the error throw expression
                                 (eval `(let ((error-data ,(coerce arg-data 'vector)))
                                          ,(third einfo))))
                               ;; Unknown code, report what we can
                               (error "Unknown GPU error ~A with arguments: ~{~A(~A)~^ ~}"
                                      eid
                                      (loop for j from 3 to size
                                         collect (mem-aref ptr :int32 (+ i j))
                                         collect (mem-aref ptr :float (+ i j)))))))
                       (incf i size))))
          ;; Wipe the buffer to avoid reporting errors twice.
          ;; Don't reuse ptr because the pointer can be changed
          ;; by cuda context recovery restarts.
          (memset (cuda-host-blk-handle buffer) 0 (cuda-host-blk-size buffer)))))))

;;; Error recovery

(def (function e) cuda-recover ()
  (cuda-reallocate (cuda-current-context)))
