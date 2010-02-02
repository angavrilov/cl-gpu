;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

(deflayer cuda-target)

(def macro with-cuda-target (&body code)
  `(with-active-layers (cuda-target) ,@code))

(def layered-method generate-c-code :in cuda-target ((obj gpu-global-var))
  (format nil "~A ~A"
          (if (or (constant-var? obj) (dynarray-var? obj))
              "__constant__"
              "__device__")
          (call-next-method)))

(def layered-method generate-c-code :in cuda-target ((obj gpu-function))
  (format nil "~A ~A"
          (if (typep obj 'gpu-kernel)
              "__global__"
              "__device__")
          (call-next-method)))

(def function lookup-cuda-module (module-id)
  (let* ((context (cuda-current-context))
         (instance (gethash-with-init module-id (cuda-context-module-hash context)
                                      (with-cuda-target
                                        (load-gpu-module-instance module-id)))))
    (unless (car (gpu-module-instance-change-sentinel instance))
      (with-cuda-target
        (upgrade-gpu-module-instance module-id instance)))
    instance))

(setf *gpu-module-lookup-fun* #'lookup-cuda-module)

;;; Instance

(defstruct (cuda-module-instance (:include gpu-module-instance))
  handle)

;; Instance-global variables

(def class* cuda-global-var ()
  ((instance :type cuda-module-instance)
   (var-decl :type gpu-global-var)
   (blk      :type cuda-linear)
   (buffer   :type cuda-mem-array))
  (:documentation "Generic implementation of a module-global var"))

(def constructor cuda-global-var
  (let* ((module (cuda-module-instance-handle (instance-of -self-)))
         (name (c-name-of (var-decl-of -self-)))
         (blk (cuda-module-get-var module name))
         (size (cuda-linear-size blk))
         (buffer (make-instance 'cuda-mem-array
                                :blk blk :size size
                                :elt-type :uint8 :elt-size 1
                                :dims (to-uint32-vector (list size))
                                :strides (to-uint32-vector (list size)))))
    (setf (blk-of -self-) blk
          (buffer-of -self-) buffer)))

;; ----

(def class* cuda-static-global (cuda-global-var)
  ()
  (:documentation "A statically allocated module-global var"))

(def constructor cuda-static-global
  (let* ((decl (var-decl-of -self-))
         (stub-buf (buffer-of -self-))
         (arr-dim (or (dimension-mask-of decl) '(1)))
         (res-buf (buffer-displace stub-buf :foreign-type (item-type-of decl)
                                   :dimensions (coerce arr-dim 'list))))
    (buffer-fill stub-buf 0)
    (setf (slot-value res-buf 'displaced-to) nil)
    (setf (buffer-of -self-) res-buf)))

(def method gpu-global-value ((obj cuda-static-global))
  (buffer-of obj))

(def method (setf gpu-global-value) (value (obj cuda-static-global))
  (unless (eq value (buffer-of obj))
    (if (arrayp value)
        (copy-full-buffer value (buffer-of obj))
        (error "Cannot set the value of a static array global.")))
  (values (buffer-of obj)))

(def method freeze-module-item ((obj cuda-static-global))
  (when (cuda-linear-valid-p (blk-of obj))
    (buffer-as-array (buffer-of obj))))

;; ----

(def class* cuda-scalar-global (cuda-static-global)
  ()
  (:documentation "A scalar statically-allocated global var"))

(def method gpu-global-value ((obj cuda-scalar-global))
  (row-major-bref (buffer-of obj) 0))

(def method (setf gpu-global-value) (value (obj cuda-scalar-global))
  (setf (row-major-bref (buffer-of obj) 0) value))

(def method freeze-module-item ((obj cuda-scalar-global))
  (when (cuda-linear-valid-p (blk-of obj))
    (gpu-global-value obj)))

;; ----

(def class* cuda-dynarray-global (cuda-global-var)
  ((value    :type (or cuda-mem-array null) :initform nil)
   (wipe-cb  :type function))
  (:documentation "A dynamic array module global"))

(def constructor cuda-dynarray-global
  (let* ((stub-buf (buffer-of -self-))
         (arg-buf (buffer-displace stub-buf :foreign-type :uint32 :size t)))
    (buffer-fill stub-buf 0)
    (setf (buffer-of -self-) arg-buf)
    ;; Cache a single instance to allow easy deletef
    (setf (wipe-cb-of -self-)
          (lambda (blk)
            (when (and (cuda-linear-valid-p (blk-of -self-))
                       (value-of -self-)
                       (eq blk (slot-value (value-of -self-) 'blk)))
              (setf (gpu-global-value -self-) nil)
              (warn "Contents of global ~S were deallocated while in use."
                    (name-of (var-decl-of -self-))))))))

(def method gpu-global-value ((obj cuda-dynarray-global))
  (value-of obj))

(def function %upload-dynarray-descriptor (buffer decl value)
  (check-type value cuda-mem-array)
  (assert (numberp (buffer-refcnt value)))
  (with-slots (blk phys-offset size dims elt-type strides) value
    (with-cuda-context ((cuda-linear-context blk))
      (let* ((rq-dims (dimension-mask-of decl))
             (rq-rank (length rq-dims))
             (dim-buf (make-array (+ 1 1 rq-rank rq-rank) :element-type 'uint32)))
        ;; Verify constraints
        (declare (dynamic-extent dim-buf))
        (unless (= rq-rank (length dims))
          (error "Array rank mismatch: ~A instead of ~A" dims rq-dims))
        (unless (equal elt-type (item-type-of decl))
          (error "Array type mismatch: ~A instead of ~A" elt-type (item-type-of decl)))
        (loop
           for new-dim across dims
           for old-dim across rq-dims
           when (and old-dim (/= old-dim new-dim))
           do (error "Dimensions ~A violate constraint ~A" dims rq-dims))
        ;; Form the attribute vector
        (setf (aref dim-buf 0) (+ (cuda-linear-ensure-handle blk) phys-offset))
        (setf (aref dim-buf 1) size)
        (copy-array-data dims 0 dim-buf 2 rq-rank)
        (copy-array-data strides 0 dim-buf (+ 2 rq-rank) rq-rank)
        ;; Upload the attribute vector
        (copy-full-buffer dim-buf buffer)))))

(def macro mem-array-wipe-cb-list (buf)
  `(cuda-linear-wipe-cb-list (slot-value ,buf 'blk)))

(def method (setf gpu-global-value) (value (-self- cuda-dynarray-global))
  (unless (eq value (value-of -self-))
    (if value
        (progn
          (%upload-dynarray-descriptor (buffer-of -self-) (var-decl-of -self-) value)
          ;; Increase the reference count and register the wipe cb
          (ref-buffer value)
          (push (wipe-cb-of -self-) (mem-array-wipe-cb-list value)))
        (buffer-fill (buffer-of -self-) 0))
    ;; Save the new value and detach the old one
    (let ((old-val (value-of -self-)))
      (setf (value-of -self-) value)
      (when old-val
        (deletef (mem-array-wipe-cb-list old-val) (wipe-cb-of -self-))
        (deref-buffer old-val))))
  (values value))

(def method freeze-module-item ((-self- cuda-dynarray-global))
  (value-of -self-))

(def method kill-frozen-object ((item cuda-dynarray-global) (obj cuda-mem-array))
  (deletef (mem-array-wipe-cb-list obj) (wipe-cb-of item))
  (setf (value-of item) nil)
  (deref-buffer obj))

;;; Module instantiation

(def layered-method instantiate-module-item
  :in cuda-target ((item gpu-global-var) instance &key old-value)
  ;; Determine the type based on the var properties:
  (let* ((var-class (cond ((dynarray-var? item) 'cuda-dynarray-global)
                          ((array-var? item) 'cuda-static-global)
                          (t 'cuda-scalar-global)))
         (var-instance (make-instance var-class :var-decl item :instance instance)))
    (when old-value
      (unless (ignore-errors
                (setf (gpu-global-value var-instance) old-value)
                t)
        (warn "Dropping incompatible value ~A for field ~A" old-value (name-of item))))
    (values var-instance)))

(def layered-method load-gpu-module-instance :in cuda-target ((module gpu-module))
  (format t "Loading CUDA module ~A~%" (name-of module))
  (make-cuda-module-instance :handle (cuda-load-module (compiled-code-of module))))

(def layered-method upgrade-gpu-module-instance :in cuda-target ((module gpu-module) instance)
  (let ((old-handle (cuda-module-instance-handle instance)))
    (cuda-unload-module old-handle)
    (loop
       (with-simple-restart (retry-load "Retry loading the module")
         (format t "Reloading CUDA module ~A~%" (name-of module))
         (setf (cuda-module-instance-handle instance)
               (cuda-load-module (compiled-code-of module)))
         (return)))))

(def layered-method upgrade-gpu-module-instance :in cuda-target :around ((module gpu-module) instance)
  ;; Verify the context early
  (with-cuda-context ((cuda-module-context (cuda-module-instance-handle instance)))
    (call-next-method)))
