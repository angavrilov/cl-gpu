;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file implements wrappers for CUDA module
;;; variables and kernel functions.
;;;

(in-package :cl-gpu)

;;; Buffer coercion

(def generic coerce-to-cuda-mem (object &key can-copy for-write)
  (:documentation "Converts the object to a cuda-mem-array instance.")
  (:method ((object null) &key can-copy for-write)
    (declare (ignore can-copy for-write))
    (values nil))
  (:method ((object cuda-mem-array) &key can-copy for-write)
    (declare (ignore can-copy for-write))
    (values object))
  (:method :after ((object cuda-debug-mem-array) &key can-copy for-write)
    (when (and for-write (consp can-copy))
      (push (curry #'update-buffer-mirror object) (car can-copy))))
  (:method ((object cuda-host-array) &key can-copy for-write)
    (declare (ignore can-copy for-write))
    (if (cuda-can-map-host-blk? (slot-value object 'blk))
        (values (buffer-displace object :mapping :gpu))
        (call-next-method)))
  (:method ((object t) &key can-copy for-write)
    (cond ((not (bufferp object))
           (error "Not a buffer: ~A" object))
          ((null can-copy)
           (error "Cannot coerce to CUDA buffer without copy: ~A" object))
          (t
           (let ((buf (make-cuda-array (buffer-dimensions object)
                                       :foreign-type (buffer-foreign-type object)
                                       :debug nil)))
             (when (consp can-copy)
               (push buf (cdr can-copy))
               (when for-write
                 (push (curry #'copy-full-buffer buf object) (car can-copy))))
             (copy-full-buffer object buf)
             (values buf t))))))

(def function cuda-cleanup-copies (copy-list)
  (restart-case
      (progn
        (cuda-context-synchronize)
        (mapc #'funcall (car copy-list)))
    (recover-and-continue ()
      :report "Recover the CUDA context and return."
      :test cuda-may-recover?
      (cuda-recover)))
  (mapc #'deref-buffer (cdr copy-list))
  nil)

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
         (buffer (make-instance (if (and *cuda-debug*
                                         (not (typep -self- 'cuda-dynarray-global)))
                                    'cuda-debug-mem-array
                                    'cuda-mem-array)
                                :blk blk :size size
                                :elt-type +gpu-uint8-type+ :elt-size 1
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
    (if (bufferp value)
        (copy-full-buffer value (buffer-of obj))
        (error "Cannot set the value of a static array global.")))
  (values (buffer-of obj)))

(def method freeze-module-item ((obj cuda-static-global))
  (when (cuda-linear-valid-p (blk-of obj))
    (buffer-as-array (buffer-of obj))))

(def method update-buffer-mirror ((obj cuda-static-global))
  (update-buffer-mirror (buffer-of obj)))

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
   (wipe-cb  :type function)
   (recover-cb :type function))
  (:documentation "A dynamic array module global"))

(def constructor cuda-dynarray-global
  (let* ((my-blk (blk-of -self-))
         (stub-buf (buffer-of -self-))
         (arg-buf (buffer-displace stub-buf :foreign-type :uint32 :size t)))
    (flet ((wipe-cb (cur blk)
             (when (and (cuda-linear-valid-p cur)
                        (value-of -self-)
                        (eq blk (slot-value (value-of -self-) 'blk)))
               (setf (gpu-global-value -self-) nil)
               (warn "Contents of global ~S were deallocated while in use."
                     (name-of (var-decl-of -self-)))))
           (recover-cb (cur target)
             (declare (ignore cur target))
             (aif (value-of -self-)
                  (%upload-dynarray-descriptor arg-buf (var-decl-of -self-) it)
                  (buffer-fill arg-buf 0))))
      (buffer-fill stub-buf 0)
      (setf (buffer-of -self-) arg-buf)
      ;; Define a wipe callback
      (setf (cuda-linear-wipe-cb my-blk)
            (make-weak-pointer (setf (wipe-cb-of -self-) #'wipe-cb)))
      ;; Define a recovery callback
      (setf (cuda-linear-recover-cb my-blk)
            (make-weak-pointer (setf (recover-cb-of -self-) #'recover-cb))))))

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

(def method (setf gpu-global-value) (value (-self- cuda-dynarray-global))
  (let ((value (coerce-to-cuda-mem value)))
    (unless (eq value (value-of -self-))
      (if value
          (progn
            (%upload-dynarray-descriptor (buffer-of -self-) (var-decl-of -self-) value)
            ;; Establish the reference (includes increasing the refcnt)
            (cuda-linear-reference (blk-of -self-) (slot-value value 'blk)))
          (buffer-fill (buffer-of -self-) 0))
      ;; Save the new value and detach the old one
      (let ((old-val (value-of -self-)))
        (setf (value-of -self-) value)
        (when old-val
          (cuda-linear-unreference (blk-of -self-) (slot-value old-val 'blk)))))
    (values value)))

(def method freeze-module-item ((-self- cuda-dynarray-global))
  (aif (value-of -self-) (ref-buffer it)))

(def method kill-frozen-object ((item cuda-dynarray-global) (obj cuda-mem-array))
  (setf (value-of item) nil)
  (deref-buffer obj))

(def method update-buffer-mirror ((obj cuda-dynarray-global))
  (update-buffer-mirror (value-of obj)))

;;; CUDA kernels

(def class* cuda-kernel (closer-mop:funcallable-standard-object)
  ((instance :type cuda-module-instance)
   (fun-decl :type gpu-kernel)
   (fun-obj  :type cuda-function))
  (:documentation "Generic implementation of a module-global var")
  (:metaclass closer-mop:funcallable-standard-class))

(def constructor cuda-kernel
  (let* ((module (cuda-module-instance-handle (instance-of -self-)))
         (decl (fun-decl-of -self-))
         (fobj (cuda-module-get-function module (c-name-of decl))))
    (setf (fun-obj-of -self-) fobj)
    (let ((fun (funcall (invoker-fun-of decl) -self-)))
      (closer-mop:set-funcallable-instance-function -self- fun))))

(def function generate-arg-setter (obj offset fhandle copy-list effects)
  (with-slots (name item-type dimension-mask static-asize
                    include-size? included-dims
                    include-extent? included-strides) obj
    (if (null dimension-mask)
        `(,(cuda-param-setter-name item-type) ,fhandle ,offset ,name)
        (let ((wofs (+ offset +cuda-ptr-size+ -4))
              (dynarr (null static-asize))
              (written? (if (member obj (side-effects-writes effects)) t)))
          (flet ((setter (value)
                   `(cuda-param-set-uint32 ,fhandle ,(incf wofs 4) ,value)))
            (let ((obj (make-symbol (symbol-name name)))
                  (items (list
                          (when (and dynarr include-size?)
                            (setter 'size))
                          (let ((items (when dynarr
                                         (loop for i from 0 for flag across included-dims
                                            when flag collect (setter `(aref adims ,i)))))
                                (checks (loop for i from 0 for check across dimension-mask
                                           when check collect
                                           `(assert (= (aref adims ,i) ,check)))))
                            (when (or items checks)
                              `(let ((adims dims))
                                 (declare (type (vector uint32) adims))
                                 ,@(nconc checks items))))
                          (when dynarr
                            (let ((ext (when include-extent?
                                         (setter '(aref astrides 0))))
                                  (items (loop for i from 1 for flag across included-strides
                                            when flag collect (setter `(aref astrides ,i)))))
                              (when (or ext items)
                                `(let ((astrides strides))
                                   (declare (type (vector uint32) astrides))
                                   ,@(if ext (list* ext items) items))))))))
              `(let ((,obj (coerce-to-cuda-mem ,name :can-copy ,copy-list :for-write ,written?)))
                 (declare (type cuda-mem-array ,obj))
                 (with-slots (blk elt-type size dims strides) ,obj
                   (with-cuda-context ((cuda-linear-context blk))
                     (assert (eq elt-type ,item-type))
                     (cuda-param-set-uint32 ,fhandle ,offset (cuda-linear-ensure-handle blk))
                     ,@(remove-if #'null items))))))))))

(def layered-method generate-invoker-form :in cuda-target ((obj gpu-kernel))
  (multiple-value-bind (args arg-size)
      (compute-field-layout obj 0)
    (with-unique-names (kernel ivec fobj fhandle bx by bz gx gy copy-list)
      (bind ((effects (side-effects-of obj))
             ((:values rq-arg-names key-arg-specs aux-arg-specs)
              (loop for arg in (arguments-of obj)
                 if (keyword-of arg)
                 collect `((,(keyword-of arg) ,(name-of arg))
                           ,(default-value-of arg)) into kwd
                 else if (default-value-of arg)
                 collect `(,(name-of arg) ,(default-value-of arg)) into aux
                 else collect (name-of arg) into rq
                 finally (return (values rq kwd aux))))
             (write-sets
              (loop for gvar in (side-effects-writes effects)
                 when (typep gvar 'gpu-global-var)
                 collect `(,(gensym "GVAR") (svref ,ivec ,(index-of gvar)))))
             (arg-setters
              (mapcar (lambda (arg)
                        (destructuring-bind (obj offset size) arg
                          (declare (ignore size))
                          (generate-arg-setter obj offset fhandle copy-list effects)))
                      args)))
        `(lambda (,kernel)
           (declare (type cuda-kernel ,kernel))
           (let* ((,fobj (fun-obj-of ,kernel))
                  (,ivec (gpu-module-instance-item-vector (instance-of ,kernel)))
                  ,@write-sets)
             (declare (type cuda-function ,fobj)
                      (ignorable ,ivec))
             (lambda (,@rq-arg-names &key
                 ((:thread-cnt-x ,bx) 1) ((:thread-cnt-y ,by) 1)
                 ((:thread-cnt-z ,bz) 1) ((:block-cnt-x ,gx) 1)
                 ((:block-cnt-y ,gy) 1) ,@key-arg-specs
                 ,@(if aux-arg-specs `(&aux ,@aux-arg-specs)))
               (with-cuda-context ((cuda-function-context ,fobj))
                 (let ((,copy-list (cons nil nil)))
                   (declare (dynamic-extent ,copy-list))
                   (with-cuda-recover (,(format nil "launching ~A of module ~A"
                                                (name-of obj) (name-of *cur-gpu-module*))
                                        :block-inner t
                                        :on-retry (setf (car ,copy-list) nil))
                     (when *cuda-debug*
                       (cuda-context-synchronize)
                       ,@(mapcar (lambda (x)
                                   `(push (curry #'update-buffer-mirror ,(first x))
                                          (car ,copy-list)))
                                 write-sets))
                     (let* ((,fhandle (cuda-function-ensure-handle ,fobj)))
                       ,@arg-setters
                       (cuda-invoke cuParamSetSize ,fhandle ,arg-size)
                       (cuda-invoke cuFuncSetBlockShape ,fhandle ,bx ,by ,bz)
                       (cuda-invoke cuLaunchGrid ,fhandle ,gx ,gy)))
                   (when (or *cuda-debug* (cdr ,copy-list))
                     (cuda-cleanup-copies ,copy-list)))))))))))

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

(def layered-method instantiate-module-item
  :in cuda-target ((item gpu-kernel) instance &key old-value)
  (declare (ignore old-value))
  (make-instance 'cuda-kernel :fun-decl item :instance instance))


(def layered-method load-gpu-module-instance :in cuda-target ((module gpu-module))
  (format t "Loading CUDA module ~A~%" (name-of module))
  (let* ((handle (cuda-load-module (compiled-code-of module)))
         (instance (make-cuda-module-instance :handle handle)))
    (cuda-context-queue-finalizer (cuda-module-context handle) instance handle)
    (setf (cuda-module-error-table handle) (error-table-of module))
    instance))

(def layered-method upgrade-gpu-module-instance :in cuda-target ((module gpu-module) instance)
  (let ((old-handle (cuda-module-instance-handle instance)))
    (cancel-finalization instance)
    (deallocate old-handle)
    (let ((new-handle
           (loop
              (with-simple-restart (retry-load "Retry loading the module")
                (format t "Reloading CUDA module ~A~%" (name-of module))
                (return (cuda-load-module (compiled-code-of module)))))))
      (setf (cuda-module-instance-handle instance) new-handle)
      (setf (cuda-module-error-table new-handle) (error-table-of module))
      (cuda-context-queue-finalizer (cuda-module-context new-handle) instance new-handle))))

(def layered-method upgrade-gpu-module-instance :in cuda-target :around ((module gpu-module) instance)
  ;; Verify the context early
  (with-cuda-context ((cuda-module-context (cuda-module-instance-handle instance)))
    (call-next-method)))

;;; Code compilation

(def layered-method compile-object :in cuda-target ((function gpu-function))
  (unless (and (slot-boundp function 'body)
               (body-of function))
    (prepare-for-compile function)
    (compute-side-effects (form-of function))
    (flatten-statements (form-of function))
    (setf (side-effects-of function)
          (gpu-var-side-effects (side-effects-of (form-of function))))
    (setf (body-of function)
          (with-output-to-string (stream)
            (emit-code-newline stream)
            (emit-c-code (form-of function) stream :inside-block? t)))
    (dolist (arg (arguments-of function))
      (setf (includes-locked? arg) t))))

(def layered-method compile-object :in cuda-target ((module gpu-module))
  (call-next-method)
  (setf (compiled-code-of module)
        (cuda-compile-kernel (generate-c-code module))))

(def layered-method post-compile-object :in cuda-target ((kernel gpu-kernel))
  (let ((form (generate-invoker-form kernel)))
    (setf (invoker-form-of kernel) form
          (invoker-fun-of kernel) (coerce form 'function))))
