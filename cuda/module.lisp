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

(deflayer cuda-target (gpu-target))

(def reserved-c-names
    "__device__" "__constant__" "__shared__" "__global__")

(def macro with-cuda-target (&body code)
  `(with-active-layers (cuda-target) ,@code))

(def method call-with-target ((target (eql :cuda)) thunk)
  (with-cuda-target (funcall thunk)))

(def layered-method generate-c-code :in cuda-target ((obj gpu-global-var))
  (format nil "~A ~A"
          (if (or (constant-var? obj) (dynarray-var? obj))
              "__constant__"
              "__device__")
          (call-next-method)))

(def layered-method generate-c-code :in cuda-target ((obj gpu-function))
  (format nil "extern \"C\" ~A ~A"
          (if (typep obj 'gpu-kernel)
              "__global__"
              "__device__")
          (call-next-method)))

(def layered-method c-type-string :in cuda-target (type)
  (case type
    (:int64 "long long")
    (:uint64 "unsigned long long")
    (otherwise (call-next-method))))

(def layered-method c-type-string :in cuda-target ((type cons))
  (case (first type)
    (:tuple (let ((size (second type))
                  (base (third type)))
              (unless (and (> size 0)
                           (<= size (case base
                                      ((:double :int64) 2)
                                      (t 4))))
                (error "Invalid size ~A for tuple of ~A" size base))
              (format nil "~A~A"
                      (case base
                        (:int8 "char") (:uint8 "uchar")
                        (:int16 "short") (:uint16 "ushort")
                        (:int32 "int") (:uint32 "uint")
                        (:int64 "longlong")
                        (:float "float") (:double "double")
                        (t (error "Invalid tuple type ~A" base)))
                      size)))
    (otherwise (call-next-method))))

(def layered-method c-type-size :in cuda-target (type)
  (case type
    (:pointer +cuda-ptr-size+)
    (otherwise (call-next-method))))

(def layered-method c-type-alignment :in cuda-target (type)
  (case type
    (:pointer +cuda-ptr-size+)
    (otherwise (call-next-method))))

(def (c-code-emitter :in cuda-target) tuple (&rest args)
  (emit "make_~A" (c-type-string (form-c-type-of -form-)))
  (emit-separated -stream- args ","))

(def (c-code-emitter :in cuda-target) untuple (tuple)
  (if (has-merged-assignment? -form-)
      (with-c-code-block (-stream-)
        (let ((tuple/type (form-c-type-of tuple)))
          (code (c-type-string tuple/type) " TMP = " tuple ";" #\Newline)
          (loop for i from 0 below (second tuple/type)
             and name in '("x" "y" "z" "w")
             when (emit-merged-assignment -stream- -form- i
                                          (format nil "TMP.~A" name))
             do (code ";"))))
      (code tuple ".x")))

(def layered-method emit-abort-command :in cuda-target (stream exception args)
  (declare (ignore exception args))
  (format stream "__trap();")
  (emit-code-newline stream))

(def function lookup-cuda-module (module-id)
  (let* ((context (or (cuda-current-context)
                      (error "No CUDA context is active.")))
         (instance (gethash-with-init module-id (cuda-context-module-hash context)
                                      (with-cuda-target
                                        (load-gpu-module-instance module-id)))))
    (unless (car (gpu-module-instance-change-sentinel instance))
      (with-cuda-target
        (upgrade-gpu-module-instance module-id instance)))
    instance))


(def method target-module-lookup-fun ((target (eql :cuda)))
  #'lookup-cuda-module)

(setf *current-gpu-target* :cuda)

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
    ;; Define a wipe callback
    (setf (wipe-cb-of -self-)
          (lambda (cur blk)
            (when (and (cuda-linear-valid-p cur)
                       (value-of -self-)
                       (eq blk (slot-value (value-of -self-) 'blk)))
              (setf (gpu-global-value -self-) nil)
              (warn "Contents of global ~S were deallocated while in use."
                    (name-of (var-decl-of -self-))))))
    (setf (cuda-linear-wipe-cb (blk-of -self-))
          (make-weak-pointer (wipe-cb-of -self-)))))

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
  (values value))

(def method freeze-module-item ((-self- cuda-dynarray-global))
  (aif (value-of -self-) (ref-buffer it)))

(def method kill-frozen-object ((item cuda-dynarray-global) (obj cuda-mem-array))
  (setf (value-of item) nil)
  (deref-buffer obj))

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

(def function generate-arg-setter (obj offset fhandle)
  (with-slots (name item-type dimension-mask static-asize
                    include-size? included-dims
                    include-extent? included-strides) obj
    (if (null dimension-mask)
        `(,(cuda-param-setter-name item-type) ,fhandle ,offset ,name)
        (let ((wofs (+ offset +cuda-ptr-size+ -4))
              (dynarr (null static-asize)))
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
              `(let ((,obj ,name))
                 (check-type ,obj cuda-mem-array)
                 (with-slots (blk elt-type size dims strides) ,obj
                   (with-cuda-context ((cuda-linear-context blk))
                     (assert (equal elt-type ',item-type))
                     (cuda-param-set-uint32 ,fhandle ,offset (cuda-linear-ensure-handle blk))
                     ,@(remove-if #'null items))))))))))

(def layered-method generate-invoker-form :in cuda-target ((obj gpu-kernel))
  (multiple-value-bind (args arg-size)
      (compute-field-layout obj 0)
    (with-unique-names (kernel fobj fhandle bx by bz gx gy)
      (bind (((:values rq-arg-names key-arg-specs aux-arg-specs)
              (loop for arg in (arguments-of obj)
                 if (keyword-of arg)
                 collect `((,(keyword-of arg) ,(name-of arg))
                           ,(default-value-of arg)) into kwd
                 else if (default-value-of arg)
                 collect `(,(name-of arg) ,(default-value-of arg)) into aux
                 else collect (name-of arg) into rq
                 finally (return (values rq kwd aux))))
             (arg-setters
              (mapcar (lambda (arg)
                        (destructuring-bind (obj offset size) arg
                          (declare (ignore size))
                          (generate-arg-setter obj offset fhandle)))
                      args)))
        `(lambda (,kernel)
           (declare (type cuda-kernel ,kernel))
           (let ((,fobj (fun-obj-of ,kernel)))
             (declare (type cuda-function ,fobj))
             (lambda (,@rq-arg-names &key
                 ((:block-x ,bx) 1) ((:block-y ,by) 1) ((:block-z ,bz) 1)
                 ((:grid-x ,gx) 1) ((:grid-y ,gy) 1) ,@key-arg-specs
                 ,@(if aux-arg-specs `(&aux ,@aux-arg-specs)))
               (let* ((,fhandle (cuda-function-ensure-handle ,fobj)))
                 (with-cuda-context ((cuda-function-context ,fobj))
                   ,@arg-setters
                   (cuda-invoke cuParamSetSize ,fhandle ,arg-size)
                   (cuda-invoke cuFuncSetBlockShape ,fhandle ,bx ,by ,bz)
                   (cuda-invoke cuLaunchGrid ,fhandle ,gx ,gy))))))))))

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
         (module (make-cuda-module-instance :handle handle)))
    (cuda-context-queue-finalizer (cuda-module-context handle) module handle)
    module))

(def layered-method upgrade-gpu-module-instance :in cuda-target ((module gpu-module) instance)
  (let ((old-handle (cuda-module-instance-handle instance)))
    (cancel-finalization module)
    (cuda-unload-module old-handle)
    (let ((new-handle
           (loop
              (with-simple-restart (retry-load "Retry loading the module")
                (format t "Reloading CUDA module ~A~%" (name-of module))
                (return (cuda-load-module (compiled-code-of module)))))))
      (setf (cuda-module-instance-handle instance) new-handle)
      (cuda-context-queue-finalizer (cuda-module-context new-handle) module new-handle))))

(def layered-method upgrade-gpu-module-instance :in cuda-target :around ((module gpu-module) instance)
  ;; Verify the context early
  (with-cuda-context ((cuda-module-context (cuda-module-instance-handle instance)))
    (call-next-method)))

;;; Code compilation

(def layered-method compile-object :in cuda-target ((function gpu-function))
  (unless (and (slot-boundp function 'body)
               (body-of function))
    (propagate-c-types (form-of function) :upper-type :void)
    (compute-side-effects (form-of function))
    (flatten-statements (form-of function))
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
