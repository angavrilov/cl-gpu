;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

;;; Code generation function prototypes

(def layered-function generate-c-code (obj)
  (:documentation "Produces a C code representation of object"))

(def layered-function generate-var-ref (obj)
  (:documentation "Returns expression for referencing this var"))

(def layered-function generate-array-dim (obj idx)
  (:documentation "Generates accessor for the dimension"))

(def layered-function generate-array-size (obj)
  (:documentation "Generates accessor for the full array size."))

(def layered-function generate-array-extent (obj)
  (:documentation "Generates accessor for the full array extent."))

(def layered-function generate-array-stride (obj idx)
  (:documentation "Generates accessor for the stride"))

(def layered-function compute-field-layout (obj start-offset)
  (:documentation "Aligns the fields in the object")
  (:method ((objs list) start-offset)
    (let* ((fields (mapcan (lambda (item)
                             (multiple-value-bind (items rofs)
                                 (compute-field-layout item start-offset)
                               (setf start-offset rofs)
                               items))
                           objs)))
      (values fields start-offset))))

(def layered-function generate-invoker-form (obj)
  (:documentation "Creates a lambda form to invoke the kernel."))

;;; Global variables

(def layered-method generate-c-code ((var gpu-global-var))
  (with-slots (c-name item-type dimension-mask static-asize) var
    (cond
      ;; Fixed-size array
      (static-asize
       (format nil "~A ~A[~A];"
               (c-type-string item-type) c-name
               static-asize))
      ;; Dynamic array
      (dimension-mask
       (format nil "struct {~%  ~A *val;
  unsigned size;~%  unsigned dim[~A];
  unsigned ext;~%  unsigned step[~A];~%} ~A;"
               (c-type-string item-type)
               (length dimension-mask)
               (1- (length dimension-mask))
               c-name))
      ;; Scalar
      (t (format nil "~A ~A;" (c-type-string item-type) c-name)))))

(def layered-method generate-var-ref ((obj gpu-global-var))
  (with-slots (c-name) obj
    (if (dynarray-var? obj)
        (format nil "~A.val" c-name)
        c-name)))

(def layered-method generate-array-dim ((obj gpu-global-var) idx)
  (with-slots (c-name dimension-mask) obj
    (or (aref dimension-mask idx)
        (format nil "~A.dim[~A]" c-name idx))))

(def layered-method generate-array-size ((obj gpu-global-var))
  (with-slots (c-name dimension-mask static-asize) obj
    (assert dimension-mask)
    (or static-asize
        (format nil "~A.size" c-name))))

(def layered-method generate-array-extent ((obj gpu-global-var))
  (with-slots (c-name static-asize) obj
    (or static-asize
        (format nil "~A.ext" c-name))))

(def layered-method generate-array-stride ((obj gpu-global-var) idx)
  (with-slots (c-name dimension-mask static-asize) obj
    (cond (static-asize
           (reduce #'* dimension-mask :start (1+ idx)))
          (dimension-mask
           (assert (and (>= idx 0)
                        (< idx (1- (length dimension-mask)))))
           (format nil "~A.step[~A]" c-name idx))
          (t (error "Not an array")))))

;;; Function arguments

(def layered-method generate-c-code ((obj gpu-argument))
  (with-slots (c-name item-type dimension-mask static-asize
                      include-size? included-dims
                      include-extent? included-strides) obj
    (cond ((null dimension-mask)
           (format nil "~A ~A" (c-type-string item-type) c-name))
          (static-asize
           (format nil "~A *~A" (c-type-string item-type) c-name))
          (t
           (list*
            (format nil "~A *~A" (c-type-string item-type) c-name)
            (if (not static-asize)
                (nconc (if include-size?
                           (list (format nil "unsigned ~A__D" c-name)))
                       (loop for i from 0 for flag across included-dims
                          when flag collect
                            (format nil "unsigned ~A__D~A" c-name i))
                       (if include-extent?
                           (list (format nil "unsigned ~A__X" c-name)))
                       (loop for i from 0 for flag across included-strides
                          when flag collect
                            (format nil "unsigned ~A__S~A" c-name i)))))))))

(def layered-method compute-field-layout ((obj gpu-argument) start-offset)
  (with-slots (c-name item-type dimension-mask static-asize
                      include-size? included-dims
                      include-extent? included-strides) obj
    (let* ((base-type (if dimension-mask :pointer item-type))
           (size (c-type-size base-type)))
      (align-for-typef start-offset base-type)
      (when (and dimension-mask (null static-asize))
        (let* ((woffset (+ start-offset size)))
          (align-for-typef woffset :uint32)
          (incf woffset
                (* (c-type-size :uint32)
                   (+ (if include-size? 1 0)
                      (if include-extent? 1 0)
                      (loop for flag across included-dims count flag)
                      (loop for flag across included-strides count flag))))
          (setf size (- woffset start-offset))))
      (values (list (list obj start-offset size))
              (+ start-offset size)))))

(def layered-method generate-var-ref ((obj gpu-argument))
  (with-slots (c-name) obj
    c-name))

(def macro with-ensure-unlocked ((obj expr) &body code)
  `(progn
     (unless ,expr
       (assert (not (includes-locked? ,obj)))
       (setf ,expr t))
     ,@code))

(def layered-method generate-array-dim ((obj gpu-argument) idx)
  (with-slots (c-name dimension-mask included-dims) obj
    (or (aref dimension-mask idx)
        (with-ensure-unlocked (obj (aref included-dims idx))
          (format nil "~A__D~A" c-name idx)))))

(def layered-method generate-array-size ((obj gpu-argument))
  (with-slots (c-name dimension-mask static-asize include-size?) obj
    (assert dimension-mask)
    (or static-asize
        (with-ensure-unlocked (obj include-size?)
          (format nil "~A__D" c-name)))))

(def layered-method generate-array-extent ((obj gpu-argument))
  (with-slots (c-name static-asize include-extent?) obj
    (or static-asize
        (with-ensure-unlocked (obj include-extent?)
          (format nil "~A__X" c-name)))))

(def layered-method generate-array-stride ((obj gpu-argument) idx)
  (with-slots (c-name dimension-mask static-asize included-strides) obj
    (cond (static-asize
           (reduce #'* dimension-mask :start (1+ idx)))
          (dimension-mask
           (with-ensure-unlocked (obj (aref included-strides idx))
             (format nil "~A__S~A" c-name idx)))
          (t (error "Not an array")))))

(def layered-method compute-field-layout ((obj gpu-function) start-offset)
  (compute-field-layout (arguments-of obj) start-offset))

;;; Top-level constructs

(def layered-method generate-c-code ((obj gpu-function))
  (with-slots (c-name return-type arguments body) obj
    (format nil "~A ~A(~{~A~^, ~}) {~%~A~%}~%"
            (c-type-string return-type) c-name
            (flatten (mapcar #'generate-c-code arguments))
            body)))

(def layered-method generate-c-code ((obj gpu-module))
  (with-slots (globals functions kernels) obj
    (format nil "/*Globals*/~%~%~{~A~%~}~%/*Functions*/~%~%~{~A~%~}~%/*Kernels*/~%~%~{~A~%~}"
            (mapcar #'generate-c-code globals)
            (mapcar #'generate-c-code functions)
            (mapcar #'generate-c-code kernels))))

;;; Code forms

(defparameter *c-code-indent* 2)
(defparameter *c-code-indent-step* 2)

(def function emit-code-newline (stream)
  (format stream "~&~v,0T" *c-code-indent*))

(def macro with-indented-c-code (&body code)
  `(let ((*c-code-indent* (+ *c-code-indent* *c-code-indent-step*)))
     ,@code))

(def layered-function emit-c-code (form stream &key))

(def layered-function emit-call-c-code (name form stream &key)
  (:method (name form stream &key)
    (declare (ignore stream))
    (error "Unsupported function: ~A in ~S" name (unwalk-form form))))

(def layered-function emit-assn-c-code (name form stream &key)
  (:method (name form stream &key)
    (declare (ignore stream))
    (error "Unsupported l-value function: ~A in ~S" name (unwalk-form form))))

(def definer c-code-emitter (name args &body code)
  (make-builtin-handler-method
   ;; Builtin prototype + method name
   name args (if assn? 'emit-assn-c-code 'emit-call-c-code)
   ;; Body
   code
   :method-args `(-stream- &key)
   :prefix `(macrolet ((emit (format &rest args)
                         `(format -stream- ,format ,@args))
                       (recurse (form &rest args)
                         `(emit-c-code ,form -stream- ,@args))))))

(def layered-methods emit-c-code
  ;; Delegate function calls
  (:method ((form free-application-form) stream &key)
    (emit-call-c-code (operator-of form) form stream))

  (:method ((form setf-application-form) stream &key)
    (emit-assn-c-code (operator-of form) form stream))

  ;; Constants

  (:method ((form constant-form) stream &key)
    (let ((value (value-of form))
          (type (form-c-type-of form)))
      (ecase type
        ((:int32 :double)
         (format stream "~A" value))
        ((:float)
         (format stream "~Af" value))
        ((:uint32)
         (format stream "~AU" value))
        ((:int8 :uint8 :int16 :uint16)
         (format stream "((~A)~A)" (c-type-string type) value)))))

  ;; Assignment
  (:method ((form setq-form) stream &key)
    (let ((gpu-var (ensure-gpu-var (variable-of form))))
      (princ (generate-var-ref gpu-var) stream))
    (princ " = " stream)
    (emit-c-code (value-of form) stream))

  (:method ((form walked-lexical-variable-reference-form) stream &key)
    (let ((gpu-var (ensure-gpu-var form)))
      (princ (generate-var-ref gpu-var) stream)))

  ;; Verbatim inline code
  (:method ((form verbatim-code-form) stream &key)
    (let ((flags nil))
      (flet ((recurse (obj)
               (emit-c-code obj stream)
               (setf flags nil)))
        (dolist (item (body-of form))
          (typecase item
            (constant-form
             (atypecase (value-of item)
               (string  (princ it stream))
               (character
                (if (eql it #\Newline)
                    (emit-code-newline stream)
                    (princ it stream)))
               (keyword (push it flags))
               (t (recurse item))))
            (t (recurse item)))))))

  ;; Program block
  (:method ((form implicit-progn-mixin) stream &key)
    (dolist (item (body-of form))
      (emit-code-newline stream)
      (emit-c-code item stream)
      (princ ";" stream)))

  (:method ((form progn-form) stream &key)
    (princ "{" stream)
    (with-indented-c-code
      (call-next-method))
    (emit-code-newline stream)
    (princ "}" stream))

  (:method ((form expr-progn-form) stream &key)
    (princ "(" stream)
    (with-indented-c-code
      (emit-code-newline stream)
      (loop
         for i from 0
         for item in (body-of form)
         when (> i 0)
         do (progn
              (princ "," stream)
              (emit-code-newline stream))
         do (emit-c-code item stream))
      (emit-code-newline stream))
    (princ ")" stream))
  )

(def c-code-emitter raw-aref (var index)
  (let ((gpu-var (ensure-gpu-var var)))
    (emit "~A[" (generate-var-ref gpu-var))
    (recurse index)
    (emit "]")))

(def c-code-emitter (setf raw-aref) (var index)
  (let ((gpu-var (ensure-gpu-var var)))
    (emit "~A[" (generate-var-ref gpu-var))
    (recurse index)
    (emit "] = ")
    (recurse -value-)))

(def function emit-aref-core (var indexes stream)
  (let* ((gpu-var (ensure-gpu-var var))
         (rank (length (dimension-mask-of gpu-var))))
    (assert (= (length indexes) rank))
    (format stream "~A[" (generate-var-ref gpu-var))
    (loop for i from 0 and idx in indexes
       when (> i 0) do
         (princ "+" stream)
       do (emit-c-code idx stream)
       when (< i (1- rank)) do
         (format stream "*~A" (generate-array-stride gpu-var i)))
    (princ "]" stream)))

(def c-code-emitter aref (var &rest indexes)
  (emit-aref-core var indexes -stream-))

(def c-code-emitter (setf aref) (var &rest indexes)
  (emit-aref-core var indexes -stream-)
  (emit " = ")
  (recurse -value-))

(def c-code-emitter array-total-size (var)
  (emit "~A" (generate-array-size (ensure-gpu-var var))))

(def c-code-emitter array-raw-extent (var)
  (emit "~A" (generate-array-extent (ensure-gpu-var var))))

(def c-code-emitter array-dimension (var dimidx)
  (emit "~A" (generate-array-dim (ensure-gpu-var var)
                                 (ensure-constant dimidx))))

(def c-code-emitter array-raw-stride (var dimidx)
  (emit "~A" (generate-array-stride (ensure-gpu-var var)
                                    (ensure-constant dimidx))))
