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

;;; Local variables

(def layered-method generate-c-code ((var gpu-local-var))
  (with-slots (c-name item-type dimension-mask static-asize) var
    (cond
      ;; Fixed-size array
      (static-asize
       (format nil "~A ~A[~A]"
               (c-type-string item-type) c-name
               static-asize))
      ;; Dynamic array
      (dimension-mask
       (error "Dynamic local arrays not supported."))
      ;; Scalar
      (t (format nil "~A ~A" (c-type-string item-type) c-name)))))

(def layered-method generate-var-ref ((obj gpu-local-var))
  (c-name-of obj))

(def layered-method generate-array-dim ((obj gpu-local-var) idx)
  (with-slots (dimension-mask) obj
    (aref dimension-mask idx)))

(def layered-method generate-array-size ((obj gpu-local-var))
  (with-slots (static-asize) obj
    (assert static-asize)
    static-asize))

(def layered-method generate-array-extent ((obj gpu-local-var))
  (with-slots (static-asize) obj
    (assert static-asize)
    static-asize))

(def layered-method generate-array-stride ((obj gpu-local-var) idx)
  (with-slots (c-name dimension-mask static-asize) obj
    (assert static-asize)
    (reduce #'* dimension-mask :start (1+ idx))))

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

(def layered-function emit-c-code (form stream &key &allow-other-keys))

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
                         `(emit-c-code ,form -stream- ,@args))
                       (code (&rest args)
                         `(progn ,@(mapcar (lambda (arg)
                                             (typecase arg
                                               (string `(princ ,arg -stream-))
                                               (t `(recurse ,arg))))
                                           args)))))))

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
        ((:void))
        ((:int32 :double)
         (format stream "~A" value))
        ((:float)
         (format stream "~Af" value))
        ((:uint32)
         (format stream "~AU" value))
        ((:boolean)
         (princ (if value "1" "0") stream))
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

  ;; Local variables
  (:method ((form lexical-variable-binding-form) stream &key)
    (let ((type (form-c-type-of form)))
      (unless (gpu-variable-of form)
        (setf (gpu-variable-of form)
              (make-local-var (name-of form) type :from-c-type? t)))
      (cond ((typep (gpu-variable-of form) 'gpu-local-var)
             (princ (generate-c-code (gpu-variable-of form)) stream)
              (awhen (initial-value-of form)
                (princ " = " stream)
                (emit-c-code it stream))
             (princ ";" stream))
            (t (format stream "/* skip: ~A */" (name-of form))))))

  ;; Program block
  (:method :around ((form implicit-progn-mixin) stream &key inside-block?)
    (if inside-block?
        (call-next-method)
        (progn
          (princ "{" stream)
          (with-indented-c-code
            (emit-code-newline stream)
            (call-next-method))
          (emit-code-newline stream)
          (princ "}" stream))))

  (:method ((form implicit-progn-mixin) stream &key)
    (loop
       for item in (body-of form)
       for first = t then nil
       do
         (when (or first (not (nop-form? item)))
           (unless first
             (emit-code-newline stream))
           (emit-c-code item stream :inside-block? t)
           (princ ";" stream))))

  (:method :around ((form lexical-variable-binder-form) stream &key)
    (call-next-layered-method form stream :inside-block? nil))

  (:method ((form lexical-variable-binder-form) stream &key)
    (dolist (binding (bindings-of form))
      (emit-c-code binding stream)
      (emit-code-newline stream))
    (call-next-method))

  (:method :around ((form expr-progn-form) stream &key)
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

  ;; Tag body
  (:method ((form tagbody-form) stream &key)
    ;; Assign label identifiers
    (dolist (item (body-of form))
      (when (and (typep item 'go-tag-form)
                 (null (c-name-of item)))
        (setf (c-name-of item) (make-local-c-name (name-of item)))))
    ;; Output C code
    (loop
       for tail on (body-of form)
       for first = t then nil
       do (atypecase (car tail)
            (go-tag-form
             (let ((*c-code-indent* (- *c-code-indent* *c-code-indent-step*)))
               (unless first
                 (emit-code-newline stream))
               (format stream "~A:" (c-name-of it)))
             ;; Insert a semicolon if ends with a label
             (unless (cdr tail)
               (emit-code-newline stream)
               (format stream "; /*end*/")))
            (t
             (unless first
               (emit-code-newline stream))
             (emit-c-code it stream :inside-block? t)
             (princ ";" stream)))))

  (:method ((form go-tag-form) stream &key)
    (declare (ignore form stream))
    ;; Can only be inside tagbody
    (assert nil))

  (:method ((form go-form) stream &key)
    (unless (c-name-of (tag-of form))
      (error "Uninitialized GO tag: ~S" (name-of form)))
    (format stream "goto ~A" (c-name-of (tag-of form))))

  ;; Block & return
  (:method ((form block-form) stream &key)
    ;; Assign the label
    (when (null (c-name-of form))
      (setf (c-name-of form) (make-local-c-name (name-of form))))
    ;; Output C code
    (call-next-method)
    (emit-code-newline stream)
    (format stream "~A: ; /*end block*/" (c-name-of form)))

  (:method ((form return-from-form) stream &key)
    (unless (c-name-of (target-block-of form))
      (error "Uninitialized RETURN-FROM tag: ~S" (name-of form)))
    (assert (nop-form? (result-of form)))
    (format stream "goto ~A" (c-name-of (target-block-of form))))

  ;; If
  (:method ((form if-form) stream &key)
    (princ "if (" stream)
    (emit-c-code (condition-of form) stream)
    (princ ") {" stream)
    (with-indented-c-code
      (emit-code-newline stream)
      (emit-c-code (then-of form) stream :inside-block? t))
    (emit-code-newline stream)
    (princ "}" stream)
    (unless (nop-form? (else-of form))
      (princ " else {" stream)
      (with-indented-c-code
        (emit-code-newline stream)
        (emit-c-code (else-of form) stream :inside-block? t))
      (emit-code-newline stream)
      (princ "}" stream)))

  
  )
