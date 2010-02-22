;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

(def generic expand-gpu-type (name args)
  (:documentation "Expands type aliases defined through gpu-type.")
  (:method ((name t) args)
    (declare (ignore name args))
    (values)))

(def (definer e :available-flags "e") gpu-type (name args &body code)
  ;; Does a deftype that is available to GPU code
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (defmethod expand-gpu-type ((name (eql ',name)) args)
       (values (block ,name
                 (destructuring-bind ,args args
                   ,@code))
               t))
     ,@(if (null (getf -options- :gpu-only))
           `((def (type :export ,(getf -options- :export)) ,name ,args
               ,@code)))))

(def function parse-atomic-type (type-spec)
  (or (lisp-to-foreign-type type-spec)
      (error "Unknown atomic type: ~S" type-spec)))

(def function parse-global-type (type-spec)
  (multiple-value-bind (rspec expanded?)
      (expand-gpu-type (ensure-car type-spec)
                       (if (consp type-spec) (cdr type-spec)))
    (cond (expanded?
           (parse-global-type rspec))
          ((consp type-spec)
           (ecase (first type-spec)
             (array
              (destructuring-bind (&optional item-type dim-spec) (rest type-spec)
                (when (or (null item-type) (eql item-type '*)
                          (null dim-spec) (eql dim-spec '*))
                  (error "Insufficiently specific type spec: ~S" type-spec))
                (values (parse-atomic-type item-type)
                        (if (numberp dim-spec)
                            (make-array dim-spec :initial-element nil)
                            (coerce (mapcar (lambda (x) (if (numberp x) x nil)) dim-spec)
                                    'vector)))))
             (vector
              (destructuring-bind (&optional item-type dim-spec) (rest type-spec)
                (when (or (null item-type) (eql item-type '*))
                  (error "Insufficiently specific type spec: ~S" type-spec))
                (values (parse-atomic-type item-type)
                        (if (numberp dim-spec) (vector dim-spec) (vector nil)))))))
          (t
           (parse-atomic-type type-spec)))))

(def function parse-local-type (type-spec)
  (if (unknown-type? type-spec)
      nil
      (multiple-value-bind (item dims)
          (parse-global-type type-spec)
        (if dims `(:pointer ,item) item))))

;;; Type properties

(def layered-function c-type-string (type)
  (:documentation "Return a string that represents the type in C")
  (:method ((type cons))
    (ecase (first type)
      (:pointer
       (format nil "~A*"
               (c-type-string (or (second type) :void))))))
  (:method (type)
    (ecase type
      (:void "void")
      (:pointer "void*")
      (:boolean "int")
      (:float "float")
      (:double "double")
      (:uint8 "unsigned char")
      (:int8 "char")
      (:uint16 "unsigned short")
      (:int16 "short")
      (:uint32 "unsigned int")
      (:int32 "int"))))

(def layered-function c-type-size (type)
  (:documentation "Return the size of the type in bytes")
  (:method ((type cons))
    (ecase (first type)
      (:pointer (c-type-size :pointer))))
  (:method (type)
    (ecase type
      (:boolean 4)
      (:float 4)
      (:double 8)
      (:uint8 1)
      (:int8 1)
      (:uint16 2)
      (:int16 2)
      (:uint32 4)
      (:int32 4))))

(def layered-function c-type-alignment (type)
  (:documentation "Return the alignment requirement of the type in bytes")
  (:method ((type cons))
    (ecase (first type)
      (:pointer (c-type-alignment :pointer))))
  (:method (type)
    (ecase type
      (:boolean 4)
      (:float 4)
      (:double 8)
      (:uint8 1)
      (:int8 1)
      (:uint16 2)
      (:int16 2)
      (:uint32 4)
      (:int32 4))))

(def layered-function c-int-range (type)
  (:documentation "Return the integer range of the type, or NIL NIL.")
  (:method (type)
    (case type
      (:uint8 (values 0 (1- (ash 1 8))))
      (:int8 (values (- (ash 1 7)) (1- (ash 1 7))))
      (:uint16 (values 0 (1- (ash 1 16))))
      (:int16 (values (- (ash 1 15)) (1- (ash 1 15))))
      (:uint32 (values 0 (1- (ash 1 32))))
      (:int32 (values (- (ash 1 31)) (1- (ash 1 31))))
      (:uint64 (values 0 (1- (ash 1 64))))
      (:int64 (values (- (ash 1 63)) (1- (ash 1 63))))
      (t (values nil nil)))))

(def layered-function can-promote-type? (src dest)
  (:documentation "Checks if a cast from src to dest is possible.")
  (:method (src dest)
    (if (equal src dest)
        t
        (bind (((:values s-min s-max) (c-int-range src))
               ((:values d-min d-max) (c-int-range dest)))
          (cond ((and s-min d-min)
                 (if (and (<= d-min s-min) (>= d-max s-max)) t :warn))
                (s-min
                 (case dest
                   ((:float :double :boolean) t)))
                (d-min
                 (case src
                   ((:float :double) :warn)))
                ((and (eq src :float) (eq dest :double))
                 t)
                ((and (eq src :double) (eq dest :float))
                 :warn))))))

(def (function i) align-offset (offset alignment)
  (logand (+ offset alignment -1) (lognot (1- alignment))))

(def (function i) align-for-type (offset type)
  (align-offset offset (c-type-alignment type)))

(define-modify-macro align-for-typef (type) align-for-type)

;;; Misc checks

(def function gpu-var-ref-type (form)
  (check-type form walked-lexical-variable-reference-form)
  (let ((defn (definition-of form)))
    (aif (gpu-variable-of defn)
         (if (dimension-mask-of it)
             `(:pointer ,(item-type-of it))
             (item-type-of it))
         (form-c-type-of defn))))

(def function verify-cast (src-type-or-form dest-type form &key prefix (warn? t) error-on-warn? allow)
  (let* ((stype (atypecase src-type-or-form
                  (constant-form (propagate-c-types it :upper-type dest-type))
                  (walked-form (form-c-type-of it))
                  (t it)))
         (status (or (member stype allow :test #'equal)
                     (can-promote-type? stype dest-type))))
    (cond ((or (null status)
               (and (eq status :warn) error-on-warn?))
           (error "Cannot cast ~S to ~S in ~A~@[~S~%Of form ~]~S"
                  stype dest-type (or prefix "")
                  (if (and (not (eq src-type-or-form form))
                           (typep src-type-or-form 'walked-form))
                      (unwalk-form src-type-or-form))
                  (unwalk-form form)))
          ((and warn? (eq status :warn))
           (warn "Implicit cast of ~S to ~S in ~A~@[~S~%Of form ~]~S"
                 stype dest-type (or prefix "")
                 (if (and (not (eq src-type-or-form form))
                          (typep src-type-or-form 'walked-form))
                     (unwalk-form src-type-or-form))
                 (unwalk-form form))))
    dest-type))

(def function int-value-matches-type? (type value)
  (multiple-value-bind (min max) (c-int-range type)
    (and min (<= min value) (<= value max))))

(def (function i) array-c-type? (type)
  (and (consp type)
       (eq (car type) :pointer)))

(def function verify-array (arr)
  (let ((arr/type (form-c-type-of arr)))
    (unless (array-c-type? arr/type)
      (error "Must be an array: ~S" (unwalk-form arr)))))

(def function verify-array-var (arr)
  (unless (typep arr 'walked-lexical-variable-reference-form)
    (error "Must be a variable: ~S" (unwalk-form arr)))
  (verify-array arr))

;;; Utilities for builtins

(def function find-common-type (arg-types type-table)
  (let ((index
         (reduce #'max
                 (mapcar (lambda (atype)
                           (or (position (ensure-c-type-of atype) type-table) 0))
                         arg-types)
                 :initial-value 0)))
    (elt type-table index)))

(def function ensure-common-type (arg-types form &key prefix types)
  (aprog1 (find-common-type arg-types types)
    (dolist (arg arg-types)
      (verify-cast arg it form :prefix prefix))))

(def function splice-constant-arg (form value)
  (with-form-object (const 'constant-form form :value value)
    (push const (arguments-of form))))

;;; Local var creation

(def function make-local-c-name (name)
  (unique-c-name name (unique-name-tbl-of *cur-gpu-function*)))

(def function make-local-var (name type-spec &key from-c-type?)
  (multiple-value-bind (item-type dims)
      (if from-c-type?
          (aprog1 type-spec
            (check-type it keyword))
          (parse-global-type type-spec))
    (when (and dims (not (every #'numberp dims)))
      (error "Local arrays must have fixed dimensions: ~S" type-spec))
    (make-instance 'gpu-local-var
                   :name name
                   :c-name (make-local-c-name name)
                   :item-type item-type :dimension-mask dims)))

;;; Type propagation engine

(def layered-function propagate-c-types (form &key upper-type))

(def layered-function propagate-call-arg-types (name form &key upper-type)
  (:method (name form &key upper-type)
    (declare (ignore name upper-type))
    (dolist (item (arguments-of form))
      (propagate-c-types item))))

(def layered-function compute-call-type (name form &key upper-type)
  (:method (name form &key upper-type)
    (declare (ignore upper-type))
    (error "Unsupported function: ~A in ~S" name (unwalk-form form))))

(def layered-function propagate-assn-arg-types (name form &key upper-type)
  (:method (name form &key upper-type)
    (declare (ignore name upper-type))
    (dolist (item (arguments-of form))
      (propagate-c-types item))
    (propagate-c-types (value-of form))))

(def layered-function compute-assn-type (name form &key upper-type)
  (:method (name form &key upper-type)
    (declare (ignore upper-type))
    (error "Unsupported l-value function: ~A in ~S" name (unwalk-form form))))

(def layered-methods propagate-c-types
  (:method :around (form &key upper-type)
    (declare (ignore upper-type))
    (let ((rtype (call-next-method)))
      (when (null rtype)
        (error "Cannot determine type of: ~S" (unwalk-form form)))
      (setf (form-c-type-of form) rtype)))

  (:method ((form the-form) &key upper-type)
    (declare (ignore upper-type))
    (let* ((cast-type (parse-local-type (declared-type-of form)))
           (arg-type (propagate-c-types (value-of form) :upper-type cast-type)))
      (when (null cast-type)
        (setf cast-type arg-type))
      (unless (equal cast-type arg-type)
        (verify-cast arg-type cast-type form :warn? nil)
        (change-class form 'cast-form))
      cast-type))

  (:method ((form walked-lexical-variable-reference-form) &key upper-type)
    (declare (ignore upper-type))
    (gpu-var-ref-type form))

  (:method ((form constant-form) &key upper-type)
    (if (typep (value-of form) 'character)
        (setf (value-of form) (char-code (value-of form))
              upper-type (or upper-type :uint8)))
    (atypecase (value-of form)
      (integer
       (cond ((int-value-matches-type? upper-type it)
              upper-type)
             ((int-value-matches-type? :int32 it)
              :int32)
             ((int-value-matches-type? :uint32 it)
              :uint32)
             ((int-value-matches-type? :int64 it)
              :int64)
             ((int-value-matches-type? :uint64 it)
              :uint64)
             (t
              (error "Integer value ~A is too big." it))))
      (single-float
       (if (eq upper-type :double)
           :double :float))
      (double-float :double)
      (keyword      :keyword)
      (boolean
       (cond ((and (null it) (eq upper-type :void))
              :void)
             (t :boolean)))
      (t (error "Cannot use constant ~S in C code." it))))

  (:method ((form setq-form) &key upper-type)
    (let* ((target-type (gpu-var-ref-type (variable-of form)))
           (val-type (propagate-c-types (value-of form)
                                        :upper-type target-type)))
      (when (eq (ensure-car target-type) :pointer)
        (error "Assignment of arrays is not supported."))
      (verify-cast val-type target-type (variable-of form)
                   :prefix "assignment to ")
      (if (eq upper-type :void) :void target-type)))

  (:method ((form values-form) &key upper-type)
    (let ((args (values-of form)))
      (cond ((and (consp upper-type)
                  (eq (first upper-type) :values))
             (unless (= (length args) (length (rest upper-type)))
               (error "Expecting ~A values, found ~A: ~S"
                      (length (rest upper-type))
                      (length args) (unwalk-form form)))
             (loop for arg in args and type in (rest upper-type)
                do (propagate-c-types arg :upper-type type)))
            (args
             (propagate-c-types (first args) :upper-type upper-type)
             (dolist (arg (rest args))
               (propagate-c-types arg :upper-type :void))))
      (if (or (null args)
              (eq upper-type :void))
          :void
          (list* :values (mapcar #'form-c-type-of args)))))

  (:method ((form verbatim-code-form) &key upper-type)
    (declare (ignore upper-type))
    (dolist (item (body-of form))
      (typecase item
        (constant-form
         (typecase (value-of item)
           ((or string character keyword))
           (t (propagate-c-types item))))
        (t (propagate-c-types item))))
    (form-c-type-of form))

  (:method ((form implicit-progn-mixin) &key upper-type)
    (if (null (body-of form))
        :void
        (progn
          (dolist (item (butlast (body-of form)))
            (propagate-c-types item :upper-type :void))
          (let ((rtype (propagate-c-types (car (last (body-of form)))
                                          :upper-type upper-type)))
            (if (eq upper-type :void) :void rtype)))))

  (:method ((form tagbody-form) &key upper-type)
    (declare (ignore upper-type))
    (dolist (item (body-of form))
      (propagate-c-types item :upper-type :void))
    :void)

  (:method ((form go-tag-form) &key upper-type)
    (declare (ignore form upper-type))
    :void)

  (:method ((form go-form) &key upper-type)
    (declare (ignore upper-type))
    (unless (tag-of form)
      (error "Unknown GO tag: ~S" (name-of form)))
    :void)

  (:method ((form block-form) &key upper-type)
    (setf (form-c-type-of form) upper-type)
    (let ((inner-t (call-next-method))
          (outer-t (form-c-type-of form)))
      (if (null outer-t)
          (setf outer-t inner-t)
          (unless (eq outer-t :void)
            (verify-cast inner-t outer-t form :allow '(:void))))
      outer-t))

  (:method ((form return-from-form) &key upper-type)
    (declare (ignore upper-type))
    (let ((blk (target-block-of form)))
      (unless blk
        (error "Unknown return tag: ~S" (name-of form)))
      (let ((vtype (propagate-c-types (result-of form)
                                      :upper-type (form-c-type-of blk)))
            (btype (form-c-type-of blk)))
        (if (null btype)
            (setf (form-c-type-of blk) vtype)
            (unless (eq btype :void)
              (verify-cast vtype btype form)))
        :void)))

  (:method ((form if-form) &key upper-type)
    (verify-cast (propagate-c-types (condition-of form) :upper-type :boolean)
                 :boolean form :prefix "condition of ")
    (let ((rt-t (propagate-c-types (then-of form) :upper-type upper-type))
          (rt-e (propagate-c-types (else-of form) :upper-type upper-type)))
      (verify-cast rt-e rt-t form :prefix "else branch of")
      rt-t))

  (:method ((form lexical-variable-binding-form) &key upper-type)
    (declare (ignore upper-type))
    (let* ((pdecls (declarations-of (parent-of form)))
           (decl-spec (find-form-by-name (name-of form) pdecls
                                         :type 'type-declaration-form))
           (decl-type (if decl-spec
                          (parse-local-type (declared-type-of decl-spec))))
           (init-form (initial-value-of form))
           (init-type (propagate-c-types init-form :upper-type decl-type)))
      (cond ((null decl-type)
             (setf decl-type init-type))
            ;; nil means no initialization
            ((and (not (eq decl-type :boolean))
                  (nil-constant? init-form))
             (setf (initial-value-of form) nil
                   init-form nil))
            (t
             (verify-cast init-type decl-type form
                          :prefix "variable initialization ")))
      (when (array-c-type? decl-type)
        (setf (gpu-variable-of form)
              (if init-form
                  (ensure-gpu-var init-form)
                  (make-local-var (name-of form) (declared-type-of decl-spec))))
        (unless (array-var? (gpu-variable-of form))
          (error "Must be an array variable: ~S" (unwalk-form form))))
      decl-type))

  (:method ((form lexical-variable-binder-form) &key upper-type)
    (declare (ignore upper-type))
    (dolist (binding (bindings-of form))
      (propagate-c-types binding))
    (call-next-method))

  ;; Delegate calls and assignments
  (:method ((form free-application-form) &key upper-type)
    (propagate-call-arg-types (operator-of form) form :upper-type upper-type)
    (compute-call-type (operator-of form) form :upper-type upper-type))

  (:method ((form setf-application-form) &key upper-type)
    (propagate-assn-arg-types (operator-of form) form :upper-type upper-type)
    (compute-assn-type (operator-of form) form :upper-type upper-type)))

;;; Call & assignment definers

(def definer type-arg-walker (name args &body code)
  (make-builtin-handler-method
   ;; Builtin prototype + method name
   name args (if assn? 'propagate-assn-arg-types 'propagate-call-arg-types)
   ;; Body
   code
   :method-args `(&key ((:upper-type -upper-type-)))
   :top-decls `((ignorable -upper-type-))
   :prefix `(macrolet ((recurse (form &key upper-type)
                         `(propagate-c-types ,form :upper-type ,upper-type))))))

(def definer type-computer (name args &body code)
  (make-builtin-handler-method
   ;; Builtin prototype + method name
   name args (if assn? 'compute-assn-type 'compute-call-type)
   ;; Body
   (let* ((all-fix-args (append rq-args (mapcar #'first opt-args))))
     `((declare (ignorable ,@all-fix-args ,@(ensure-list rest-arg)))
       (symbol-macrolet ((-arguments/type- (mapcar #'form-c-type-of -arguments-))
                         ,@(if assn?
                               `((-value/type- (form-c-type-of -value-))))
                         ,@(mapcar (lambda (arg)
                                     `(,(make-type-arg arg) (form-c-type-of ,arg)))
                                   all-fix-args)
                         ,@(if rest-arg
                               `((,(make-type-arg rest-arg)
                                   (mapcar #'form-c-type-of ,rest-arg)))))
         ,@code)))
   :method-args `(&key ((:upper-type -upper-type-)))
   :top-decls `((ignorable -upper-type-))))


