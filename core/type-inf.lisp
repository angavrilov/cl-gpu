;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file defines core C type management functions
;;; and type inference rules for special forms.
;;;

(in-package :cl-gpu)

(def function parse-atomic-type (type-spec &key form)
  (or (lisp-to-foreign-type type-spec)
      (gpu-code-error form "Unknown atomic type: ~S" type-spec)))

(def function parse-global-type (type-spec &key form)
  (multiple-value-bind (rspec expanded?)
      (expand-gpu-type (ensure-car type-spec)
                       (if (consp type-spec) (cdr type-spec)))
    (cond (expanded?
           (parse-global-type rspec :form form))
          ((consp type-spec)
           (ecase (first type-spec)
             (array
              (destructuring-bind (&optional item-type dim-spec) (rest type-spec)
                (when (or (null item-type) (eql item-type '*)
                          (null dim-spec) (eql dim-spec '*))
                  (gpu-code-error form "Insufficiently specific type spec: ~S" type-spec))
                (values (parse-atomic-type item-type :form form)
                        (if (numberp dim-spec)
                            (make-array dim-spec :initial-element nil)
                            (coerce (mapcar (lambda (x) (if (numberp x) x nil)) dim-spec)
                                    'vector)))))
             (vector
              (destructuring-bind (&optional item-type dim-spec) (rest type-spec)
                (when (or (null item-type) (eql item-type '*))
                  (gpu-code-error form "Insufficiently specific type spec: ~S" type-spec))
                (values (parse-atomic-type item-type :form form)
                        (if (numberp dim-spec) (vector dim-spec) (vector nil)))))))
          (t
           (parse-atomic-type type-spec :form form)))))

(def function parse-local-type (type-spec &key form)
  (if (unknown-type? type-spec)
      nil
      (cond ((and (consp type-spec)
                  (eq (car type-spec) 'tuple))
             `(:tuple ,(third type-spec)
                      ,(parse-atomic-type (second type-spec) :form form)))
            (t
             (multiple-value-bind (item dims)
                 (parse-global-type type-spec :form form)
               (if dims `(:pointer ,item) item))))))

(def function get-local-var-decl-type (pdecls name)
  (aif (find-form-by-name name pdecls :type 'type-declaration-form)
       (values (parse-local-type (declared-type-of it) :form it)
               (declared-type-of it))))

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
      (:pointer (c-type-size :pointer))
      (:tuple (* (second type)
                 (c-type-size (third type))))))
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
      (:pointer (c-type-alignment :pointer))
      (:tuple (* (extract-power-of-two (second type))
                 (c-type-alignment (third type))))))
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
      (:uint8 (values 0 (1- (ash 1 8)) 8))
      (:int8 (values (- (ash 1 7)) (1- (ash 1 7)) 8))
      (:uint16 (values 0 (1- (ash 1 16)) 16))
      (:int16 (values (- (ash 1 15)) (1- (ash 1 15)) 16))
      (:uint32 (values 0 (1- (ash 1 32)) 32))
      (:int32 (values (- (ash 1 31)) (1- (ash 1 31)) 32))
      (:uint64 (values 0 (1- (ash 1 64)) 64))
      (:int64 (values (- (ash 1 63)) (1- (ash 1 63)) 64))
      (t (values nil nil nil)))))

(def (function i) values-c-type? (type)
  (and (consp type)
       (eq (car type) :values)))

(def layered-function can-promote-type? (src dest &key silent-signed?)
  (:documentation "Checks if a cast from src to dest is possible.")
  (:method (src dest &key silent-signed?)
    (if (equal src dest)
        t
        (bind (((:values s-min s-max s-bits) (c-int-range src))
               ((:values d-min d-max d-bits) (c-int-range dest)))
          (cond ((and s-min d-min)
                 (cond ((and (<= d-min s-min) (>= d-max s-max)) t)
                       ((and silent-signed?
                             (or (zerop d-min) (zerop s-min))
                             (= s-bits d-bits))
                        t)
                       (t :warn)))
                ;; Integer vs floating-point
                (s-min
                 (case dest
                   ((:float :double :boolean) t)))
                (d-min
                 (case src
                   ((:float :double) :warn)))
                ;; Floating-point
                ((and (eq src :float) (eq dest :double))
                 t)
                ((and (eq src :double) (eq dest :float))
                 :warn)
                ;; Recurse into (values) types
                ((and (values-c-type? src) (values-c-type? dest))
                 (let ((match (mapcar (rcurry #'can-promote-type?
                                              :silent-signed? silent-signed?)
                                      (rest src) (rest dest))))
                   (cond ((< (length src) (length dest))
                          nil)
                         ((some #'null match)
                          nil)
                         ((or (> (length src) (length dest))
                              (member :warn match))
                          :warn)
                         (t t)))))))))

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

(def function verify-cast (src-type-or-form dest-type form &key
                                            prefix (warn? t) error-on-warn?
                                            allow (silent-signed? t))
  (let* ((stype (atypecase src-type-or-form
                  (constant-form (propagate-c-types it :upper-type dest-type))
                  (walked-form (form-c-type-of it))
                  (t it)))
         (status (or (member stype allow :test #'equal)
                     (can-promote-type? stype dest-type
                                        :silent-signed? silent-signed?))))
    (flet ((report (func msg)
             (funcall func form msg
                      stype dest-type prefix
                      (if (and (not (eq src-type-or-form form))
                               (typep src-type-or-form 'walked-form))
                          (unwalk-form src-type-or-form)))))
      (cond ((or (null status)
                 (and (eq status :warn) error-on-warn?))
             (report #'gpu-code-error "Cannot cast ~S to ~S in~@[ ~A~]~@[ ~S~]"))
            ((and warn? (eq status :warn))
             (report #'warn-gpu-style "Implicit cast of ~S to ~S in~@[ ~A~]~@[ ~S~]"))))
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
      (gpu-code-error arr "Must be an array."))))

(def function verify-array-var (arr)
  (unless (typep arr 'walked-lexical-variable-reference-form)
    (gpu-code-error arr "Must be a variable."))
  (verify-array arr))

(def function wrap-values-type (types)
  (cond ((cdr types)
         (list* :values types))
        (types
         (car types))
        (t :void)))

(def function unwrap-values-type (type)
  (if (values-c-type? type) (rest type) (list type)))

;;; Utilities for builtins

(def function find-common-type (arg-types type-table)
  (let ((index
         (reduce #'max
                 (mapcar (lambda (atype)
                           (or (position (ensure-c-type-of atype) type-table) 0))
                         arg-types)
                 :initial-value 0)))
    (elt type-table index)))

(def function ensure-common-type (arg-types form &key prefix types (silent-signed? t))
  (aprog1 (find-common-type arg-types types)
    (dolist (arg arg-types)
      (verify-cast arg it form :prefix prefix :silent-signed? silent-signed?))))

(def function splice-constant-arg (form value)
  (with-form-object (const 'constant-form form :value value)
    (push const (arguments-of form))))

;;; Local var creation

(def function make-local-c-name (name)
  (unique-c-name name (unique-name-tbl-of *cur-gpu-function*)))

(def function check-fixed-dims (form dims type-spec)
  (when (and dims (not (every #'numberp dims)))
    (gpu-code-error form "Local arrays must have fixed dimensions: ~S" type-spec)))

(def function make-local-var (name type-spec &key from-c-type? (c-name (make-local-c-name name)) form)
  (multiple-value-bind (item-type dims)
      (if from-c-type?
          type-spec
          (parse-global-type type-spec :form form))
    (check-fixed-dims form dims type-spec)
    (make-instance 'gpu-local-var
                   :name name :c-name c-name
                   :item-type item-type :dimension-mask dims)))

(def function make-shared-var (identity type-spec decl-type &key form)
  (or (find identity (shared-vars-of *cur-gpu-function*) :key #'identity-of)
      (multiple-value-bind (item-type dims)
          (if type-spec (parse-global-type type-spec :form form) decl-type)
        (check-fixed-dims form dims (or type-spec decl-type))
        (let* ((name (name-of identity))
               (var (make-instance 'gpu-shared-var :name name
                                   :c-name (make-local-c-name name)
                                   :item-type item-type :dimension-mask dims
                                   :identity identity)))
          (push var (shared-vars-of *cur-gpu-function*))
          var))))

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
    (gpu-code-error form "Unsupported function: ~A" name)))

(def layered-function propagate-assn-arg-types (name form &key upper-type)
  (:method (name form &key upper-type)
    (declare (ignore name upper-type))
    (dolist (item (arguments-of form))
      (propagate-c-types item))
    (propagate-c-types (value-of form))))

(def layered-function compute-assn-type (name form &key upper-type)
  (:method (name form &key upper-type)
    (declare (ignore upper-type))
    (gpu-code-error form "Unsupported l-value function: ~A" name)))

(def layered-methods propagate-c-types
  ;; Generic
  (:method :around (form &key upper-type)
    (declare (ignore upper-type))
    (let ((rtype (call-next-method)))
      (when (null rtype)
        (gpu-code-error form "Cannot determine type."))
      (setf (form-c-type-of form) rtype)))

  (:method ((form walked-form) &key upper-type)
    (declare (ignore upper-type))
    (gpu-code-error form "This form is not supported in GPU code."))

  ;; Type cast
  (:method ((form the-form) &key upper-type)
    (declare (ignore upper-type))
    (let* ((cast-type (parse-local-type (declared-type-of form) :form form))
           (arg-type (propagate-c-types (value-of form) :upper-type cast-type)))
      (when (null cast-type)
        (setf cast-type arg-type))
      (unless (equal cast-type arg-type)
        (verify-cast arg-type cast-type form :warn? nil)
        (change-class form 'cast-form))
      cast-type))

  ;; Variable reference
  (:method ((form walked-lexical-variable-reference-form) &key upper-type)
    (declare (ignore upper-type))
    (gpu-var-ref-type form))

  ;; Constants
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
              (gpu-code-error form "Integer value ~A is too big." it))))
      (single-float
       (if (eq upper-type :double)
           :double :float))
      (double-float :double)
      (keyword      :keyword)
      (boolean
       (cond ((and (null it) (eq upper-type :void))
              :void)
             (t :boolean)))
      (t (gpu-code-error form "Cannot use constant ~S in C code." it))))

  ;; Assignment
  (:method ((form setq-form) &key upper-type)
    (let* ((target-type (gpu-var-ref-type (variable-of form)))
           (val-type (propagate-c-types (value-of form)
                                        :upper-type target-type)))
      (when (eq (ensure-car target-type) :pointer)
        (gpu-code-error form "Assignment of arrays is not supported."))
      (verify-cast val-type target-type (variable-of form)
                   :prefix "assignment to"
                   :silent-signed? nil)
      (if (eq upper-type :void) :void target-type)))

  (:method ((form multiple-value-setq-form) &key upper-type)
    (bind ((var-types (mapcar #'gpu-var-ref-type (variables-of form)))
           (target-type (wrap-values-type var-types))
           (value-type (propagate-c-types (value-of form) :upper-type target-type))
           (type-list (unwrap-values-type value-type)))
      (loop for var in (variables-of form)
         for var-type in var-types
         for rtype = type-list then (cdr rtype)
         for val-type = (car rtype)
         do (progn
              (unless val-type
                (gpu-code-error form "Too few values: ~S in multiple-value-setq." value-type))
              (when (eq (ensure-car val-type) :pointer)
                (gpu-code-error form "Assignment of arrays is not supported."))
              (verify-cast val-type var-type var
                           :prefix "assignment to"
                           :silent-signed? nil)))
      (if (eq upper-type :void) :void (first type-list))))

  ;; Value group
  (:method ((form values-form) &key upper-type)
    (let ((args (values-of form))
          (upper-types (unwrap-values-type upper-type)))
      (cond ((values-c-type? upper-type)
             (unless (>= (length args) (length (rest upper-type)))
               (gpu-code-error form "Expecting ~A values, found only ~A."
                               (length (rest upper-type))
                               (length args)))
             (loop for arg in args
                for rtype = upper-types then (cdr rtype)
                do (propagate-c-types arg :upper-type (if rtype (car rtype) :void))))
            (args
             (propagate-c-types (first args) :upper-type upper-type)
             (dolist (arg (rest args))
               (propagate-c-types arg :upper-type (if upper-type :void)))))
      (if (or (null args)
              (eq upper-type :void))
          :void
          (let ((child-types (mapcar #'form-c-type-of args)))
            (wrap-values-type (if upper-type
                                  (subseq child-types 0 (length upper-types))
                                  child-types))))))

  ;; Verbatim code
  (:method ((form verbatim-code-form) &key upper-type)
    (declare (ignore upper-type))
    (do-verbatim-code (item flags form :flatten? t)
      (typecase item
        ((or string character))
        (t
         (let* ((upper (getf flags :type))
                (rtype (propagate-c-types item :upper-type upper)))
           (when upper
             (verify-cast rtype upper form :prefix "inline argument"
                          :silent-signed? nil))))))
    (form-c-type-of form))

  ;; Blocks
  (:method ((form implicit-progn-mixin) &key upper-type)
    (if (null (body-of form))
        :void
        (progn
          (dolist (item (butlast (body-of form)))
            (propagate-c-types item :upper-type :void))
          (let ((rtype (propagate-c-types (car (last (body-of form)))
                                          :upper-type upper-type)))
            (if (eq upper-type :void) :void rtype)))))

  (:method ((form implicit-progn-with-declarations-mixin) &key upper-type)
    (declare (ignore upper-type))
    (with-optimize-context (form)
      (call-next-method)))

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
      (gpu-code-error form "Unknown GO tag: ~S" (name-of form)))
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
        (gpu-code-error form "Unknown return tag: ~S" (name-of form)))
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
                 :boolean form :prefix "condition of")
    (let ((rt-t (propagate-c-types (then-of form) :upper-type upper-type))
          (rt-e (propagate-c-types (else-of form) :upper-type upper-type)))
      (verify-cast rt-e rt-t form :prefix "else branch of")
      rt-t))

  (:method ((form lexical-variable-binding-form) &key upper-type values-init-type)
    (declare (ignore upper-type))
    (bind ((pdecls (declarations-of (parent-of form)))
           ((:values decl-type lisp-decl-type)
            (get-local-var-decl-type pdecls (name-of form)))
           (init-form (initial-value-of form))
           (init-type (or values-init-type
                          (when (typep init-form 'free-application-form)
                            (case (operator-of init-form)
                              (unevaluated
                               (prog1
                                   (propagate-c-types (first (arguments-of init-form))
                                                      :upper-type decl-type)
                                 (setf (initial-value-of form) nil)))
                              (make-array
                               (destructuring-bind (dims &key element-type)
                                   (mapcar #'unwrap-keyword-const (arguments-of init-form))
                                 (let* ((etype (if element-type (ensure-constant element-type) 'single-float))
                                        (dtype `(array ,etype ,(ensure-list (ensure-constant dims)))))
                                   (if lisp-decl-type
                                       (unless (equal lisp-decl-type dtype)
                                         (gpu-code-error init-form
                                                         "Initform does not agree with declaration ~A"
                                                         lisp-decl-type))
                                       (setf lisp-decl-type dtype))
                                   (setf (initial-value-of form) nil
                                         init-form nil)
                                   (parse-local-type dtype :form init-form))))))
                          (propagate-c-types init-form :upper-type decl-type))))
      (cond ((null decl-type)
             (setf decl-type init-type))
            ;; nil means no initialization
            ((and (not (eq decl-type :boolean))
                  (nil-constant? init-form))
             (setf (initial-value-of form) nil
                   init-form nil))
            (t
             (verify-cast init-type decl-type form
                          :prefix "variable initialization"
                          :silent-signed? nil)))
      (cond ((shared-identity-of form)
             (when (or (initial-value-of form) values-init-type)
               (gpu-code-error form "Shared variables cannot have initialization forms."))
             (setf (gpu-variable-of form)
                   (make-shared-var (shared-identity-of form)
                                    lisp-decl-type decl-type)))
            ((array-c-type? decl-type)
             (when values-init-type
               (gpu-code-error form "Arrays cannot be assigned through (values)."))
             (setf (gpu-variable-of form)
                   (if init-form
                       (ensure-gpu-var init-form)
                       (make-local-var (name-of form) lisp-decl-type)))
             (unless (array-var? (gpu-variable-of form))
               (gpu-code-error form "Must be an array variable."))))
      decl-type))

  (:method ((form lexical-variable-binder-form) &key upper-type)
    (declare (ignore upper-type))
    (dolist (binding (bindings-of form))
      (propagate-c-types binding))
    (call-next-method))

  (:method ((form multiple-value-bind-form) &key upper-type)
    (declare (ignore upper-type))
    (bind ((decls (declarations-of form))
           (rq-type (wrap-values-type
                     (loop for var in (bindings-of form)
                        collect (get-local-var-decl-type decls (name-of var)))))
           (value-type (propagate-c-types (value-of form) :upper-type rq-type)))
      (loop for var in (bindings-of form)
         for rtype = (unwrap-values-type value-type) then (cdr rtype)
         for val-type = (car rtype)
         do (unless val-type
              (gpu-code-error form "Too few values: ~S in multiple-value-bind." value-type))
         do (propagate-c-types var :values-init-type val-type)))
    (call-next-method))

  (:method ((form multiple-value-prog1-form) &key upper-type)
    (aprog1
        (propagate-c-types (first-form-of form) :upper-type upper-type)
      (dolist (item (other-forms-of form))
        (propagate-c-types item :upper-type :void))))

  (:method ((form unwind-protect-form) &key upper-type)
    (aprog1
        (propagate-c-types (protected-form-of form) :upper-type upper-type)
      (dolist (item (cleanup-form-of form))
        (propagate-c-types item :upper-type :void))))

  ;; Delegate calls and assignments
  (:method ((form free-application-form) &key upper-type)
    (propagate-call-arg-types (operator-of form) form :upper-type upper-type)
    (compute-call-type (operator-of form) form :upper-type upper-type))

  (:method ((form setf-application-form) &key upper-type)
    (propagate-assn-arg-types (operator-of form) form :upper-type upper-type)
    (compute-assn-type (operator-of form) form :upper-type upper-type)))

;;; Call & assignment definers

(def macro with-type-arg-walker-lexicals (&body code)
  `(macrolet ((recurse (form &key upper-type)
                `(propagate-c-types ,form :upper-type ,upper-type)))
     ,@code))

(def definer type-arg-walker (name args &body code)
  (make-builtin-handler-method
   ;; Builtin prototype + method name
   name args (if assn? 'propagate-assn-arg-types 'propagate-call-arg-types)
   ;; Body
   code
   :method-args `(&key ((:upper-type -upper-type-)))
   :top-decls `((ignorable -upper-type-))
   :prefix `(with-type-arg-walker-lexicals)))

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


