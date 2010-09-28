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

(def function parse-global-type (type-spec &key form)
  (atypecase (parse-lisp-type type-spec :error-cb (curry #'gpu-code-error form))
    (gpu-array-type
     (unless (specific-type-p it)
       (gpu-code-error form "Insufficiently specific type spec: ~S" type-spec))
     (values (item-type-of it) (coerce (dimensions-of it) 'vector)))
    (t
     (unless (specific-type-p it)
       (gpu-code-error form "Type annotation is too abstract: ~S" type-spec))
     it)))

(def function parse-local-type (type-spec &key form)
  (atypecase (parse-lisp-type type-spec :error-cb (curry #'gpu-code-error form))
    (gpu-array-type
     (reintern-as-class it (default-pointer-type)))
    (t it)))

(def function get-local-var-decl-type (pdecls name)
  (aif (find-form-by-name name pdecls :type 'type-declaration-form)
       (values (parse-local-type (declared-type-of it) :form it)
               (declared-type-of it))))

;;; Type properties

(def function c-int-range (type)
  (if (typep type 'gpu-integer-type)
      (values (min-value-of type) (max-value-of type))))

(def (function i) <=limit (val limit)
  (or (null limit)
      (and val (<= val limit))))

(def (function i) >=limit (val limit)
  (or (null limit)
      (and val (>= val limit))))

(def layered-function can-promote-type? (src dest &key &allow-other-keys)
  (:documentation "Checks if a cast from src to dest is possible.")
  (:argument-precedence-order dest src)
  (:method ((src gpu-type) (dest gpu-type) &key)
    (eq src dest))
  (:method ((src gpu-type) (dest gpu-any-type) &key) t)
  (:method ((src gpu-no-type) (dest gpu-type) &key) t)
  (:method ((src gpu-number-type) (dest gpu-number-type) &key)
    (cond ((and (>=limit (min-value-of src) (min-value-of dest))
                (<=limit (max-value-of src) (max-value-of dest))) t)
          (t :warn)))
  (:method ((src gpu-integer-type) (dest gpu-native-integer-type) &key silent-signed?)
    (cond ((and silent-signed?
                (let ((size (ash 1 (1- (* 8 (native-type-byte-size dest))))))
                  (and (>=limit (min-value-of src) (- size))
                       (<=limit (max-value-of src) (1- (* 2 size))))))
           t)
          (t (call-next-method))))
  (:method ((src gpu-float-type) (dest gpu-integer-type) &key)
    :warn)
  (:method ((src gpu-double-float-type) (dest gpu-single-float-type) &key)
    :warn)
  (:method ((src gpu-integer-type) (dest gpu-boolean-type) &key)
    t)
  (:method ((src gpu-values-type) (dest gpu-values-type) &rest flags)
    (let* ((sv (values-of src))
           (dv (values-of dest))
           (match (mapcar (lambda (s d)
                            (apply #'can-promote-type? s d flags))
                          sv dv)))
      (cond ((< (length sv) (length dv))
             nil)
            ((some #'null match)
             nil)
            ((or (> (length sv) (length dv))
                 (member :warn match))
             :warn)
            (t t)))))

(def function fc-or-null (fun &rest args)
  (if (some #'null args) nil
      (reduce fun args)))

(def function fc-non-null (fun &rest args)
  (let ((args (remove-if-not #'identity args)))
    (if args (reduce fun args))))

(def macro range-union-call ((type1 type2) call-form)
  "Appends :min-value and :max-value parameters to the call."
  (append call-form
          `(:min-value
            (fc-or-null #'min (min-value-of ,type1) (min-value-of ,type2))
            :max-value
            (fc-or-null #'max (max-value-of ,type1) (max-value-of ,type2)))))

(def function make-foreign-type-with-limits (type1 type2 rtype with-limits?)
  (if with-limits?
      (range-union-call (type1 type2) (make-foreign-gpu-type rtype))
      (make-foreign-gpu-type rtype)))

(def layered-function join-arithmetic-types (type1 type2 &key with-limits?)
  (:documentation "Computes a common promoted type for arithmetic ops.")
  (:method ((type1 gpu-type) (type2 gpu-type) &key with-limits? form)
    (declare (ignore with-limits?))
    (gpu-code-error form "Cannot find an arithmetic supertype of ~A and ~A" type1 type2))
  (:method ((type1 gpu-number-type) (type2 gpu-number-type) &key with-limits?)
    (if with-limits?
        (range-union-call (type1 type2) (make-instance 'gpu-number-type))
        (make-instance 'gpu-number-type)))
  (:method ((type1 gpu-number-type) (type2 gpu-float-type) &key with-limits?)
    (make-foreign-type-with-limits type1 type2 :float with-limits?))
  (:method ((type1 gpu-number-type) (type2 gpu-double-float-type) &key with-limits?)
    (make-foreign-type-with-limits type1 type2 :double with-limits?))
  (:method ((type1 gpu-float-type) (type2 gpu-number-type) &key with-limits?)
    (make-foreign-type-with-limits type1 type2 :float with-limits?))
  (:method ((type1 gpu-double-float-type) (type2 gpu-number-type) &key with-limits?)
    (make-foreign-type-with-limits type1 type2 :double with-limits?))
  (:method ((type1 gpu-integer-type) (type2 gpu-integer-type) &key with-limits?)
    (if with-limits?
        (range-union-call (type1 type2) (make-instance 'gpu-number-type))
        (make-instance 'gpu-integer-type)))
  (:method ((type1 gpu-native-integer-type) (type2 gpu-native-integer-type) &key with-limits?)
    (bind ((rtype (svref +gpu-integer-foreign-ids+
                         (max (position (foreign-type-of type1) +gpu-integer-foreign-ids+)
                              (position (foreign-type-of type2) +gpu-integer-foreign-ids+)))))
      (make-foreign-type-with-limits type1 type2 rtype with-limits?))))

(def layered-function promote-type-to-variable (type)
  (:documentation "Upgrades the type of an init expressison to one of a variable.")
  (:method ((type gpu-type)) type)
  (:method ((type gpu-native-number-type))
    (make-instance (class-of type))))

(def layered-function promote-type-to-arithmetic (type)
  (:documentation "Upgrades the type to one that may be an arithmetic result. E.g. forces ints to >=int32")
  (:method ((type gpu-type)) type)
  (:method ((type gpu-native-integer-type))
    (if (member (foreign-type-of type) '(:int8 :uint8 :int16 :uint16))
        (reintern-as-class type 'gpu-int32-type)
        (call-next-method))))

(def layered-function promote-type-to-float (type)
  (:documentation "Upgrades the type to one that may be a float function. E.g. forces ints to >=int32")
  (:method ((type gpu-type)) type)
  (:method ((type gpu-number-type))
    (reintern-as-class type 'gpu-single-float-type))
  (:method ((type gpu-single-float-type)) type)
  (:method ((type gpu-double-float-type)) type))

(def macro range-intersection-call ((type1 type2) call-form)
  "Appends :min-value and :max-value parameters to the call."
  (append call-form
          `(:min-value
            (fc-non-null #'max (min-value-of ,type1) (min-value-of ,type2))
            :max-value
            (fc-non-null #'min (max-value-of ,type1) (max-value-of ,type2)))))

(def layered-function compute-casted-type (cast-type src-type &key)
  (:documentation "Derives a concrete resulting type of a cast.")
  (:method ((cast-type gpu-type) (src-type gpu-type) &key) cast-type)
  (:method ((cast-type gpu-any-type) (src-type gpu-type) &key) src-type)
  (:method ((cast-type null) (src-type gpu-type) &key) src-type)
  (:method ((cast-type gpu-number-type) (src-type gpu-number-type) &key)
    (range-intersection-call (cast-type src-type)
                             (make-instance (class-of src-type))))
  (:method ((cast-type gpu-float-type) (src-type gpu-number-type) &key)
    (range-intersection-call (cast-type src-type)
                             (make-foreign-gpu-type (if (eq (foreign-type-of src-type) :double)
                                                        :double :float))))
  (:method ((cast-type gpu-native-float-type) (src-type gpu-number-type) &key)
    (range-intersection-call (cast-type src-type)
                             (make-instance (class-of cast-type))))
  (:method ((cast-type gpu-native-integer-type) (src-type gpu-number-type) &key)
    (range-intersection-call (cast-type src-type)
                             (make-instance (class-of cast-type)))))

;;; Alignment helpers

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
             (make-instance (default-pointer-type) :item-type (item-type-of it))
             (item-type-of it))
         (form-c-type-of defn))))

(def function verify-cast (src-type-or-form dest-type form &key
                                            prefix (warn? t) error-on-warn?
                                            allow (silent-signed? t))
  (let* ((stype (atypecase src-type-or-form
                  (constant-form (propagate-c-types it :upper-type dest-type))
                  (walked-form (form-c-type-of it))
                  (t it)))
         (status (or (typecase allow
                       (cons (member stype allow :test #'eq))
                       (function (funcall allow stype)))
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

(def (function i) array-c-type? (type)
  (or (typep type 'gpu-array-type)
      (typep type 'gpu-pointer-type)))

(def function verify-array (arr)
  (let ((arr/type (form-c-type-of arr)))
    (unless (array-c-type? arr/type)
      (gpu-code-error arr "Must be an array."))))

(def function verify-array-var (arr)
  (unless (typep arr 'walked-lexical-variable-reference-form)
    (gpu-code-error arr "Must be a variable."))
  (verify-array arr))

(def function wrap-values-type (types)
  (if (or (null types) (cdr types))
      (make-instance 'gpu-values-type :values types)
      (car types)))

(def function unwrap-values-type (type)
  (if (typep type 'gpu-values-type) (values-of type) (list type)))

(def function coerce-to-void (upper-type value-type)
  (if (eq upper-type +gpu-void-type+) +gpu-void-type+ value-type))

;;; Utilities for builtins

(def function ensure-common-type (arg-types form &key
                                            prefix (silent-signed? t) with-limits?
                                            (promotion #'identity))
  (aprog1 (funcall promotion
                   (reduce (lambda (a b) (join-arithmetic-types a b :form form :with-limits? with-limits?))
                           (mapcar #'ensure-c-type-of arg-types)))
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
      (unless (specific-type-p rtype)
        (gpu-code-error form "Cannot determine type."))
      (setf (form-c-type-of form) rtype)))

  (:method ((form walked-form) &key upper-type)
    (declare (ignore upper-type))
    (gpu-code-error form "This form is not supported in GPU code."))

  ;; Type cast
  (:method ((form the-form) &key upper-type)
    (declare (ignore upper-type))
    (let* ((cast-type (parse-local-type (declared-type-of form) :form form))
           (arg-type (propagate-c-types (value-of form) :upper-type cast-type))
           (casted-type (compute-casted-type cast-type arg-type)))
      (unless (eq casted-type arg-type)
        (verify-cast arg-type casted-type form :warn? nil)
        (change-class form 'cast-form))
      casted-type))

  ;; Variable reference
  (:method ((form walked-lexical-variable-reference-form) &key upper-type)
    (declare (ignore upper-type))
    (gpu-var-ref-type form))

  ;; Constants
  (:method ((form constant-form) &key upper-type)
    (if (typep (value-of form) 'character)
        (setf (value-of form) (char-code (value-of form))
              upper-type (compute-casted-type upper-type +gpu-uint8-type+)))
    (atypecase (value-of form)
      (integer
       (let ((vt (make-gpu-integer-from-range it it)))
         (cond ((and upper-type (eq (can-promote-type? vt upper-type) t))
                (compute-casted-type upper-type vt))
               ((not (specific-type-p vt))
                (gpu-code-error form "Integer value ~A is too big." it))
               (t vt))))
      (single-float
       (compute-casted-type upper-type
                            (make-instance 'gpu-single-float-type :min-value it :max-value it)))
      (double-float
       (make-instance 'gpu-double-float-type :min-value it :max-value it))
      (keyword
       +gpu-keyword-type+)
      (boolean
       (if it +gpu-boolean-type+
           (coerce-to-void upper-type +gpu-boolean-type+)))
      (t (gpu-code-error form "Cannot use constant ~S in C code." it))))

  ;; Assignment
  (:method ((form setq-form) &key upper-type)
    (let* ((target-type (gpu-var-ref-type (variable-of form)))
           (val-type (propagate-c-types (value-of form)
                                        :upper-type target-type)))
      (when (array-c-type? target-type)
        (gpu-code-error form "Assignment of arrays is not supported."))
      (verify-cast val-type target-type (variable-of form)
                   :prefix "assignment to"
                   :silent-signed? nil)
      (coerce-to-void upper-type target-type)))

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
              (when (array-c-type? var-type)
                (gpu-code-error form "Assignment of arrays is not supported."))
              (verify-cast val-type var-type var
                           :prefix "assignment to"
                           :silent-signed? nil)))
      (coerce-to-void upper-type (first type-list))))

  ;; Value group
  (:method ((form values-form) &key upper-type)
    (let ((args (values-of form))
          (upper-types (unwrap-values-type upper-type)))
      (cond ((typep upper-type 'gpu-values-type)
             (unless (>= (length args) (length upper-types))
               (gpu-code-error form "Expecting ~A values, found only ~A."
                               (length upper-types)
                               (length args)))
             (loop for arg in args
                for rtype = upper-types then (cdr rtype)
                do (propagate-c-types arg :upper-type (if rtype (car rtype) +gpu-void-type+))))
            (args
             (propagate-c-types (first args) :upper-type upper-type)
             (dolist (arg (rest args))
               (propagate-c-types arg :upper-type (if upper-type +gpu-void-type+)))))
      (let ((child-types (mapcar #'form-c-type-of args)))
        (wrap-values-type (if upper-type
                              (subseq child-types 0 (length upper-types))
                              child-types)))))

  ;; Verbatim code
  (:method ((form verbatim-code-form) &key upper-type)
    (declare (ignore upper-type))
    (do-verbatim-code (item flags form :flatten? t)
      (typecase item
        ((or string character))
        (t
         (let* ((upper (aif (getf flags :type)
                            (parse-lisp-type it :error-cb (curry #'gpu-code-error form))))
                (rtype (propagate-c-types item :upper-type upper)))
           (when upper
             (verify-cast rtype upper form :prefix "inline argument"
                          :silent-signed? nil))))))
    (form-c-type-of form))

  ;; Blocks
  (:method ((form implicit-progn-mixin) &key upper-type)
    (if (null (body-of form))
        +gpu-void-type+
        (progn
          (dolist (item (butlast (body-of form)))
            (propagate-c-types item :upper-type +gpu-void-type+))
          (let ((rtype (propagate-c-types (car (last (body-of form)))
                                          :upper-type upper-type)))
            (coerce-to-void upper-type rtype)))))

  (:method ((form implicit-progn-with-declarations-mixin) &key upper-type)
    (declare (ignore upper-type))
    (with-optimize-context (form)
      (call-next-method)))

  (:method ((form tagbody-form) &key upper-type)
    (declare (ignore upper-type))
    (dolist (item (body-of form))
      (propagate-c-types item :upper-type +gpu-void-type+))
    +gpu-void-type+)

  (:method ((form go-tag-form) &key upper-type)
    (declare (ignore form upper-type))
    +gpu-void-type+)

  (:method ((form go-form) &key upper-type)
    (declare (ignore upper-type))
    (unless (tag-of form)
      (gpu-code-error form "Unknown GO tag: ~S" (name-of form)))
    +gpu-void-type+)

  (:method ((form block-form) &key upper-type)
    (setf (form-c-type-of form) upper-type)
    (let ((inner-t (call-next-method))
          (outer-t (form-c-type-of form)))
      (if (null outer-t)
          (setf outer-t inner-t)
          (unless (eq outer-t +gpu-void-type+)
            (verify-cast inner-t outer-t form :allow '(#.+gpu-void-type+))))
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
            (unless (eq btype +gpu-void-type+)
              (verify-cast vtype btype form)))
        +gpu-void-type+)))

  (:method ((form if-form) &key upper-type)
    (verify-cast (propagate-c-types (condition-of form) :upper-type +gpu-boolean-type+)
                 +gpu-boolean-type+ form :prefix "condition of")
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
      (cond ;; nil means no initialization
        ((and (not (member decl-type '(nil #.+gpu-boolean-type+ #.+gpu-any-type+)))
              (nil-constant? init-form))
         (setf (initial-value-of form) nil
               init-form nil))
        (t
         (setf decl-type
               (compute-casted-type decl-type
                                    (if (assigned-to? form)
                                        (promote-type-to-variable init-type)
                                        init-type)))
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
        (propagate-c-types item :upper-type +gpu-void-type+))))

  (:method ((form unwind-protect-form) &key upper-type)
    (aprog1
        (propagate-c-types (protected-form-of form) :upper-type upper-type)
      (dolist (item (cleanup-form-of form))
        (propagate-c-types item :upper-type +gpu-void-type+))))

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


