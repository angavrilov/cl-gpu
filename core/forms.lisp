;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

(deflayer gpu-target)

;; Array functions - they make sense on the GPU side.

(def (function ei) array-raw-extent (arr)
  "Returns the raw size of a pitched array."
  (array-total-size arr))

(def (function ei) array-raw-stride (arr idx)
  "Returns the raw stride of a pitched array."
  (loop with prod = 1
     for i from (1+ idx) below (array-rank arr)
     do (setf prod (* prod (array-dimension arr i)))
     finally (return prod)))

(def (function ei) raw-aref (arr index)
  "Access a pitched array with a raw index."
  (row-major-aref arr index))

(def (function ei) (setf raw-aref) (value arr index)
  "Access a pitched array with a raw index."
  (setf (row-major-aref arr index) value))

;; Some ad-hoc attribute definitions

(def form-attribute-accessor result-type)
(def form-attribute-accessor gpu-variable
  :type (or gpu-variable null) :forms name-definition-form)
(def form-attribute-accessor assigned-to?
  :type boolean :forms (lexical-variable-binding-form
                        function-argument-form))

;; A wrapper for global variables

(def (form-class :export nil) global-var-binding-form (name-definition-form)
  ((gpu-variable :type (or gpu-variable null))))

;; An expression progn form (C comma operator)

(def (form-class :export nil) expr-progn-form (progn-form)
  ())

;; A SETF form.

(def (form-class :export nil) setf-application-form (application-form)
  ((value :ast-link t)))

(def (walker-method :in gpu-target) setf
  (if (> (length -form-) 3)
      (recurse `(progn
                  ,@(loop
                       :for (name value) :on (cdr -form-) :by #'cddr
                       :collect `(setf ,name ,value))))
      (with-form-object (setf 'setq-form -parent-)
        (let ((target (recurse (second -form-) setf)))
          (typecase target
            (variable-reference-form
             (setf (variable-of setf) target))
            ((or free-application-form lexical-application-form)
             (change-class setf 'setf-application-form
                           :operator (operator-of target)
                           :arguments (arguments-of target))
             (dolist (arg (arguments-of target))
               (setf (parent-of arg) setf)))
            (t
             (error "Not an lvalue form: ~S" (second -form-)))))
        (setf (value-of setf) (recurse (third -form-) setf)))))

(def unwalker setf-application-form (value)
  `(setf (,(operator-of -form-)
           ,@(recurse-on-body (arguments-of -form-)))
         ,(recurse value)))

;; A verbatim inline C form

(defmacro inline-verbatim (&whole full (ret-type) &body code)
  (declare (ignore ret-type code))
  (error "This form cannot be used in ordinary lisp code: ~S" full))

(def (form-class :export nil) verbatim-code-form (implicit-progn-mixin)
  ((result-type)))

(def (walker-method :in gpu-target) inline-verbatim
  (destructuring-bind ((ret-type) &rest code) (rest -form-)
    (with-form-object (vcode 'verbatim-code-form -parent-
                             :result-type ret-type)
      (setf (body-of vcode)
            (mapcar (lambda (form) (recurse form vcode)) code)))))

(def unwalker verbatim-code-form (result-type)
  `(inline-verbatim (,result-type)
     ,@(recurse-on-body (body-of -form-))))

;;;

(def function ensure-gpu-var (ref)
  (unless (typep ref 'walked-lexical-variable-reference-form)
    (error "Must be a local variable reference: ~S" (unwalk-form ref)))
  (let ((defn (definition-of ref)))
    (unless defn
      (error "Undefined variable reference: ~S" (unwalk-form ref)))
    (or (gpu-variable-of defn)
        (error "Unallocated variable reference: ~S" (unwalk-form ref)))))

(def function ensure-constant (obj)
  (unless (typep obj 'constant-form)
    (error "Must be a constant: ~S" (unwalk-form obj)))
  (value-of obj))

(def function unknown-type? (type)
  (or (null type)
      (eql type t)))

(def function pull-global-refs (tree)
  "Convert all special refs to &aux arguments."
  (with-accessors ((top-args arguments-of)
                   (top-decls declarations-of)) tree
    (let ((arg-cache nil))
      (flet ((close-special-var (form)
               (let ((arg-def
                      (or (cdr (assoc (name-of form) arg-cache))
                          ;; Add a new aux argument
                          (let ((name (name-of form))
                                (type (or (declared-type-of form) t)))
                            ;; Use type declarations from top level
                            (awhen (find-form-by-name name top-decls
                                                      :type 'type-declaration-form)
                              (unless (subtypep (declared-type-of it) type)
                                (error "Bad type specialization of ~S: ~S is not subtype of ~S"
                                       name (declared-type-of it) type))
                              (setf type (declared-type-of it)))
                            ;; Use type declarations via (the ... *foo*)
                            (when (and (unknown-type? type)
                                       (typep (parent-of form) 'the-form))
                              (setf type (declared-type-of (parent-of form))))
                            ;; Assert that the type is known
                            (when (unknown-type? type)
                              (error "Unknown global variable type: ~S" name))
                            ;; Create the argument
                            (let* ((new-id (make-symbol (string name)))
                                   (arg (with-form-object (arg 'auxiliary-function-argument-form tree
                                                               :name new-id :usages nil)
                                          (setf (default-value-of arg)
                                                (walk-form `(locally (declare (special ,name))
                                                              ,name)
                                                           :parent arg)))))
                              (nconcf top-args (list arg))
                              (push (cons name arg) arg-cache)
                              (with-form-object (decl 'type-declaration-form tree
                                                      :name new-id :declared-type type)
                                (nconcf top-decls (list decl)))
                              arg)))))
                 (change-class form 'walked-lexical-variable-reference-form
                               :name (name-of arg-def) :definition arg-def)
                 (push form (usages-of arg-def))
                 nil)))
        (map-ast (lambda (form)
                   (if (member form top-args)
                       nil ; Don't process top args to avoid cycles
                       (typecase form
                         (special-variable-reference-form
                          (close-special-var form))
                         (unwalked-lexical-variable-reference-form
                          (error "Closing over lexical variables is not supported: ~S"
                                 (unwalk-form form)))
                         (lexical-variable-binding-form
                          (when (special-binding? form)
                            (error "Binding special variables is not supported: ~S"
                                   (unwalk-form form)))
                          form)
                         (setq-form
                          (unless (typep (variable-of form) 'walked-lexical-variable-reference-form)
                            (error "Setting non-lexical variables is not supported: ~S"
                                   (unwalk-form form)))
                          form)
                         (t form))))
                 tree)))))

(def function preprocess-tree (tree global-vars)
  (annotate-binding-usage (list* tree
                                 (mapcar #'form-of global-vars)))
  (pull-global-refs tree)
  tree)
