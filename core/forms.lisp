;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

(deflayer gpu-target)

;; Some ad-hoc attribute definitions

(def form-attribute-accessor form-c-type)
(def form-attribute-accessor gpu-variable
  :type (or gpu-variable null) :forms name-definition-form)
(def form-attribute-accessor assigned-to?
  :type boolean :forms (lexical-variable-binding-form
                        function-argument-form))
(def form-attribute-accessor c-name
  :forms (go-tag-form block-form))
(def form-attribute-accessor is-expression?
  :forms (implicit-progn-mixin if-form))
(def form-attribute-accessor is-merged-assignment?
  :forms setq-form)

;; A wrapper for global variables

(def (form-class :export nil) global-var-binding-form (name-definition-form)
  ((gpu-variable :type (or gpu-variable null))))

;; Forced cast

(def (macro e) cast (type body)
  `(the ,type ,body))

(def (form-class :export nil) cast-form (the-form)
  ())

(def (walker :in gpu-target) cast
  (with-form-object (cast 'cast-form -parent- :declared-type (second -form-))
    (setf (value-of cast) (recurse (third -form-) cast))))

(def unwalker cast-form (value)
  `(cast ,(declared-type-of -form-) ,(recurse value)))

;; Values

(def (form-class :export nil) values-form ()
  ((values :ast-link t)))

(def (walker :in gpu-target) values
  (with-form-object (values 'values-form -parent-)
    (setf (values-of values)
          (mapcar (lambda (x) (recurse x values)) (rest -form-)))))

(def unwalker values-form (values)
  `(values ,@(recurse-on-body values)))

;; Multiple value setq

(def (form-class :export nil) multiple-value-setq-form ()
  ((variables :ast-link t)
   (value :ast-link t)
   (is-merged-assignment? nil)))

(def (walker :in gpu-target) multiple-value-setq
  (with-form-object (setq 'multiple-value-setq-form -parent-)
    (setf (variables-of setq)
          (mapcar (lambda (x) (recurse x setq)) (second -form-)))
    (setf (value-of setq) (recurse (third -form-) setq))))

(def unwalker multiple-value-setq-form (variables value)
  `(multiple-value-setq ,(recurse-on-body variables) ,(recurse value)))

;; A SETF form.

(def (form-class :export nil) setf-application-form (application-form)
  ((value :ast-link t)))

(def (walker :in gpu-target) setf
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
            (values-form
             (change-class setf 'multiple-value-setq-form
                           :variables (values-of target))
             (adjust-parents (variables-of setf)))
            ((or free-application-form lexical-application-form)
             (change-class setf 'setf-application-form
                           :operator (operator-of target)
                           :arguments (arguments-of target))
             (adjust-parents (arguments-of setf)))
            (t
             (error "Not an lvalue form: ~S" (unwalk-form target)))))
        (setf (value-of setf) (recurse (third -form-) setf)))))

(def unwalker setf-application-form (value)
  `(setf (,(operator-of -form-)
           ,@(recurse-on-body (arguments-of -form-)))
         ,(recurse value)))

;; A verbatim inline C form

(defmacro inline-verbatim (&whole full (ret-type) &body code)
  (declare (ignore ret-type code))
  (error "This form cannot be used in ordinary lisp code: ~S" full))

(def (form-class :export nil) verbatim-code-form ()
  ((body)
   (form-c-type)
   (is-expression?)))

(def (walker :in gpu-target) inline-verbatim
  (destructuring-bind ((ret-type &key statement?) &rest code)
      (rest -form-)
    (with-form-object (vcode 'verbatim-code-form -parent-
                             :form-c-type ret-type
                             :is-expression? (not statement?))
      (setf (body-of vcode)
            (mapcar (lambda (form) (recurse form vcode)) code)))))

(def unwalker verbatim-code-form (body form-c-type is-expression?)
  `(inline-verbatim (,form-c-type :statement? ,(not is-expression?))
     ,@(recurse-on-body body)))

(def function parse-verbatim-flag (list-pos flags)
  (ecase (value-of (car list-pos))
    (:stmt (setf (getf flags :stmt) t))
    (:return?
     (setf (getf flags :return-nth) 0))
    (:return
     (setf (getf flags :return-nth) 0
           (getf flags :force-return) t))
    (:return-nth?
     (setf (getf flags :return-nth) (cadr list-pos)
           list-pos (cdr list-pos)))
    (:return-nth
     (setf (getf flags :return-nth) (cadr list-pos)
           list-pos (cdr list-pos)
           (getf flags :force-return) t)))
  (values list-pos flags))

(def macro do-verbatim-code ((item flags form) &body code)
  (with-unique-names (list-pos)
    `(let ((,flags nil))
       (do ((,list-pos (body-of ,form) (cdr ,list-pos)))
           ((null ,list-pos))
         (symbol-macrolet ((,item (car ,list-pos)))
           (if (and (typep ,item 'constant-form)
                    (keywordp (value-of ,item)))
               (setf (values ,list-pos ,flags)
                     (parse-verbatim-flag ,list-pos ,flags))
               (progn
                 ,@code
                 (setf ,flags nil))))))))

;;; AND & OR - parse them as ordinary function calls

(def (walker :in gpu-target) or
  (with-form-object (appl 'free-application-form -parent-
                          :operator 'or)
    (setf (arguments-of appl)
          (mapcar (lambda (f) (recurse f appl)) (rest -form-)))))

(def (walker :in gpu-target) and
  (with-form-object (appl 'free-application-form -parent-
                          :operator 'and)
    (setf (arguments-of appl)
          (mapcar (lambda (f) (recurse f appl)) (rest -form-)))))

;;; Macros

(def generic expand-gpu-macro (name form env)
  (:method (name form env)
    (declare (ignore name env))
    form))

(def layered-method hu.dwim.walker::walker-macroexpand-1 :in gpu-target ((form cons) &optional env)
  (let ((rvalue (expand-gpu-macro (first form) form env)))
    (if (eq rvalue form)
        (call-next-method)
        (values rvalue t))))

(def (definer e) gpu-macro (name args &body code)
  "Like compiler-macro, but for GPU code."
  (with-unique-names (whole env vname)
    (when (eq (first args) '&whole)
      (setf whole (second args)
            args (cddr args)))
    (awhen (position '&environment args)
      (setf env (nth (1+ it) args)
            args (append (subseq args 0 it)
                         (subseq args (+ it 2)))))
    `(defmethod expand-gpu-macro ((,vname (eql ',name)) ,whole ,env)
       (declare (ignore ,vname)
                (ignorable ,env))
       (destructuring-bind ,args (cdr ,whole)
         ,@code))))

;;;

(def function has-merged-assignment? (form)
  (atypecase (parent-of form)
    ((or multiple-value-setq-form setq-form)
     (is-merged-assignment? it))
    (t nil)))

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

(def function ensure-int-constant (obj)
  (aprog1 (ensure-constant obj)
    (unless (typep it 'integer)
      (error "Must be an integer constant: ~S" it))))

(def function constant-number-value (form)
  (and (typep form 'constant-form)
       (numberp (value-of form))
       (value-of form)))

(def function power-of-two (value)
  (if (and (integerp value)
           (> value 0)
           (= (logand value (1- value)) 0))
      (values (round (log value 2)) (1- value))))

(def function nil-constant? (obj)
  (and (typep obj 'constant-form)
       (eq (value-of obj) nil)))

(def function ensure-c-type-of (obj)
  (typecase obj
    (walked-form (form-c-type-of obj))
    (t obj)))

(def function make-lexical-binding (parent &key (name (make-symbol "_T")) initial-value c-type)
  (with-form-object (binding 'lexical-variable-binding-form parent
                             :name name :initial-value initial-value)
    (setf (form-c-type-of binding) (ensure-c-type-of c-type))))

(def function make-lexical-var (definition parent)
  (with-form-object (var `walked-lexical-variable-reference-form parent
                         :name (name-of definition)
                         :definition definition)
    (setf (form-c-type-of var)
          (form-c-type-of definition))))

(def function nop-form? (obj)
  (or (null obj)
      (nil-constant? obj)
      (and (typep obj 'values-form)
           (null (values-of obj)))))

(def function unknown-type? (type)
  (case type
    ((nil t number real) t)))

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
