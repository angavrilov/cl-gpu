;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file implements function inlining and global
;;; variable reference extraction.
;;;

(in-package :cl-gpu)

;;; Function inlining

(def function remove-func-decls (decls funcs)
  (reduce (lambda (decls func)
            (remove-if (lambda (decl) (and (typep decl 'function-declaration-form)
                                      (eq (name-of decl) (name-of func))))
                       decls))
          funcs :initial-value decls))

(def function match-call-arguments (arg-mixin arg-vals)
  "Create a mapping of function arguments to their values."
  (bind (((:values rq-args opt-args key-args aux-args)
          (loop for arg in (bindings-of arg-mixin)
             when (typep arg 'required-function-argument-form)
             collect arg into rq-args
             else when (typep arg 'optional-function-argument-form)
             collect arg into opt-args
             else when (typep arg 'keyword-function-argument-form)
             collect (cons arg (default-value-of arg)) into key-args
             else when (typep arg 'auxiliary-function-argument-form)
             collect (cons arg (default-value-of arg)) into aux-args
             else do (error "Invalid argument type: ~S" arg)
             finally (return (values rq-args opt-args key-args aux-args))))
         (val-cnt (length arg-vals))
         (rq-cnt (length rq-args))
         (opt-cnt (length opt-args))
         (nkey-cnt (min val-cnt (+ rq-cnt opt-cnt)))
         (rq-vals (subseq arg-vals 0 rq-cnt))
         (opt-vals (subseq arg-vals rq-cnt nkey-cnt))
         (rest-vals (subseq arg-vals nkey-cnt)))
    (append (loop for arg in rq-args and val in rq-vals
               collect (cons arg val))
            (loop for arg in opt-args
               and val-list = opt-vals then #'cdr
               for val = (or (car val-list) (default-value-of arg))
               collect (cons arg val))
            (loop with res-args = nil
               for (key val) on rest-vals by #'cddr
               for item = (assoc (if (typep key 'constant-form)
                                     (value-of key)
                                     (error "Must be a keyword constant: ~S" key))
                                 key-args :key #'effective-keyword-name-of)
               when item
               do (progn
                    (setf (cdr item) val)
                    (push item res-args)
                    (deletef key-args item))
               else do (if (allow-other-keys? arg-mixin)
                           (push (cons (make-lexical-binding arg-mixin) val) res-args)
                           (error "Unknown key argument in call to ~S: ~S ~S"
                                  (if (typep arg-mixin 'named-walked-form)
                                      (name-of arg-mixin) 'lambda)
                                  (value-of key) (unwalk-form val)))
               finally (progn
                         (return (nconc (nreverse res-args) key-args))))
            aux-args)))

(def layered-function inline-functions (form)
  (:method ((form t))
    (do-ast-links (subform form :rewrite t)
      (setf subform (inline-functions subform)))
    form))

(defvar *function-inline-stack* nil)

(def function inline-one-call (parent func args)
  (when (member func *function-inline-stack*)
    (error "Recursive call detected: ~S" (unwalk-form func)))
  (bind ((new-func (deep-copy-ast func :parent parent))
         (arg-map (match-call-arguments new-func args)))
    (change-class new-func (if (typep new-func 'block-lambda-function-form)
                               'block-let-form 'let*-form))
    (setf (bindings-of new-func)
          (mapcar (lambda (item)
                    (change-class (car item) 'lexical-variable-binding-form)
                    (setf (initial-value-of (car item)) (cdr item)
                          (parent-of (cdr item)) (car item)))
                  arg-map))
    (let ((*function-inline-stack*
           (list* new-func func *function-inline-stack*)))
      (inline-functions new-func))))

(def layered-methods inline-functions
  ;; Unlink local function bindings
  (:method ((form function-binding-form))
    (inline-functions
     (wrap-body-in-form (parent-of form) (body-of form)
                        ;; TODO: extract the useful decls if any
                        :declarations (remove-func-decls (declarations-of form)
                                                         (bindings-of form)))))

  ;; Flatten macros
  (:method ((form macrolet-form))
    (inline-functions
     (wrap-body-in-form (parent-of form) (body-of form)
                        :declarations (declarations-of form))))

  (:method ((form symbol-macrolet-form))
    (inline-functions
     (wrap-body-in-form (parent-of form) (body-of form)
                        :declarations (declarations-of form))))

  ;; Don't recurse into lexical functions
  (:method ((form function-form))
    (if (typep form 'function-definition-form)
        (call-next-method)
        form))

  ;; Reject random calls
  (:method ((form lexical-application-form))
    (error "Unwalked lexical function calls not supported: ~S"
           (unwalk-form form)))

  ;; Inline specific call variants
  (:method ((form walked-lexical-application-form))
    (inline-one-call (parent-of form) (definition-of form) (arguments-of form)))

  (:method ((form lambda-application-form))
    (inline-one-call (parent-of form) (operator-of form) (arguments-of form))))

(def function remove-functions (form)
  "Remove all remaining function bodies."
  (rewrite-ast form (lambda (parent field form)
                      (declare (ignore field))
                      (if (and (typep form 'function-form)
                               (not (typep form 'function-definition-form)))
                          (with-form-object (obj 'function-form parent))
                          form))))

(def function inline-all-functions (form)
  (remove-functions (inline-functions form)))

;;; Global extraction

(def function pull-global-refs (tree)
  "Convert all special refs to &aux arguments."
  (with-accessors ((top-args bindings-of)
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
                                                               :name new-id)
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
                         (t form))))
                 tree)))))

(def function preprocess-tree (tree global-vars)
  ;; Inline functions
  (setf tree (inline-all-functions tree))
  ;; Mark bindings that are destructively assigned to.
  ;; This also checks absence of assignments to special vars.
  (mark-mutated-vars tree)
  ;; Convert global var refs to &aux args.
  (pull-global-refs tree)
  ;; Collect the binding usage lists.
  (annotate-binding-usage (list* tree (mapcar #'form-of global-vars)))
  tree)

