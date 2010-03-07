;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file implements function inlining and special
;;; variable reference extraction.
;;;

(in-package :cl-gpu)

;;; Function inlining utils

(def function remove-func-decls (decls funcs)
  (reduce (lambda (decls func)
            (remove-form-by-name decls (name-of func) :type 'function-declaration-form))
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
               and val-list = opt-vals then (cdr val-list)
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

(def function join-call-arguments (arg-map)
  (mapcar (lambda (item)
            (aprog1 (car item)
              (change-class it 'lexical-variable-binding-form)
              (setf (initial-value-of it) (cdr item))
              (when (cdr item)
                (setf (parent-of (cdr item)) it))))
          arg-map))

(def function find-function-definition (parent arg &key force-copy?)
  (atypecase arg
    (walked-lexical-function-object-form
     (values (deep-copy-ast (definition-of it) :parent parent)
             (definition-of it)))
    (walked-lexical-variable-reference-form
     (find-function-definition parent
                               (initial-value-of (definition-of it))
                               :force-copy? t))
    (lambda-function-form
     (values (if force-copy?
                 (deep-copy-ast it :parent parent)
                 (prog1 it
                   (setf (parent-of it) parent)))
             it))))

;;; The inliner core

(def layered-function inline-functions (form)
  (:method ((form t))
    (do-ast-links (subform form :rewrite t)
      (setf subform (inline-functions subform)))
    form))

;;; Inline stack check

(defvar *function-inline-stack* nil)

(def function check-non-recursive (func form)
  (when (member func *function-inline-stack*)
    (error "Recursive call detected: ~S" (unwalk-form form))))

(def macro with-inline-stack (objects &body code)
  `(let ((*function-inline-stack*
          (list* ,@objects *function-inline-stack*)))
     ,@code))

;;; Simple call inliner

(def function inline-one-call (form func args &key no-copy?)
  (check-non-recursive func form)
  (bind ((new-func (if no-copy? func
                       (deep-copy-ast func :parent (parent-of form))))
         (arg-map (match-call-arguments new-func args)))
    (change-class new-func (if (typep new-func 'block-lambda-function-form)
                               'block-let-form 'let*-form))
    (setf (bindings-of new-func) (join-call-arguments arg-map))
    (with-inline-stack (new-func func)
      (inline-functions new-func))))

;;; Form inline rules

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
    (inline-one-call form (definition-of form) (arguments-of form)))

  (:method ((form lambda-application-form))
    (inline-one-call form (operator-of form) (arguments-of form) :no-copy? t)))

;;; Multiple-value-call handler

(def layered-method inline-functions ((form multiple-value-call-form))
  (bind (((:values defn src-defn)
          (find-function-definition (parent-of form) (function-designator-of form))))
    (if (null defn)
        (call-next-method)
        (bind ((args (loop for arg in (bindings-of defn)
                        when (or (typep arg 'optional-function-argument-form)
                                 (typep arg 'required-function-argument-form))
                        collect arg
                        else do (typecase arg
                                  (rest-function-argument-form
                                   (unless (find-form-by-name (name-of arg) (declarations-of defn)
                                                              :type 'variable-ignorable-declaration-form)
                                     (error "The rest argument must be ignored in multiple-value-call"))
                                   (remove-form-by-name! (declarations-of defn) (name-of arg)
                                                         :type 'variable-declaration-form))
                                  (t
                                   (error "Unsupported arg type in multiple-value-call inline: ~S"
                                          arg))))))
          (check-non-recursive src-defn form)
          (change-class defn (if (typep defn 'block-lambda-function-form)
                                 'block-multiple-value-bind-form 'multiple-value-bind-form))
          (setf (bindings-of defn) (join-call-arguments (mapcar #'list args)))
          (setf (value-of defn) (if (null (cdr (arguments-of form)))
                                    (car (arguments-of form))
                                    (error "Only one argument allowed in multiple-value-call")))
          (adjust-parents (value-of defn))
          (with-inline-stack (defn src-defn)
            (inline-functions defn))))))

;;; Misc cleanup

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

;;; Special variable extraction.

(defparameter *cur-special-bindings* nil)

(def function walk-tree-with-specials (tree visitor)
  "Recurse into the tree, tracking the active special bindings."
  (labels ((recurse-block (form)
             (dolist (decl (declarations-of form))
               (recurse form 'declarations decl))
             (dolist (item (body-of form))
               (recurse form 'body item))
             (funcall visitor form))
           (recurse-let (form)
             (let ((new-bindings *cur-special-bindings*))
               (dolist (var (bindings-of form))
                 (recurse form 'bindings var)
                 (when (special-binding? var)
                   (push (cons (name-of var) var) new-bindings)))
               (let ((*cur-special-bindings* new-bindings))
                 (recurse-block form))))
           (recurse (parent field form)
             (declare (ignore parent field))
             (typecase form
               (let*-form
                (let ((*cur-special-bindings* *cur-special-bindings*))
                  (dolist (var (bindings-of form))
                    (recurse form 'bindings var)
                    (when (special-binding? var)
                      (push (cons (name-of var) var) *cur-special-bindings*)))
                  (recurse-block form)))
               (let-form
                (recurse-let form))
               (multiple-value-bind-form
                (recurse form 'value (value-of form))
                (recurse-let form))
               (t
                (enum-ast-links form #'recurse)
                (funcall visitor form)))))
    (declare (dynamic-extent #'recurse #'recurse-block))
    (recurse nil nil tree)))

(def function pull-global-refs (tree)
  "Convert all special refs to &aux arguments, and bindings to lexical vars."
  (with-accessors ((top-args bindings-of)
                   (top-decls declarations-of)) tree
    (let ((arg-cache nil))
      (labels ((close-special-var (form)
                 (when (typep (parent-of form)
                              '(or setq-form multiple-value-setq-form))
                   (warn "Write to special variable ~S has been localized."
                         (name-of form)))
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
                                 :name (name-of arg-def) :definition arg-def)))
               (visitor (form)
                 (typecase form
                   (special-variable-reference-form
                    (aif (assoc (name-of form) *cur-special-bindings*)
                         ;; Once the functions are completely inlined,
                         ;; dynamic scope is equal to lexical, and
                         ;; specials can be converted to normal vars.
                         (change-class form 'walked-lexical-variable-reference-form
                                       :definition (cdr it))
                         (close-special-var form)))
                   (unwalked-lexical-variable-reference-form
                    (error "Closing over lexical variables is not supported: ~S"
                           (unwalk-form form)))
                   ((or lexical-variable-binder-form multiple-value-bind-form)
                    (dolist (var (bindings-of form))
                      (setf (special-binding? var) nil))))))
        (dolist (item (body-of tree))
          (walk-tree-with-specials item #'visitor))))))

(def function preprocess-tree (tree global-vars)
  ;; Inline functions
  (setf tree (inline-all-functions tree))
  ;; Convert global var refs to &aux args.
  (pull-global-refs tree)
  ;; Mark bindings that are destructively assigned to.
  (mark-mutated-vars tree)
  ;; Collect the binding usage lists.
  (annotate-binding-usage (list* tree (mapcar #'form-of global-vars)))
  tree)

