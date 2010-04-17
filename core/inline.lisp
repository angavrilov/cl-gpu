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

(def function match-call-arguments (arg-mixin arg-vals &key form)
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
             else do (gpu-code-error arg "Invalid argument type.")
             finally (return (values rq-args opt-args key-args aux-args))))
         (val-cnt (length arg-vals))
         (rq-cnt (length rq-args))
         (opt-cnt (length opt-args))
         (nkey-cnt (min val-cnt (+ rq-cnt opt-cnt)))
         (rq-vals (if (< val-cnt rq-cnt)
                      (gpu-code-error form "Too few arguments in call, required:~{ ~A~}"
                                      (mapcar #'name-of rq-args))
                      (subseq arg-vals 0 rq-cnt)))
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
                                     (gpu-code-error key "Must be a keyword constant."))
                                 key-args :key #'effective-keyword-name-of)
               when item
               do (progn
                    (setf (cdr item) val)
                    (push item res-args)
                    (deletef key-args item))
               else do (if (allow-other-keys? arg-mixin)
                           (push (cons (make-lexical-binding arg-mixin) val) res-args)
                           (gpu-code-error val "Unknown key argument in call to ~S: ~S"
                                           (if (typep arg-mixin 'named-walked-form)
                                               (name-of arg-mixin) 'lambda)
                                           (value-of key)))
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
    (gpu-code-error form "Recursive call detected.")))

(def macro with-inline-stack (objects &body code)
  `(let ((*function-inline-stack*
          (list* ,@objects *function-inline-stack*)))
     ,@code))

;;; Simple call inliner

(def function inline-one-call (form func args &key no-copy?)
  (check-non-recursive func form)
  (bind ((new-func (if no-copy? func
                       (deep-copy-ast func :parent (parent-of form))))
         (arg-map (match-call-arguments new-func args :form form)))
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
    (gpu-code-error form "Unwalked lexical function calls not supported."))

  ;; Inline specific call variants
  (:method ((form walked-lexical-application-form))
    (inline-one-call form (definition-of form) (arguments-of form)))

  (:method ((form free-application-form))
    (aif (symbol-gpu-function (operator-of form))
         (inline-one-call form (form-of it) (arguments-of form))
         (call-next-method)))

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
                                     (gpu-code-error form "The rest argument must be ignored in multiple-value-call."))
                                   (remove-form-by-name! (declarations-of defn) (name-of arg)
                                                         :type 'variable-declaration-form))
                                  (t
                                   (gpu-code-error arg "Unsupported arg type in multiple-value-call inline."))))))
          (check-non-recursive src-defn form)
          (change-class defn (if (typep defn 'block-lambda-function-form)
                                 'block-multiple-value-bind-form 'multiple-value-bind-form))
          (setf (bindings-of defn) (join-call-arguments (mapcar #'list args)))
          (setf (value-of defn) (if (null (cdr (arguments-of form)))
                                    (car (arguments-of form))
                                    (gpu-code-error form "Only one argument allowed in multiple-value-call.")))
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
(defparameter *cur-catch-tags* nil)

(def function walk-tree-with-specials (tree visitor)
  "Recurse into the tree, tracking the active special bindings & catch tags."
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
               (catch-form
                (let* ((tag (ensure-constant (tag-of form)))
                       (*cur-catch-tags* (list* (cons tag form) *cur-catch-tags*)))
                  (enum-ast-links form #'recurse)
                  (funcall visitor form)))
               (t
                (enum-ast-links form #'recurse)
                (funcall visitor form)))))
    (declare (dynamic-extent #'recurse #'recurse-block))
    (recurse nil nil tree)))

(def function lexicalize-dynamic-refs (tree)
  "Convert all special refs to &aux arguments, bindings to lexical vars and catch to block."
  (with-accessors ((top-args bindings-of)
                   (top-decls declarations-of)) tree
    (let ((arg-cache nil))
      (labels ((close-special-var (form)
                 (when (typep (parent-of form)
                              '(or setq-form multiple-value-setq-form))
                   (warn-gpu-style form "Write to a special variable has been localized."))
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
                                (gpu-code-error form "Unknown global variable type: ~S" name))
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
                    (gpu-code-error form "Closing over lexical variables is not supported."))
                   ((or lexical-variable-binder-form multiple-value-bind-form)
                    (dolist (var (bindings-of form))
                      (setf (special-binding? var) nil)))
                   (catch-form
                    (change-class form 'block-form :name (ensure-constant (tag-of form))))
                   (throw-form
                    (aif (assoc (ensure-constant (tag-of form)) *cur-catch-tags*)
                         ;; Likewise, convert throw to return-from
                         (change-class form 'return-from-form
                                       :target-block (cdr it) :result (value-of form))
                         (gpu-code-error form "Throw without a catch."))))))
        (dolist (item (body-of tree))
          (walk-tree-with-specials item #'visitor))))))

;;; Shared variable instance assignment

(def function assign-shared-identities (tree)
  (map-ast (lambda (form)
             (typecase form
               (lexical-variable-binding-form
                (when (find-form-by-name (name-of form)
                                         (declarations-of (parent-of form))
                                         :type 'shared-declaration-form)
                  (setf (shared-identity-of form)
                        (make-instance 'gpu-shared-identity :name (name-of form))))))
             form)
           tree))

(def function preprocess-function (tree)
  (assign-shared-identities tree)
  tree)

(def function preprocess-tree (tree global-vars)
  (assign-shared-identities tree)
  ;; Inline functions
  (setf tree (inline-all-functions tree))
  ;; Convert global var refs to &aux args.
  (lexicalize-dynamic-refs tree)
  ;; Mark bindings that are destructively assigned to.
  (mark-mutated-vars tree)
  ;; Collect the binding usage lists.
  (annotate-binding-usage (list* tree (mapcar #'form-of global-vars)))
  tree)

