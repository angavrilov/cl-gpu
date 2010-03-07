;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file defines a transformation that pulls
;;; forms mapped to C statements out of expressions.
;;;

(in-package :cl-gpu)

;;; Predicate that determines if a node is a statement

(def layered-function is-statement-call? (name form)
  (:method (name form)
    (declare (ignore name form))
    nil))

(def layered-function is-statement-assn? (name form)
  (:method (name form)
    (declare (ignore name form))
    nil))

(def definer is-statement? (name args &body code)
  (make-builtin-handler-method
   ;; Builtin prototype + method name
   name args (if assn? 'is-statement-assn? 'is-statement-call?)
   ;; Body
   (let ((all-args (flatten (list rq-args (mapcar #'first opt-args) rest-arg))))
     `((declare (ignorable ,@all-args))
       ,@code))))

(def layered-function is-statement? (form)
  ;; Built-in functions:
  (:method ((form free-application-form))
    (is-statement-call? (operator-of form) form))
  (:method ((form setf-application-form))
    (is-statement-assn? (operator-of form) form))
  ;; Special forms:
  ;;  blocks
  (:method ((form implicit-progn-mixin)) t)
  ;; (implies: block catch eval-when progv tagbody locally .*let.* labels)
  (:method ((form progn-form)) :defer) ; override
  (:method ((form unwind-protect-form)) t)
  (:method ((form multiple-value-prog1-form)) t)
  ;;  if
  (:method ((form if-form)) :defer)
  ;;  control transfer
  (:method ((form return-from-form)) t)
  (:method ((form go-form)) t)
  (:method ((form go-tag-form)) t)
  (:method ((form throw-form)) t)
  ;;  function calls
  ;(:method ((form application-form)) t)
  ;(:method ((form multiple-value-call-form)) nil)
  ;;  assignments
  (:method ((form setq-form)) nil)
  (:method ((form multiple-value-setq-form)) nil)
  ;;  expressions
  (:method ((form the-form)) nil)
  (:method ((form constant-form)) nil)
  (:method ((form variable-reference-form)) nil)
  (:method ((form load-time-value-form)) nil)
  ;;(:method ((form function-form)) nil)
  ;;  values
  (:method ((form values-form))
    (/= (length (values-of form)) 1))
  ;;  inline C
  (:method ((form verbatim-code-form))
    (not (is-expression? form))))

;;; Convert value return to side effects

(def function compute-self-assign-conflicts (vals vars)
  "Computes which vars would cause setq and psetq results to differ."
  (let* ((vdefs (mapcar #'definition-of vars))
         (effects
          (reduce (lambda (effects arg-effects)
                    (bind (((:values r-a-w w-a-r)
                            (side-effect-conflicts (car effects) arg-effects)))
                      (cons (join-side-effects (car effects) arg-effects)
                            (union (cdr effects) (union r-a-w w-a-r)))))
                  (mapcar (lambda (arg var)
                            (join-side-effects (side-effects-of arg)
                                               (make-side-effects :writes (list var))))
                          vals vdefs)
                  :initial-value (cons nil nil))))
    (intersection vdefs (cdr effects))))

(define-modify-macro push-assignments! (varlist) push-assignments)

(def layered-function push-assignments (form varlist)
  ;; Generic case
  (:method (form varlist)
    (if (= (length varlist) 1)
        (with-form-object (set 'setq-form (parent-of form) :value form)
          (setf (parent-of form) set)
          (setf (variable-of set)
                (copy-ast-form (first varlist) :parent set))
          (setf (form-c-type-of set) form))
        (with-form-object (set 'multiple-value-setq-form (parent-of form)
                               :value form)
          (setf (parent-of form) set)
          (setf (variables-of set)
                (mapcar (lambda (var) (copy-ast-form var :parent set))
                        varlist))
          (setf (form-c-type-of set) form))))

  ;; Never-returning forms
  (:method ((form tagbody-form) varlist)
    (declare (ignore varlist))
    form)

  (:method ((form return-from-form) varlist)
    (declare (ignore varlist))
    form)

  (:method ((form go-form) varlist)
    (declare (ignore varlist))
    form)

  (:method ((form throw-form) varlist)
    (declare (ignore varlist))
    form)

  ;; Control structures
  (:method ((form if-form) varlist)
    (push-assignments! (then-of form) varlist)
    (push-assignments! (else-of form) varlist)
    form)

  (:method ((form values-form) varlist)
    (assert (<= (length varlist) (length (values-of form))))
    (let ((side-effect-conflicts
           (when (side-effects-of form)
             (compute-self-assign-conflicts (values-of form) varlist))))
      ;; Allocate a let or progn depending on the necessity
      ;; of side effect conflict resolution:
      (with-form-object (blk (if side-effect-conflicts
                                 'let-form 'progn-form)
                               (parent-of form)
                               :body (copy-list (values-of form)))
        (adjust-parents (body-of blk))
        (setf (form-c-type-of blk) :void)
        ;; Allocate new temporary vars
        (let ((aux-vars
               (loop for defn in side-effect-conflicts
                  for binding = (make-lexical-binding blk :c-type defn)
                  collect (cons defn binding))))
          ;; Push assignments
          (loop
             for tail on (body-of blk)
             and var in varlist
             for aux-var = (assoc (definition-of var) aux-vars)
             for res-var = (if aux-var (make-lexical-var (cdr aux-var)) var)
             do (push-assignments! (car tail) (list res-var)))
          ;; If used temporaries, forward values to the real targets
          (when aux-vars
            (setf (bindings-of blk) (mapcar #'cdr aux-vars))
            (appendf (body-of blk)
                     (loop for aux-var in aux-vars
                        collect (make-lexical-assignment (car aux-var) (cdr aux-var) blk))))))))

  (:method ((form block-form) varlist)
    (dolist (ret (usages-of form))
      (push-assignments! (result-of ret) varlist))
    (call-next-method))

  (:method ((form implicit-progn-mixin) varlist)
    (let ((tail (last (body-of form))))
      (push-assignments! (car tail) varlist))
    form)

  (:method ((form multiple-value-prog1-form) varlist)
    (push-assignments! (first-form-of form) varlist)
    form)

  (:method ((form unwind-protect-form) varlist)
    (push-assignments! (protected-form-of form) varlist)
    form))

;;; Generic nested statement extraction

(def layered-function flatten-statements (form))

(defparameter *nested-trigger-set* nil)

(def function extract-nested-statements (form &key (start-with form) (pull-root? nil))
  "Pulls out statements nested inside an expression."
  (let ((nested-vars nil)
        (nested-assns nil)
        (*nested-trigger-set* nil))
    (labels ((extract-stmt (form)
               (let* ((binding (make-lexical-binding nil :c-type form))
                      (var (make-lexical-var binding (parent-of form))))
                 (push binding nested-vars)
                 (push (push-assignments form (list var)) nested-assns)
                 var))
             ;; Main recursive handler
             (unnest-rec (parent field form)
               (declare (ignore parent field))
               (if (typep form 'walked-form)
                   (let ((stmt? (is-statement? form)))
                     ;; Deferred statement forms may be rendered
                     ;; both as expressions and statements. This
                     ;; code ensures that they are promoted to
                     ;; full statements if they contain any non-
                     ;; deferred ones.
                     (when (eq stmt? :defer)
                       (let ((*nested-trigger-set* t))
                         (catch 'statement-found
                           (recurse form)
                           ;; Executed only if no statements inside.
                           (setf (is-expression? form) t)
                           (return-from unnest-rec form))))
                     ;; Deferred handling failed, or impossible.
                     (if stmt?
                         (if *nested-trigger-set*
                             ;; Within deferred: promote. May cascade.
                             (throw 'statement-found t)
                             (extract-stmt form))
                         (progn
                           (recurse form)
                           form)))
                   form))
             ;; Recursion thunk
             (recurse (form)
               (rewrite-ast-links form #'unnest-rec)))
      ;; Perform the recursion.
      (if pull-root?
          (setf start-with
                (if (listp start-with)
                    (mapcar (lambda (item) (unnest-rec form nil item)) start-with)
                    (unnest-rec form nil start-with)))
          (recurse start-with))
      ;; If any statements found, wrap in a let form.
      (values (if (or nested-vars nested-assns)
                  (let ((new-body ; recurse into extracted statements:
                         (list* form (mapcar #'flatten-statements nested-assns))))
                    (with-form-object (wrapper (if nested-vars 'let-form 'progn-form)
                                               (parent-of form) :body (nreverse new-body))
                      (adjust-parents (body-of wrapper))
                      (when nested-vars
                        (setf (bindings-of wrapper) (nreverse nested-vars))
                        (adjust-parents (bindings-of wrapper)))))
                  form)
              start-with))))

;;; Specialization for various statements

(define-modify-macro flatten-statements! () flatten-statements)

(def layered-methods flatten-statements
  ;; Generic: assume it's an expression
  (:method ((form walked-form))
    (extract-nested-statements form))

  (:method ((form cons))
    (mapcar #'flatten-statements form))

  ;; Block-like
  (:method ((form implicit-progn-mixin))
    (flatten-statements! (body-of form))
    form)

  (:method ((form implicit-progn-with-declarations-mixin))
    (with-optimize-context (form)
      (call-next-method)))

  ;; Variable bindings: may split
  (:method ((form lexical-variable-binder-form))
    (flet (;; Splits the binding statement after the point
           (split (tail init-stmt)
             ;; The initform is being converted to a side effect
             (setf (initial-value-of (car tail)) nil)
             ;; If there are some more bindings, split
             (when (cdr tail)
               (with-form-object (inner (class-of form) form
                                        :bindings (cdr tail)
                                        :body (body-of form))
                 (setf (cdr tail) nil) ; cut the binding list
                 (adjust-parents (bindings-of inner))
                 (adjust-parents (body-of inner))
                 ;; Split declarations
                 (loop for decl in (declarations-of form)
                    :if (and (typep decl 'named-walked-form)
                             (find-form-by-name (name-of decl)
                                                (bindings-of form)))
                    :collect decl :into cur-decls
                    :else :collect decl :into inner-decls
                    :finally
                      (setf (declarations-of form) cur-decls
                            (declarations-of inner) inner-decls))
                 (adjust-parents (declarations-of inner))
                 ;; Replace the outer body
                 (setf (body-of form) (list inner))))
             ;; Push the side effect statement and walk the body.
             (push init-stmt (body-of form))
             (return-from flatten-statements
               (call-next-method)))
           ;; Assignment push helper
           (push-assn (form binding)
             (push-assignments form (list (make-lexical-var binding nil)))))
      ;; Walk bindings, looking for split points
      (loop for tail on (bindings-of form)
         :for binding = (car tail)
         :for value = (initial-value-of binding)
         :when value
         :do (if (is-statement? value)
                 (split tail (push-assn value binding))
                 (let ((ext (extract-nested-statements value)))
                   (unless (eq ext value)
                     (split tail (push-assn ext binding))))))
      ;; Walk the body
      (call-next-method)))

  ;; Assignments
  (:method ((form setq-form))
    (let* ((value (value-of form))
           (stmt? (is-statement? value)))
      (cond ((and stmt? (or (typep value 'free-application-form)
                            (typep value 'verbatim-code-form)))
             ;; Some forms return values, but do it explicitly in
             ;; their code generators:
             (setf (is-merged-assignment? form) t)
             (extract-nested-statements form :start-with value))
            (stmt? ; Statement as value
             (let ((alt (push-assignments value (list (variable-of form)))))
               (assert (is-statement? alt))
               (flatten-statements alt)))
            (t
             (extract-nested-statements form)))))

  (:method ((form multiple-value-setq-form))
    (let* ((value (value-of form))
           (stmt? (is-statement? value)))
      (cond ((and stmt? (or (typep value 'free-application-form)
                            (typep value 'verbatim-code-form)))
             (setf (is-merged-assignment? form) t)
             (extract-nested-statements form :start-with value))
            (stmt?
             (let ((alt (push-assignments value (variables-of form))))
               (assert (is-statement? alt))
               (flatten-statements alt)))
            (t
             (error "Could not inline multiple-value assignment: ~S"
                    (unwalk-form form))))))

  ;; Miscellaneous statements
  (:method ((form values-form))
    ;; Demote to progn
    (flatten-statements
     (with-form-object (progn 'progn-form (parent-of form)
                              :body (copy-list (values-of form)))
       (adjust-parents (body-of progn)))))

  (:method ((form return-from-form))
    (if (nop-form? (result-of form))
        form
        (with-form-object (progn 'progn-form (parent-of form)
                                 :body (list (result-of form)
                                             form))
          (setf (result-of form) nil)
          (adjust-parents (body-of progn))
          (flatten-statements! (first (body-of progn))))))

  (:method ((form if-form))
    (flatten-statements! (then-of form))
    (flatten-statements! (else-of form))
    (multiple-value-bind (new-form new-condition)
        (extract-nested-statements form :start-with (condition-of form) :pull-root? t)
      (setf (condition-of form) new-condition)
      new-form))

  (:method ((form multiple-value-bind-form))
    ;; Convert to let
    (let ((vars (mapcar #'make-lexical-var (bindings-of form))))
      (with-form-object (let 'let-form (parent-of form)
                             :bindings (bindings-of form)
                             :declarations (declarations-of form)
                             :body (list* (push-assignments (value-of form) vars)
                                          (body-of form)))
        (adjust-parents (bindings-of let))
        (adjust-parents (declarations-of let))
        (adjust-parents (body-of let))
        (flatten-statements! (body-of let)))))

  (:method ((form multiple-value-prog1-form))
    ;; Demote to progn
    (flatten-statements
     (with-form-object (progn 'progn-form (parent-of form)
                              :body (list* (first-form-of form)
                                           (other-forms-of form)))
       (adjust-parents (body-of progn)))))

  (:method ((form unwind-protect-form))
    (flatten-statements! (protected-form-of form))
    (flatten-statements! (cleanup-form-of form))
    form)

  ;; Verbatim code
  (:method ((form verbatim-code-form))
    (let ((exprs nil))
      (do-verbatim-code (item flags form)
        (if (getf flags :stmt)
            (flatten-statements! item)
            (push item exprs)))
      (multiple-value-bind (new-form new-exprs)
          (extract-nested-statements form :start-with exprs :pull-root? t)
        (unless (eq new-form form)
          (let ((table (loop for a in exprs and b in new-exprs
                          unless (eq a b) collect (cons a b))))
            (setf (body-of form)
                  (mapcar (lambda (item) (aif (assoc item table) (cdr it) item))
                          (body-of form)))))
        new-form))))
