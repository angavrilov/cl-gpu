;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

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
   code))

(def layered-function is-statement? (form)
  (:method ((form application-form))
    (is-statement-call? (operator-of form) form))
  (:method ((form setf-application-form))
    (is-statement-assn? (operator-of form) form))
  ;; Built-in forms
  (:method ((form block-form)) t)
  (:method ((form lexical-variable-binder-form)) t)
  (:method ((form return-from-form)) t)
  (:method ((form setq-form)) nil)
  (:method ((form multiple-value-setq-form)) nil)
  (:method ((form implicit-progn-mixin)) t)
  (:method ((form tagbody-form)) t)
  (:method ((form the-form)) nil)
  (:method ((form go-form)) t)
  (:method ((form if-form)) :defer)
  (:method ((form constant-form)) nil)
  (:method ((form variable-reference-form)) nil)
  (:method ((form verbatim-code-form))
    (not (is-expression? form))))

;;; Value return to side effects conversion

(def macro push-assignments! (place varlist)
  `(setf ,place (push-assignments ,place ,varlist)))

(def layered-function push-assignments (form varlist)
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

  (:method ((form tagbody-form) varlist)
    (declare (ignore varlist))
    form)

  (:method ((form return-from-form) varlist)
    (declare (ignore varlist))
    form)

  (:method ((form go-form) varlist)
    (declare (ignore varlist))
    form)

  (:method ((form if-form) varlist)
    (push-assignments! (then-of form) varlist)
    (push-assignments! (else-of form) varlist)
    form)

  (:method ((form values-form) varlist)
    (assert (<= (length varlist) (length (values-of form))))
    (with-form-object (progn 'progn-form (parent-of form)
                             :body (copy-list (values-of form)))
      (setf (form-c-type-of progn) :void)
      (loop
         for tail on (body-of progn)
         and var in varlist
         do (push-assignments! (car tail) (list var)))))

  (:method ((form block-form) varlist)
    (dolist (ret (usages-of form))
      (push-assignments! (result-of ret) varlist))
    (call-next-method))

  (:method ((form implicit-progn-mixin) varlist)
    (let ((tail (last (body-of form))))
      (push-assignments! (car tail) varlist))
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
               (let ((binding
                      (with-form-object (binding 'lexical-variable-binding-form nil
                                                 :name (make-symbol "_T")
                                                 :initial-value nil)
                        (setf (form-c-type-of binding) (form-c-type-of form))
                        (push binding nested-vars))))
                 (aprog1
                     (make-lexical-var binding (parent-of form))
                   (push (push-assignments form (list it)) nested-assns))))
             (unnest-rec (parent field form)
               (declare (ignore parent field))
               (if (typep form 'walked-form)
                   (let ((stmt? (is-statement? form)))
                     (when (eq stmt? :defer)
                       (let ((*nested-trigger-set* t))
                         (catch 'statement-found
                           (recurse form)
                           (setf (is-expression? form) t)
                           (return-from unnest-rec form))))
                     (if stmt?
                         (if *nested-trigger-set*
                             (throw 'statement-found t)
                             (extract-stmt form))
                         (progn
                           (recurse form)
                           form)))
                   form))
             (recurse (form)
               (rewrite-ast-fields form #'unnest-rec)))
      (if pull-root?
          (setf start-with (unnest-rec form nil start-with))
          (recurse start-with))
      (values (if (or nested-vars nested-assns)
                  (with-form-object (wrapper (if nested-vars 'let-form 'progn-form)
                                             (parent-of form)
                                             :body (nreverse (list* form
                                                                    (mapcar #'flatten-statements
                                                                            nested-assns))))
                    (adjust-parents (body-of wrapper))
                    (when nested-vars
                      (setf (bindings-of wrapper) (nreverse nested-vars))
                      (adjust-parents (bindings-of wrapper))))
                  form)
              start-with))))

;;; Specialization for various statements

(def layered-methods flatten-statements
  (:method ((form walked-form))
    (extract-nested-statements form))

  (:method ((form implicit-progn-mixin))
    (setf (body-of form)
          (mapcar #'flatten-statements (body-of form)))
    form)

  (:method ((form lexical-variable-binder-form))
    (flet ((split (tail init-stmt)
             (setf (initial-value-of (car tail)) nil)
             (when (cdr tail)
               (with-form-object (inner 'let-form form
                                        :bindings (cdr tail)
                                        :body (body-of form))
                 (setf (cdr tail) nil) ; split the bindings
                 (adjust-parents (bindings-of inner))
                 (adjust-parents (body-of inner))
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
                 (setf (body-of form) (list inner))))
             (push init-stmt (body-of form))
             (return-from flatten-statements
               (call-next-method)))
           (push-assn (form binding)
             (push-assignments form (list (make-lexical-var binding nil)))))
      (loop for tail on (bindings-of form)
         :for binding = (car tail)
         :for value = (initial-value-of binding)
         :when value
         :do (if (is-statement? value)
                 (split tail (push-assn value binding))
                 (let ((ext (extract-nested-statements value)))
                   (unless (eq ext value)
                     (split tail (push-assn ext binding))))))
      (call-next-method)))

  (:method ((form setq-form))
    (let* ((value (value-of form))
           (stmt? (is-statement? value)))
      (cond ((and stmt? (typep value 'application-form))
             (setf (is-merged-assignment? form) t)
             (extract-nested-statements form :start-with value))
            (stmt?
             (let ((alt (push-assignments value (list (variable-of form)))))
               (assert (is-statement? alt))
               (flatten-statements alt)))
            (t
             (extract-nested-statements form)))))

  (:method ((form multiple-value-setq-form))
    (let* ((value (value-of form))
           (stmt? (is-statement? value)))
      (cond ((and stmt? (typep value 'application-form))
             (setf (is-merged-assignment? form) t)
             (extract-nested-statements form :start-with value))
            (stmt?
             (let ((alt (push-assignments value (list (variable-of form)))))
               (assert (is-statement? alt))
               (flatten-statements alt)))
            (t
             (error "Could not inline multiple-value assignment: ~S"
                    (unwalk-form form))))))

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
          (setf (first (body-of progn))
                (flatten-statements (first (body-of progn)))))))

  (:method ((form if-form))
    (setf (then-of form) (flatten-statements (then-of form)))
    (setf (else-of form) (flatten-statements (else-of form)))
    (multiple-value-bind (new-form new-condition)
        (extract-nested-statements form :start-with (condition-of form)
                                   :pull-root? t)
      (setf (condition-of form) new-condition)
      new-form))
  )
