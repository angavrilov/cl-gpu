;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file defines the side effect computation.
;;;

(in-package :cl-gpu)

(def layered-function side-effects-call (name form)
  (:method (name form)
    (declare (ignore name form))
    nil))

(def layered-function side-effects-assn (name form)
  (:method (name form)
    (declare (ignore name form))
    nil))

(def definer side-effects (name args &body code)
  (make-builtin-handler-method
   ;; Builtin prototype + method name
   name args (if assn? 'side-effects-assn 'side-effects-call)
   ;; Body
   (let ((all-args (flatten (list rq-args (mapcar #'first opt-args) rest-arg))))
     `((declare (ignorable ,@all-args))
       ,@code))
   :let-decls (if assn? '((ignorable -value-)))))

(def function join-side-effects (arg1 arg2)
  "Returns merged side effect structures."
  (cond ((null arg1) arg2)
        ((null arg2) arg1)
        (t (make-side-effects
            :reads (union (side-effects-reads arg1)
                          (side-effects-reads arg2))
            :writes (union (side-effects-writes arg1)
                           (side-effects-writes arg2))))))

(define-modify-macro join-side-effects! (effects2) join-side-effects)

(def function side-effect-conflicts (arg1 arg2)
  "Returns read-after-write and write-after-read conflicts."
  (when (and arg1 arg2)
    (values (intersection (side-effects-reads arg2)
                          (side-effects-writes arg1))
            (intersection (side-effects-writes arg2)
                          (side-effects-reads arg1)))))

(def function remove-side-effects (effects items)
  "Removes items from all side effect lists."
  (awhen effects
    (let ((reads (set-difference (side-effects-reads effects) items))
          (writes (set-difference (side-effects-writes effects) items)))
      (when (or reads writes)
        (make-side-effects :reads reads :writes writes)))))

(def function ensure-side-effects (obj)
  (or obj (make-side-effects)))

(def function gpu-var-side-effects (obj)
  (let ((effects (ensure-side-effects obj)))
    (make-side-effects
     :reads (mapcar #'gpu-variable-of (side-effects-reads effects))
     :writes (mapcar #'gpu-variable-of (side-effects-writes effects)))))

(def layered-function compute-side-effects (form))

(def function compute-arg-side-effects (args form)
  (reduce (lambda (effects arg)
            (bind ((arg-effects
                    (compute-side-effects arg))
                   ((:values r-a-w w-a-r)
                    (side-effect-conflicts effects arg-effects)))
              (when r-a-w
                (warn-gpu-style form "Read-after-write conflicts on ~S"
                                (mapcar #'name-of r-a-w)))
              (when w-a-r
                (warn-gpu-style form "Write-after-read conflicts on ~S"
                                (mapcar #'name-of w-a-r)))
              (join-side-effects effects arg-effects)))
          args :initial-value nil))

(def layered-methods compute-side-effects
  (:method :around (form)
    (setf (side-effects-of form) (call-next-method)))

  (:method ((form walked-form))
    (let ((result nil))
      (do-ast-links (form2 form)
        (join-side-effects! result (compute-side-effects form2)))
      result))

  ;; Variables
  (:method ((form walked-lexical-variable-reference-form))
    (if (assigned-to? (definition-of form))
        (make-side-effects :reads (list (definition-of form)))))

  ;; Assigment forms
  (:method ((form setq-form))
    (let* ((vdef (definition-of (variable-of form)))
           (seff (make-side-effects :writes (list vdef))))
      (assert (assigned-to? vdef))
      (setf (side-effects-of (variable-of form)) seff)
      (join-side-effects seff (compute-side-effects (value-of form)))))

  (:method ((form multiple-value-setq-form))
    (reduce (lambda (effects var)
              (let* ((vdef (definition-of var))
                     (seff (make-side-effects :writes (list vdef))))
                (assert (assigned-to? vdef))
                (setf (side-effects-of var) seff)
                (join-side-effects effects seff)))
            (variables-of form)
            :initial-value (compute-side-effects (value-of form))))

  ;; Variable definition forms discard items
  (:method ((form lexical-variable-binder-form))
    (remove-side-effects (call-next-method) (bindings-of form)))

  (:method ((form lambda-function-form))
    (remove-side-effects (call-next-method) (bindings-of form)))

  ;; Calls may have their own effects
  (:method ((form free-application-form))
    (let ((arg-effects (compute-arg-side-effects (arguments-of form) form)))
      (join-side-effects arg-effects
                         (side-effects-call (operator-of form) form))))

  (:method ((form setf-application-form))
    (let* ((all-args (list* (value-of form) (arguments-of form)))
           (arg-effects (compute-arg-side-effects all-args form)))
      (join-side-effects arg-effects
                         (side-effects-assn (operator-of form) form)))))


