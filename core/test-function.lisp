;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file implements a function for testing
;;; gpu-function definitions in the REPL.
;;;

(in-package :cl-gpu)

(def function wrap-constant (value)
  (make-instance 'constant-form :value value :parent nil :source nil))

(def function is-default-arg-match? (pair)
  (and (typep (car pair) 'function-argument-form-with-default-value)
       (eq (cdr pair) (default-value-of (car pair)))))

(def function buffer-type-spec (buffer)
  `(array ,(buffer-element-type buffer) ,(buffer-rank buffer)))

(def function find-arg-type-spec (pair)
  (bind ((arg (car pair))
         (value (cdr pair))
         (type-decl (find-form-by-name (name-of arg)
                                       (declarations-of (parent-of arg))
                                       :type 'type-declaration-form)))
    (cond (type-decl
           (declared-type-of type-decl))
          ((bufferp (value-of value))
           (buffer-type-spec (value-of value)))
          (t
           (or (foreign-to-lisp-type (propagate-c-types value))
               (error "Could not determine type of: ~A" value))))))

(def function keyword-arg< (a b)
  (cond ((and (typep a 'keyword-function-argument-form)
              (typep b 'keyword-function-argument-form))
         (string< (symbol-name (effective-keyword-name-of a))
                  (symbol-name (effective-keyword-name-of b))))
        ((typep b 'keyword-function-argument-form)
         t)
        (t nil)))

(def function make-test-module-key (arg-map types)
  (loop for arg in arg-map and type in types
     when (typep (car arg) 'keyword-function-argument-form)
     collect (effective-keyword-name-of (car arg))
     collect type))

(def layered-function prepare-for-compile (obj)
  (:method ((function gpu-function))
    (propagate-c-types (form-of function) :upper-type +gpu-void-type+))
  (:method ((function test-gpu-kernel))
    (assert (null (globals-of *cur-gpu-module*)))
    (bind ((body (form-of function))
           (last-entry (last (body-of body)))
           (rtypes (propagate-c-types body :upper-type nil)))
      ;; Splice assignments to retrieve the return vals
      (loop for rtype in (unwrap-values-type rtypes)
         for i from 0
         for vname = (gensym "TEST-RES")
         for gvar = (make-gpu-global-var vname rtype nil
                                         :c-name (format nil "TEST_RES_~A" i))
         for ref = (make-lexical-var (form-of gvar))
         do (setf (assigned-to? (form-of gvar)) t)
         collect gvar into gvars
         collect ref into refs
         finally (progn
                   (setf (globals-of *cur-gpu-module*) gvars)
                   (setf (car last-entry)
                         (push-assignments-here (car last-entry) refs)))))))

(def function build-test-function (fobj key arg-map types)
  (bind (((:values anames adecls cargs)
          (loop for arg in arg-map and type in types
             for aname = (gensym "ARG")
             collect aname into anames
             collect `(type ,type ,aname) into adecls
             when (typep (car arg) 'keyword-function-argument-form)
               collect (effective-keyword-name-of (car arg)) into cargs
             collect aname into cargs
             finally (return (values anames adecls cargs))))
         (kspec `(:kernel f (,@anames)
                   (declare ,@adecls)
                   (,(name-of fobj) ,@cargs)))
         (*global-inline-callback*
          (lambda (func form)
            (declare (ignore form))
            (weak-set-addf (usages-of func) fobj)))
         (module (parse-gpu-module-spec (list kspec)))
         (kernel (first (kernels-of module)))
         (entry (cons key module)))
    (change-class kernel 'test-gpu-kernel)
    (compile-gpu-module module)
    (push entry (test-modules-of fobj))
    entry))

(def function invoke-test-function (module args)
  (bind ((items (get-module-instance-items module))
         (kernel (first (kernels-of module))))
    (apply (svref items (index-of kernel))
           (mapcar #'value-of args))
    (values-list
     (loop for var in (globals-of module)
        collect (gpu-global-value (svref items (index-of var)))))))

(def (function e) test-gpu-function (function-name &rest args)
  (bind ((fobj (or (symbol-gpu-function function-name)
                   (error "Unknown GPU function: ~S" function-name)))
         (arg-map (match-call-arguments (form-of fobj)
                                        (mapcar #'wrap-constant args)))
         (filt-map (remove-if #'is-default-arg-match? arg-map))
         (sort-map (stable-sort filt-map #'keyword-arg< :key #'car))
         (types (mapcar #'find-arg-type-spec sort-map))
         (key (make-test-module-key sort-map types))
         (module (or (assoc key (test-modules-of fobj) :test #'equal)
                     (build-test-function fobj key sort-map types))))
    (invoke-test-function (cdr module) (mapcar #'cdr sort-map))))
