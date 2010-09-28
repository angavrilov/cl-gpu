;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file defines a metaclass that supports maintaining
;;; a single instance for every initarg value set.
;;;

(in-package :cl-gpu)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (def class interned-class (closer-mop:standard-class)
    ((lookup-table :documentation "A table for interned instance lookup")
     (lookup-arg-key-fun :documentation "Computes a lookup key from initargs")
     (lookup-obj-key-fun :documentation "Computes a lookup key from an instance")
     (obj-initargs-fun :documentation "Computes initargs from an instance")))

  (def method closer-mop:validate-superclass ((class interned-class) (superclass standard-class))
    t))

(def method closer-mop:finalize-inheritance :after ((class interned-class))
  (with-slots (lookup-table lookup-arg-key-fun lookup-obj-key-fun obj-initargs-fun) class
    (bind ((initargs (closer-mop:class-default-initargs class))
           ((:values keys vals slots args)
            (loop for slot in (remove-if-not (lambda (slot)
                                               (and (closer-mop:slot-definition-initargs slot)
                                                    (eq (closer-mop:slot-definition-allocation slot)
                                                        :instance)))
                                             (closer-mop:class-slots class))
               for name = (closer-mop:slot-definition-name slot)
               for args = (closer-mop:slot-definition-initargs slot)
               for initfun = (or (third (assoc (first args) initargs))
                                 (closer-mop:slot-definition-initfunction slot)
                                 (constantly nil))
               for vname = (gensym "ARG")
               for refexpr = `(slot-value obj ',name)
               do (when (rest args)
                    (warn "Multiple initargs for interned-class slot ~A: ~S" name args))
               collect `((,(first args) ,vname) (funcall ,initfun)) into keys
               collect vname into vals
               collect refexpr into slots
               nconc (list (first args) refexpr) into iargs
               finally (return (values keys vals slots iargs))))
           (arg-key-fun (eval `(lambda (&key ,@keys) (list ,@vals))))
           (obj-key-fun (eval `(lambda (obj) (declare (type ,class obj) (ignorable obj)) (list ,@slots))))
           (obj-args-fun (eval `(lambda (obj) (declare (type ,class obj) (ignorable obj)) (list ,@args))))
           (new-table #-(or ecl openmcl) (make-weak-hash-table :test #'equal :weakness :value)
                      #+openmcl (make-hash-table :test #'equal :weak :value)
                      #+ecl (make-hash-table :test #'equal))
           (old-table (if (slot-boundp class 'lookup-table)
                          lookup-table)))
      (setf lookup-table new-table
            lookup-arg-key-fun arg-key-fun
            lookup-obj-key-fun obj-key-fun
            obj-initargs-fun obj-args-fun)
      (when old-table
        (maphash (lambda (key val)
                   (declare (ignore key))
                   (setf (gethash (funcall obj-key-fun val) new-table) val))
                 old-table)))))

(def method make-instance ((class interned-class) &rest args)
  (unless (closer-mop:class-finalized-p class)
    (closer-mop:finalize-inheritance class))
  (with-slots (lookup-table lookup-arg-key-fun) class
    (gethash-with-init (apply lookup-arg-key-fun args)
                       lookup-table
                       (call-next-method))))

(def class interned-object (standard-object)
  ()
  (:metaclass interned-class))

(def method initarg-values-of ((object interned-object))
  (let ((class (class-of object)))
    (declare (type interned-class class))
    (with-slots (obj-initargs-fun) class
      (funcall obj-initargs-fun object))))

(def method make-load-form ((object interned-object) &optional env)
  (declare (ignore env))
  `(make-instance ',(class-name (class-of object)) ,@(initarg-values-of object)))

(def method print-object ((object interned-object) stream)
  (print-unreadable-object (object stream :type t :identity nil)
    (format stream "~{~S~^ ~}" (initarg-values-of object))))

(def function reintern-as-class (instance class &rest new-args)
  "Interns an object of the specified class, using initargs derived from the instance."
  (apply #'make-instance class (append new-args (list* :allow-other-keys t (initarg-values-of instance)))))

#|
(def class test-interned (interned-object)
  ((a :initarg :a :initform 3)
   (b :initarg :b :initform "foo")
   (c :initarg :x :initform nil))
  (:metaclass interned-class))
|#
