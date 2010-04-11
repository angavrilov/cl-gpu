;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file defines error and warning conditions
;;; used by the GPU code translator and runtime.
;;;

(in-package :cl-gpu)

(defparameter *cur-gpu-function* nil)
(defparameter *cur-gpu-module* nil)

(def (condition* e) gpu-code-condition (condition)
  ((gpu-module *cur-gpu-module*)
   (gpu-function *cur-gpu-function*)
   (enclosing-form nil)))

(def (condition* e) gpu-code-error (gpu-code-condition error)
  ())

(def (condition* e) simple-gpu-code-error (simple-error gpu-code-error)
  ())

(def function gpu-code-error (form message &rest args)
  (error 'simple-gpu-code-error :enclosing-form form
         :format-control message :format-arguments args))

(def (condition* e) gpu-code-warning (gpu-code-condition warning)
  ())

(def (condition* e) gpu-code-style-warning (gpu-code-warning style-warning)
  ())

(def (condition* e) simple-gpu-code-style-warning (simple-warning gpu-code-style-warning)
  ())

(def function warn-gpu-style (form message &rest args)
  (warn 'simple-gpu-code-style-warning :enclosing-form form
        :format-control message :format-arguments args))
