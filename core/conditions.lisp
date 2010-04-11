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

(def (condition* e) gpu-runtime-error (error)
  ((gpu-module nil)
   (call-stack nil)
   (thread-idx nil)
   (block-idx nil)))

(def method print-object :after ((obj gpu-runtime-error) stream)
  (unless *print-escape*
    (format stream "~&GPU function: ~:[(unknown)~;~:*~{~S~^ in ~}~]" (call-stack-of obj))
    (when (gpu-module-of obj)
      (format stream " of module ~S" (gpu-module-of obj)))
    (awhen (or (thread-idx-of obj) (block-idx-of obj))
      (format stream "~&Thread ~S, block ~S"
              (thread-idx-of obj) (block-idx-of obj)))))

(def (condition* e) simple-gpu-runtime-error (simple-error gpu-runtime-error)
  ())

(def function simple-gpu-runtime-error (module stack message thread block &rest args)
  (error 'simple-gpu-runtime-error :gpu-module module :call-stack stack
         :thread-idx thread :block-idx block
         :format-control message :format-arguments args))
