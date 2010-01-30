;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

(deflayer cuda-target)

(def macro with-cuda-target (&body code)
  `(with-active-layers (cuda-target) ,@code))

(def layered-method generate-c-code :in cuda-target ((obj gpu-global-var))
  (with-slots (constant-var?) obj
    (format nil "~A ~A"
            (if (or constant-var? (dynarray-var? obj))
                "__constant__"
                "__device__")
            (call-next-method))))

(def layered-method generate-c-code :in cuda-target ((obj gpu-function))
  (format nil "~A ~A"
          (if (typep obj 'gpu-kernel)
              "__global__"
              "__device__")
          (call-next-method)))
