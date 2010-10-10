;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(load-system :hu.dwim.asdf)

(in-package :hu.dwim.asdf)

;; Auto-detect CUDA
#-(or cuda (and ecl (not (and dffi dlopen))))
(load-system :cffi)

#-(or cuda (and ecl (not (and dffi dlopen))))
(when (ignore-errors (cffi:load-foreign-library '(:default "libcuda")) t)
  (pushnew :cuda *features*))

;; System definition
(defsystem :cl-gpu
  :class hu.dwim.system
  :author ("Alexander Gavrilov <angavrilov@gmail.com>")
  :licence "LLGPL"
  :description "A library for writing GPU kernels in a subset of CL"
  :depends-on (:cl-gpu.core
               #+cuda :cl-gpu.cuda))

