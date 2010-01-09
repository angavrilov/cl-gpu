;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :common-lisp-user)

(defpackage :cl-gpu
  (:documentation "A library for writing GPU code in a subset of CL")

  (:use :alexandria
        :anaphora
        :contextl
        :bordeaux-threads
        :cffi
        :hu.dwim.common-lisp
        :hu.dwim.def
        :hu.dwim.defclass-star
        :hu.dwim.util
        :metabang-bind
        :hu.dwim.walker)

  (:export #:cuda-driver-error #:buffer
           #:int-8 #:int-16 #:int-32
           #:uint-8 #:uint-16 #:uint-32
           #:cuda-context #:cuda-context-device))
