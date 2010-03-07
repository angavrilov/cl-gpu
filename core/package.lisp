;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :common-lisp-user)

(defpackage :cl-gpu
  (:documentation "A library for writing GPU code in a subset of CL")
  (:nicknames #:gpu)

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

  (:import-from #:hu.dwim.walker
                #:recurse #:recurse-on-body
                #:variable #:value #:body #:declarations
                #:walk-environment/augment #:bindings
                #:-augment- #:with-current-form
                #:unwalk-declarations
                #:do-list-collect #:make-declaration)

  (:export #:cuda-driver-error #:buffer
           #:int8 #:int16 #:int32 #:int64
           #:uint8 #:uint16 #:uint32 #:uint64
           #:cuda-context #:cuda-context-device
           #:gpu-optimize))
