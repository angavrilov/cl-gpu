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
        :trivial-garbage
        :cffi
        :hu.dwim.common-lisp
        :hu.dwim.def
        :hu.dwim.defclass-star
        :hu.dwim.util
        :metabang-bind
        :hu.dwim.walker
        :cl-gpu.buffers
        :cl-gpu.buffers/types
        :cl-gpu.buffers/impl)

  (:import-from #:hu.dwim.walker
                #:recurse #:recurse-on-body
                #:variable #:value #:body #:declarations
                #:walk-environment/augment #:walk-environment/augment!
                #:bindings #:function-name?
                #:-augment- #:with-current-form
                #:unwalk-declarations
                #:do-list-collect #:make-declaration)

  (:import-from #:cl-gpu.buffers
                #:to-uint32-vector #:with-memoize #:gethash-with-init
                #:with-slot-values)

  (:export #:cuda-driver-error
           #:cuda-context #:cuda-context-device
           #:gpu-optimize :shared))

(hu.dwim.common::export-external-symbols :cl-gpu.buffers :cl-gpu)

