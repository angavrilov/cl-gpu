;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :common-lisp-user)

(defpackage :cl-gpu.test
  (:use :contextl
        :hu.dwim.common
        :hu.dwim.def
        :hu.dwim.stefil
        :cl-gpu))

(in-package :cl-gpu.test)

(defsuite* (test :in root-suite) ()
  (run-child-tests))
