;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by the authors.
;;;
;;; See LICENCE for details.

(load-system :hu.dwim.asdf)

(in-package :hu.dwim.asdf)

(defsystem :cl-gpu.test
  :class hu.dwim.test-system
  :author ("Alexander Gavrilov <angavrilov@gmail.com>")
  :licence "LLGPL"
  :description "Test suite for cl-gpu"
  :depends-on (:hu.dwim.def+hu.dwim.stefil
               :hu.dwim.stefil+swank
               :cl-gpu)
  :components ((:module "test"
                :components ((:file "utils" :depends-on ("package"))
                             (:file "package")))))
