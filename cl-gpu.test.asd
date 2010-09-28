;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(load-system :hu.dwim.asdf)

(in-package :hu.dwim.asdf)

(defsystem :cl-gpu.test
  :class hu.dwim.test-system
  :author ("Alexander Gavrilov <angavrilov@gmail.com>")
  :licence "LLGPL"
  :description "Test suite for cl-gpu"
  :depends-on (:hu.dwim.stefil+hu.dwim.def
               :cl-gpu)
  :components ((:module "test"
                :components ((:file "utils" :depends-on ("package"))
                             (:file "buffers" :depends-on ("utils"))
                             (:file "translator" :depends-on ("buffers"))
                             (:file "cuda-driver" :depends-on ("buffers" "translator"))
                             (:file "package")))))
