;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(defsystem "cl-gpu.buffers"
  :defsystem-depends-on ("hu.dwim.asdf")
  :class "hu.dwim.asdf:hu.dwim.system"
  :author ("Alexander Gavrilov <angavrilov@gmail.com>")
  :licence "LLGPL"
  :description "A helper library that provides a generalized buffer interface."
  :depends-on ("cffi"
               #+sbcl "sb-vector-io"
               "hu.dwim.util"
               "hu.dwim.def+contextl"
               "trivial-garbage")
  :components ((:module "buffers"
                :components ((:file "package")
                             (:file "utils" :depends-on ("package"))
                             (:file "interned-class" :depends-on ("utils"))
                             (:file "typedefs" :depends-on ("interned-class"))
                             (:file "utils-array" :depends-on ("typedefs"))
                             (:file "buffers" :depends-on ("typedefs" "utils-array"))
                             (:file "foreign-buf" :depends-on ("buffers"))))))
