;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(load-system :hu.dwim.asdf)

(in-package :hu.dwim.asdf)

(pushnew :cuda *features*)

(defsystem :cl-gpu
  :class hu.dwim.system
  :author ("Alexander Gavrilov <angavrilov@gmail.com>")
  :licence "LLGPL"
  :description "A library for writing GPU kernels in a subset of CL"
  :depends-on (:cffi
               :bordeaux-threads
               :hu.dwim.walker)
  :components ((:module "core"
                :components ((:file "package")
                             (:file "typedefs" :depends-on ("package"))
                             (:file "utils" :depends-on ("typedefs"))
                             (:file "buffers" :depends-on ("typedefs" "utils"))
                             (:file "foreign-buf" :depends-on ("buffers"))
                             (:file "gpu-module" :depends-on ("buffers"))
                             (:file "forms" :depends-on ("gpu-module"))
                             (:file "type-inf" :depends-on ("gpu-module" "forms"))
                             (:file "codegen" :depends-on ("type-inf"))
                             (:file "unnest" :depends-on ("type-inf"))
                             (:file "builtins" :depends-on ("type-inf" "codegen" "unnest"))
                             (:file "syntax" :depends-on ("type-inf"))))
               (:module "cuda"
                :depends-on ("core")
                :components (#+cuda (:file "driver-api")
                             #+cuda (:file "device-mem" :depends-on ("driver-api"))
                             (:file "nvcc")
                             #+cuda (:file "module" :depends-on ("device-mem" "nvcc"))))))
