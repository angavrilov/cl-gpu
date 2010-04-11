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
               :hu.dwim.walker
               :trivial-garbage)
  :components ((:module "core"
                :components ((:file "package")
                             (:file "conditions" :depends-on ("package"))
                             (:file "typedefs" :depends-on ("package" "conditions"))
                             (:file "utils" :depends-on ("typedefs"))
                             (:file "buffers" :depends-on ("typedefs" "utils"))
                             (:file "foreign-buf" :depends-on ("buffers"))
                             (:file "gpu-module" :depends-on ("buffers"))
                             (:file "forms" :depends-on ("gpu-module"))
                             (:file "inline" :depends-on ("forms"))
                             (:file "type-inf" :depends-on ("gpu-module" "forms"))
                             (:file "codegen" :depends-on ("type-inf"))
                             (:file "unnest" :depends-on ("type-inf" "side-effects"))
                             (:file "side-effects" :depends-on ("gpu-module" "forms"))
                             (:file "builtins" :depends-on ("type-inf" "codegen" "unnest" "side-effects"))
                             (:file "syntax" :depends-on ("inline" "type-inf"))))
               #+cuda
               (:module "cuda"
                :depends-on ("core")
                :components ((:file "driver-lib")
                             (:file "driver-api" :depends-on ("driver-lib"))
                             (:file "pitched-copy" :depends-on ("driver-api"))
                             (:file "device-mem" :depends-on ("driver-api" "pitched-copy"))
                             (:file "nvcc")
                             (:file "target" :depends-on ("driver-api" "nvcc"))
                             (:file "module" :depends-on ("device-mem" "target" "nvcc"))))))
