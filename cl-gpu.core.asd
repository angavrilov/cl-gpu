;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(defsystem "cl-gpu.core"
  :defsystem-depends-on ("hu.dwim.asdf")
  :class "hu.dwim.asdf:hu.dwim.system"
  :author ("Alexander Gavrilov <angavrilov@gmail.com>")
  :licence "LLGPL"
  :description "Core part of the GPU code translator."
  :depends-on (:cffi
               :bordeaux-threads
               :hu.dwim.walker
               :cl-gpu.buffers)
  :components ((:module "core"
                :components ((:file "package")
                             (:file "conditions" :depends-on ("package"))
                             (:file "utils" :depends-on ("conditions"))
                             (:file "gpu-module" :depends-on ("utils"))
                             (:file "forms" :depends-on ("gpu-module"))
                             (:file "inline" :depends-on ("forms"))
                             (:file "type-inf" :depends-on ("gpu-module" "forms"))
                             (:file "codegen" :depends-on ("type-inf"))
                             (:file "unnest" :depends-on ("type-inf" "side-effects"))
                             (:file "side-effects" :depends-on ("gpu-module" "forms"))
                             (:file "builtins" :depends-on ("type-inf" "codegen" "unnest" "side-effects"))
                             (:file "syntax" :depends-on ("inline" "type-inf"))
                             (:file "test-function" :depends-on ("syntax"))))))
