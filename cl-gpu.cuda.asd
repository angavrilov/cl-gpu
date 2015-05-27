;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(defsystem :cl-gpu.cuda
  :defsystem-depends-on (:hu.dwim.asdf)
  :class "hu.dwim.asdf:hu.dwim.system"
  :author ("Alexander Gavrilov <angavrilov@gmail.com>")
  :licence "LLGPL"
  :description "CUDA integration for the GPU code translator."
  :depends-on (:cl-gpu.core)
  :components ((:module "cuda"
                :components ((:file "driver-lib")
                             (:file "driver-api" :depends-on ("driver-lib"))
                             (:file "pitched-copy" :depends-on ("driver-api"))
                             (:file "device-mem" :depends-on ("driver-api" "pitched-copy"))
                             (:file "nvcc")
                             (:file "target" :depends-on ("driver-api" "nvcc"))
                             (:file "module" :depends-on ("device-mem" "target" "nvcc"))))))

(defmethod perform :after ((o load-op) (c (eql (find-system :cl-gpu.cuda))))
  (pushnew :cuda *features*))
