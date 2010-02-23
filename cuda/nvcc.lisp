;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file implements access to the NVidia C compiler.
;;;

(in-package :cl-gpu)

(defconstant +cuda-ptr-size+ 4
  "GPU pointer size in bytes.")

(defvar *nvcc* "nvcc")
(defvar *nvcc-flags* "")
(defvar *nvcc-cubin* nil)

(defvar *print-kernel-code* nil)

(defun cuda-compile-kernel (code)
  (with-memoize ((list code *nvcc-cubin* *nvcc-flags*)
                 :test #'equal)
    (with-temp-file (srcname sstream "/tmp/cudakernel.cu")
        (write-string code sstream)
      (let* ((outname (make-pathname :type (if *nvcc-cubin* "cubin" "ptx")
                                     :defaults srcname))
             (cmd (format nil "~A ~A ~A -m~A --output-file=~A ~A"
                          *nvcc*
                          (if *nvcc-cubin* "--cubin" "--ptx")
                          *nvcc-flags*
                          (* +cuda-ptr-size+ 8)
                          outname srcname)))
        (when *print-kernel-code*
          (format t "Compiling:~%~A" code))
        (unwind-protect
             (progn
               (format t "Running command:~%  ~A~%" cmd)
               (let ((rv (system-command cmd)))
                 (unless (= rv 0)
                   (error "Compilation failed: ~A~%" rv)))
               (with-open-file (out outname)
                 (let ((buffer (make-string (file-length out)
                                            :element-type 'base-char)))
                   (read-sequence buffer out)
                   (when *print-kernel-code*
                     (format t "Result:~%~A" buffer))
                   buffer)))
          (delete-if-exists outname))))))
