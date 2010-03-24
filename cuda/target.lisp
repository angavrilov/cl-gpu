;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file defines the CUDA code generation target,
;;; including the relevant layer, C types and builtins.
;;;

(in-package :cl-gpu)

;;; Target definition

(deflayer cuda-target (gpu-target))

(def reserved-c-names
    "__device__" "__constant__" "__shared__" "__global__")

(def macro with-cuda-target (&body code)
  `(with-active-layers (cuda-target) ,@code))

(def method call-with-target ((target (eql :cuda)) thunk)
  (with-cuda-target (funcall thunk)))

(def function lookup-cuda-module (module-id)
  (let* ((context (or (cuda-current-context)
                      (error "No CUDA context is active.")))
         (instance (gethash-with-init module-id (cuda-context-module-hash context)
                                      (with-cuda-target
                                        (load-gpu-module-instance module-id)))))
    (unless (car (gpu-module-instance-change-sentinel instance))
      (with-cuda-target
        (upgrade-gpu-module-instance module-id instance)))
    instance))

(def method target-module-lookup-fun ((target (eql :cuda)))
  #'lookup-cuda-module)

(setf *current-gpu-target* :cuda)

;;; Toplevel object code generation

(def layered-method generate-c-code :in cuda-target ((obj gpu-global-var))
  (format nil "~A ~A"
          (if (or (constant-var? obj) (dynarray-var? obj))
              "__constant__"
              "__device__")
          (call-next-method)))

(def layered-method generate-c-code :in cuda-target ((obj gpu-function))
  (format nil "extern \"C\" ~A ~A"
          (if (typep obj 'gpu-kernel)
              "__global__"
              "__device__")
          (call-next-method)))

;;; C types

(def layered-method c-type-string :in cuda-target (type)
  (case type
    (:int64 "long long")
    (:uint64 "unsigned long long")
    (otherwise (call-next-method))))

(def layered-method c-type-string :in cuda-target ((type cons))
  (case (first type)
    (:tuple (let ((size (second type))
                  (base (third type)))
              (unless (and (> size 0)
                           (<= size (case base
                                      ((:double :int64) 2)
                                      (t 4))))
                (error "Invalid size ~A for tuple of ~A" size base))
              (format nil "~A~A"
                      (case base
                        (:int8 "char") (:uint8 "uchar")
                        (:int16 "short") (:uint16 "ushort")
                        (:int32 "int") (:uint32 "uint")
                        (:int64 "longlong")
                        (:float "float") (:double "double")
                        (t (error "Invalid tuple type ~A" base)))
                      size)))
    (otherwise (call-next-method))))

(def layered-method c-type-size :in cuda-target (type)
  (case type
    (:pointer +cuda-ptr-size+)
    (otherwise (call-next-method))))

(def layered-method c-type-alignment :in cuda-target (type)
  (case type
    (:pointer +cuda-ptr-size+)
    (otherwise (call-next-method))))

;;; Abort command

(def layered-method emit-abort-command :in cuda-target (stream exception args)
  (declare (ignore exception args))
  (format stream "__trap();")
  (emit-code-newline stream))

;;; Built-in functions

(def (c-code-emitter :in cuda-target) tuple (&rest args)
  (emit "make_~A" (c-type-string (form-c-type-of -form-)))
  (emit-separated -stream- args ","))

(def (c-code-emitter :in cuda-target) untuple (tuple)
  (if (has-merged-assignment? -form-)
      (with-c-code-block (-stream-)
        (let ((tuple/type (form-c-type-of tuple)))
          (code (c-type-string tuple/type) " TMP = " tuple ";" #\Newline)
          (loop for i from 0 below (second tuple/type)
             and name in '("x" "y" "z" "w")
             when (emit-merged-assignment -stream- -form- i
                                          (format nil "TMP.~A" name))
             do (code ";"))))
      (code tuple ".x")))

