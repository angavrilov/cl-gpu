;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

(def class* gpu-variable ()
  ((name           :documentation "Lisp name of the variable")
   (c-name         :documentation "C name of the variable")
   (item-type      :documentation "Type without array dimensions")
   (dimension-mask :documentation "If array, vector of fixed dims")
   (static-asize   :documentation "Full dimension if all dims constant."))
  (:documentation "Common name of a global variable or parameter."))

(def generic array-var? (obj)
  (:method ((obj gpu-variable)) (dimension-mask-of obj)))

(def generic dynarray-var? (obj)
  (:method ((obj gpu-variable))
    (and (dimension-mask-of obj)
         (not (static-asize-of obj)))))

(def method initialize-instance :after ((obj gpu-variable) &key &allow-other-keys)
  (with-slots (dimension-mask static-asize) obj
    (unless (slot-boundp obj 'static-asize)
      (setf static-asize
            (if (and dimension-mask (every #'numberp dimension-mask))
                (reduce #'* dimension-mask)
                nil)))))

(def class* gpu-global-var (gpu-variable)
  ((index          :documentation "Ordinal index for fast access.")
   (constant-var?  nil :accessor constant-var? :type boolean
                   :documentation "Specifies allocation in constant memory."))
  (:documentation "A global variable in a GPU module."))

(def class* gpu-argument (gpu-variable)
  ((includes-locked? nil :accessor includes-locked? :type boolean)
   (include-size?    nil :accessor include-size? :type boolean)
   (included-dims    nil :documentation "Mask of dimensions to append.")
   (include-extent?  nil :accessor include-extent? :type boolean)
   (included-strides nil :documentation "Mask of strides to append."))
  (:documentation "A GPU function or kernel parameter."))

(def method initialize-instance :after ((obj gpu-argument) &key &allow-other-keys)
  (with-slots (dimension-mask included-dims included-strides) obj
    (when dimension-mask
      (unless included-dims
        (setf included-dims
              (make-array (length dimension-mask) :initial-element nil)))
      (unless included-strides
        (setf included-strides
              (make-array (1- (length dimension-mask)) :initial-element nil))))))

(def class* gpu-function ()
  ((name           :documentation "Lisp name of the function")
   (c-name         :documentation "C name of the function")
   (return-type    :documentation "Return type")
   (arguments      :documentation "List of arguments")
   (body           :documentation "Body tree"))
  (:documentation "A function usable on the GPU"))

(def class* gpu-kernel (gpu-function)
  ((index          :documentation "Ordinal for fast access"))
  (:default-initargs :return-type :void)
  (:documentation "A kernel callable from the host"))

(def class* gpu-module ()
  ((name            :documentation "Lisp name of the module")
   (globals         :documentation "List of global variables")
   (functions       :documentation "List of helper functions")
   (kernels         :documentation "List of kernel functions")
   (index-table     (make-hash-table)
                    :documentation "An index assignment table")
   (compiled-code   :documentation "Code string")
   (change-sentinel (cons nil nil)
                    :documentation "Used to trigger module reloads"))
  (:documentation "A module that can be loaded to the GPU."))

(defstruct gpu-module-instance
  module change-sentinel var-vector kernel-vector)

