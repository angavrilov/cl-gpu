;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file defines miscellaneous utility functions;
;;; some of them depend on the lisp implementation.
;;;

(in-package :cl-gpu.buffers)

;;; Pointer type

(defconstant +foreign-pointer-type-name+
  #+ecl 'si:foreign-data
  #+openmcl 'ccl:macptr
  #+sbcl 'sb-alien:system-area-pointer)

;;; Misc

(declaim (inline ensure-cdr *->nil nil->*))

(def function ensure-cdr (item)
  (if (consp item) (cdr item)))

(def function *->nil (x)
  (if (eq x '*) nil x))

(def function nil->* (x)
  (or x '*))

(def function extract-power-of-two (value)
  "Returns the integer part of base-2 logarithm and the remainder."
  (loop for i from 0
     when (or (logtest value 1) (<= value 0))
     return (values i value)
     else do (setf value (ash value -1))))

(def function to-uint32-vector (list)
  (make-array (length list) :element-type '(unsigned-byte 32) :initial-contents list))

;;; Memoize

(def macro gethash-with-init (key table init-expr)
  "Looks up the key in the table. When not found, lazily initializes with init-expr."
  (with-unique-names (item found)
    (once-only (key table)
      `(multiple-value-bind (,item ,found) (gethash ,key ,table)
         (if ,found ,item
             (setf (gethash ,key ,table) ,init-expr))))))

(def macro with-memoize ((key &rest flags) &body code)
  "Memoizes the result of the code block using key; flags are passed to make-hash-table."
  `(gethash-with-init ,key
                      (load-time-value (make-hash-table ,@flags))
                      (progn ,@code)))

;; Memory

(defcfun "memcpy" :pointer
  (dest :pointer)
  (src :pointer)
  (count :unsigned-int))

(defcfun "memmove" :pointer
  (dest :pointer)
  (src :pointer)
  (count :unsigned-int))

(defcfun "memset" :pointer
  (dest :pointer)
  (byte :int)
  (count :unsigned-int))

;; Slots

(def macro with-slot-values (slots object &body code)
  (once-only (object)
    `(let ,(loop for spec in slots
              for name = (ensure-car spec)
              for slot = (if (consp spec) (second spec) `(quote ,name))
              collect `(,name (slot-value ,object ,slot)))
       ,@code)))

