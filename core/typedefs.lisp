;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

(deftype uint8 () '(unsigned-byte 8))
(deftype uint16 () '(unsigned-byte 16))
(deftype uint32 () '(unsigned-byte 32))
(deftype uint64 () '(unsigned-byte 64))

(deftype int8 () '(signed-byte 8))
(deftype int16 () '(signed-byte 16))
(deftype int32 () '(signed-byte 32))
(deftype int64 () '(signed-byte 64))

(let ((cvtable (make-hash-table :test #'eq)))
  (def function canonify-foreign-type (type)
    "Computes a canonic form of a C type."
    (or (gethash type cvtable) type))
  ;; Init
  (dolist (item '(:char :short :int :long :long-long :llong))
    (setf (gethash item cvtable)
          (svref  #(:int8 :int8 :int16 :int32 :int32 :int64 :int64 :int64 :int64)
                  (foreign-type-size item))))
  (dolist (item '(:unsigned-char :unsigned-short :unsigned-int
                  :unsigned-long :unsigned-long-long :uchar
                  :ushort :uint :ulong :ullong))
    (setf (gethash item cvtable)
          (svref  #(:uint8 :uint8 :uint16 :uint32 :uint32 :uint64 :uint64 :uint64 :uint64)
                  (foreign-type-size item))))
  (dolist (item '(:int8 :int16 :int32 :int64 :uint8 :uint16 :uint32 :uint64
                  :float :double :void))
    (setf (gethash item cvtable) item)))

(let ((cvtable (make-hash-table :test #'equal)))
  (def function canonify-lisp-type (type)
    "Computes a canonic form of a lisp type."
    (or (gethash type cvtable)
        (setf (gethash type cvtable)
              (or (cond ((subtypep type 'unsigned-byte)
                         (cond ((subtypep type '(unsigned-byte 8)) 'uint8)
                               ((subtypep type '(unsigned-byte 16)) 'uint16)
                               ((subtypep type '(unsigned-byte 32)) 'uint32)
                               ((subtypep type '(unsigned-byte 64)) 'uint64)))
                        ((subtypep type 'signed-byte)
                         (cond ((subtypep type '(signed-byte 8)) 'int8)
                               ((subtypep type '(signed-byte 16)) 'int16)
                               ((subtypep type '(signed-byte 32)) 'int32)
                               ((subtypep type '(signed-byte 64)) 'int64)))
                        ((subtypep type 'single-float) 'single-float)
                        ((subtypep type 'double-float) 'double-float))
                  type))))
  ;; Init
  (dolist (item '(uint8 uint16 uint32 uint64 int8 int16 int32 int64
                  single-float double-float fixnum))
    (setf (gethash item cvtable) item))
  (dolist (item '((unsigned-byte 8) (unsigned-byte 16) (unsigned-byte 32) (unsigned-byte 64)
                  (signed-byte 8) (signed-byte 16) (signed-byte 32) (signed-byte 64)))
    (canonify-lisp-type item)
    (canonify-lisp-type (upgraded-array-element-type item))))

(def (function e) foreign-to-lisp-type (type)
  "Converts a foreign type to a lisp supertype."
  (case (canonify-foreign-type type)
    (:int8 'int8) (:int16 'int16) (:int32 'int32) (:int64 'int64)
    (:uint8 'uint8) (:uint16 'uint16) (:uint32 'uint32) (:uint64 'uint64)
    (:float 'single-float) (:double 'double-float)))

(def (function e) lisp-to-foreign-type (type)
  "Converts a lisp type to a foreign supertype."
  (case (canonify-lisp-type type)
    (int8 :int8) (int16 :int16) (int32 :int32) (int64 :int64)
    (uint8 :uint8) (uint16 :uint16) (uint32 :uint32) (uint64 :uint64)
    (single-float :float) (double-float :double)))

(let ((cvtable (make-hash-table :test #'eq)))
  (def (function e) foreign-to-lisp-elt-type (type)
    "Converts a foreign type to an equivalent lisp array elt type. NIL if none."
    (gethash type cvtable))
  (dolist (type '(:int8 :int16 :int32 :int64 :uint8 :uint16 :uint32 :uint64
                  :float :double))
    (setf (gethash type cvtable)
          (let* ((cvent (foreign-to-lisp-type type))
                 (elttype (upgraded-array-element-type cvent)))
            (if (subtypep elttype cvent) elttype nil)))))

(let ((cvtable (make-hash-table :test #'equal)))
  (def (function e) lisp-to-foreign-elt-type (type)
    "Converts a lisp type to an equivalent foreign type. NIL if none."
    (multiple-value-bind (res found) (gethash type cvtable)
      (if found res
          (setf (gethash type cvtable)
                (let ((cvent (lisp-to-foreign-type type)))
                  (if (subtypep (foreign-to-lisp-type cvent) type)
                      cvent nil))))))
  ;; Init
  (dolist (item '(uint8 uint16 uint32 uint64 int8 int16 int32 int64
                  single-float double-float))
    (lisp-to-foreign-elt-type item)
    (lisp-to-foreign-elt-type (upgraded-array-element-type item))))
