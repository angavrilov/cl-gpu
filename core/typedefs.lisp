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

(def function canonify-foreign-type (type)
  "Computes a canonic form of a C type."
  (with-memoize (type :test #'eq)
    (case type
      ((:char :short :int :long :long-long :llong)
       (svref  #(:int8 :int8 :int16 :int32 :int32 :int64 :int64 :int64 :int64)
               (foreign-type-size type)))
      ((:uchar :unsigned-char :ushort :unsigned-short :uint
               :unsigned-int :ulong :unsigned-long
               :unsigned-long-long :ullong)
       (svref  #(:uint8 :uint8 :uint16 :uint32 :uint32 :uint64 :uint64 :uint64 :uint64)
               (foreign-type-size type)))
      (t type))))

(def function canonify-lisp-type (type)
  "Computes a canonic form of a lisp type."
  (with-memoize (type :test #'equal)
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
        type)))

(def (function e) foreign-to-lisp-type (type)
  "Converts a foreign type to a lisp supertype."
  (case (canonify-foreign-type type)
    (:int8 'int8) (:int16 'int16) (:int32 'int32) (:int64 'int64)
    (:uint8 'uint8) (:uint16 'uint16) (:uint32 'uint32) (:uint64 'uint64)
    (:float 'single-float) (:double 'double-float) (:void nil)))

(def (function e) lisp-to-foreign-type (type)
  "Converts a lisp type to a foreign supertype."
  (case (canonify-lisp-type type)
    (int8 :int8) (int16 :int16) (int32 :int32) (int64 :int64)
    (uint8 :uint8) (uint16 :uint16) (uint32 :uint32) (uint64 :uint64)
    (single-float :float) (double-float :double) ((nil) :void)))

(def (function e) foreign-to-lisp-elt-type (type)
  "Converts a foreign type to an equivalent lisp array elt type. NIL if none."
  (with-memoize (type :test #'eq)
    (let* ((cvent (foreign-to-lisp-type type))
           (elttype (upgraded-array-element-type cvent)))
      (if (subtypep elttype cvent) elttype nil))))

(def (function e) lisp-to-foreign-elt-type (type)
  "Converts a lisp type to an equivalent foreign type. NIL if none."
  (with-memoize (type :test #'equal)
    (let ((cvent (lisp-to-foreign-type type)))
      (if (subtypep (foreign-to-lisp-type cvent) type) cvent nil))))

