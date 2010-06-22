;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file defines a few convenient lisp types,
;;; a set of classes that describe the type system
;;; of the code converter, and various conversions.
;;;

(in-package :cl-gpu)

;;; GPU-specific type expanders

(def generic expand-gpu-type (name args)
  (:documentation "Expands type aliases defined through gpu-type.")
  (:method ((name t) args)
    (declare (ignore name args))
    (values)))

(def (definer e :available-flags "e") gpu-type (name args &body code)
  ;; Does a deftype that is available to GPU code
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (defmethod expand-gpu-type ((name (eql ',name)) args)
       (values (block ,name
                 (destructuring-bind ,args args
                   ,@code))
               t))
     ,@(if (null (getf -options- :gpu-only))
           `((def (type :export ,(getf -options- :export)) ,name ,args
               ,@code)))))

;;; Some convenient types

(def gpu-type uint8 () '(unsigned-byte 8))
(def gpu-type uint16 () '(unsigned-byte 16))
(def gpu-type uint32 () '(unsigned-byte 32))
(def gpu-type uint64 () '(unsigned-byte 64))

(def gpu-type int8 () '(signed-byte 8))
(def gpu-type int16 () '(signed-byte 16))
(def gpu-type int32 () '(signed-byte 32))
(def gpu-type int64 () '(signed-byte 64))

(def (type e) tuple (item &optional size)
  (check-type size (or null unsigned-byte))
  `(simple-array ,item (,size)))

;;; Misc

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

;;; Classes that describe types as used by the library
;;;
;;; NOTE: Concrete type objects are interned, which allows
;;;       meaningfull comparison with EQ, but requires that
;;;       their field values be constant.
;;;

(def class gpu-type ()
  ()
  (:documentation "A GPU value type"))

(def generic lisp-type-of (type)
  (:documentation "Returns the lisp typespec based on the gpu type"))

(def class gpu-concrete-type (gpu-type interned-object)
  ()
  (:metaclass interned-class)
  (:documentation "A specific GPU value type."))

(def class gpu-number-type (gpu-concrete-type)
  ((min-value :initform nil :initarg :min-value :reader min-value-of)
   (max-value :initform nil :initarg :max-value :reader max-value-of))
  (:metaclass interned-class)
  (:documentation "A numeric GPU value type."))

(def function lisp-real-type-name (type tag)
  (let ((min (min-value-of type))
        (max (max-value-of type)))
    (if (or min max)
        (list tag (or min '*) (or max '*))
        tag)))

(def method lisp-type-of ((type gpu-number-type))
  (lisp-real-type-name type 'real))

(def class gpu-integer-type (gpu-number-type)
  ()
  (:metaclass interned-class)
  (:documentation "An integer GPU value type."))

(def method lisp-type-of ((type gpu-integer-type))
  (lisp-real-type-name type 'integer))

(def class gpu-float-type (gpu-number-type)
  ()
  (:metaclass interned-class)
  (:documentation "An floating-point GPU value type."))

(def method lisp-type-of ((type gpu-float-type))
  (lisp-real-type-name type 'float))

;; Native types

(def class gpu-native-type ()
  ()
  (:documentation "An abstract class denoting types that map to foreign ones."))

(def generic make-foreign-gpu-type (id &key)
  (:documentation "Returns a gpu-type object for the specified CFFI type."))

(def layered-function foreign-type-of (type)
  (:documentation "Returns the CFFI descriptor for a native type"))

(def layered-function native-type-c-string (type)
  (:documentation "Returns the C string for a native type"))

(def layered-function native-type-byte-size (type)
  (:documentation "Returns the byte size of a native type"))

(def layered-function native-type-alignment (type)
  (:documentation "Returns the byte alignment requirements of a native type"))

(def macro def-native-type-info (class fid cstring size alignment &key min-limit max-limit not-number?)
  `(progn
     ,(if not-number?
          `(def method make-foreign-gpu-type ((id (eql ,fid)) &key)
             (make-instance ',class))
          `(def method make-foreign-gpu-type ((id (eql ,fid)) &key (min-value ,min-limit) (max-value ,max-limit))
             (make-instance ',class
                            :min-value ,(if min-limit `(max min-value ,min-limit) 'min-value)
                            :max-value ,(if max-limit `(min max-value ,max-limit) 'max-value))))
     (let ((foo)) ; Make defconstant non-toplevel to avoid
                  ; compile-time evaluation
       (declare (ignore foo))
       (defconstant ,(symbolicate "+" class "+") (make-foreign-gpu-type ,fid)))
     (def layered-method foreign-type-of ((type ,class))
       ,fid)
     ,(when cstring
            `(def layered-method native-type-c-string ((type ,class))
               ,cstring))
     (def layered-method native-type-byte-size ((type ,class))
       ,size)
     (def layered-method native-type-alignment ((type ,class))
       ,alignment)))

;; Floating-point native types

(def class gpu-single-float-type (gpu-float-type gpu-native-type)
  ()
  (:metaclass interned-class)
  (:documentation "A single-precision floating-point GPU value type."))

(def-native-type-info gpu-single-float-type :float "float" 4 4)

(def method lisp-type-of ((type gpu-single-float-type))
  (lisp-real-type-name type 'single-float))

(def class gpu-double-float-type (gpu-float-type gpu-native-type)
  ()
  (:metaclass interned-class)
  (:documentation "A double-precision floating-point GPU value type."))

(def-native-type-info gpu-double-float-type :double "double" 8 8)

(def method lisp-type-of ((type gpu-double-float-type))
  (lisp-real-type-name type 'double-float))

;; Integer native types

(macrolet ((mkints (&rest items)
             (loop for (foreign-id c-name bytes signed?) in items
                for class-name = (symbolicate '#:gpu- foreign-id '#:-type)
                for magnitude = (ash 1 (- (* 8 bytes) (if signed? 1 0)))
                for min-value = (if signed? (- magnitude) 0)
                for max-value = (1- magnitude)
                append `((def class ,class-name (gpu-integer-type gpu-native-type)
                           ()
                           (:metaclass interned-class)
                           (:default-initargs :min-value ,min-value :max-value ,max-value)
                           (:documentation "A native integer GPU value type."))
                         (def method initialize-instance :before ((self ,class-name) &key min-value max-value)
                           (assert (and (>= min-value ,min-value) (<= max-value ,max-value))))
                         (def-native-type-info ,class-name ,foreign-id ,c-name ,bytes ,bytes
                                               :min-limit ,min-value :max-limit ,max-value))
                into classes
                collect `((subtypep type '(integer ,min-value ,max-value))
                          (make-instance ',class-name))
                into from-lisp
                collect `((and (>= min-value ,min-value) (<= max-value ,max-value))
                          (make-instance ',class-name :min-value min-value :max-value max-value))
                into from-range
                finally
                  (return `(progn
                             ,@classes
                             (def function make-gpu-integer-from-lisp-type (type)
                               (cond ,@from-lisp
                                     (t (make-instance 'gpu-integer-type))))
                             (def function make-gpu-integer-from-range (min-value max-value)
                               (cond ((or (null min-value) (null max-value))
                                      (make-instance 'gpu-integer-type
                                                     :min-value min-value :max-value max-value))
                                     ,@from-range
                                     (t (make-instance 'gpu-integer-type
                                                       :min-value min-value :max-value max-value)))))))))
  (mkints (:int8 "signed char" 1 t)
          (:uint8 "unsigned char" 1 nil)
          (:int16 "short" 2 t)
          (:uint16 "unsigned short" 2 nil)
          (:int32 "int" 4 t)
          (:uint32 "unsigned int" 4 nil)
          (:int64 nil 8 t)
          (:uint64 nil 8 nil)))

;; Boolean native type

(def class gpu-boolean-type (gpu-concrete-type gpu-native-type)
  ()
  (:metaclass interned-class)
  (:documentation "A boolean GPU value type."))

(def-native-type-info gpu-boolean-type :boolean "int" 4 4 :not-number? t)

(def method lisp-type-of ((type gpu-boolean-type))
  'boolean)

;;; Lisp type parsing code

(def generic do-parse-lisp-type (name type-spec &key form)
  (:documentation "Parses the lisp type specifier")
  (:method ((name t) type-spec &key form)
    (declare (ignore name))
    ;; Fallback code using subtypep provided by the implementation.
    (cond ((subtypep type-spec 'integer)
           (make-gpu-integer-from-lisp-type type-spec))
          ((subtypep type-spec 'float)
           (cond ((subtypep type-spec 'single-float)
                  (make-instance 'gpu-single-float-type))
                 ((subtypep type-spec 'double-float)
                  (make-instance 'gpu-double-float-type))
                 (t
                  (make-instance 'gpu-float-type))))
          ((subtypep type-spec 'number)
           (make-instance 'gpu-number-type))
          ((subtypep type-spec 'boolean)
           (make-instance 'gpu-boolean-type))
          (t
           (gpu-code-error form "Unsupported type spec: ~S" type-spec)))))

(def definer lisp-type-parser (name args &body code)
  `(defmethod do-parse-lisp-type ((-name- (eql ',name)) -whole- &key ((:form -form-)))
     (declare (ignorable -name- -form-))
     (flet ((-recurse- (type)
              (parse-lisp-type type :form -form-)))
       (block ,name
         (destructuring-bind ,args (ensure-cdr -whole-)
           ,@code)))))

(def function parse-lisp-type (type-spec &key form)
  (multiple-value-bind (rspec expanded?)
      (expand-gpu-type (ensure-car type-spec) (ensure-cdr type-spec))
    (if expanded?
        (parse-lisp-type rspec :form form)
        (do-parse-lisp-type (ensure-car type-spec) type-spec :form form))))

(def macro with-no-stars (vars &body code)
  `(let ,(loop for var in vars collect `(,var (if (eq ,var '*) nil ,var)))
     ,@code))

;; Parsers:

(def lisp-type-parser fixnum ()
  (make-instance 'gpu-int32-type))

(def lisp-type-parser integer (&optional min-value max-value)
  (with-no-stars (min-value max-value)
    (make-gpu-integer-from-range min-value max-value)))

(def lisp-type-parser unsigned-byte (&optional bits)
  (with-no-stars (bits)
    (if bits
        (make-gpu-integer-from-range 0 (1- (ash 1 bits)))
        (make-instance 'gpu-integer-type :min-value 0))))

(def lisp-type-parser signed-byte (&optional bits)
  (with-no-stars (bits)
    (if bits
        (make-gpu-integer-from-range (- (ash 1 (1- bits))) (1- (ash 1 (1- bits))))
        (make-instance 'gpu-integer-type))))

(def lisp-type-parser float (&optional min-value max-value)
  (with-no-stars (min-value max-value)
    (make-instance 'gpu-float-type :min-value min-value :max-value max-value)))

(def lisp-type-parser single-float (&optional min-value max-value)
  (with-no-stars (min-value max-value)
    (make-instance 'gpu-single-float-type :min-value min-value :max-value max-value)))

(def lisp-type-parser double-float (&optional min-value max-value)
  (with-no-stars (min-value max-value)
    (make-instance 'gpu-double-float-type :min-value min-value :max-value max-value)))

;;; Type conversion functions

(def (function e) foreign-to-lisp-type (type)
  "Converts a foreign type to a lisp supertype."
  (lisp-type-of (make-foreign-gpu-type (canonify-foreign-type type))))

(def (function e) foreign-to-lisp-elt-type (type)
  "Converts a foreign type to an equivalent lisp array elt type. NIL if none."
  (with-memoize (type :test #'eq)
    (let* ((cvent (foreign-to-lisp-type type))
           (elttype (upgraded-array-element-type cvent)))
      (if (subtypep elttype cvent) elttype nil))))

(def (function e) lisp-to-foreign-type (type)
  "Converts a lisp type to a foreign supertype."
  (foreign-type-of (parse-lisp-type type)))

(def (function e) lisp-to-foreign-elt-type (type)
  "Converts a lisp type to an equivalent foreign type. NIL if none."
  (with-memoize (type :test #'equal)
    (let ((cvent (lisp-to-foreign-type type)))
      (if (subtypep (foreign-to-lisp-type cvent) type) cvent nil))))
