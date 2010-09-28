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

(in-package :cl-gpu.buffers)

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

(def method print-object :around ((object gpu-concrete-type) stream)
  ;; Don't print the GPU- prefix:
  (format stream "#<~A" (subseq (symbol-name (class-name (class-of object))) 4))
  (call-next-method)
  (princ ">" stream))

(def method print-object ((object gpu-concrete-type) stream)
  (awhen (initarg-values-of object)
    (format stream "~{ ~S~}" it)))

(def class gpu-any-type (gpu-concrete-type)
  ()
  (:metaclass interned-class)
  (:documentation "An unspecified type."))

(let () (def constant +gpu-any-type+ (make-instance 'gpu-any-type)))

(def method lisp-type-of ((type gpu-any-type)) t)

(def class gpu-no-type (gpu-concrete-type)
  ()
  (:metaclass interned-class)
  (:documentation "A nonexistant type."))

(let () (def constant +gpu-no-type+ (make-instance 'gpu-no-type)))

(def method lisp-type-of ((type gpu-no-type)) nil)

(def class gpu-keyword-type (gpu-concrete-type)
  ()
  (:metaclass interned-class)
  (:documentation "A keyword type."))

(let () (def constant +gpu-keyword-type+ (make-instance 'gpu-keyword-type)))

(def method lisp-type-of ((type gpu-keyword-type)) 'keyword)

;; Numbers

(def class gpu-number-type (gpu-concrete-type)
  ((min-value :initform nil :initarg :min-value :reader min-value-of)
   (max-value :initform nil :initarg :max-value :reader max-value-of))
  (:metaclass interned-class)
  (:documentation "A numeric GPU value type."))

(def function lisp-real-type-name (type tag &key (cv #'identity) (min-cv cv) (max-cv cv))
  (let ((min (min-value-of type))
        (max (max-value-of type)))
    (if (or min max)
        (list tag
              (if min (funcall min-cv min) '*)
              (if max (funcall max-cv max) '*))
        tag)))

(def method lisp-type-of ((type gpu-number-type))
  (lisp-real-type-name type 'real))

(def method print-object ((object gpu-number-type) stream)
  (with-slot-values (min-value max-value) object
    (when (or min-value max-value)
      (format stream " ~A..~A" (or min-value '*) (or max-value '*)))))

(def class gpu-integer-type (gpu-number-type)
  ()
  (:metaclass interned-class)
  (:documentation "An integer GPU value type."))

(def method lisp-type-of ((type gpu-integer-type))
  (lisp-real-type-name type 'integer :min-cv #'ceiling :max-cv #'floor))

(def class gpu-float-type (gpu-number-type)
  ()
  (:metaclass interned-class)
  (:documentation "An floating-point GPU value type."))

(def method lisp-type-of ((type gpu-float-type))
  (lisp-real-type-name type 'float :cv #'float))

;; Native types

(def class gpu-native-type (gpu-type)
  ()
  (:documentation "An abstract class denoting types that map to foreign ones."))

(def layered-function specific-type-p (type)
  (:documentation "Determines if the type is fully specific")
  (:method ((type gpu-type)) nil)
  (:method ((type null)) nil)
  (:method ((type gpu-native-type)) t))

(def class gpu-native-number-type (gpu-number-type gpu-native-type)
  ()
  (:metaclass interned-class)
  (:documentation "An abstract class denoting foreign number types."))

(def generic make-foreign-gpu-type (id &key)
  (:documentation "Returns a gpu-type object for the specified CFFI type.")
  (:method ((id gpu-type) &key)
    id))

(def generic foreign-type-of (type)
  (:documentation "Returns the CFFI descriptor for a native type")
  (:method ((type t))
    nil))

(def layered-function native-type-c-string (type)
  (:documentation "Returns the C string for a native type"))

(def (function i) c-type-string (type) (native-type-c-string type))

(def generic native-type-byte-size (type)
  (:documentation "Returns the byte size of a native type"))

(def (function i) c-type-size (type) (native-type-byte-size type))

(def layered-function native-type-alignment (type)
  (:documentation "Returns the byte alignment requirements of a native type"))

(def (function i) c-type-alignment (type) (native-type-alignment type))

(def generic native-type-ref (type ptr offset)
  (:documentation "Reads the native type value from the specified ptr & offset."))

(def generic (setf native-type-ref) (value type ptr offset)
  (:documentation "Writes the native type value to the specified ptr & offset."))

(def macro def-native-type-info (class fid cstring size alignment &key min-limit max-limit not-number?)
  `(progn
     ,(if not-number?
          `(def method make-foreign-gpu-type ((id (eql ,fid)) &key)
             (make-instance ',class))
          `(def method make-foreign-gpu-type ((id (eql ,fid)) &key min-value max-value)
             (make-instance ',class
                            :min-value ,(if min-limit `(max (or min-value ,min-limit) ,min-limit) 'min-value)
                            :max-value ,(if max-limit `(min (or max-value ,max-limit) ,max-limit) 'max-value))))
     (let () ; Make defconstant non-toplevel to avoid
             ; compile-time evaluation
       (def constant ,(symbolicate "+" class "+") (make-foreign-gpu-type ,fid)))
     (def method foreign-type-of ((type ,class))
       ,fid)
     ,(when cstring
            `(def layered-method native-type-c-string ((type ,class))
               ,cstring))
     (def-native-type-refs ,class ,fid ,size ,alignment)))

(def macro def-native-type-refs (class fid size alignment
                                       &aux (fp-type +foreign-pointer-type-name+))
  (assert (= size (foreign-type-size fid)))
  `(progn
     (def method native-type-byte-size ((type ,class))
       ,size)
     (def layered-method native-type-alignment ((type ,class))
       ,alignment)
     (def method native-type-ref ((type ,class) (ptr ,fp-type) offset)
       (mem-ref ptr ,fid offset))
     (def method native-type-ref ((type ,class) (ptr function) offset)
       (with-foreign-object (tmp ,fid)
         (funcall ptr tmp offset ,size t)
         (mem-ref tmp ,fid)))
     (def method (setf native-type-ref) (value (type ,class) (ptr ,fp-type) offset)
       (setf (mem-ref ptr ,fid offset) value))
     (def method (setf native-type-ref) (value (type ,class) (ptr function) offset)
       (with-foreign-object (tmp ,fid)
        (prog1
            (setf (mem-ref tmp ,fid) value)
          (funcall ptr tmp offset ,size nil))))))

;; Floating-point native types

(def class gpu-native-float-type (gpu-float-type gpu-native-number-type)
  ()
  (:metaclass interned-class)
  (:documentation "An abstract class denoting foreign float types."))

(def class gpu-single-float-type (gpu-native-float-type)
  ()
  (:metaclass interned-class)
  (:documentation "A single-precision floating-point GPU value type."))

(def-native-type-info gpu-single-float-type :float "float" 4 4)

(def method lisp-type-of ((type gpu-single-float-type))
  (lisp-real-type-name type 'single-float :cv (lambda (x) (float x 1.0))))

(def class gpu-double-float-type (gpu-native-float-type)
  ()
  (:metaclass interned-class)
  (:documentation "A double-precision floating-point GPU value type."))

(def-native-type-info gpu-double-float-type :double "double" 8 8)

(def method lisp-type-of ((type gpu-double-float-type))
  (lisp-real-type-name type 'double-float :cv (lambda (x) (float x 1.0d0))))

;; Integer native types

(def class gpu-native-integer-type (gpu-integer-type gpu-native-number-type)
  ()
  (:metaclass interned-class)
  (:documentation "An abstract class denoting foreign integer types."))

(macrolet ((mkints (&rest items)
             (loop for (foreign-id c-name bytes signed?) in items
                for class-name = (symbolicate '#:gpu- foreign-id '#:-type)
                for magnitude = (ash 1 (- (* 8 bytes) (if signed? 1 0)))
                for min-value = (if signed? (- magnitude) 0)
                for max-value = (1- magnitude)
                append `((def class ,class-name (gpu-native-integer-type)
                           ()
                           (:metaclass interned-class)
                           (:default-initargs :min-value ,min-value :max-value ,max-value)
                           (:documentation "A native integer GPU value type."))
                         (def method initialize-instance :before ((self ,class-name) &key min-value max-value)
                           (assert (and (>= min-value ,min-value) (<= max-value ,max-value))))
                         (def method print-object ((object ,class-name) stream)
                           (with-slot-values (min-value max-value) object
                             (let ((min (if (> min-value ,min-value) min-value))
                                   (max (if (< max-value ,max-value) max-value)))
                               (when (or min max)
                                 (format stream " ~A..~A" (or min 'min) (or max 'max))))))
                         (def-native-type-info ,class-name ,foreign-id ,c-name ,bytes ,bytes
                                               :min-limit ,min-value :max-limit ,max-value))
                into classes
                collect `((subtypep type '(integer ,min-value ,max-value))
                          (make-instance ',class-name))
                into from-lisp
                collect `((and (>= min-value ,min-value) (<= max-value ,max-value))
                          (make-instance ',class-name :min-value min-value :max-value max-value))
                into from-range
                collect foreign-id into fids
                finally
                  (return `(progn
                             ,@classes
                             (def (constant e :test 'equalp) +gpu-integer-foreign-ids+ ,(coerce fids 'vector))
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

;; Compound types

(def class gpu-compound-type (gpu-concrete-type)
  ()
  (:metaclass interned-class)
  (:documentation "An abstract class for compound types."))

(def class gpu-values-type (gpu-compound-type)
  ((values :initform nil :initarg :values :reader values-of))
  (:metaclass interned-class)
  (:documentation "A multiple-value return type."))

(def method print-object ((object gpu-values-type) stream)
  (awhen (values-of object)
    (format stream "~{ ~S~}" it)))

(def method lisp-type-of ((type gpu-values-type))
  (list* 'values (mapcar #'lisp-type-of (values-of type))))

(def layered-method specific-type-p ((type gpu-values-type))
  (every #'specific-type-p (values-of type)))

(let ()
  (def constant +gpu-void-type+ (make-instance 'gpu-values-type)))

(def layered-method native-type-c-string ((type (eql (make-instance 'gpu-values-type))))
  "void")

(def class gpu-container-type (gpu-compound-type)
  ((item-type :initform nil :initarg :item-type :reader item-type-of)
   (dimensions :initform nil :initarg :dimensions :reader dimensions-of))
  (:metaclass interned-class)
  (:documentation "A value container/reference type."))

(def method print-object ((object gpu-container-type) stream)
  (with-slot-values (item-type dimensions) object
    (when (or item-type dimensions)
      (format stream " ~:[*~;~:*~S~]~@[ ~S~]"
              item-type (mapcar #'nil->* dimensions)))))

(def class gpu-array-type (gpu-container-type)
  ()
  (:metaclass interned-class)
  (:documentation "An array value types."))

(def method lisp-type-of ((type gpu-array-type))
  `(array ,(aif (item-type-of type) (lisp-type-of it) '*)
          ,(aif (dimensions-of type)
                (mapcar #'nil->* it)
                '*)))

(def layered-method specific-type-p ((type gpu-array-type))
  (and (dimensions-of type) (specific-type-p (item-type-of type))))

(def class gpu-pointer-type (gpu-container-type)
  ()
  (:metaclass interned-class)
  (:documentation "A generic pointer value type."))

(def layered-method specific-type-p ((type gpu-pointer-type))
  t)

(def layered-method native-type-c-string ((type gpu-pointer-type))
  (if (specific-type-p (item-type-of type))
      (concatenate 'string (native-type-c-string (item-type-of type)) "*")
      "void*"))

(def class gpu-32b-pointer-type (gpu-pointer-type gpu-native-type)
  ()
  (:metaclass interned-class)
  (:documentation "A 32-bit pointer value type."))

(def-native-type-refs gpu-32b-pointer-type :uint32 4 4)

(def class gpu-64b-pointer-type (gpu-pointer-type gpu-native-type)
  ()
  (:metaclass interned-class)
  (:documentation "A 64-bit pointer value type."))

(def-native-type-refs gpu-64b-pointer-type :uint64 8 8)

(def layered-function default-pointer-type ()
  (:documentation "Returns the class of the usual pointer type. To be redefined by targets.")
  (:method () 'gpu-pointer-type))

;; Tuple (fixed vector) type

(def class gpu-tuple-type (gpu-compound-type gpu-native-type)
  ((item-type :initform nil :initarg :item-type :reader item-type-of)
   (size :initform nil :initarg :size :reader size-of)
   (item-lisp-type)
   (item-byte-size))
  (:metaclass interned-class)
  (:documentation "A tuple type."))

(def method print-object ((object gpu-tuple-type) stream)
  (with-slot-values (item-type size) object
    (when (or item-type size)
      (format stream " ~:[*~;~:*~S~]~@[ ~S~]" item-type size))))

(def constructor gpu-tuple-type
  (with-slots (item-type item-lisp-type item-byte-size) -self-
    ;; Cache some properties of the item for use in native-type-ref
    (when item-type
      (setf item-lisp-type (lisp-type-of item-type)
            item-byte-size (ignore-errors (native-type-byte-size item-type))))))

(def method lisp-type-of ((type gpu-tuple-type))
  `(tuple ,(aif (item-type-of type) (lisp-type-of it) '*)
          ,(nil->* (size-of type))))

(def layered-method specific-type-p ((type gpu-tuple-type))
  (and (size-of type) (specific-type-p (item-type-of type))))

(def method native-type-byte-size ((type gpu-tuple-type))
  (* (native-type-byte-size (item-type-of type)) (size-of type)))

(def layered-method native-type-alignment ((type gpu-tuple-type))
  (* (extract-power-of-two (size-of type))
     (native-type-alignment (item-type-of type))))

(def method native-type-ref ((type gpu-tuple-type) ptr offset)
  (with-slot-values (item-type item-lisp-type item-byte-size size) type
    (assert (and item-byte-size size))
    (let ((result (make-array (list size) :element-type item-lisp-type)))
      (loop for i from 0 below size
         do (setf (row-major-aref result i)
                  (native-type-ref item-type ptr (+ offset (* i item-byte-size)))))
      result)))

(def method (setf native-type-ref) (value (type gpu-tuple-type) ptr offset)
  (check-type value array)
  (with-slot-values (item-type item-byte-size size) type
    (assert (and item-byte-size size))
    (assert (= size (array-total-size value)))
    (loop for i from 0 below size
       do (setf (native-type-ref item-type ptr (+ offset (* i item-byte-size)))
                (row-major-aref value i)))
    value))

;;; Lisp type parsing code

(def generic do-parse-lisp-type (name type-spec &key error-cb)
  (:documentation "Parses the lisp type specifier")
  (:method ((name t) type-spec &key error-cb)
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
           (funcall (or error-cb #'error) "Unsupported type spec: ~S" type-spec))))
  (:method ((name gpu-type) type-spec &key error-cb)
    (declare (ignore error-cb))
    (assert (eq name type-spec))
    name))

(def definer lisp-type-parser (name args &body code)
  `(defmethod do-parse-lisp-type ((-name- (eql ',name)) -whole- &key ((:error-cb -error-cb-)))
     (declare (ignorable -name- -error-cb-))
     (flet ((-recurse- (type)
              (if (eq type '*) nil
                  (parse-lisp-type type :error-cb -error-cb-)))
            (-error- (&rest args)
              (apply (or -error-cb- #'error) args)))
       (block ,name
         (destructuring-bind ,args (ensure-cdr -whole-)
           ,@code)))))

(def function parse-lisp-type (type-spec &key error-cb)
  (multiple-value-bind (rspec expanded?)
      (expand-gpu-type (ensure-car type-spec) (ensure-cdr type-spec))
    (if expanded?
        (parse-lisp-type rspec :error-cb error-cb)
        (do-parse-lisp-type (ensure-car type-spec) type-spec :error-cb error-cb))))

(def macro with-no-stars (vars &body code)
  `(let ,(loop for var in vars collect `(,var (if (eq ,var '*) nil ,var)))
     ,@code))

;; Parsers:

(def lisp-type-parser t ()
  (make-instance 'gpu-any-type))

(def lisp-type-parser nil ()
  (make-instance 'gpu-no-type))

(def lisp-type-parser fixnum ()
  (make-instance 'gpu-int32-type))

(def lisp-type-parser real (&optional min-value max-value)
  (with-no-stars (min-value max-value)
    (make-instance 'gpu-number-type :min-value min-value :max-value max-value)))

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

(def lisp-type-parser values (&rest items)
  (make-instance 'gpu-values-type :values (mapcar #'-recurse- items)))

(def lisp-type-parser array (&optional (item-type '*) dims)
  (let ((dimv (cond ((or (null dims) (eq dims '*))
                     nil)
                    ((numberp dims)
                     (loop for i from 1 to dims collect nil))
                    ((consp dims)
                     (mapcar #'*->nil dims))
                    (t
                     (-error- "Invalid array type spec: ~S" -whole-)))))
    (make-instance 'gpu-array-type :item-type (-recurse- item-type) :dimensions dimv)))

(def lisp-type-parser vector (&optional (item-type '*) dim)
  (make-instance 'gpu-array-type :item-type (-recurse- item-type) :dimensions (list (*->nil dim))))

(def lisp-type-parser tuple (item-type &optional size)
  (make-instance 'gpu-tuple-type :item-type (-recurse- item-type) :size (*->nil size)))

;;; Type conversion functions

(def (function e) lisp-to-gpu-type (type)
  "Converts a foreign type to an equivalent gpu-type object. NIL if none."
  (with-memoize (type :test #'equal)
    (if (typep type 'gpu-type)
        type
        (let* ((cvtype (parse-lisp-type type))
               (cvltype (lisp-type-of cvtype)))
          (if (subtypep cvltype type) cvtype nil)))))

(def (function e) foreign-to-gpu-type (type)
  "Converts a foreign type to a gpu-type object."
  (make-foreign-gpu-type (canonify-foreign-type type)))

(def (function e) foreign-to-lisp-type (type)
  "Converts a foreign type to a lisp supertype."
  (lisp-type-of (foreign-to-gpu-type type)))

(def (function e) foreign-to-lisp-elt-type (type)
  "Converts a foreign type to an equivalent lisp array elt type. NIL if none."
  (with-memoize (type :test #'eq)
    (let* ((cvent (foreign-to-lisp-type type))
           (elttype (upgraded-array-element-type cvent)))
      (if (subtypep elttype cvent) elttype nil))))

(def (function e) lisp-to-foreign-type (type)
  "Converts a lisp type to a foreign supertype."
  (foreign-type-of (lisp-to-gpu-type type)))

(def (function e) lisp-to-foreign-elt-type (type)
  "Converts a lisp type to an equivalent foreign type. NIL if none."
  (with-memoize (type :test #'equal)
    (let ((cvent (lisp-to-foreign-type type)))
      (if (subtypep (foreign-to-lisp-type cvent) type) cvent nil))))
