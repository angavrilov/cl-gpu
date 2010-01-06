;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by the authors.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

#+ccl
(def function double-offset-fixup (ivector)
  (case (ccl::typecode ivector)
    ((#.target::subtag-double-float-vector
      #+64-bit-target #.target::subtag-s64-vector
      #+64-bit-target #.target::subtag-u64-vector
      #+64-bit-target #.target::subtag-fixnum-vector)
     #.(- target::misc-dfloat-offset
          target::misc-data-offset))
    (otherwise 0)))

#+ccl
(def function array-ivector-range (arr)
  (multiple-value-bind (ivector ofs)
      (ccl::array-data-and-offset arr)
    (let* ((typecode (ccl::typecode ivector))
           (fixup (double-offset-fixup ivector))
           (base (+ fixup (ccl::subtag-bytes typecode ofs)))
           (size (ccl::subtag-bytes typecode (array-total-size arr))))
      (values ivector base size))))

(def (function e) write-array-bytes (array stream)
  #+ecl (write-sequence (ext:array-raw-data array) stream)
  #+ccl (multiple-value-bind (ivector base size)
            (array-ivector-range array)
          (ccl:stream-write-ivector stream ivector base size))
  #-(or ecl ccl) (error "Not implemented"))

(def (function e) read-array-bytes (array stream)
  #+ecl (read-sequence (ext:array-raw-data array) stream)
  #+ccl (multiple-value-bind (ivector base size)
            (array-ivector-range array)
          (ccl:stream-read-ivector stream ivector base size))
  #-(or ecl ccl) (error "Not implemented"))

(def function array-type-tag (array)
  (ecase (array-element-type array)
    (single-float #x46525241)
    (double-float #x44525241)))

(def function array-type-by-tag (tag)
  (ecase tag
    (#x46525241 'single-float)
    (#x44525241 'double-float)))

(def (function e) write-array (array stream)
  (let* ((dims (list* (array-type-tag array)
                      (array-rank array)
                      (array-dimensions array)))
         (header (make-array (length dims)
                             :element-type '(unsigned-byte 32)
                             :initial-contents dims)))
    (declare (dynamic-extent dims header))
    (write-array-bytes header stream)
    (write-array-bytes array stream)))

(def (function e) read-array (array stream &key allocate)
  (let ((base-hdr (make-array 2 :element-type '(unsigned-byte 32))))
    (declare (dynamic-extent base-hdr))
    (read-array-bytes base-hdr stream)
    (let ((type (array-type-by-tag (aref base-hdr 0)))
          (dims (make-array (aref base-hdr 1)
                            :element-type '(unsigned-byte 32))))
      (declare (dynamic-extent dims))
      (read-array-bytes dims stream)
      (if array
          (let ((adims (array-dimensions array)))
            (unless (equal (upgraded-array-element-type type)
                           (array-element-type array))
              (error "Type mismatch: array ~A, file ~A"
                     (array-element-type array) type))
            (unless (and (eql (length dims) (length adims))
                         (every #'eql adims dims))
              (error "Dimension mismatch: array ~A, file ~A" adims dims)))
          (progn
            (unless allocate
              (cerror "allocate" "Array is nil in read-array"))
            (setf array (make-array (coerce dims 'list)
                                    :element-type type))))
      (read-array-bytes array stream)
      array)))
