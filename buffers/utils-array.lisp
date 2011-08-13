;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file defines utility functions for working
;;; with arrays; some depend on the implementation.
;;;

(in-package :cl-gpu.buffers)

;;; Array element size

#+sbcl
(sb-ext:defglobal %%size-shift-table%%
    (let ((arr (make-array (1+ sb-vm:widetag-mask) :initial-element nil)))
      (loop
         for info across sb-vm:*specialized-array-element-type-properties*
         for shift = (and (subtypep (sb-vm:saetp-specifier info) 'number)
                          (not (sb-vm:saetp-fixnum-p info))
                          (case (sb-vm:saetp-n-bits info)
                            (8 0) (16 1) (32 2) (64 3) (128 4)))
         when shift
         do (setf (svref arr (sb-vm:saetp-typecode info)) shift))
      arr)
  "A table of element size shifts for supported array element types.")

#+sbcl
(declaim (inline vector-elt-shift-of)
         (ftype (function (t) (integer 0 4)) vector-elt-shift-of))

#+sbcl
(defun vector-elt-shift-of (obj)
  (declare (optimize (safety 0)))
  (or (svref %%size-shift-table%%
             (if (sb-vm::%other-pointer-p obj)
                 (sb-kernel:%other-pointer-widetag obj)
                 0))
      (error "Not an array, or element type not supported: ~A" obj)))

;;; Array I/O

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

#+ecl
(defun %array-pointer (arr)
  "Return the address of array's data."
  (check-type arr array)
  (ffi:c-inline (arr) (:object) :object
    "switch (#0->array.elttype) {
       case aet_object: FEerror(\"Not a specialized array: ~S\", 1, #0); break;
       case aet_bit: FEerror(\"Cannot get a pointer to a bit array: ~S\", 1, #0); break;
       default: @(return) = ecl_make_pointer((#0)->array.self.b8);
     }"))

(def macro with-pointer-to-array ((ptr-var arr) &body code)
  #+ccl
  (with-unique-names (iv bv)
    `(multiple-value-bind (,iv ,bv) (array-ivector-range ,arr)
       (ccl:with-pointer-to-ivector (,ptr-var ,iv)
         (incf-pointer ,ptr-var ,bv)
         ,@code)))
  #+sbcl
  (with-unique-names (iv bv ev)
    `(sb-kernel:with-array-data ((,iv ,arr) (,bv 0) (,ev))
       (declare (ignore ,ev))
       (sb-sys:with-pinned-objects (,iv)
         (let ((,ptr-var (sb-sys:sap+ (sb-sys:vector-sap ,iv)
                                      (ash ,bv (vector-elt-shift-of ,iv)))))
           ,@code))))
  #+ecl
  `(let ((,ptr-var (%array-pointer ,arr)))
     ,@code)
  #-(or ecl ccl sbcl)
  (error "Not implemented"))

(def macro with-lisp-array-ref (((ptr-var) arr index elt-size) &body code)
  `(with-pointer-to-array (,ptr-var ,arr)
     (incf-pointer ,ptr-var (* ,index ,elt-size))
     ,@code))

(def (function e) write-array-bytes (array stream)
  "Write the contents of the array to a binary stream."
  #+ecl (write-sequence (ext:array-raw-data array) stream)
  #+ccl (multiple-value-bind (ivector base size)
            (array-ivector-range array)
          (ccl:stream-write-ivector stream ivector base size))
  #+sbcl (sb-kernel:with-array-data ((vector array)
                                     (start 0)
                                     (end (array-total-size array)))
           (vector-elt-shift-of vector) ; assert it is safe
           (sb-vector-io:write-vector-data vector stream :start start :end end))
  #-(or ecl ccl sbcl) (error "Not implemented"))

(def (function e) read-array-bytes (array stream)
  "Restore the contents of the array from a binary stream."
  #+ecl (read-sequence (ext:array-raw-data array) stream)
  #+ccl (multiple-value-bind (ivector base size)
            (array-ivector-range array)
          (ccl:stream-read-ivector stream ivector base size))
  #+sbcl (sb-kernel:with-array-data ((vector array)
                                     (start 0)
                                     (end (array-total-size array)))
           (vector-elt-shift-of vector) ; assert it is safe
           (sb-vector-io:read-vector-data vector stream :start start :end end))
  #-(or ecl ccl sbcl) (error "Not implemented"))

(def function array-type-tag (array)
  (ecase (lisp-to-foreign-elt-type
          (array-element-type array))
    (:float #x46525241)
    (:double #x44525241)
    (:int8 #x62525241)
    (:uint8 #x42525241)
    (:int16 #x73525241)
    (:uint16 #x53525241)
    (:int32 #x69525241)
    (:uint32 #x49525241)
    (:int64 #x71525241)
    (:uint64 #x51525241)))

(def function array-type-by-tag (tag)
  (foreign-to-lisp-elt-type
   (ecase tag
     (#x46525241 :float)
     (#x44525241 :double)
     (#x62525241 :int8)
     (#x42525241 :uint8)
     (#x73525241 :int16)
     (#x53525241 :uint16)
     (#x69525241 :int32)
     (#x49525241 :uint32)
     (#x71525241 :int64)
     (#x51525241 :uint64))))

(def (function e) write-array (array stream)
  "Store the array to a binary stream with meta-data."
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
  "Restore data written using write-array."
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

;;; Array copying

(def function %portable-copy-array-data (src-array src-ofs dest-array dest-ofs count)
  (declare (type fixnum src-ofs dest-ofs count)
           (type array src-array dest-array))
  (if (and (eq dest-array src-array) (> dest-ofs src-ofs))
      (let ((dest-base (+ dest-ofs count -1))
            (src-base (+ src-ofs count -1)))
        (declare (type fixnum dest-base src-base))
        (dotimes (i count)
          (declare (fixnum i))
          (setf (row-major-aref dest-array (- dest-base i))
                (row-major-aref src-array (- src-base i)))))
      (dotimes (i count)
        (declare (fixnum i))
        (setf (row-major-aref dest-array (+ dest-ofs i))
              (row-major-aref src-array (+ src-ofs i))))))

(def function %copy-array-data (src-array src-ofs dest-array dest-ofs count)
  (declare (type fixnum src-ofs dest-ofs count)
           (type array src-array dest-array))
  #+ccl
  (bind (((:values src-vec src-delta)   (ccl::array-data-and-offset src-array))
         ((:values dest-vec dest-delta) (ccl::array-data-and-offset dest-array))
         (src-index (+ src-delta src-ofs))
         (dest-index (+ dest-delta dest-ofs)))
    (cond
      ;; Reference vector
      ((and (ccl::gvectorp src-vec) (ccl::gvectorp dest-vec))
       (ccl::%copy-gvector-to-gvector src-vec src-index dest-vec dest-index count))
      ;; Unaligned bit move
      ((and (or (bit-vector-p src-vec) (bit-vector-p dest-vec))
            (or (logtest src-index 7) (logtest dest-index 7) (logtest count 7)))
       (%portable-copy-array-data src-vec src-index dest-vec dest-index count))
      ;; Raw data vector
      ((and (ccl::ivectorp src-vec) (ccl::ivectorp dest-vec))
       (let* ((type (ccl::typecode dest-vec))
              (fixup (double-offset-fixup dest-vec))
              (src-byte (+ fixup (ccl::subtag-bytes type src-index)))
              (dest-byte (+ fixup (ccl::subtag-bytes type dest-index)))
              (byte-count (ccl::subtag-bytes type count)))
         (assert (eql type (ccl::typecode src-vec)))
         (ccl::%copy-ivector-to-ivector src-vec src-byte dest-vec dest-byte byte-count)))
      ;; Unknown
      (t (error "Array type mismatch"))))
  #+ecl
  (if (and (eq dest-array src-array) (> dest-ofs src-ofs))
      (%portable-copy-array-data src-array src-ofs dest-array dest-ofs count)
      (ffi:c-inline
          (dest-array dest-ofs src-array src-ofs count)
          (:object :int :object :int :int) :void
        "ecl_copy_subarray(#0,#1,#2,#3,#4)"
        :one-liner t :side-effects t))
  #+sbcl
  (sb-kernel:with-array-data ((src-vector src-array)
                              (src-index src-ofs)
                              (src-end (+ src-ofs count)))
    (declare (ignore src-end))
    (sb-kernel:with-array-data ((dest-vector dest-array)
                                (dest-index dest-ofs)
                                (dest-end (+ dest-ofs count)))
      (declare (ignore dest-end))
      (bind ((src-shift  (ignore-errors (vector-elt-shift-of src-vector)))
             (dest-shift (ignore-errors (vector-elt-shift-of dest-vector))))
        (cond ((and src-shift dest-shift (= src-shift dest-shift))
               (sb-sys:with-pinned-objects (src-vector dest-vector)
                 (let ((src-ptr  (sb-sys:sap+ (sb-sys:vector-sap src-vector)
                                              (ash src-index src-shift)))
                       (dest-ptr (sb-sys:sap+ (sb-sys:vector-sap dest-vector)
                                              (ash dest-index dest-shift)))
                       (size (ash count src-shift)))
                   (if (eq src-vector dest-vector)
                       (memmove dest-ptr src-ptr size)
                       (memcpy dest-ptr src-ptr size)))))
              (t
               (%portable-copy-array-data src-array src-ofs dest-array dest-ofs count))))))
  #-(or ccl ecl sbcl)
  (%portable-copy-array-data src-array src-ofs dest-array dest-ofs count))

(declaim (ftype (function (fixnum fixnum fixnum fixnum t) fixnum)
                adjust-copy-count))

(def function adjust-copy-count (src-size src-ofs dest-size dest-ofs count)
  (assert (and (>= dest-ofs 0) (>= src-ofs 0)))
  (if (eql count t)
      (max 0 (min (- dest-size dest-ofs)
                  (- src-size src-ofs)))
      (progn
        (assert (and (>= count 0)
                     (<= (+ dest-ofs count) dest-size)
                     (<= (+ src-ofs count) src-size))
                (count)
                "Copy region out of bounds: ~A (~A left) -> ~A (~A left): ~A elements."
                src-ofs (- src-size src-ofs)
                dest-ofs (- dest-size dest-ofs) count)
        count)))

(def (function e) copy-array-data (src-array src-ofs dest-array dest-ofs count)
  "Copy data elements between arrays. If count is t, it is deduced."
  (assert (equal (array-element-type dest-array)
                 (array-element-type src-array)))
  (let ((dest-size (array-total-size dest-array))
        (src-size (array-total-size src-array)))
    ;; Verify/deduce the region:
    (let ((rcount (adjust-copy-count src-size src-ofs dest-size dest-ofs count)))
      ;; Copy data
      (when (> rcount 0)
        (%copy-array-data src-array src-ofs dest-array dest-ofs rcount))
      ;; Return the count
      rcount)))
