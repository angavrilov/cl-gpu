;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu.test)

(defsuite* (test/utils :in test))

(defun ignore-warning (condition)
  (declare (ignore condition))
  (muffle-warning))

(def test test/utils/r-w-array ()
  (with-standard-io-syntax
    (let ((arr (make-array 5 :element-type 'single-float
                             :initial-contents '(0.0f0 0.1f0 0.2f0 0.3f0 0.4f0))))
      (with-open-file (s "cl-gpu.test.tmp"
                         :direction :output
                         :if-exists :supersede
                         :element-type '(unsigned-byte 8))
        (write-array arr s))
      (let ((res (with-open-file (s "cl-gpu.test.tmp"
                                    :direction :input
                                    :element-type '(unsigned-byte 8))
                   (read-array nil s :allocate t)))
            (res2 (with-open-file (s "cl-gpu.test.tmp"
                                     :direction :input
                                     :element-type '(unsigned-byte 8))
                    (read-array (make-array 5 :element-type 'single-float) s))))
        (delete-file "cl-gpu.test.tmp")
        (is (equal (array-dimensions arr)
                   (array-dimensions res)))
        (is (every #'eql arr res))
        (is (every #'eql arr res2))))))

(def test test/utils/copy-gvector ()
  (let ((arr1 (make-array 5 :initial-contents '(1 2 3 4 5)))
        (arr2 (make-array 5 :initial-contents '(1 2 3 4 5))))
    (is (= (copy-array-data arr1 0 arr2 2 t) 3))
    (is (every #'eql arr2 #(1 2 1 2 3)))
    (is (= (copy-array-data arr1 0 arr2 0 t) 5))
    (is (every #'eql arr1 arr2))
    (is (= (copy-array-data arr2 0 arr2 2 t) 3))
    (is (every #'eql arr2 #(1 2 1 2 3)))
    (copy-array-data arr1 0 arr2 0 t)
    (is (= (copy-array-data arr2 2 arr2 0 t) 3))
    (is (every #'eql arr2 #(3 4 5 4 5)))))

(def test test/utils/copy-ivector ()
  (let ((arr1 (make-array 5 :element-type '(unsigned-byte 32)
                          :initial-contents '(1 2 3 4 5)))
        (arr2 (make-array 5 :element-type '(unsigned-byte 32)
                          :initial-contents '(1 2 3 4 5))))
    (is (= (copy-array-data arr1 0 arr2 2 t) 3))
    (is (every #'eql arr2 #(1 2 1 2 3)))
    (is (= (copy-array-data arr1 0 arr2 0 t) 5))
    (is (every #'eql arr1 arr2))
    (is (= (copy-array-data arr2 0 arr2 2 t) 3))
    (is (every #'eql arr2 #(1 2 1 2 3)))
    (copy-array-data arr1 0 arr2 0 t)
    (is (= (copy-array-data arr2 2 arr2 0 t) 3))
    (is (every #'eql arr2 #(3 4 5 4 5)))))

(def test test/utils/copy-bvector ()
  (let ((arr1 (make-array 16 :element-type 'bit
                          :initial-contents '(1 0 1 1 0 0 1 0 0 1 0 0 1 1 0 1)))
        (arr2 (make-array 16 :element-type 'bit
                          :initial-contents '(1 0 1 1 0 0 1 0 0 1 0 0 1 1 0 1))))
    (is (= (copy-array-data arr1 0 arr2 8 t) 8))
    (is (every #'eql arr2 #*1011001010110010))
    (is (= (copy-array-data arr1 0 arr2 0 t) 16))
    (is (every #'eql arr1 arr2))
    (is (= (copy-array-data arr2 0 arr2 8 t) 8))
    (is (every #'eql arr2 #*1011001010110010))
    (copy-array-data arr1 0 arr2 0 t)
    (is (= (copy-array-data arr2 8 arr2 0 t) 8))
    (is (every #'eql arr2 #*0100110101001101))
    (copy-array-data arr1 0 arr2 0 t)
    (copy-array-data arr2 2 arr2 6 6)
    (is (every #'eql arr2 #*1011001100101101))
    (copy-array-data arr1 0 arr2 0 t)
    (copy-array-data arr2 6 arr2 2 6)
    (is (every #'eql arr2 #*1010010001001101))))

(def function type-equal (t1 t2)
  (and (subtypep t1 t2) (subtypep t2 t1)))

(def test test/utils/types ()
  (is (equal (cl-gpu.buffers::canonify-foreign-type :unsigned-short) :uint16))
  (is (equal (lisp-to-foreign-type '(unsigned-byte 10)) :int16))
  (is (equal (lisp-to-foreign-type '(unsigned-byte 16)) :uint16))
  (is (equal (lisp-to-foreign-type '(unsigned-byte 17)) :int32))
  (is (equal (lisp-to-foreign-elt-type '(unsigned-byte 10)) nil))
  (is (equal (lisp-to-foreign-elt-type '(unsigned-byte 32)) :uint32))
  (is (type-equal (foreign-to-lisp-type :uint16) 'uint16))
  (is (type-equal (foreign-to-lisp-elt-type :uint32)
                  (upgraded-array-element-type 'uint32))))
