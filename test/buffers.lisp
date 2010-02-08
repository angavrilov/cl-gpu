;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu.test)

(defsuite* (test/buffers :in test))

(defun zero-buffer? (buf)
  (loop for i from 0 below (buffer-size buf)
     always (= (row-major-bref buf i) 0)))

(defun buffer-filled? (buf value &key (start 0) (end (buffer-size buf)))
  (loop for i from start below end
     always (= (row-major-bref buf i) value)))

(defun set-index-buffer (buf)
  (loop for i from 0 below (buffer-size buf)
     do (setf (row-major-bref buf i) (float i))))

(defun index-buffer? (buf &key (start 0) (end (buffer-size buf)) (shift 0))
  (loop for i from start below end
     always (= (row-major-bref buf i) (- i shift))))

(def test test/buffers/basic-buffer (buffer)
  (is (= (buffer-rank buffer) 2))
  (is (equal (buffer-dimensions buffer) '(5 5)))
  (is (= (buffer-size buffer) 25))
  (is (eql (buffer-foreign-type buffer) :float))
  (is (eql (buffer-element-type buffer) 'single-float))
  (is (eq (buffer-fill buffer 0) buffer))
  (is (zero-buffer? buffer))
  (set-index-buffer buffer)
  (is (index-buffer? buffer))
  (is (equal (loop for i from 0 below 5
                collect (bref buffer 1 i))
             (loop for i from 0 below 5 collect (float (+ i 5)))))
  (buffer-fill buffer 0.5 :start 2 :end 23)
  (is (index-buffer? buffer :start 0 :end 2))
  (is (index-buffer? buffer :start 23))
  (is (buffer-filled? buffer 0.5 :start 2 :end 23)))

(def test test/buffers/copy-array-buffer (buffer)
  (let ((arr1 (make-array '(5 5) :element-type 'single-float))
        (arr2 (make-array 25 :element-type 'single-float :initial-element 1.0)))
    (buffer-fill buffer 0)
    (copy-full-buffer buffer arr2)
    (is (zero-buffer? arr2))
    (set-index-buffer arr1)
    (copy-full-buffer arr1 buffer)
    (is (index-buffer? buffer))
    (buffer-fill arr2 0.5)
    (copy-buffer-data arr2 2 buffer 2 (- 23 2))
    (is (index-buffer? buffer :start 0 :end 2))
    (is (index-buffer? buffer :start 23))
    (is (buffer-filled? buffer 0.5 :start 2 :end 23))))

(def test test/buffers/copy-buffer-buffer (buffer1 buffer2 buffer3 buffer4)
  (buffer-fill buffer1 0)
  (is (zero-buffer? buffer1))
  (copy-full-buffer buffer1 buffer2)
  (is (zero-buffer? buffer2))
  (copy-full-buffer buffer1 buffer3)
  (is (zero-buffer? buffer3))
  (copy-full-buffer buffer1 buffer4)
  (is (zero-buffer? buffer4))
  (set-index-buffer buffer4)
  (is (index-buffer? buffer4))
  (copy-full-buffer buffer4 buffer3)
  (is (index-buffer? buffer3))
  (copy-full-buffer buffer3 buffer2)
  (is (index-buffer? buffer2))
  (copy-full-buffer buffer2 buffer1)
  (is (index-buffer? buffer1))
  (copy-buffer-data buffer1 1 buffer4 4 3)
  (is (index-buffer? buffer4 :start 4 :end 7 :shift 3))
  (copy-buffer-data buffer2 4 buffer1 1 3)
  (is (index-buffer? buffer1 :start 1 :end 4 :shift -3)))

(def test test/buffers/displaced-buffer (buffer1 buffer2)
  (let* ((arr1 (buffer-displace buffer1 :offset 5 :dimensions '(2 5)))
         (arr2 (buffer-displace buffer2 :offset 5 :dimensions '(2 5)))
         (tmp (make-array '(5 5) :element-type 'single-float :initial-element 0.0))
         (tmp1 (buffer-displace tmp :offset 5 :dimensions '(2 5))))
    (set-index-buffer buffer1)
    (is (index-buffer? arr1 :shift -5)) ; displaced read
    (buffer-fill arr1 0.0)              ; displaced fill
    (is (index-buffer? buffer1 :start 0 :end 5))
    (is (buffer-filled? buffer1 0.0 :start 5 :end 15))
    (is (index-buffer? buffer1 :start 15))
    (set-index-buffer arr1)             ; displaced write
    (is (index-buffer? buffer1 :start 5 :end 15 :shift 5))
    (set-index-buffer buffer2)
    (buffer-fill buffer1 0.0)
    (copy-full-buffer arr2 arr1)        ; displaced dev -> dev
    (is (buffer-filled? buffer1 0.0 :start 0 :end 5))
    (is (index-buffer? buffer1 :start 5 :end 15))
    (is (buffer-filled? buffer1 0.0 :start 15))
    (buffer-fill arr1 0.0)
    (is (zero-buffer? buffer1))
    (set-index-buffer tmp)
    (copy-full-buffer tmp1 arr1)        ; displaced host -> dev
    (is (buffer-filled? buffer1 0.0 :start 0 :end 5))
    (is (index-buffer? buffer1 :start 5 :end 15))
    (is (buffer-filled? buffer1 0.0 :start 15))
    (buffer-fill tmp 0.0)
    (is (zero-buffer? tmp))
    (copy-full-buffer arr1 tmp1)        ; displaced dev -> host
    (is (buffer-filled? tmp 0.0 :start 0 :end 5))
    (is (index-buffer? tmp :start 5 :end 15))
    (is (buffer-filled? tmp 0.0 :start 15))))

(def test test/buffers/all (buffer1 buffer2 buffer3 buffer4)
  (test/buffers/basic-buffer buffer1)
  (test/buffers/copy-array-buffer buffer1)
  (test/buffers/copy-buffer-buffer buffer1 buffer2 buffer3 buffer4)
  (test/buffers/displaced-buffer buffer1 buffer2))

(def test test/buffers/copy (buffer1 buffer2 buffer3 buffer4)
  (test/buffers/copy-buffer-buffer buffer1 buffer2 buffer3 buffer4)
  (test/buffers/displaced-buffer buffer1 buffer2))

(defvar *foreign-arr1*)
(defvar *foreign-arr2*)
(defvar *foreign-arr3*)
(defvar *foreign-arr4*)

(def fixture foreign-arrs
  (setf *foreign-arr1* (make-foreign-array '(5 5) :element-type 'single-float :initial-element 0.0))
  (setf *foreign-arr2* (make-foreign-array '(5 5) :element-type 'single-float :initial-element 0.0))
  (setf *foreign-arr3* (make-foreign-array 25 :element-type 'single-float :initial-element 0.0))
  (setf *foreign-arr4* (make-foreign-array 25 :element-type 'single-float :initial-element 0.0))
  (unwind-protect
       (-body-)
    (deref-buffer *foreign-arr1*)
    (deref-buffer *foreign-arr2*)
    (deref-buffer *foreign-arr3*)
    (deref-buffer *foreign-arr4*)))

(def test test/buffers/foreign ()
  (with-fixture foreign-arrs
    (test/buffers/all *foreign-arr1* *foreign-arr2* *foreign-arr3* *foreign-arr4*)))
