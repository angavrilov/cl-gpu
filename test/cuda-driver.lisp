;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu.test)

(defsuite* (test/cuda-driver :in test))

(def test test/cuda-driver/errors ()
  (let ((cond (handler-case
                  (cuda-device-attr 0 123455)
                (cuda-driver-error (cond)
                  cond))))
    (is (eql (cl-gpu::method-name cond) 'cl-gpu::cuDeviceGetAttribute))
    (is (eql (cl-gpu::error-code cond) :error-invalid-value))))

(defvar *cuda-ctx*)
(defvar *cuda-arr1*)

(defixture cuda-context
  (:setup (setf *cuda-ctx* (cuda-create-context 0))
          (setf *cuda-arr1* (cuda-make-array '(5 5) :element-type 'single-float
                                             :pitch-elt-size 4 :initial-element 0.0)))
  (:teardown (cuda-destroy-context *cuda-ctx*)))

(def test verify-wrap-pitch (blk offset size commands)
  (cl-gpu::%cuda-linear-wrap-pitch blk offset size
                                   (lambda (&rest data)
                                     (is (equal (list* :chunk data) (pop commands))))
                                   (lambda (&rest data)
                                     (is (equal (list* :rows data) (pop commands)))))
  (is (eql commands nil)))

(def test test/cuda-driver/wrap-pitch ()
  (with-fixture cuda-context
    (let ((blk (slot-value *cuda-arr1* 'cl-gpu::blk)))
      (verify-wrap-pitch blk 0 (* 4 5)
                         '((:chunk 0 0 20)))
      (verify-wrap-pitch blk (* 4 5) 4
                         '((:chunk 64 0 4)))
      (verify-wrap-pitch blk (* 4 5) (* 4 6)
                         '((:chunk 64 0 20)
                           (:chunk 128 20 4)))
      (verify-wrap-pitch blk 0 (* 4 10)
                         '((:rows 0 20 64 0 2)))
      (verify-wrap-pitch blk (* 4 5) (* 4 10)
                         '((:rows 0 20 64 1 2)))
      (verify-wrap-pitch blk (* 4 4) (* 4 14)
                         '((:chunk 16 0 4)
                           (:rows 4 20 64 1 2)
                           (:chunk 192 44 12))))))

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
     always (= (row-major-bref buf i) (+ i shift))))

(def test test/cuda-driver/basic-buffer ()
  (with-fixture cuda-context
    (is (= (buffer-rank *cuda-arr1*) 2))
    (is (equal (buffer-dimensions *cuda-arr1*) '(5 5)))
    (is (= (buffer-size *cuda-arr1*) 25))
    (is (eql (buffer-foreign-type *cuda-arr1*) :float))
    (is (eql (buffer-element-type *cuda-arr1*) 'single-float))
    (is (eq (buffer-fill *cuda-arr1* 0) *cuda-arr1*))
    (is (zero-buffer? *cuda-arr1*))
    (set-index-buffer *cuda-arr1*)
    (is (index-buffer? *cuda-arr1*))
    (is (equal (loop for i from 0 below 5
                  collect (bref *cuda-arr1* 1 i))
               (loop for i from 0 below 5 collect (float (+ i 5)))))
    (buffer-fill *cuda-arr1* 0.5 :start 2 :end 23)
    (is (index-buffer? *cuda-arr1* :start 0 :end 2))
    (is (index-buffer? *cuda-arr1* :start 23))
    (is (buffer-filled? *cuda-arr1* 0.5 :start 2 :end 23))))

(def test test/cuda-driver/copy-array-buffer ()
  (with-fixture cuda-context
    (let ((arr1 (make-array '(5 5) :element-type 'single-float))
          (arr2 (make-array 25 :element-type 'single-float :initial-element 1.0)))
      (buffer-fill *cuda-arr1* 0)
      (copy-full-buffer *cuda-arr1* arr2)
      (is (zero-buffer? arr2))
      (set-index-buffer arr1)
      (copy-full-buffer arr1 *cuda-arr1*)
      (is (index-buffer? *cuda-arr1*))
      (buffer-fill arr2 0.5)
      (copy-buffer-data arr2 2 *cuda-arr1* 2 (- 23 2))
      (is (index-buffer? *cuda-arr1* :start 0 :end 2))
      (is (index-buffer? *cuda-arr1* :start 23))
      (is (buffer-filled? *cuda-arr1* 0.5 :start 2 :end 23)))))



