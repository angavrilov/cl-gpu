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
(defvar *cuda-arr2*)
(defvar *cuda-arr3*)
(defvar *cuda-arr4*)

(defixture cuda-context
  (:setup (setf *cuda-ctx* (cuda-create-context 0))
          (setf *cuda-arr1* (cuda-make-array '(5 5) :element-type 'single-float
                                             :pitch-elt-size 4 :initial-element 0.0))
          (setf *cuda-arr2* (cuda-make-array '(5 5) :element-type 'single-float
                                             :pitch-elt-size 4 :initial-element 0.0))
          (setf *cuda-arr3* (cuda-make-array '(5 5) :element-type 'single-float
                                             :pitch-elt-size 16 :initial-element 0))
          (setf *cuda-arr4* (cuda-make-array '(5 5) :element-type 'single-float
                                             :initial-element 0)))
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

(def test test/cuda-driver/cuda-linear-buffer ()
  (with-fixture cuda-context
    (test/buffers/all *cuda-arr1* *cuda-arr2* *cuda-arr3* *cuda-arr4*)
    (is (equal (handler-case
                   (copy-buffer-data *cuda-arr1* 0 *cuda-arr2* 1 6)
                 (simple-warning (w)
                   (format nil "~A" w)))
               "Misaligned pitch copy, falling back to intermediate host array."))))

(def test test/cuda-driver/cuda-linear-foreign-copy ()
  (with-fixtures (cuda-context foreign-arrs)
    (test/buffers/copy *cuda-arr1* *foreign-arr2* *cuda-arr3* *foreign-arr4*)))
