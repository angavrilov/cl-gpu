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
  (setf *cuda-ctx* (cuda-create-context 0))
  (unwind-protect
       (progn
         (setf *cuda-arr1* (make-cuda-array '(5 5) :element-type 'single-float
                                            :pitch-elt-size 4 :initial-element 0.0))
         (setf *cuda-arr2* (make-cuda-array '(5 5) :element-type 'single-float
                                            :pitch-elt-size 4 :initial-element 0.0))
         (setf *cuda-arr3* (make-cuda-array '(5 5) :element-type 'single-float
                                            :pitch-elt-size 16 :initial-element 0))
         (setf *cuda-arr4* (make-cuda-array '(5 5) :element-type 'single-float
                                            :initial-element 0))
         (-body-))
    (cuda-destroy-context *cuda-ctx*)))

(defvar *cuda-host-arr1*)
(defvar *cuda-host-arr2*)
(defvar *cuda-host-arr3*)
(defvar *cuda-host-arr4*)

(def fixture cuda-host-arrs
  (with-deref-buffers ((*cuda-host-arr1*
                        (make-cuda-host-array '(5 5) :element-type 'single-float :initial-element 0.0))
                       (*cuda-host-arr2*
                        (make-cuda-host-array '(5 5) :element-type 'single-float :initial-element 0.0))
                       (*cuda-host-arr3*
                        (make-cuda-host-array 25 :element-type 'single-float :initial-element 0.0))
                       (*cuda-host-arr4*
                        (make-cuda-host-array 25 :element-type 'single-float :initial-element 0.0)))
    (-body-)))

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

(def test test/cuda-driver/cuda-linear-host-copy ()
  (with-fixtures (cuda-context cuda-host-arrs)
    (test/buffers/copy *cuda-host-arr1* *cuda-arr2* *cuda-host-arr3* *cuda-arr4*)))

(def gpu-type test-array (&optional (len '*))
  `(array single-float (2 ,len 4)))

(def test test/cuda-driver/cuda-module-vars ()
  (with-fixture cuda-context
    (let ((module (cl-gpu::parse-gpu-module-spec
                   '((:variable foo single-float)
                     (:variable bar test-array)
                     (:variable baz (test-array 6))))))
      (cl-gpu::compile-gpu-module module)
      (symbol-macrolet ((instance (cl-gpu::get-module-instance module))
                        (foo-var (aref (cl-gpu::gpu-module-instance-item-vector instance) 0))
                        (bar-var (aref (cl-gpu::gpu-module-instance-item-vector instance) 1))
                        (baz-var (aref (cl-gpu::gpu-module-instance-item-vector instance) 2))
                        (foo-val (cl-gpu::gpu-global-value foo-var))
                        (bar-val (cl-gpu::gpu-global-value bar-var))
                        (baz-val (cl-gpu::gpu-global-value baz-var)))
        (is (eq instance instance))
        ;; Initially zero
        (is (eql foo-val 0.0))
        (is (eql bar-val nil))
        (is (zero-buffer? baz-val))
        ;; Fill in static data
        (setf foo-val 3.0)
        (is (eql foo-val 3.0))
        (set-index-buffer baz-val)
        (is (index-buffer? baz-val))
        ;; Attach an array
        (setf bar-val (make-cuda-array '(2 10 4) :foreign-type :float :pitch-elt-size 16))
        (is (bufferp bar-val))
        (is (eql (deref-buffer bar-val) 1))
        (is (equal (rest (coerce (buffer-as-array (cl-gpu::buffer-of bar-var)) 'list))
                   '(80 2 10 4 1280 640 64)))
        ;; Fill the attached array
        (set-index-buffer bar-val)
        (is (index-buffer? bar-val))
        ;; Force a reload
        (let ((old-handle (cl-gpu::cuda-module-instance-handle instance)))
          (reinitialize-instance module)
          (is (not (eql old-handle (cl-gpu::cuda-module-instance-handle instance)))))
        ;; Verify that the values are still there
        (is (eql foo-val 3.0))
        (is (index-buffer? bar-val))
        (is (index-buffer? baz-val))
        ;; Verify auto-wipe
        (handler-bind ((warning #'ignore-warning))
          (deref-buffer bar-val))
        (is (eq bar-val nil))
        (is (zero-buffer? (cl-gpu::buffer-of bar-var)))))))

(declaim (type single-float *test-global-val*))
(defparameter *test-global-val* 0.7)

(def test test/cuda-driver/cuda-module-args ()
  (with-fixture cuda-context
    (let ((module (cl-gpu::parse-gpu-module-spec
                   `((:global foo single-float)
                     (:global bar test-array)
                     (:global baz (test-array 6))
                     (:global buf (vector uint32 7))
                     (:kernel foo (foo bar &key baz)
                       (declare (type single-float foo)
                                (type test-array bar)
                                (type (test-array 6) baz))
                       (setf (aref buf 0) (gpu::inline-verbatim (:uint32)
                                            "(unsigned)" bar)
                             (aref buf 1) (array-total-size bar)
                             (aref buf 2) (array-dimension bar 1)
                             (aref buf 3) (array-raw-extent bar)
                             (aref buf 4) (array-raw-stride bar 0)
                             (aref buf 5) (array-raw-stride bar 1)
                             (aref buf 6) (gpu::inline-verbatim (:uint32)
                                            "(unsigned)" baz)
                             (aref bar 1 1 1) (+ foo *test-global-val*)))))))
      (setf *last-tested-module* module)
      (setf *current-gpu-target* :cuda)
      (cl-gpu::compile-gpu-module module)
      (let* ((instance (cl-gpu::get-module-instance module))
             (items (cl-gpu::gpu-module-instance-item-vector instance))
             (baz (cl-gpu::gpu-global-value (aref items 2)))
             (kernel (aref items 4))
             (result (cl-gpu::gpu-global-value (aref items 3)))
             (ptr (cl-gpu::cuda-linear-handle (slot-value baz 'cl-gpu::blk))))
        (is (zero-buffer? result))
        (funcall kernel 0.5 baz :baz baz)
        (is (every #'=
                   (buffer-as-array result)
                   (list ptr 48 6 48 24 4 ptr)))
        (is (= (bref baz 1 1 1) 1.2))))))

(def test test/cuda-driver/compute ()
  (with-fixture cuda-context
    (test/translator/compute :cuda)))

(def function cuda-allocate-dummy-block ()
  (make-cuda-array 10)
  (values nil nil nil nil nil))

(def function cuda-context-block-cnt (ctx)
  (length (cl-gpu::weak-set-snapshot (cl-gpu::cuda-context-blocks ctx))))

(def test test/cuda-driver/gc/blocks ()
  (with-fixture cuda-context
    (let ((cnt (cuda-context-block-cnt *cuda-ctx*)))
      (cuda-allocate-dummy-block)
      (is (= (1+ cnt) (cuda-context-block-cnt *cuda-ctx*)))
      (is (null (cl-gpu::cuda-context-destroy-queue *cuda-ctx*)))
      (tg:gc :full t)
      (tg:gc :full t)
      (tg:gc :full t)
      #+openmcl (ccl::drain-termination-queue)
      (is (= cnt (cuda-context-block-cnt *cuda-ctx*)))
      (is (typep (car (cl-gpu::cuda-context-destroy-queue *cuda-ctx*))
                 'cl-gpu::cuda-linear))
      (with-deref-buffer (buf (make-cuda-array 10))
        (declare (ignore buf))
        (is (null (cl-gpu::cuda-context-destroy-queue *cuda-ctx*)))))))

