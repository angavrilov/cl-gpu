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

(def test test/cuda-driver/cuda-module-vars ()
  (with-fixture cuda-context
    (let ((module (make-instance 'cl-gpu::gpu-module :name nil
                                 :globals (list (make-instance 'cl-gpu::gpu-global-var
                                                               :name 'foo :c-name "foo" :index 0
                                                               :item-type :float :dimension-mask nil)
                                                (make-instance 'cl-gpu::gpu-global-var
                                                               :name 'bar :c-name "bar" :index 1
                                                               :item-type :float :dimension-mask #(2 nil 4))
                                                (make-instance 'cl-gpu::gpu-global-var
                                                               :name 'baz :c-name "baz" :index 2
                                                               :item-type :float :dimension-mask #(2 6 4)))
                                 :functions nil :kernels nil)))
      (setf (cl-gpu::compiled-code-of module)
            (cl-gpu::with-cuda-target
              (cl-gpu::cuda-compile-kernel (cl-gpu::generate-c-code module))))
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
        (setf bar-val (cuda-make-array '(2 10 4) :foreign-type :float :pitch-elt-size 16))
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

