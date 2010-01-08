;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

(def (generic e) bufferp (buffer)
  (:documentation "Determines if the argument is a buffer")
  (:method ((buffer array)) t))

(deftype buffer ()
  "A buffer is an arrayish object; includes normal arrays."
  '(satisfies bufferp))

(def (generic e) buffer-as-array (buffer)
  (:documentation "Returns the buffer converted to an array.")
  (:method ((buffer array))
    ;; Second value: shared flag, i.e. it is the buffer.
    (values buffer t))
  (:method ((buffer t))
    (let ((array (make-array (buffer-dimensions buffer)
                             :element-type (buffer-element-type buffer))))
      (copy-full-buffer buffer array)
      (values array nil))))

(def (generic e) buffer-element-type (buffer)
  (:documentation "Returns the lisp element type of the buffer.")
  (:method ((buffer array))
    (array-element-type buffer)))

(def (generic e) buffer-rank (buffer)
  (:documentation "Returns the array rank of the buffer.")
  (:method ((buffer array))
    (array-rank buffer)))

(def (generic e) buffer-dimensions (buffer)
  (:documentation "Returns the dimension list of the buffer.")
  (:method ((buffer array))
    (array-dimensions buffer))
  (:method ((buffer t))
    (loop for i from 0 to (1- (buffer-rank buffer))
       collect (buffer-dimension buffer i))))

(def (generic e) buffer-dimension (buffer axis)
  (:documentation "Returns the nth dimension of the buffer.")
  (:method ((buffer array) axis)
    (array-dimension buffer axis)))

(def (generic e) buffer-size (buffer)
  (:documentation "Returns the total element count of the buffer.")
  (:method ((buffer array))
    (array-total-size buffer)))

(def (generic e) buffer-row-major-index (buffer &rest indexes)
  (:documentation "Returns a linear index element of an element.")
  (:method ((buffer array) &rest indexes)
    (apply #'array-row-major-index buffer indexes))
  (:method ((buffer t) &rest indexes)
    (loop with index = (car indexes)
       for i from 1
       for idx in (cdr indexes)
       do (setf index
                (+ (* (buffer-dimension buffer i) index) idx))
       finally (return index))))

(def (generic e) bref (buffer &rest indexes)
  (:documentation "Returns an element of the buffer.")
  (:method ((buffer array) &rest indexes)
    (apply #'aref buffer indexes))
  (:method ((buffer t) &rest indexes)
    (row-major-bref buffer (apply #'buffer-row-major-index buffer indexes))))

(def (generic e) (setf bref) (value buffer &rest indexes)
  (:documentation "Updates an element of the buffer.")
  (:method (value (buffer array) &rest indexes)
    (setf (apply #'aref buffer indexes) value))
  (:method (value (buffer t) &rest indexes)
    (setf (row-major-bref buffer (apply #'buffer-row-major-index buffer indexes))
          value)))

(def (generic e) row-major-bref (buffer index)
  (:documentation "Returns an element of the buffer.")
  (:method ((buffer array) index)
    (row-major-aref buffer index)))

(def (generic e) (setf row-major-bref) (value buffer index)
  (:documentation "Updates an element of the buffer.")
  (:method (value (buffer array) index)
    (setf (row-major-aref buffer index) value)))

(def (generic e) %copy-buffer-data (src dst src-offset dst-offset count)
  (:documentation "Copies a subset of elements from src to dst. Counts and offsets must be correct.")
  (:method ((src array) (dst array) src-offset dst-offset count)
    (copy-array-data src src-offset dst dst-offset count))
  (:method ((src t) (dst t) src-offset dst-offset count)
    ;; Fallback to a temporary array. This requires
    ;; a type check to avoid an infinite recursion.
    (cond ((arrayp src) (error "Cannot copy array -> ~A" dst))
          ((arrayp dst) (error "Cannot copy ~A -> array" src))
          (t
           (let ((tmp (make-array count :element-type (buffer-element-type src))))
             (%copy-buffer-data src tmp src-offset 0 count)
             (%copy-buffer-data tmp dst 0 dst-offset count))))))

(def (function e) copy-buffer-data (src src-offset dest dest-offset count)
  "Copies a subset of elements from src to dst. A safe wrapper around %copy-buffer-data."
  (let* ((src-size (buffer-size src))
         (dest-size (buffer-size dest))
         (rcount (adjust-copy-count src-size src-offset dest-size dest-offset count)))
    (when (> rcount 0)
      (%copy-buffer-data src dest src-offset dest-offset rcount))))

(def (function e) copy-full-buffer (src dest)
  "Copies all contents of src to dst. Requires same size and elttype."
  (unless (eq src dest) ;; No-op if the same buffer
    (let ((src-size (buffer-size src))
          (dest-size (buffer-size dest)))
      (assert (= src-size dest-size))
      (%copy-buffer-data src dest 0 0 src-size))))
