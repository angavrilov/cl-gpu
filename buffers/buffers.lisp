;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file defines an interface for a generic buffer
;;; object that resembles an ordinary Common Lisp array.
;;; The interface is immediately implemented for arrays.
;;;

(in-package :cl-gpu.buffers)

(def (generic e) bufferp (buffer)
  (:documentation "Determines if the argument is a buffer. Returns :foreign if based on foreign types.")
  (:method ((buffer t)) nil)
  (:method ((buffer array)) :lisp))

(deftype buffer ()
  "A buffer is an arrayish object; includes normal arrays."
  '(satisfies bufferp))

(def (generic e) buffer-refcnt (buffer)
  (:documentation "Returns the reference count of the buffer, t if not counted, or nil if invalid.")
  (:method ((buffer array)) t))

(def (generic e) ref-buffer (buffer)
  (:documentation "Increase the reference count on the buffer. Returns the buffer.")
  (:method ((buffer t)) buffer)
  (:method :around ((buffer t))
    (let ((cnt (buffer-refcnt buffer)))
      (when (numberp cnt)
        (call-next-method))
      buffer)))

(def (generic e) deref-buffer (buffer)
  (:documentation "Decrease the reference count on the buffer. Returns post-value of buffer-refcnt.")
  (:method ((buffer t)) t)
  (:method :around ((buffer null)) nil)
  (:method :around ((buffer t))
    (let ((cnt (buffer-refcnt buffer)))
      (if (numberp cnt)
          (progn
            (call-next-method)
            (if (> cnt 1) (1- cnt) nil))
          cnt))))

(def (macro e) with-deref-buffer ((var buffer-expr) &body code)
  (with-unique-names (tmp)
    `(let (,tmp)
       (unwind-protect
            (let ((,var (setf ,tmp ,buffer-expr)))
              ,@code)
         (deref-buffer ,tmp)))))

(def (macro e) with-deref-buffers (bindings &body code)
  (if (null (cdr bindings))
      `(with-deref-buffer ,(car bindings) ,@code)
      `(with-deref-buffer ,(car bindings)
         (with-deref-buffers ,(cdr bindings)
           ,@code))))

(def (generic e) buffer-displace (buffer &key
                                         offset byte-offset ; Offset in elements of the original
                                         element-type foreign-type ; May not always be supported
                                         size dimensions) ; Size is a shortcut for one dimension
  (:documentation "Returns a buffer displaced to the current one. The reference count is shared and unchanged.")
  (:method ((buffer array) &key
            (byte-offset 0 b-ofs-p)
            (offset (if b-ofs-p
                        (/ byte-offset (foreign-type-size (buffer-foreign-type buffer)))
                        0))
            (foreign-type nil f-type-p)
            (element-type (if f-type-p
                              (foreign-to-lisp-elt-type foreign-type)
                              (array-element-type buffer)))
            (size nil)
            (dimensions (or size (array-dimensions buffer))))
    (if (eql dimensions t)
        (setf dimensions (- (array-total-size buffer) offset)))
    (unless (equal (array-element-type buffer)
                   (upgraded-array-element-type element-type))
      (error "Cannot change the element type of an array: ~S -> ~S"
             (array-element-type buffer) element-type))
    (make-array dimensions :element-type element-type :displaced-to buffer :displaced-index-offset offset)))

(def (generic e) buffer-displacement (buffer)
  (:documentation "Returns information about this buffer's displacement.")
  (:method ((buffer array))
    (array-displacement buffer)))

(def (generic e) buffer-as-array (buffer &key no-copy)
  (:documentation "Returns the buffer converted to an array.")
  (:method ((buffer array) &key no-copy)
    (declare (ignore no-copy))
    ;; Second value: shared flag, i.e. it is the buffer.
    (values buffer t))
  (:method ((buffer t) &key no-copy)
    (let ((array (make-array (buffer-dimensions buffer)
                             :element-type (buffer-element-type buffer))))
      (unless no-copy
        (copy-full-buffer buffer array))
      (values array nil))))

(def (generic e) buffer-element-type (buffer)
  (:documentation "Returns the lisp element type of the buffer.")
  (:method ((buffer array))
    (array-element-type buffer))
  (:method ((buffer t))
    (foreign-to-lisp-elt-type (buffer-foreign-type buffer))))

(def (generic e) buffer-gpu-type (buffer)
  (:documentation "Returns the gpu-type object for the elements of the buffer.")
  (:method ((buffer t))
    (lisp-to-gpu-type (buffer-element-type buffer))))

(def (generic e) buffer-foreign-type (buffer)
  (:documentation "Returns the foreign element type of the buffer.")
  (:method ((buffer array))
    (lisp-to-foreign-elt-type (array-element-type buffer)))
  (:method ((buffer t))
    (foreign-type-of (buffer-gpu-type buffer))))

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
    (loop with index = 0
       for idx in indexes
       for i from 0
       do (let ((dim (buffer-dimension buffer i)))
            (unless (and (>= idx 0) (< idx dim))
              (error "Index ~S is out of bounds for ~S" indexes buffer))
            (setf index (+ (* dim index) idx)))
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

(def (generic e) buffer-fill (buffer value &key start end)
  (:documentation "Fills the buffer with the specified value.")
  (:method ((buffer vector) value &key (start 0) end)
    (fill buffer value :start start :end end))
  (:method ((buffer array) value &key (start 0) end)
    (loop for i from start below (or end (array-total-size buffer))
       do (setf (row-major-aref buffer i) value))
    buffer)
  (:method ((buffer t) value &key (start 0) end)
    (loop for i from start below (or end (buffer-size buffer))
       do (setf (row-major-bref buffer i) value))
    buffer))

(def generic %copy-buffer-data (src dst src-offset dst-offset count)
  (:documentation "Copies a subset of elements from src to dst. Counts, offsets and types must be correct.")
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

(def function print-buffer (name obj stream)
  (print-unreadable-object (obj stream)
    (let ((refcnt (buffer-refcnt obj))
          (pcnt (or *print-length* 10))
          (size (buffer-size obj)))
      (format stream "~A &~A ~S ~A" name (or refcnt 0)
              (buffer-dimensions obj)
              (if (eql (bufferp obj) :foreign)
                  (or (buffer-foreign-type obj)
                      (buffer-gpu-type obj))
                  (buffer-element-type obj)))
      (if refcnt
          (or (ignore-errors
                (format stream ":~{ ~A~}~:[~;...~]"
                        (loop for i from 0 below (min size pcnt)
                           collect (row-major-bref obj i))
                        (> size pcnt))
                t)
              (format stream " (data inaccessible)"))
          (format stream " (DEAD)")))))

(def function check-buffer-same-type (src dst)
  (unless
      (if (or (eql (bufferp src) :foreign)
              (eql (bufferp dst) :foreign))
          (eq (buffer-gpu-type src) (buffer-gpu-type dst))
          (equal (buffer-element-type src) (buffer-element-type dst)))
    (error "Element type mismatch: ~S and ~S" src dst)))

(def (function e) copy-buffer-data (src src-offset dest dest-offset count)
  "Copies a subset of elements from src to dst. A safe wrapper around %copy-buffer-data."
  (let* ((src-size (buffer-size src))
         (dest-size (buffer-size dest))
         (rcount (adjust-copy-count src-size src-offset dest-size dest-offset count)))
    (when (> rcount 0)
      (check-buffer-same-type src dest)
      (%copy-buffer-data src dest src-offset dest-offset rcount))))

(def (function e) copy-full-buffer (src dest)
  "Copies all contents of src to dst. Requires same size and elttype."
  (unless (eq src dest) ;; No-op if the same buffer
    (let ((src-size (buffer-size src))
          (dest-size (buffer-size dest)))
      (assert (= src-size dest-size))
      (check-buffer-same-type src dest)
      (%copy-buffer-data src dest 0 0 src-size))))

(def (function e) write-buffer (buffer stream)
  (write-array (buffer-as-array buffer) stream))

(def (function e) read-buffer (buffer stream)
  (multiple-value-bind (array samep)
      (buffer-as-array buffer :no-copy t)
    (read-array array stream)
    (unless samep
      (copy-full-buffer array buffer))))
