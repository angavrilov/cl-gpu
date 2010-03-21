;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file implements the generic buffer interface for
;;; a buffer based on CUDA linear memory allocations.
;;;

(in-package :cl-gpu)

(def function compute-linear-strides (blk dims elt-size pitch-level)
  (assert (> pitch-level 0))
  (if (and (cuda-linear-pitched-p blk)
           (< pitch-level (length dims)))
      (nconc (compute-strides (append (butlast dims pitch-level)
                                      (list (/ (cuda-linear-pitch blk) elt-size)))
                              (/ (cuda-linear-extent blk) elt-size))
             (rest
              (compute-strides (last dims pitch-level)
                               (/ (cuda-linear-width blk) elt-size))))
      (compute-strides dims (/ (cuda-linear-extent blk) elt-size))))

(def class* cuda-mem-array (abstract-foreign-buffer)
  ((blk :type cuda-linear)
   (phys-offset :type fixnum :initform 0 :documentation "Pitched byte offset")
   (strides :type (vector uint32)))
  (:automatic-accessors-p nil))

(def method print-object ((obj cuda-mem-array) stream)
  (print-buffer "CUDA Array" obj stream))

(def (function e) make-cuda-array (dims &key (element-type 'single-float)
                                        (foreign-type (lisp-to-foreign-elt-type element-type) ft-p)
                                        pitch-elt-size (pitch-level 1)
                                        initial-element)
  (unless foreign-type
    (error "Invalid cuda array type: ~S" (if ft-p foreign-type element-type)))
  (let* ((dims (ensure-list dims))
         (head-dims (butlast dims pitch-level))
         (tail-dims (last dims pitch-level))
         (size (reduce #'* dims))
         (elt-size (foreign-type-size foreign-type)))
    (when (and initial-element
               (not (eql initial-element 0))
               (not (case elt-size ((1 2 4) t))))
      (error "Cannot fill arrays of type: ~S" foreign-type))
    (let* ((blk (cuda-alloc-linear (reduce #'* tail-dims :initial-value elt-size)
                                   (reduce #'* head-dims)
                                   :pitch-for pitch-elt-size))
           (strides (compute-linear-strides blk dims elt-size pitch-level))
           (buffer (make-instance 'cuda-mem-array :blk blk :size size
                                  :elt-type foreign-type :elt-size elt-size
                                  :dims (to-uint32-vector dims)
                                  :strides (to-uint32-vector strides))))
      (if initial-element
          (buffer-fill buffer initial-element)
          buffer))))

(def method buffer-displace ((buffer cuda-mem-array) &key
                             byte-offset foreign-type size dimensions
                             offset element-type)
  (declare (ignore offset element-type))
  (with-slots (blk log-offset) buffer
    ;; Verify pitch alignment
    (let* ((elt-size (foreign-type-size foreign-type))
           (new-offset (+ log-offset byte-offset))
           (phys-offset (cuda-linear-adjust-offset blk new-offset))
           (rank (length dimensions))
           (byte-size (* size elt-size))
           (width (cuda-linear-width blk))
           (pitch-level (if (cuda-linear-pitched-p blk)
                            (loop ; Find which of the dims fit in the pitch line
                               for rdims on dimensions
                               for i downfrom rank ; 1 <= i <= rank
                               for rsize = (reduce #'* rdims :initial-value elt-size)
                               if (<= rsize width)
                               do (progn
                                    (unless (or (= i rank) (= rsize width))
                                      (error "Inner dimensions ~S*~A = ~A don't match pitch width ~A"
                                             rdims elt-size rsize width))
                                    (return i))
                               finally (error "Inner dimension of ~S*~A doesn't fit in pitch width ~A"
                                              dimensions elt-size width))
                            rank)))
      (when (cuda-linear-pitched-p blk)
        (unless (or (= (mod new-offset width) 0)
                    (<= (+ (mod new-offset width) byte-size) width))
          (error "Byte offset ~A+~A not aligned to pitch width ~A"
                 log-offset byte-offset width)))
      ;; Everything seems OK, create the new descriptor
      (make-instance 'cuda-mem-array :blk blk :size size
                     :displaced-to buffer :log-offset new-offset :phys-offset phys-offset
                     :elt-type foreign-type :elt-size elt-size
                     :dims (to-uint32-vector dimensions)
                     :strides (to-uint32-vector (compute-linear-strides blk dimensions
                                                                        elt-size pitch-level))))))

(def method buffer-refcnt ((buffer cuda-mem-array))
  (with-slots (blk) buffer
    (if (cuda-linear-valid-p blk)
        (if (cuda-linear-module blk)
            t ; Module-backed blocks cannot be deallocated
            (cuda-linear-refcnt blk))
        nil)))

(def method ref-buffer ((buffer cuda-mem-array))
  (with-slots (blk) buffer
    (incf (cuda-linear-refcnt blk))))

(def method deref-buffer ((buffer cuda-mem-array))
  (with-slots (blk) buffer
    (unless (> (decf (cuda-linear-refcnt blk)) 0)
      (deallocate blk))))

(def method row-major-bref ((buffer cuda-mem-array) index)
  (with-slots (blk log-offset elt-type elt-size) buffer
    (with-foreign-object (tmp elt-type)
      (%cuda-linear-dh-transfer blk tmp (+ (* index elt-size) log-offset) elt-size t)
      (mem-ref tmp elt-type))))

(def method (setf row-major-bref) (value (buffer cuda-mem-array) index)
  (with-slots (blk log-offset elt-type elt-size) buffer
    (with-foreign-object (tmp elt-type)
      (prog1
          (setf (mem-ref tmp elt-type) value)
        (%cuda-linear-dh-transfer blk tmp (+ (* index elt-size) log-offset) elt-size nil)))))

(def method buffer-fill ((buffer cuda-mem-array) value &key start end)
  (with-slots (blk log-offset elt-type elt-size) buffer
    (let ((count (- end start)))
      (if (eql value 0) ; Fill with 0 in binary mode, so that it
                        ; works with any type
          (%cuda-linear-memset blk (* start elt-size) (* count elt-size) :uint8 0 :offset log-offset)
          (%cuda-linear-memset blk start count elt-type value :offset log-offset)))))

(def method %copy-buffer-data ((src array) (dst cuda-mem-array) src-offset dst-offset count)
  (with-slots (blk log-offset elt-size) dst
    (with-pointer-to-array (ptr src)
      (%cuda-linear-dh-transfer blk (inc-pointer ptr (* src-offset elt-size))
                                (+ (* dst-offset elt-size) log-offset)
                                (* count elt-size) nil))))

(def method %copy-buffer-data ((src cuda-mem-array) (dst array) src-offset dst-offset count)
  (with-slots (blk log-offset elt-size) src
    (with-pointer-to-array (ptr dst)
      (%cuda-linear-dh-transfer blk (inc-pointer ptr (* dst-offset elt-size))
                                (+ (* src-offset elt-size) log-offset)
                                (* count elt-size) t))))

(def method %copy-buffer-data ((src cuda-mem-array) (dst cuda-mem-array) src-offset dst-offset count)
  (with-slots ((s-blk blk) (s-log-offset log-offset) elt-size) src
    (with-slots ((d-blk blk) (d-log-offset log-offset)) dst
      (when (eql (%cuda-linear-dd-transfer s-blk (+ (* src-offset elt-size) s-log-offset)
                                           d-blk (+ (* dst-offset elt-size) d-log-offset)
                                           (* count elt-size)
                                           :return-if-mismatch :misaligned)
                 :misaligned)
        (warn "Misaligned pitch copy, falling back to intermediate host array.")
        (call-next-method)))))

(def method %copy-buffer-data ((src foreign-array) (dst cuda-mem-array) src-offset dst-offset count)
  (with-slots ((s-blk blk) (s-log-offset log-offset) elt-size) src
    (with-slots ((d-blk blk) (d-log-offset log-offset)) dst
      (%cuda-linear-dh-transfer d-blk
                                (inc-pointer (foreign-block-ptr s-blk)
                                             (+ (* src-offset elt-size) s-log-offset))
                                (+ (* dst-offset elt-size) d-log-offset)
                                (* count elt-size) nil))))

(def method %copy-buffer-data ((src cuda-mem-array) (dst foreign-array) src-offset dst-offset count)
  (with-slots ((s-blk blk) (s-log-offset log-offset) elt-size) src
    (with-slots ((d-blk blk) (d-log-offset log-offset)) dst
      (%cuda-linear-dh-transfer s-blk
                                (inc-pointer (foreign-block-ptr d-blk)
                                             (+ (* dst-offset elt-size) d-log-offset))
                                (+ (* src-offset elt-size) s-log-offset)
                                (* count elt-size) t))))
