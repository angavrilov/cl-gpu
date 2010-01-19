;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

(def function compute-linear-strides (blk dims elt-size pitch-level)
  (if (cuda-linear-pitched-p blk)
      (nconc (compute-strides (butlast dims pitch-level)
                              (cuda-linear-pitch blk)
                              (cuda-linear-extent blk))
             (compute-strides (last dims pitch-level)
                              elt-size
                              (cuda-linear-width blk)))
      (compute-strides dims elt-size (cuda-linear-extent blk))))

(def class* cuda-mem-array ()
  ((blk :type cuda-linear)
   (displaced-to :type (or cuda-mem-array null) :initform nil
                 :documentation "CUDA array this one is displaced to")
   (log-offset :type fixnum :initform 0 :documentation "Logical byte offset")
   (phys-offset :type fixnum :initform 0 :documentation "Pitched byte offset")
   (size :type fixnum :documentation "Array size in elements")
   (elt-type :type t)
   (elt-size :type fixnum)
   (dims :type (vector uint32))
   (strides :type (vector uint32)))
  (:automatic-accessors-p nil))

(def (function e) cuda-make-array (dims &key (element-type 'single-float)
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

(def method print-object ((obj cuda-mem-array) stream)
  (with-slots (blk dims elt-type size) obj
    (print-unreadable-object (obj stream)
      (format stream "CUDA Array &~A ~S ~A" (cuda-linear-refcnt blk)
              (coerce dims 'list) elt-type)
      (if (cuda-linear-valid-p blk)
          (or (ignore-errors
                (with-cuda-context ((cuda-linear-context blk))
                  (format stream ":~{ ~A~}~:[~;...~]"
                          (loop for i from 0 below (min size (or *print-length* 10))
                             collect (row-major-bref obj i))
                          (> size 10)))
                t)
              (format stream " (data inaccessible)"))
          (format stream " (DEAD)")))))

(def method buffer-displace ((buffer cuda-mem-array) &key
                             (offset 0 ofs-p)
                             (byte-offset (if ofs-p (* offset (slot-value buffer 'elt-size)) 0))
                             (element-type nil elt-type-p)
                             (foreign-type (if elt-type-p
                                               (lisp-to-foreign-elt-type element-type)
                                               (slot-value buffer 'elt-type)))
                             (size nil)
                             (dimensions (or size (buffer-dimensions buffer))))
  (with-slots (blk log-offset) buffer
    (let* ((elt-size (foreign-type-size foreign-type))
           (new-offset (+ log-offset byte-offset))
           (phys-offset (cuda-linear-adjust-offset blk new-offset)))
      (when (eql dimensions t)
        (setf dimensions (floor (- (cuda-linear-size blk) new-offset) elt-size)))
      (let* ((dims (ensure-list dimensions))
             (rank (length dims))
             (size (reduce #'* dims))
             (byte-size (* size elt-size))
             (width (cuda-linear-width blk))
             (pitch-level (if (cuda-linear-pitched-p blk)
                              (loop ; Find which of the dims fit in the pitch line
                                 for rdims on dims
                                 for i downfrom rank
                                 for rsize = (reduce #'* rdims :initial-value elt-size)
                                 if (<= rsize width)
                                 do (progn
                                      (unless (or (= i rank) (= rsize width))
                                        (error "Inner dimensions ~S*~A = ~A don't match pitch width ~A"
                                               rdims elt-size rsize width))
                                      (return i))
                                 finally (error "Inner dimension of ~S*~A doesn't fit in pitch width ~A"
                                                dims elt-size width))
                              rank)))
        (when (cuda-linear-pitched-p blk)
          (unless (or (= (mod new-offset width) 0)
                      (<= (+ (mod new-offset width) byte-size) width))
            (error "Byte offset ~A+~A not aligned to pitch width ~A"
                   log-offset byte-offset width)))
        (with-slots (size elt-type elt-size) buffer
          (when (> (+ byte-offset byte-size) (* size elt-size))
            (error "Specified dimensions ~A ~S exceed the original size ~A ~A"
                   foreign-type dims size elt-type)))
        ;; Everything seems OK, create the new descriptor
        (make-instance 'cuda-mem-array :blk blk :size size
                       :displaced-to buffer :log-offset new-offset :phys-offset phys-offset
                       :elt-type foreign-type :elt-size elt-size
                       :dims (to-uint32-vector dims)
                       :strides (to-uint32-vector (compute-linear-strides blk dims elt-size pitch-level)))))))

(def method buffer-displacement ((buffer cuda-mem-array))
  (with-slots (displaced-to log-offset) buffer
    (if displaced-to
        (with-slots (elt-size (s-log-offset log-offset)) displaced-to
          (values displaced-to (/ (- log-offset s-log-offset) elt-size)))
        (values nil 0))))

(def method bufferp ((buffer cuda-mem-array)) t)

(def method ref-buffer ((buffer cuda-mem-array))
  (with-slots (blk) buffer
    (when (> (cuda-linear-refcnt blk) 0)
      (incf (cuda-linear-refcnt blk)))))

(def method deref-buffer ((buffer cuda-mem-array))
  (with-slots (blk) buffer
    (when (> (cuda-linear-refcnt blk) 0)
      (let ((live (> (decf (cuda-linear-refcnt blk)) 0)))
        (unless live (cuda-free-linear blk))
        live))))

(def method buffer-foreign-type ((buffer cuda-mem-array))
  (slot-value buffer 'elt-type))

(def method buffer-rank ((buffer cuda-mem-array))
  (length (slot-value buffer 'dims)))

(def method buffer-dimensions ((buffer cuda-mem-array))
  (coerce (slot-value buffer 'dims) 'list))

(def method buffer-dimension ((buffer cuda-mem-array) axis)
  (aref (slot-value buffer 'dims) axis))

(def method buffer-size ((buffer cuda-mem-array))
  (slot-value buffer 'size))

(def method row-major-bref ((buffer cuda-mem-array) index)
  (with-slots (blk log-offset size elt-type elt-size) buffer
    (unless (and (>= index 0) (< index size))
      (error "Linear index ~S is out of bounds for ~S" index buffer))
    (with-foreign-object (tmp elt-type)
      (%cuda-linear-dh-transfer blk tmp (+ (* index elt-size) log-offset) elt-size t)
      (mem-ref tmp elt-type))))

(def method (setf row-major-bref) (value (buffer cuda-mem-array) index)
  (with-slots (blk log-offset size elt-type elt-size) buffer
    (unless (and (>= index 0) (< index size))
      (error "Linear index ~S is out of bounds for ~S" index buffer))
    (with-foreign-object (tmp elt-type)
      (prog1
          (setf (mem-ref tmp elt-type) value)
        (%cuda-linear-dh-transfer blk tmp (+ (* index elt-size) log-offset) elt-size nil)))))

(def method buffer-fill ((buffer cuda-mem-array) value &key (start 0) end)
  (with-slots (blk log-offset size elt-type elt-size) buffer
    (unless (and (>= start 0) (< start size)
                 (or (null end)
                     (and (>= end start) (<= end size))))
      (error "Bad fill range ~A...~A for ~S" start end buffer))
    (let ((count (- (or end size) start)))
      (if (eql value 0) ; Fill with 0 in binary mode, so that it
                        ; works with any type
          (%cuda-linear-memset blk (* start elt-size) (* count elt-size) :uint8 0 :offset log-offset)
          (%cuda-linear-memset blk start count elt-type value :offset log-offset)))
    buffer))

(def method %copy-buffer-data ((src array) (dst cuda-mem-array) src-offset dst-offset count)
  (with-slots (blk log-offset elt-type elt-size) dst
    (unless (equal (buffer-foreign-type src) elt-type)
      (error "Element type mismatch: ~S and ~S" src dst))
    (with-pointer-to-array (ptr src)
      (%cuda-linear-dh-transfer blk (inc-pointer ptr (* src-offset elt-size))
                                (+ (* dst-offset elt-size) log-offset)
                                (* count elt-size) nil))))

(def method %copy-buffer-data ((src cuda-mem-array) (dst array) src-offset dst-offset count)
  (with-slots (blk log-offset elt-type elt-size) src
    (unless (equal elt-type (buffer-foreign-type dst))
      (error "Element type mismatch: ~S and ~S" src dst))
    (with-pointer-to-array (ptr dst)
      (%cuda-linear-dh-transfer blk (inc-pointer ptr (* dst-offset elt-size))
                                (+ (* src-offset elt-size) log-offset)
                                (* count elt-size) t))))

(def method %copy-buffer-data ((src cuda-mem-array) (dst cuda-mem-array) src-offset dst-offset count)
  (with-slots ((s-blk blk) (s-log-offset log-offset) (s-elt-type elt-type) elt-size) src
    (with-slots ((d-blk blk) (d-log-offset log-offset) (d-elt-type elt-type)) dst
      (unless (eql s-elt-type d-elt-type)
        (error "Element type mismatch: ~S and ~S" src dst))
      (when (eql (%cuda-linear-dd-transfer s-blk (+ (* src-offset elt-size) s-log-offset)
                                           d-blk (+ (* dst-offset elt-size) d-log-offset)
                                           (* count elt-size)
                                           :return-if-mismatch :misaligned)
                 :misaligned)
        (warn "Misaligned pitch copy, falling back to intermediate host array.")
        (call-next-method)))))

