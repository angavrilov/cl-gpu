;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file implements the generic buffer interface for
;;; buffers based on CUDA device linear and pinned host
;;; memory allocations.
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

;; Debug version of cuda-mem-array that mirrors all data.
(def class* cuda-debug-mem-array (cuda-mem-array mirrored-foreign-buffer)
  ()
  (:automatic-accessors-p nil))

(def method print-object ((obj cuda-debug-mem-array) stream)
  (print-buffer "CUDA Array (DBG)" obj stream))

(def function recover-cuda-mem-array (mblk blk ref)
  (declare (ignore ref))
  (%cuda-linear-dh-transfer blk (foreign-block-handle mblk)
                            0 (foreign-block-size mblk) nil))

(def constructor cuda-debug-mem-array
  (with-slots (mirror blk displaced-to) -self-
    (unless displaced-to
      (setf (cuda-linear-recover-cb blk)
            (curry #'recover-cuda-mem-array
                   (slot-value mirror 'blk))))))

(def (function e) make-cuda-array (dims &key (element-type 'single-float)
                                        (foreign-type (lisp-to-gpu-type element-type) ft-p)
                                        pitch-elt-size (pitch-level 1)
                                        initial-element (debug *cuda-debug*))
  (unless foreign-type
    (error "Invalid cuda array type: ~S" (if ft-p foreign-type element-type)))
  (let* ((dims (ensure-list dims))
         (head-dims (butlast dims pitch-level))
         (tail-dims (last dims pitch-level))
         (size (reduce #'* dims))
         (elt-type (foreign-to-gpu-type foreign-type))
         (elt-size (native-type-byte-size elt-type)))
    (when (and initial-element
               (not (eql initial-element 0))
               (not (case elt-size ((1 2 4) t))))
      (error "Cannot fill arrays of type: ~S" foreign-type))
    (let* ((blk (cuda-alloc-linear (reduce #'* tail-dims :initial-value elt-size)
                                   (reduce #'* head-dims)
                                   :pitch-for pitch-elt-size))
           (strides (compute-linear-strides blk dims elt-size pitch-level))
           (buffer (make-instance (if debug
                                      'cuda-debug-mem-array
                                      'cuda-mem-array)
                                  :blk blk :size size
                                  :elt-type elt-type :elt-size elt-size
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
    (let* ((elt-size (native-type-byte-size foreign-type))
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
      (make-instance (class-of buffer) :blk blk :size size
                     :displaced-to buffer :log-offset new-offset :phys-offset phys-offset
                     :elt-type foreign-type :elt-size elt-size
                     :dims (to-uint32-vector dimensions)
                     :strides (to-uint32-vector (compute-linear-strides blk dimensions
                                                                        elt-size pitch-level))))))

(def macro with-cuda-array-ref (((blk-var pos-var) buffer index &key (msg "copying data")) &body code)
  (with-unique-names (log-offset-var)
    (let ((inner
           `(with-slots ((,blk-var blk) (,log-offset-var log-offset) elt-size) ,buffer
              (let ((,pos-var (+ (* ,index elt-size) ,log-offset-var)))
                ,@code))))
      (if msg
          `(with-cuda-recover (,msg) ,inner)
          inner))))

(def method row-major-bref ((buffer cuda-mem-array) index)
  (with-slots (elt-type) buffer
    (with-cuda-array-ref ((blk pos) buffer index :msg "reading the buffer element")
      (native-type-ref elt-type (curry #'%cuda-linear-dh-transfer blk) pos))))

(def method (setf row-major-bref) (value (buffer cuda-mem-array) index)
  (with-slots (elt-type) buffer
    (with-cuda-array-ref ((blk pos) buffer index :msg "writing the buffer element")
      (setf (native-type-ref elt-type (curry #'%cuda-linear-dh-transfer blk) pos) value))))

(def method buffer-fill ((buffer cuda-mem-array) value &key start end)
  (with-slots (blk log-offset elt-type elt-size) buffer
    (with-cuda-recover ("filling the buffer")
      (let ((count (- end start)))
        (if (eql value 0)     ; Fill with 0 in binary mode, so that it
                              ; works with any type
            (%cuda-linear-memset blk (* start elt-size) (* count elt-size)
                                 +gpu-uint8-type+ 0 :offset log-offset)
            (%cuda-linear-memset blk start count elt-type value :offset log-offset))))))

(def method %copy-buffer-data ((src array) (dst cuda-mem-array) src-offset dst-offset count)
  (with-cuda-array-ref ((d-blk d-pos) dst dst-offset)
    (with-lisp-array-ref ((s-ptr) src src-offset elt-size)
      (%cuda-linear-dh-transfer d-blk s-ptr d-pos (* count elt-size) nil))))

(def method %copy-buffer-data ((src cuda-mem-array) (dst array) src-offset dst-offset count)
  (with-cuda-array-ref ((s-blk s-pos) src src-offset)
    (with-lisp-array-ref ((d-ptr) dst dst-offset elt-size)
      (%cuda-linear-dh-transfer s-blk d-ptr s-pos (* count elt-size) t))))

(def method %copy-buffer-data ((src cuda-mem-array) (dst cuda-mem-array) src-offset dst-offset count)
  (with-cuda-array-ref ((s-blk s-pos) src src-offset)
    (with-cuda-array-ref ((d-blk d-pos) dst dst-offset :msg nil)
      (when (eql (%cuda-linear-dd-transfer s-blk s-pos d-blk d-pos
                                           (* count elt-size)
                                           :return-if-mismatch :misaligned)
                 :misaligned)
        (warn "Misaligned pitch copy, falling back to intermediate host array.")
        (call-next-method)))))

(def method %copy-buffer-data ((src foreign-array) (dst cuda-mem-array) src-offset dst-offset count)
  (with-cuda-array-ref ((d-blk d-pos) dst dst-offset)
    (with-foreign-array-ref ((s-ptr) src src-offset)
      (%cuda-linear-dh-transfer d-blk s-ptr d-pos (* count elt-size) nil))))

(def method %copy-buffer-data ((src cuda-mem-array) (dst foreign-array) src-offset dst-offset count)
  (with-cuda-array-ref ((s-blk s-pos) src src-offset)
    (with-foreign-array-ref ((d-ptr) dst dst-offset)
      (%cuda-linear-dh-transfer s-blk d-ptr s-pos (* count elt-size) t))))

;;; Host memory array

(def class* cuda-host-array (foreign-array)
  ()
  (:automatic-accessors-p nil))

(def method print-object ((obj cuda-host-array) stream)
  (print-buffer "CUDA Host Array" obj stream))

(def class* cuda-mapped-array (cuda-mem-array)
  ()
  (:automatic-accessors-p nil))

(def method print-object ((obj cuda-mapped-array) stream)
  (print-buffer "CUDA Mapped Host Array" obj stream))

(def (function e) make-cuda-host-array (dims &key (element-type 'single-float)
                                             (foreign-type (lisp-to-gpu-type element-type) ft-p)
                                             initial-element flags)
  (unless foreign-type
    (error "Invalid foreign array type: ~S" (if ft-p foreign-type element-type)))
  (let* ((dims (ensure-list dims))
         (size (reduce #'* dims))
         (elt-type (foreign-to-gpu-type foreign-type))
         (elt-size (native-type-byte-size elt-type)))
    (let* ((blk (cuda-alloc-host (* size elt-size) :flags flags))
           (buffer (make-instance 'cuda-host-array :blk blk :size size
                                  :elt-type elt-type :elt-size elt-size
                                  :dims (to-uint32-vector dims))))
      (if initial-element
          (buffer-fill buffer initial-element)
          buffer))))

(def method buffer-displace :around ((buffer cuda-host-array) &key
                                     byte-offset foreign-type size dimensions
                                     offset element-type mapping)
  (declare (ignore byte-offset foreign-type size dimensions offset element-type))
  (aprog1 (call-next-method)
    (ecase mapping
      ((:device :gpu)
       (with-slots (blk log-offset elt-size dims) it
         (let* ((blk (cuda-map-host-blk blk))
                (strides (compute-strides (coerce dims 'list) (/ (cuda-linear-extent blk) elt-size))))
           (change-class it 'cuda-mapped-array
                         :blk blk :phys-offset log-offset
                         :strides (to-uint32-vector strides)))))
      ;; Valid NOP
      ((nil :host)))))

(def method buffer-displace :around ((buffer cuda-mapped-array) &key
                                     byte-offset foreign-type size dimensions
                                     offset element-type mapping)
  (declare (ignore byte-offset foreign-type size dimensions offset element-type))
  (aprog1 (call-next-method)
    (ecase mapping
      (:host
       (change-class it 'cuda-host-array
                     :blk (cuda-mapped-blk-root (slot-value buffer 'blk))))
      ;; Valid NOP
      ((nil :device :gpu)))))
