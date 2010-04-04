;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file implements the generic buffer interface for
;;; a buffer based on foreign memory allocated via CFFI.
;;;

(in-package :cl-gpu)

;;; Reference counting

(defstruct counted-block
  (refcnt 1 :type fixnum)
  (handle nil))

(defgeneric invalidate (blk)
  (:documentation "Invalidates the block handle")
  (:method-combination progn :most-specific-last)
  (:method progn ((blk t))
    (cancel-finalization blk))
  (:method progn ((blk counted-block))
    (setf (counted-block-handle blk) nil)))

(defgeneric deallocate (blk)
  (:documentation "Deallocates the block")
  (:method :after ((blk t))
    (invalidate blk))
  (:method :around ((blk counted-block))
    (if (counted-block-handle blk)
        (call-next-method)
        (cerror "ignore" "This block has already been deallocated"))))

(def method buffer-refcnt ((blk counted-block))
  (if (counted-block-handle blk)
      (counted-block-refcnt blk)
      nil))

(def method ref-buffer ((blk counted-block))
  (incf (counted-block-refcnt blk)))

(def method deref-buffer ((blk counted-block))
  (unless (> (decf (counted-block-refcnt blk)) 0)
    (deallocate blk)))

(def macro delegate-buffer-refcnt ((var class) target)
  `(progn
     (def method buffer-refcnt ((,var ,class))
       (buffer-refcnt ,target))
     (def method ref-buffer ((,var ,class))
       (ref-buffer ,target))
     (def method deref-buffer ((,var ,class))
       (deref-buffer ,target))))

;;; Common code for foreign buffers

(def class* abstract-foreign-buffer ()
  ((blk :type counted-block)
   (displaced-to :type (or abstract-foreign-buffer null) :initform nil
                 :documentation "Foreign array this one is displaced to")
   (log-offset :type fixnum :initform 0
               :documentation "Offset to the start of the block")
   (size :type fixnum :documentation "Array size in elements")
   (elt-type :type t :documentation "Foreign element type")
   (elt-size :type fixnum :documentation "Element size in bytes")
   (dims :type (vector uint32) :documentation "Dimension vector"))
  (:automatic-accessors-p nil))

(def method bufferp ((buffer abstract-foreign-buffer)) :foreign)

(delegate-buffer-refcnt (buffer abstract-foreign-buffer) (slot-value buffer 'blk))

(def method buffer-foreign-type ((buffer abstract-foreign-buffer))
  (slot-value buffer 'elt-type))

(def method buffer-rank ((buffer abstract-foreign-buffer))
  (length (slot-value buffer 'dims)))

(def method buffer-dimensions ((buffer abstract-foreign-buffer))
  (coerce (slot-value buffer 'dims) 'list))

(def method buffer-dimension ((buffer abstract-foreign-buffer) axis)
  (aref (slot-value buffer 'dims) axis))

(def method buffer-size ((buffer abstract-foreign-buffer))
  (slot-value buffer 'size))

(def method buffer-row-major-index ((buffer abstract-foreign-buffer) &rest indexes)
  ;; A copy with direct dimension vector access
  (loop
     with dims of-type (vector uint32) = (slot-value buffer 'dims)
     with index fixnum = 0
     for idx in indexes
     for i from 0
     do (let ((dim (aref dims i)))
          (unless (and (>= idx 0) (< idx dim))
            (error "Index ~S is out of bounds for ~S" indexes buffer))
          (setf index (+ (* dim index) idx)))
     finally (return index)))

(def method row-major-bref :around ((buffer abstract-foreign-buffer) index)
  (with-slots (size) buffer
    (unless (and (>= index 0) (< index size))
      (error "Linear index ~S is out of bounds for ~S" index buffer))
    (call-next-method)))

(def method (setf row-major-bref) :around (value (buffer abstract-foreign-buffer) index)
  (declare (ignore value))
  (with-slots (size) buffer
    (unless (and (>= index 0) (< index size))
      (error "Linear index ~S is out of bounds for ~S" index buffer))
    (call-next-method)))

(def method buffer-fill :around ((buffer abstract-foreign-buffer) value &key (start 0) end)
  (with-slots (size) buffer
    (unless (and (>= start 0) (< start size)
                 (or (null end)
                     (and (>= end start) (<= end size))))
      (error "Bad fill range ~A...~A for ~S" start end buffer))
    (let ((endp (or end size)))
      (when (> endp start)
        (call-next-method buffer value :start start :end endp)))
    buffer))

(def method buffer-displace :around ((buffer abstract-foreign-buffer) &rest flags &key
                                     (offset 0 ofs-p)
                                     (byte-offset (if ofs-p (* offset (slot-value buffer 'elt-size)) 0))
                                     (element-type nil elt-type-p)
                                     (foreign-type (if elt-type-p
                                                       (lisp-to-foreign-elt-type element-type)
                                                       (slot-value buffer 'elt-type)))
                                     (size nil)
                                     (dimensions (or size (buffer-dimensions buffer))))
  (check-type byte-offset unsigned-byte)
  (with-slots (elt-type elt-size size) buffer
    (let ((old-byte-size (* elt-size size))
          (new-elt-size (foreign-type-size foreign-type)))
      (when (eql dimensions t)
        (setf dimensions (floor (- old-byte-size byte-offset) new-elt-size)))
      (let ((dims (ensure-list dimensions)))
        (unless (every (lambda (x) (typep x 'unsigned-byte)) dims)
          (error "Invalid dimensions: ~S" dims))
        (let* ((new-size (reduce #'* dims))
               (byte-size (* new-size new-elt-size)))
          (when (> (+ byte-offset byte-size) old-byte-size)
            (error "Specified dimensions ~A ~S exceed the original size ~A ~A"
                   foreign-type dims size elt-type))
          (apply #'call-next-method buffer
                 :byte-offset byte-offset :foreign-type foreign-type
                 :size new-size :dimensions dims
                 (remove-from-plist flags :offset :byte-offset :element-type
                                    :foreign-type :size :dimensions)))))))

(def method buffer-displacement ((buffer abstract-foreign-buffer))
  (with-slots (displaced-to log-offset) buffer
    (if displaced-to
        (with-slots (elt-size (s-log-offset log-offset)) displaced-to
          (values displaced-to (/ (- log-offset s-log-offset) elt-size)))
        (values nil 0))))


;;; Buffer backed by a host foreign block

(defstruct (foreign-block (:include counted-block))
  (size 0 :type fixnum :read-only t))

(def method deallocate ((blk foreign-block))
  (foreign-free (foreign-block-handle blk)))

(def function alloc-foreign-block (foreign-type count &key initial-element)
  (let* ((ptr (if initial-element
                  (foreign-alloc foreign-type :count count :initial-element initial-element)
                  (foreign-alloc foreign-type :count count)))
         (blk (make-foreign-block :handle ptr
                                  :size (* count (foreign-type-size foreign-type)))))
    (finalize blk (lambda () (foreign-free ptr)))
    blk))

;; The actual buffer:

(def class* foreign-array (abstract-foreign-buffer)
  ((blk :type foreign-block))
  (:automatic-accessors-p nil))

(def method print-object ((obj foreign-array) stream)
  (print-buffer "Foreign Array" obj stream))

(def (function e) make-foreign-array (dims &key (element-type 'single-float)
                                           (foreign-type (lisp-to-foreign-elt-type element-type) ft-p)
                                           initial-element)
  (unless foreign-type
    (error "Invalid foreign array type: ~S" (if ft-p foreign-type element-type)))
  (let* ((dims (ensure-list dims))
         (size (reduce #'* dims))
         (elt-size (foreign-type-size foreign-type))
         (blk (alloc-foreign-block foreign-type size :initial-element initial-element))
         (buffer (make-instance 'foreign-array :blk blk :size size
                                :elt-type foreign-type :elt-size elt-size
                                :dims (to-uint32-vector dims))))
    (values buffer)))

(def method buffer-displace ((buffer foreign-array) &key
                             byte-offset foreign-type size dimensions
                             element-type offset)
  (declare (ignore element-type offset))
  (with-slots (blk log-offset) buffer
    ;; The dimensions have been verified and canonified by the around method
    (make-instance (class-of buffer) :blk blk :size size
                   :displaced-to buffer :log-offset (+ log-offset byte-offset)
                   :elt-type foreign-type :elt-size (foreign-type-size foreign-type)
                   :dims (to-uint32-vector dimensions))))

(def method row-major-bref ((buffer foreign-array) index)
  (with-slots (blk log-offset elt-type elt-size) buffer
    (mem-ref (foreign-block-handle blk) elt-type
             (+ (* index elt-size) log-offset))))

(def method (setf row-major-bref) (value (buffer foreign-array) index)
  (with-slots (blk log-offset elt-type elt-size) buffer
    (setf (mem-ref (foreign-block-handle blk) elt-type
                   (+ (* index elt-size) log-offset))
          value)))

(defcfun "memcpy" :pointer
  (dest :pointer)
  (src :pointer)
  (count :unsigned-int))

(defcfun "memmove" :pointer
  (dest :pointer)
  (src :pointer)
  (count :unsigned-int))

(defcfun "memset" :pointer
  (dest :pointer)
  (byte :int)
  (count :unsigned-int))

(def macro with-foreign-array-ref (((ptr-var) buffer index) &body code)
  (with-unique-names (blk-var log-offset-var)
    `(with-slots ((,blk-var blk) (,log-offset-var log-offset) elt-size) ,buffer
       (let ((,ptr-var (inc-pointer (foreign-block-handle ,blk-var)
                                    (+ (* ,index elt-size) ,log-offset-var))))
         ,@code))))

(def method buffer-fill ((buffer foreign-array) value &key start end)
  (with-foreign-array-ref ((ptr) buffer start)
    (if (eql value 0)
        (memset ptr 0 (* (- end start) elt-size))
        (loop with elt-type = (slot-value buffer 'elt-type)
           for i from 0 below (- end start)
           do (setf (mem-aref ptr elt-type i) value)))))

(def method %copy-buffer-data ((src array) (dst foreign-array) src-offset dst-offset count)
  (with-foreign-array-ref ((d-ptr) dst dst-offset)
    (with-lisp-array-ref ((s-ptr) src src-offset elt-size)
      (memcpy d-ptr s-ptr (* count elt-size)))))

(def method %copy-buffer-data ((src foreign-array) (dst array) src-offset dst-offset count)
  (with-foreign-array-ref ((s-ptr) src src-offset)
    (with-lisp-array-ref ((d-ptr) dst dst-offset elt-size)
      (memcpy d-ptr s-ptr (* count elt-size)))))

(def method %copy-buffer-data ((src foreign-array) (dst foreign-array) src-offset dst-offset count)
  (with-foreign-array-ref ((s-ptr) src src-offset)
    (with-foreign-array-ref ((d-ptr) dst dst-offset)
      (memmove d-ptr s-ptr (* count elt-size)))))

;;; A mirrored foreign buffer

(def class* mirrored-foreign-buffer (abstract-foreign-buffer)
  ((mirror :type foreign-array))
  (:automatic-accessors-p nil))

(def function make-displaced-foreign-mirror (dbuf displaced-to)
  (with-slots (mirror dims size elt-size elt-type log-offset) dbuf
    (let* ((base-mirror (slot-value displaced-to 'mirror))
           (base-blk (slot-value base-mirror 'blk)))
      (setf mirror
            (make-instance 'foreign-array :blk base-blk :dims dims
                           :log-offset log-offset :size size
                           :elt-size elt-size :elt-type elt-type)))))

(def function alloc-base-foreign-mirror (dbuf)
  (with-slots (mirror dims log-offset displaced-to elt-type) dbuf
    (assert (and (null displaced-to) (= log-offset 0)))
    (setf mirror
          (make-foreign-array (coerce dims 'list) :foreign-type elt-type))
    (copy-full-buffer dbuf mirror)))

(def constructor mirrored-foreign-buffer
  (unless (slot-boundp -self- 'mirror)
    (aif (slot-value -self- 'displaced-to)
         (make-displaced-foreign-mirror -self- it)
         (alloc-base-foreign-mirror -self-))))

#+nil
(def method update-instance-for-different-class :after
  (old-inst (-self- mirrored-foreign-buffer) &key)
  (declare (ignore old-inst))
  (unless (slot-boundp -self- 'mirror)
    (alloc-base-foreign-mirror -self-)))

(def generic update-buffer-mirror (buffer)
  (:method ((buffer t)) nil)
  (:method ((buffer mirrored-foreign-buffer))
    (copy-full-buffer buffer (slot-value buffer 'mirror))))

(def method deref-buffer :after ((buffer mirrored-foreign-buffer))
  (when (null (buffer-refcnt (slot-value buffer 'blk)))
    (deref-buffer (slot-value buffer 'mirror))))

(def method (setf row-major-bref) :after (value (buffer mirrored-foreign-buffer) index)
  (setf (row-major-bref (slot-value buffer 'mirror) index) value))

(def method buffer-fill :after ((buffer mirrored-foreign-buffer) value &key start end)
  (buffer-fill (slot-value buffer 'mirror) value :start start :end end))

(def method %copy-buffer-data :after ((src t) (dst mirrored-foreign-buffer) src-offset dst-offset count)
  (%copy-buffer-data src (slot-value dst 'mirror) src-offset dst-offset count))

