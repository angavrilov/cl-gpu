;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :common-lisp-user)

(defpackage :cl-gpu.buffers/types
  (:documentation "Type classes of the buffer package")
  (:export #:gpu-type #:gpu-concrete-type #:gpu-native-type
           #:lisp-type-of #:foreign-type-of #:specific-type-p
           ;; Generic types
           #:gpu-any-type #:gpu-no-type #:gpu-void-type
           #:+gpu-any-type+ #:+gpu-no-type+ #:+gpu-void-type+
           #:gpu-keyword-type #:+gpu-keyword-type+
           ;; Numbers
           #:gpu-number-type #:gpu-native-number-type
           #:min-value-of #:max-value-of
           #:gpu-integer-type #:gpu-native-integer-type
           #:+gpu-integer-foreign-ids+
           #:gpu-int8-type #:gpu-uint8-type
           #:+gpu-int8-type+ #:+gpu-uint8-type+
           #:gpu-int16-type #:gpu-uint16-type
           #:+gpu-int16-type+ #:+gpu-uint16-type+
           #:gpu-int32-type #:gpu-uint32-type
           #:+gpu-int32-type+ #:+gpu-uint32-type+
           #:gpu-int64-type #:gpu-uint64-type
           #:+gpu-int64-type+ #:+gpu-uint64-type+
           #:gpu-float-type #:gpu-native-float-type
           #:gpu-single-float-type #:gpu-double-float-type
           #:+gpu-single-float-type+ #:+gpu-double-float-type+
           ;; Boolean
           #:gpu-boolean-type #:+gpu-boolean-type+
           ;; Compounds
           #:gpu-compound-type
           #:gpu-values-type #:values-of
           #:gpu-container-type
           #:item-type #:dimensions
           #:item-type-of #:dimensions-of
           #:gpu-array-type
           #:gpu-tuple-type #:size #:size-of
           #:item-lisp-type #:item-byte-size
           #:gpu-pointer-type #:default-pointer-type
           #:gpu-32b-pointer-type #:gpu-64b-pointer-type
           ;; Native properties
           #:native-type-byte-size #:native-type-c-string
           #:native-type-alignment
           #:c-type-alignment #:c-type-string #:c-type-size
           ;; Utils
           #:reintern-as-class
           #:make-foreign-gpu-type #:parse-lisp-type
           #:make-gpu-integer-from-range))

(defpackage :cl-gpu.buffers/impl
  (:documentation "Foreign buffer implementation objects.")
  (:use :cl-gpu.buffers/types)
  (:export #:print-buffer
           #:%copy-buffer-data
           ;; Abstract foreign
           #:counted-block
           #:counted-block-refcnt #:counted-block-handle
           #:invalidate #:deallocate
           #:delegate-buffer-refcnt
           #:abstract-foreign-buffer
           #:blk #:displaced-to #:log-offset #:size #:elt-type #:elt-size #:dims
           ;; Foreign
           #:foreign-block
           #:foreign-block-refcnt #:foreign-block-handle #:foreign-block-size
           #:alloc-foreign-block
           #:foreign-array
           ;; Mirror
           #:mirrored-foreign-buffer #:mirror #:update-buffer-mirror
           ;; Utils
           #:memcpy #:memset #:memmove
           #:with-pointer-to-array #:native-type-ref
           #:with-lisp-array-ref #:with-foreign-array-ref))

(defpackage :cl-gpu.buffers
  (:documentation "A package providing generalized buffer support.")

  (:use :alexandria
        :anaphora
        :contextl
        :trivial-garbage
        :cffi
        :hu.dwim.common-lisp
        :hu.dwim.def
        :hu.dwim.defclass-star
        :hu.dwim.util
        :metabang-bind
        :cl-gpu.buffers/types
        :cl-gpu.buffers/impl)

  (:export #:buffer
           #:int8 #:int16 #:int32 #:int64
           #:uint8 #:uint16 #:uint32 #:uint64))

