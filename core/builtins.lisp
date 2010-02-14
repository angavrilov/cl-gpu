;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

;;; AREF

(def type-computer aref (arr &rest indexes)
  (verify-array-var arr)
  (unless (= (length (dimension-mask-of (ensure-gpu-var arr)))
             (length indexes))
    (error "Incorrect array index count in ~S" (unwalk-form -form-)))
  (loop for itype in indexes/type and idx in indexes
     do (verify-cast itype :uint32 idx :prefix "aref index "
                     :allow '(:int32) :error-on-warn? t))
  (second arr/type))

(def type-arg-walker (setf aref) (arr &rest indexes)
  (let ((arr/type (recurse arr)))
    (verify-array-var arr)
    (unless (= (length (dimension-mask-of (ensure-gpu-var arr)))
               (length indexes))
      (error "Incorrect array index count in ~S" (unwalk-form -form-)))
    (dolist (idx indexes)
      (recurse idx :upper-type :uint32))
    (recurse -value- :upper-type (second arr/type))))

(def type-computer (setf aref) (arr &rest indexes)
  (loop for itype in indexes/type and idx in indexes
     do (verify-cast itype :uint32 idx :prefix "aref index "
                     :allow '(:int32) :error-on-warn? t))
  (verify-cast -value-/type (second arr/type) -form-)
  (if (eq -upper-type- :void) :void (second arr/type)))

(def function emit-aref-core (var indexes stream)
  (let* ((gpu-var (ensure-gpu-var var))
         (rank (length (dimension-mask-of gpu-var))))
    (assert (= (length indexes) rank))
    (format stream "~A[" (generate-var-ref gpu-var))
    (loop for i from 0 and idx in indexes
       when (> i 0) do
         (princ "+" stream)
       do (emit-c-code idx stream)
       when (< i (1- rank)) do
         (format stream "*~A" (generate-array-stride gpu-var i)))
    (princ "]" stream)))

(def c-code-emitter aref (var &rest indexes)
  (emit-aref-core var indexes -stream-))

(def c-code-emitter (setf aref) (var &rest indexes)
  (emit-aref-core var indexes -stream-)
  (emit " = ")
  (recurse -value-))

;;; RAW-AREF

(def type-computer raw-aref (arr index)
  (verify-array-var arr)
  (verify-cast index/type :uint32 index :prefix "raw-aref index "
               :allow '(:int32) :error-on-warn? t)
  (second arr/type))

(def type-arg-walker (setf raw-aref) (arr index)
  (let ((arr/type (recurse arr)))
    (verify-array-var arr)
    (recurse index :upper-type :uint32)
    (recurse -value- :upper-type (second arr/type))))

(def type-computer (setf raw-aref) (arr index)
  (verify-cast index/type :uint32 index :prefix "raw-aref index "
               :allow '(:int32) :error-on-warn? t)
  (verify-cast -value-/type (second arr/type) -form-)
  (if (eq -upper-type- :void) :void (second arr/type)))

(def c-code-emitter raw-aref (var index)
  (let ((gpu-var (ensure-gpu-var var)))
    (emit "~A[" (generate-var-ref gpu-var))
    (recurse index)
    (emit "]")))

(def c-code-emitter (setf raw-aref) (var index)
  (let ((gpu-var (ensure-gpu-var var)))
    (emit "~A[" (generate-var-ref gpu-var))
    (recurse index)
    (emit "] = ")
    (recurse -value-)))

;;; Misc array properties

(def type-computer array-total-size (arr)
  (verify-array-var arr)
  :uint32)

(def type-computer array-raw-extent (arr)
  (verify-array-var arr)
  :uint32)

(def type-computer array-dimension (var dimidx)
  (verify-array-var var)
  (let ((cval (ensure-int-constant dimidx))
        (gvar (ensure-gpu-var var)))
    (unless (and (>= cval 0)
                 (< cval (length (dimension-mask-of gvar))))
      (error "Dimension index out of bounds: ~S" (unwalk-form -form-))))
  :uint32)

(def type-computer array-raw-stride (var dimidx)
  (verify-array-var var)
  (let ((cval (ensure-int-constant dimidx))
        (gvar (ensure-gpu-var var)))
    (unless (and (>= cval 0)
                 (< cval (1- (length (dimension-mask-of gvar)))))
      (error "Stride index out of bounds: ~S" (unwalk-form -form-))))
  :uint32)

(def c-code-emitter array-total-size (var)
  (emit "~A" (generate-array-size (ensure-gpu-var var))))

(def c-code-emitter array-raw-extent (var)
  (emit "~A" (generate-array-extent (ensure-gpu-var var))))

(def c-code-emitter array-dimension (var dimidx)
  (emit "~A" (generate-array-dim (ensure-gpu-var var)
                                 (ensure-constant dimidx))))

(def c-code-emitter array-raw-stride (var dimidx)
  (emit "~A" (generate-array-stride (ensure-gpu-var var)
                                    (ensure-constant dimidx))))

