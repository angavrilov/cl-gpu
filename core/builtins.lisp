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
  (code " = " -value-))

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
    (code index "]")))

(def c-code-emitter (setf raw-aref) (var index)
  (let ((gpu-var (ensure-gpu-var var)))
    (emit "~A[" (generate-var-ref gpu-var))
    (code index "] = " -value-)))

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

;;; Arithmetics

(def function arithmetic-result-type (arg-types)
  (cond ((member :double arg-types) :double)
        ((member :float arg-types) :float)
        ((member :uint64 arg-types) :uint64)
        ((member :int64 arg-types) :int64)
        ((member :uint32 arg-types) :uint32)
        (t :int32)))

(def function ensure-arithmetic-result (arg-types form &key prefix)
  (aprog1 (arithmetic-result-type arg-types)
    (dolist (arg arg-types)
      (verify-cast arg it form :prefix prefix))))

(def function splice-constant-arg (form value)
  (with-form-object (const 'constant-form form :value value)
    (push const (arguments-of form))))

(def definer arithmetic-type-computer (name &key zero)
  `(def type-computer ,name (&rest args)
     (if args
         (ensure-arithmetic-result args/type -form-)
         (propagate-c-types (splice-constant-arg -form- ,zero)
                            :upper-type -upper-type-))))

(def arithmetic-type-computer + :zero 0)
(def arithmetic-type-computer - :zero 0)
(def arithmetic-type-computer * :zero 1)
(def arithmetic-type-computer / :zero 1.0)

(def type-computer 1+ (arg)
  (ensure-arithmetic-result (list arg/type) -form-))

(def type-computer 1- (arg)
  (ensure-arithmetic-result (list arg/type) -form-))

(def function emit-separated (stream args op &key (single-pfix ""))
  (assert args)
  (princ "(" stream)
  (when (null (rest args))
    (princ single-pfix stream))
  (emit-c-code (first args) stream)
  (dolist (arg (rest args))
    (princ op stream)
    (emit-c-code arg stream))
  (princ ")" stream))

(def c-code-emitter + (&rest args)
  (emit-separated -stream- args "+"))

(def c-code-emitter - (&rest args)
  (emit-separated -stream- args "-" :single-pfix "-"))

(def c-code-emitter * (&rest args)
  (emit-separated -stream- args "*"))

(def c-code-emitter / (&rest args)
  (emit-separated -stream- args "/"
                  :single-pfix
                  (if (eq (form-c-type-of -form-) :double)
                      "1.0/" "1.0f/")))

(def c-code-emitter 1+ (arg)
  (code "(" arg "+1)"))

(def c-code-emitter 1- (arg)
  (code "(" arg "-1)"))

(def type-computer abs (arg)
  (ensure-arithmetic-result (list arg/type) -form-))

(def c-code-emitter abs (arg)
  (emit "~A("
        (ecase (form-c-type-of -form-)
          (:int32 "abs")
          (:uint32 "")
          (:int64 "llabs")
          (:uint64 "")
          (:float "fabsf")
          (:double "fabs")))
  (code arg ")"))

;;; Comparisons and logical ops

#+ccl
(def gpu-macro ccl::int>0-p (arg)
  `(> ,arg 0))

(def type-computer not (arg)
  (verify-cast arg/type :boolean -form-))

(def c-code-emitter not (arg)
  (code "!" arg))

(def type-computer zerop (arg)
  (ensure-arithmetic-result (list arg/type) -form-)
  :boolean)

(def c-code-emitter zerop (arg)
  (code "(" arg "==0)"))

(declaim (inline nonzerop))
(def (function e) nonzerop (arg)
  (not (zerop arg)))

(def type-computer nonzerop (arg)
  (ensure-arithmetic-result (list arg/type) -form-)
  :boolean)

(def c-code-emitter nonzerop (arg)
  (code "(" arg "!=0)"))

(def function expand-comparison-chain (form args mode)
  (if (cddr args)
      (let ((names (loop for i from 0 below (length args)
                      collect (format-symbol nil "TV~A" i))))
        `(let ,(loop for name in names and arg in args
                  collect `(,name ,arg))
           (and ,@(if (eq mode :quadratic)
                      (loop for n1 on names append
                           (loop for n2 in (cdr n1)
                              collect `(,(car form) ,(car n1) ,n2)))
                      (loop for n1 in names and n2 in (cdr names)
                         collect `(,(car form) ,n1 ,n2))))))
      form))

(def macro def-comparison-builtin (name operator &key any-type? any-count?)
  `(progn
     ,(if any-count?
          `(def gpu-macro ,name (&whole form &rest args)
             (expand-comparison-chain form args ,any-count?)))
     (def type-computer ,name (arg1 arg2)
       ,(if any-type?
            `(or (equal arg1/type arg2/type)
                 (ensure-arithmetic-result (list arg1/type arg2/type) -form-))
            `(ensure-arithmetic-result (list arg1/type arg2/type) -form-))
       :boolean)
     (def c-code-emitter ,name (arg1 arg2)
       (code "(" arg1 ,operator arg2 ")"))))

(def-comparison-builtin > ">" :any-count? t)
(def-comparison-builtin >= ">=" :any-count? t)
(def-comparison-builtin = "==" :any-count? t)
(def-comparison-builtin /= "!=" :any-count? :quadratic)
(def-comparison-builtin < "<" :any-count? t)
(def-comparison-builtin <= "<=" :any-count? t)

(def-comparison-builtin eql "==" :any-type? t)
(def-comparison-builtin eq "==" :any-type? t)

(def type-computer and (&rest args)
  (dolist (atype args/type)
    (verify-cast atype :boolean -form-))
  :boolean)

(def c-code-emitter and (&rest args)
  (if (null args)
      (emit "1")
      (emit-separated -stream- args "&&")))

(def type-computer or (&rest args)
  (dolist (atype args/type)
    (verify-cast atype :boolean -form-))
  :boolean)

(def c-code-emitter or (&rest args)
  (if (null args)
      (emit "0")
      (emit-separated -stream- args "||")))

;;; Trigonometry etc

(def function ensure-float-result (arg-types form &key prefix)
  (aprog1 (if (member :double arg-types) :double :float)
    (dolist (arg arg-types)
      (verify-cast arg it form :prefix prefix))))

(def definer float-gpu-builtin (name float-c-name double-c-name)
  `(progn
     (def type-computer ,name (arg)
       (ensure-float-result (list arg/type) -form-))
     (def c-code-emitter ,name (arg)
       (ecase (form-c-type-of -form-)
         (:double (emit ,double-c-name))
         (:float (emit ,float-c-name)))
       (code "(" arg ")"))))

(def float-gpu-builtin sin "sinf" "sin")
(def float-gpu-builtin asin "asinf" "asin")
(def float-gpu-builtin sinh "sinhf" "sinh")
(def float-gpu-builtin asinh "asinhf" "asinh")

(def float-gpu-builtin cos "cosf" "cos")
(def float-gpu-builtin acos "acosf" "acos")
(def float-gpu-builtin cosh "coshf" "cosh")
(def float-gpu-builtin acosh "acoshf" "acosh")

(def float-gpu-builtin tan "tanf" "tan")
(def float-gpu-builtin atan "atanf" "atan")
(def float-gpu-builtin tanh "tanhf" "tanh")
(def float-gpu-builtin atanh "atanhf" "atanh")

(def float-gpu-builtin exp "expf" "exp")
(def float-gpu-builtin sqrt "sqrtf" "sqrt")

(def type-computer log (val &optional base)
  (ensure-float-result (flatten (list val/type base/type)) -form-))

(def c-code-emitter log (arg &optional base)
  (let* ((ftype (form-c-type-of -form-))
         (tail-tag (ecase ftype (:double "") (:float "f")))
         (base-val (and (typep base 'constant-form)
                        (numberp (value-of base))
                        (value-of base))))
    (cond ((and base-val
                (or (= base-val 10) (= base-val 2)))
           (emit "log~A~A(" base-val tail-tag)
           (code arg ")"))
          (base-val
           (emit "(log~A(" tail-tag)
           (code arg)
           (emit ")/~,,,,,,'EG~A)"
                 (log (float base-val 1.0d0)) tail-tag))
          (base
           (emit "(log~A(" tail-tag)
           (code arg)
           (emit ")/log~A(" tail-tag)
           (code base "))"))
          (t
           (emit "log~A(" tail-tag)
           (code arg ")")))))

(def type-computer expt (base power)
  (ensure-float-result (list base/type power/type) -form-))

(def c-code-emitter expt (base power)
  (let* ((ftype (form-c-type-of -form-))
         (tail-tag (ecase ftype (:double "") (:float "f")))
         (base-val (and (typep base 'constant-form)
                        (numberp (value-of base))
                        (value-of base))))
    (cond ((and base-val
                (or (= base-val 10) (= base-val 2)))
           (emit "exp~A~A(" base-val tail-tag)
           (code power ")"))
          (base-val
           (emit "exp~A(" tail-tag)
           (code power)
           (emit "*~,,,,,,'EG~A)"
                 (log (float base-val 1.0d0)) tail-tag))
          (t
           (emit "exp~A(" tail-tag)
           (code power)
           (emit "*log~A(" tail-tag)
           (code base "))")))))

;;; MIN & MAX

(def function treeify-list (items)
  (if (cdr items)
      (treeify-list
       (loop for obj on items by #'cddr
          collect (if (cdr obj)
                      (cons (car obj) (cadr obj))
                      (car obj))))
      (car items)))

(def function emit-function-tree (stream name item-tree)
  (if (consp item-tree)
      (progn
        (format stream "~A(" name)
        (emit-function-tree stream name (car item-tree))
        (princ "," stream)
        (emit-function-tree stream name (cdr item-tree))
        (princ ")" stream))
      (emit-c-code item-tree stream)))

(def function emit-minmax-code (stream form cmp float-fun double-fun args)
  (when (null (rest args))
    (return-from emit-minmax-code
      (emit-c-code (first args) stream)))
  (acase (form-c-type-of form)
    (:double (emit-function-tree stream double-fun (treeify-list args)))
    (:float (emit-function-tree stream float-fun (treeify-list args)))
    (t
     (princ "{" stream)
     (with-indented-c-code
       (emit-code-newline stream)
       (format stream "~A TmpV, BestV = " (c-type-string it))
       (emit-c-code (first args) stream)
       (princ ";" stream)
       (emit-code-newline stream)
       (dolist (arg (rest args))
         (princ "TmpV = " stream)
         (emit-c-code arg stream)
         (princ ";" stream)
         (emit-code-newline stream)
         (format stream "if (TmpV ~A BestV) BestV = TmpV;" cmp)
         (emit-code-newline stream))
       (when (has-merged-assignment? form)
         (emit-merged-assignment stream form 0 "BestV")
         (princ ";" stream)))
     (emit-code-newline stream)
     (princ "}" stream))))

(def is-statement? min (&rest args)
  (case (form-c-type-of -form-)
    ((:float :double) nil)
    (t (rest args))))

(def type-computer min (&rest args)
  (when (null args)
    (error "Cannot use MIN without arguments."))
  (ensure-arithmetic-result args/type -form-))

(def c-code-emitter min (&rest args)
  (emit-minmax-code -stream- -form- "<" "fminf" "fmin" args))

(def is-statement? max (&rest args)
  (case (form-c-type-of -form-)
    ((:float :double) nil)
    (t (rest args))))

(def type-computer max (&rest args)
  (when (null args)
    (error "Cannot use MAX without arguments."))
  (ensure-arithmetic-result args/type -form-))

(def c-code-emitter max (&rest args)
  (emit-minmax-code -stream- -form- ">" "fmaxf" "fmax" args))
