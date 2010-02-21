;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

;;; New builtin function definitions

;; Array functions - they make sense on the GPU side.

(def (function ei) array-raw-extent (arr)
  "Returns the raw size of a pitched array."
  (array-total-size arr))

(def (function ei) array-raw-stride (arr idx)
  "Returns the raw stride of a pitched array."
  (loop with prod = 1
     for i from (1+ idx) below (array-rank arr)
     do (setf prod (* prod (array-dimension arr i)))
     finally (return prod)))

(def (function ei) raw-aref (arr index)
  "Access a pitched array with a raw index."
  (row-major-aref arr index))

(def (function ei) (setf raw-aref) (value arr index)
  "Access a pitched array with a raw index."
  (setf (row-major-aref arr index) value))

;; Misc

(declaim (inline nonzerop))
(def (function e) nonzerop (arg)
  (not (zerop arg)))


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

(def function ensure-arithmetic-result (arg-types form &key prefix)
  (ensure-common-type arg-types form :prefix prefix
                      :types '(:int32 :uint32 :int64 :uint64 :float :double)))

(def definer delimited-builtin (name op &key zero single-pfix
                                      (typechecker 'ensure-arithmetic-result))
  `(progn
     (def type-computer ,name (&rest args)
       (if args
           (,typechecker args/type -form-)
           (propagate-c-types (splice-constant-arg -form- ,zero)
                              :upper-type -upper-type-)))
     (def c-code-emitter ,name (&rest args)
       (emit-separated -stream- args ,op :single-pfix ,single-pfix))))

(def function ensure-div-result-type (arg-types form)
  (if (cdr arg-types)
      (ensure-arithmetic-result arg-types form)
      (ensure-common-type arg-types form :types '(:float :double))))

(def delimited-builtin + "+" :zero 0)
(def delimited-builtin - "-" :zero 0 :single-pfix "-")
(def delimited-builtin * "*" :zero 1)
(def delimited-builtin / "/" :zero 1.0
  :typechecker ensure-div-result-type
  :single-pfix (if (eq (form-c-type-of -form-) :double)
                   "1.0/" "1.0f/"))

(def definer arithmetic-builtin (name args &body code)
  `(progn
     (def type-computer ,name ,args
       (ensure-arithmetic-result -arguments-/type -form-))
     (def c-code-emitter ,name ,args
       ,@code)))

(def arithmetic-builtin 1+ (arg)
  (code "(" arg "+1)"))

(def arithmetic-builtin 1- (arg)
  (code "(" arg "-1)"))

(def arithmetic-builtin abs (arg)
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

(def definer comparison-builtin (name operator &key any-type? any-count?)
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

(def comparison-builtin > ">" :any-count? t)
(def comparison-builtin >= ">=" :any-count? t)
(def comparison-builtin = "==" :any-count? t)
(def comparison-builtin /= "!=" :any-count? :quadratic)
(def comparison-builtin < "<" :any-count? t)
(def comparison-builtin <= "<=" :any-count? t)

(def comparison-builtin eql "==" :any-type? t)
(def comparison-builtin eq "==" :any-type? t)

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

;;; Bitwise logic

(def function ensure-bitwise-result (arg-types form &key prefix)
  (aprog1 (ensure-arithmetic-result arg-types form :prefix prefix)
    (when (member it '(:float :double))
      (error "Floating-point arguments not allowed: ~S" (unwalk-form form)))))

(def definer bitwise-builtin (name args &body code)
  `(progn
     (def type-computer ,name ,args
       (ensure-bitwise-result -arguments-/type -form-))
     (def c-code-emitter ,name ,args
       ,@code)))

(def delimited-builtin logand "&" :zero -1
  :typechecker ensure-bitwise-result)

(def delimited-builtin logior "|" :zero 0
  :typechecker ensure-bitwise-result)

(def delimited-builtin logxor "^" :zero 0
  :typechecker ensure-bitwise-result)

(def gpu-macro logeqv (&rest args)
  `(lognot (logxor ,@(mapcar (lambda (a) `(lognot ,a)) args))))

(def bitwise-builtin lognot (arg)
  (code "~" arg))

(def bitwise-builtin logandc1 (arg1 arg2)
  (code "(~" arg1 "&" arg2 ")"))

(def bitwise-builtin logandc2 (arg1 arg2)
  (code "(" arg1 "&~" arg2 ")"))

(def bitwise-builtin lognand (arg1 arg2)
  (code "~(" arg1 "&" arg2 ")"))

(def bitwise-builtin lognor (arg1 arg2)
  (code "~(" arg1 "|" arg2 ")"))

(def bitwise-builtin logorc1 (arg1 arg2)
  (code "(~" arg1 "|" arg2 ")"))

(def bitwise-builtin logorc2 (arg1 arg2)
  (code "(" arg1 "|~" arg2 ")"))


;;; Trigonometry etc

(def function ensure-float-result (arg-types form &key prefix)
  (ensure-common-type arg-types form :prefix prefix
                      :types '(:float :double)))

(def function float-name-tag (type)
  (ecase type (:double "") (:float "f")))

(def definer float-function-builtin (name c-name)
  `(progn
     (def type-computer ,name (arg)
       (ensure-float-result (list arg/type) -form-))
     (def c-code-emitter ,name (arg)
       (emit "~A~A(" ,c-name (float-name-tag (form-c-type-of -form-)))
       (code arg ")"))))

(def float-function-builtin sin "sin")
(def float-function-builtin asin "asin")
(def float-function-builtin sinh "sinh")
(def float-function-builtin asinh "asinh")

(def float-function-builtin cos "cos")
(def float-function-builtin acos "acos")
(def float-function-builtin cosh "cosh")
(def float-function-builtin acosh "acosh")

(def float-function-builtin tan "tan")
(def float-function-builtin atan "atan")
(def float-function-builtin tanh "tanh")
(def float-function-builtin atanh "atanh")

(def float-function-builtin exp "exp")
(def float-function-builtin sqrt "sqrt")

(def definer float-builtin (name args &body code)
  `(progn
     (def type-computer ,name ,args
       (ensure-float-result -arguments-/type -form-))
     (def c-code-emitter ,name ,args
       ,@code)))

(def float-builtin log (arg &optional base)
  (let* ((tail-tag (float-name-tag (form-c-type-of -form-)))
         (base-val (constant-number-value base)))
    (cond ((and base-val
                (or (= base-val 10) (= base-val 2)))
           (emit "log~A~A(" base-val tail-tag)
           (code arg ")"))
          (base-val
           (emit "(log~A(" tail-tag)
           (code arg)
           (emit ")/~,,,,,,'EE~A)"
                 (log (float base-val 1.0d0)) tail-tag))
          (base
           (emit "(log~A(" tail-tag)
           (code arg)
           (emit ")/log~A(" tail-tag)
           (code base "))"))
          (t
           (emit "log~A(" tail-tag)
           (code arg ")")))))

(def float-builtin expt (base power)
  (let* ((tail-tag (float-name-tag (form-c-type-of -form-)))
         (base-val (constant-number-value base)))
    (cond ((and base-val
                (or (= base-val 10) (= base-val 2)))
           (emit "exp~A~A(" base-val tail-tag)
           (code power ")"))
          (base-val
           (emit "exp~A(" tail-tag)
           (code power)
           (emit "*~,,,,,,'EE~A)"
                 (log (float base-val 1.0d0)) tail-tag))
          (t
           (emit "exp~A(" tail-tag)
           (code power)
           (emit "*log~A(" tail-tag)
           (code base "))")))))

;;; Rounding

(def form-attribute-accessor combined-arg-type
  :forms application-form)

(def function round-function-type (form)
  (acase (or (combined-arg-type-of form)
            (form-c-type-of form))
    ((:float :double) it)
    (t :float)))

(def definer float-round-builtin (name c-name)
  `(def float-builtin ,name (arg &optional divisor)
     (let ((type (round-function-type -form-)))
       (emit "~A~A(" ,c-name (float-name-tag type))
       (code arg)
       (when divisor
         (emit "/(~A)" (c-type-string type))
         (code divisor))
       (code ")"))))

(def float-round-builtin ffloor "floor")
(def float-round-builtin fceiling "ceil")
(def float-round-builtin ftruncate "trunc")
(def float-round-builtin fround "round")

(def function int-round-builtin-type (form args upper-type)
  (let ((atype (ensure-arithmetic-result args form)))
    (setf (combined-arg-type-of form) atype)
    (case atype
      ((:int32 :uint32 :int64 :uint64) atype)
      (t
       (case upper-type
         ((:int32 :uint32 :int64 :uint64) upper-type)
         (t :int32))))))

(def definer int-round-builtin (name fallback &body code)
  `(progn
     (def type-computer ,name (arg &optional divisor)
       (int-round-builtin-type -form- -arguments-/type -upper-type-))
     (def c-code-emitter ,name (arg &optional divisor)
       (bind ((div-value
               (constant-number-value divisor))
              ((:values minv maxv)
               (c-int-range (if (and (integerp div-value) (> div-value 0))
                                (form-c-type-of arg)
                                (combined-arg-type-of -form-))))
              ((:values power-2 mask-2)
               (power-of-two div-value)))
         (declare (ignorable maxv power-2 mask-2))
         (cond ((and minv (or (null divisor)
                              (eql div-value 1)))
                (recurse arg))
               ,@code
               (t
                (emit "(~A)" (c-type-string (form-c-type-of -form-)))
                (emit-call-c-code ',fallback -form- -stream-)))))))

(def int-round-builtin floor ffloor
  ((and minv power-2)
   (code "(" arg ">>" power-2 ")"))
  ((eql minv 0)
   (code "(" arg "/" divisor ")")))

(def int-round-builtin ceiling fceiling
  ((and minv power-2)
   (code "((" arg "+" mask-2 ")>>" power-2 ")"))
  ((and (eql minv 0) div-value (> div-value 0))
   (code "((" arg "+" (1- div-value) ")/" divisor ")")))

(def int-round-builtin truncate ftruncate
  ((and (eql minv 0) power-2)
   (code "(" arg ">>" power-2 ")"))
  ((eql minv 0)
   (code "(" arg "/" divisor ")")))

(def int-round-builtin round fround
  ((and minv power-2)
   (code "((" arg "+" (/ div-value 2) ")>>" power-2 ")"))
  ((and (eql minv 0) div-value (> div-value 0))
   (code "((" arg "+" (floor div-value 2) ")/" divisor ")")))


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
       (loop for arg in args and i from 0
          do (progn
               (format stream "~A TmpV~A = " (c-type-string it) i)
               (emit-c-code arg stream)
               (princ ";" stream)
               (emit-code-newline stream)))
       (loop for i from 1 below (length args)
          do (progn
               (format stream "if (TmpV~A ~A TmpV0) TmpV0 = TmpV~A;" i cmp i)
               (emit-code-newline stream)))
       (when (has-merged-assignment? form)
         (emit-merged-assignment stream form 0 "TmpV0")
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
