;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file implements type checking and code generation
;;; for standard library functions and special built-ins.
;;;

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

(def (function ei) array-raw-index (arr &rest indexes)
  "Returns the raw index of a pitched array."
  (apply #'array-row-major-index arr indexes))

(def (function ei) raw-aref (arr index)
  "Access a pitched array with a raw index."
  (row-major-aref arr index))

(def (function ei) (setf raw-aref) (value arr index)
  "Access a pitched array with a raw index."
  (setf (row-major-aref arr index) value))

;; Tuples

(def (function ei) untuple (tuple)
  (check-type tuple vector)
  (values-list (coerce tuple 'list)))

(def (function ei) tuple (&rest items)
  (coerce items 'vector))

(def (function e) tuple-aref (arr &rest indexes)
  "Access the innermost dimension of an array as a tuple."
  (bind ((size (array-dimension arr (length indexes)))
         (index (apply #'array-row-major-index arr (append indexes '(0))))
         (rv (make-array size :element-type (array-element-type arr))))
    (loop for i from 0 below size
       do (setf (row-major-aref rv i) (row-major-aref arr (+ i index))))
    rv))

(def (function e) (setf tuple-aref) (value arr &rest indexes)
  "Update the innermost dimension of an array as a tuple."
  (bind ((size (array-total-size value))
         (index (apply #'array-row-major-index arr (append indexes '(0)))))
    (loop for i from 0 below size
       do (setf (row-major-aref arr (+ i index)) (row-major-aref value i)))
    value))

(def (function e) tuple-raw-aref (arr index size)
  "Access consecutive items of an array as a tuple."
  (bind ((rv (make-array size :element-type (array-element-type arr))))
    (loop for i from 0 below size
       do (setf (row-major-aref rv i) (row-major-aref arr (+ i index))))
    rv))

(def (function e) (setf tuple-raw-aref) (value arr index size)
  "Update consecutive items of an array as a tuple."
  (loop for i from 0 below size
     do (setf (row-major-aref arr (+ i index)) (row-major-aref value i)))
  value)

;; Thread grid dimensions

(macrolet ((dimfun (name short-stem)
             `(progn
                (def (function e) ,name (&optional dimension)
                  (declare (ignore dimension))
                  (error "~S is only supported in GPU code" ',name))
                (def (symbol-macro e) ,short-stem (,name))
                (def (symbol-macro e) ,(symbolicate short-stem :-X) (,name 0))
                (def (symbol-macro e) ,(symbolicate short-stem :-Y) (,name 1))
                (def (symbol-macro e) ,(symbolicate short-stem :-Z) (,name 2)))))
  (dimfun thread-index thread-idx)
  (dimfun thread-count thread-cnt)
  (dimfun block-index block-idx)
  (dimfun block-count block-cnt))

;; Misc

(declaim (inline nonzerop))
(def (function e) nonzerop (arg)
  (not (zerop arg)))

(def (function e) barrier (&optional mode)
  (declare (ignore mode))
  (cerror "continue" "Barriers are only supported in GPU code"))

(def type-computer barrier (&optional mode)
  (unless (or (null mode) (eq mode/type :keyword))
    (gpu-code-error -form- "Barrier type must be a keyword constant."))
  :void)

;;; AREF

(def function walk-aref-type-core (form arr indexes &key (value nil v-p)
                                        (item-type-fun #'second) (index-bias 0)
                                        access-prefix-fun)
  (declare (ignore access-prefix-fun))
  (with-type-arg-walker-lexicals
    (let* ((arr/type (recurse arr))
           (item-type (funcall item-type-fun arr/type)))
      (verify-array-var arr)
      (unless (= (length (dimension-mask-of (ensure-gpu-var arr)))
                 (+ (length indexes) index-bias))
        (gpu-code-error form "Incorrect array index count."))
      (dolist (idx indexes)
        (recurse idx :upper-type :uint32))
      (when v-p
        (recurse value :upper-type item-type)))))

(def function type-aref-core (form arr indexes &key (value nil v-p)
                                   (item-type-fun #'second) (index-bias 0)
                                   access-prefix-fun)
  (declare (ignore index-bias access-prefix-fun))
  (let* ((arr/type (form-c-type-of arr))
         (item-type (funcall item-type-fun arr/type)))
    (loop for idx in indexes
       do (verify-cast idx :uint32 form :prefix "index "
                       :allow '(:int32) :error-on-warn? t))
    (when v-p
      (verify-cast value item-type form))
    item-type))

(def function emit-aref-expr (stream prefix base-fun gpu-var rank indexes)
  (format stream "(~A(~A" prefix (funcall base-fun gpu-var))
  (loop for i from 0 and idx in indexes
     do (princ "+" stream)
     do (emit-c-code idx stream)
     when (< i (1- rank)) do
     (format stream "*~A" (generate-array-stride gpu-var i)))
  (princ "))" stream))

(def (function i) generate-bound-checks? ()
  (is-optimize-level-any? :check-bounds 1 'safety 1))

(def function emit-aref-core (form var indexes stream &key value
                                   (item-type-fun #'second) (index-bias 0)
                                   (access-prefix-fun (constantly "*"))
                                   (access-base-fun #'generate-var-ref))
  (declare (ignore item-type-fun))
  (with-c-code-emitter-lexicals (stream)
    (let* ((gpu-var (ensure-gpu-var var))
           (rank (length (dimension-mask-of gpu-var)))
           (prefix (funcall access-prefix-fun form)))
      (assert (= (+ (length indexes) index-bias) rank))
      (if (generate-bound-checks?)
          ;; With bound checks
          (with-c-code-block (stream)
            (loop for i from 0 and index in indexes
               for idxname = (format nil "IDX~A" i)
               for dimname = (format nil "DIM~A" i)
               do (code "unsigned " idxname " = " index ";" #\Newline
                        "unsigned " dimname " = " (generate-array-dim gpu-var i) ";" #\Newline)
               collect idxname into idxs
               collect (list :uint32 idxname) into idxinfos
               collect (list :uint32 dimname) into diminfos
               collect (format nil "~A>=~A" idxname dimname) into checks
               finally
                 (progn
                   (code "if ")
                   (emit-separated stream checks " || ")
                   (code " ")
                   (emit-abort-command stream form "Bad index ~A for dims ~A in ~S"
                                       (list `(list ,@idxinfos) `(list ,@diminfos)
                                             `(quote (,(operator-of form) ,(name-of var)
                                                       ,@(mapcar #'unwalk-form indexes)))))
                   (force-emit-merged-assignment stream form 0
                                                 (with-output-to-string (sv)
                                                   (emit-aref-expr sv prefix access-base-fun
                                                                   gpu-var rank idxs)))
                   (when value
                     (code " = " value))
                   (code ";"))))
          ;; Without bound checks
          (progn
            (when value
              (code "("))
            (emit-aref-expr stream prefix access-base-fun
                            gpu-var rank indexes)
            (when value
              (code " = " value ")")))))))

(def definer aref-like-builtin (name &rest flags)
  `(progn
     (def type-arg-walker ,name (arr &rest indexes)
       (walk-aref-type-core -form- arr indexes ,@flags))
     (def type-computer ,name (arr &rest indexes)
       (type-aref-core -form- arr indexes ,@flags))
     (def side-effects ,name (arr &rest indexes)
       (make-side-effects :reads (list (ensure-gpu-var arr))))
     (def is-statement? ,name (arr &rest indexes)
       (generate-bound-checks?))
     (def c-code-emitter ,name (arr &rest indexes)
       (emit-aref-core -form- arr indexes -stream- ,@flags))
     (def type-arg-walker (setf ,name) (arr &rest indexes)
       (walk-aref-type-core -form- arr indexes :value -value- ,@flags))
     (def type-computer (setf ,name) (arr &rest indexes)
       (type-aref-core -form- arr indexes :value -value- ,@flags))
     (def side-effects (setf ,name) (arr &rest indexes)
       (make-side-effects :writes (list (ensure-gpu-var arr))))
     (def is-statement? (setf ,name) (arr &rest indexes)
       (generate-bound-checks?))
     (def c-code-emitter (setf ,name) (arr &rest indexes)
       (emit-aref-core -form- arr indexes -stream- :value -value- ,@flags))))

(def aref-like-builtin aref)

;;; RAW-AREF

(def function walk-raw-aref-type-core (form arr index &key (value nil v-p)
                                            item-type-fun
                                            access-prefix-fun access-range-fun)
  (declare (ignore form item-type-fun access-prefix-fun access-range-fun))
  (with-type-arg-walker-lexicals
    (let ((arr/type (recurse arr)))
      (verify-array-var arr)
      (recurse index :upper-type :uint32)
      (when v-p
        (recurse value :upper-type (second arr/type))))))

(def function type-raw-aref-core (form arr index &key (value nil v-p)
                                       (item-type-fun #'second)
                                       access-range-fun access-prefix-fun)
  (declare (ignore access-prefix-fun access-range-fun))
  (let* ((arr/type (form-c-type-of arr))
         (item-type (funcall item-type-fun arr/type)))
    (verify-cast index :uint32 form :prefix "index "
                 :allow '(:int32) :error-on-warn? t)
    (when v-p
      (verify-cast (form-c-type-of value) item-type form))
    item-type))

(def function emit-raw-aref-core (form var index stream &key value
                                       item-type-fun
                                       (access-range-fun (constantly 1))
                                       (access-prefix-fun (constantly "*")))
  (declare (ignore item-type-fun))
  (with-c-code-emitter-lexicals (stream)
    (let* ((gpu-var (ensure-gpu-var var))
           (refname (generate-var-ref gpu-var))
           (prefix (funcall access-prefix-fun form))
           (access-range (funcall access-range-fun form)))
      (if (generate-bound-checks?)
          ;; With bound checks
          (with-c-code-block (stream)
            (code "unsigned IDX = " index ";" #\Newline
                  "unsigned EXT = " (generate-array-extent gpu-var) ";" #\Newline
                  "if (IDX >= EXT")
            (when (> access-range 1)
              (code "||(IDX+" (1- access-range) ")>=EXT"))
            (code ") ")
            (emit-abort-command stream form "Bad index ~A for extent ~A in ~S"
                                (list '(:uint32 "IDX") '(:uint32 "EXT")
                                      `(quote (,(operator-of form) ,(name-of var) ,(unwalk-form index)))))
            (force-emit-merged-assignment stream form 0
                                          (format nil "(~A(~A+IDX))" prefix refname))
            (when value
              (code " = " value))
            (code ";"))
          ;; Without bound checks
          (progn
            (when value
              (code "("))
            (code "(" prefix "(" refname "+" index "))")
            (when value
              (code " = " value ")")))))))

(def definer raw-aref-like-builtin (name xargs &rest flags)
  `(progn
     (def type-arg-walker ,name (arr indexes ,@xargs)
       (walk-raw-aref-type-core -form- arr indexes ,@flags))
     (def type-computer ,name (arr indexes ,@xargs)
       (type-raw-aref-core -form- arr indexes ,@flags))
     (def side-effects ,name (arr indexes ,@xargs)
       (make-side-effects :reads (list (ensure-gpu-var arr))))
     (def is-statement? ,name (arr indexes ,@xargs)
       (generate-bound-checks?))
     (def c-code-emitter ,name (arr indexes ,@xargs)
       (emit-raw-aref-core -form- arr indexes -stream- ,@flags))
     (def type-arg-walker (setf ,name) (arr indexes ,@xargs)
       (walk-raw-aref-type-core -form- arr indexes :value -value- ,@flags))
     (def type-computer (setf ,name) (arr indexes ,@xargs)
       (type-raw-aref-core -form- arr indexes :value -value- ,@flags))
     (def side-effects (setf ,name) (arr indexes ,@xargs)
       (make-side-effects :writes (list (ensure-gpu-var arr))))
     (def is-statement? (setf ,name) (arr indexes ,@xargs)
       (generate-bound-checks?))
     (def c-code-emitter (setf ,name) (arr indexes ,@xargs)
       (emit-raw-aref-core -form- arr indexes -stream- :value -value- ,@flags))))

(def raw-aref-like-builtin raw-aref ())

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
      (gpu-code-error -form- "Dimension index out of bounds: ~S" cval)))
  :uint32)

(def type-computer array-raw-stride (var dimidx)
  (verify-array-var var)
  (let ((cval (ensure-int-constant dimidx))
        (gvar (ensure-gpu-var var)))
    (unless (and (>= cval 0)
                 (< cval (1- (length (dimension-mask-of gvar)))))
      (gpu-code-error -form- "Stride index out of bounds: ~S" cval)))
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

(def type-arg-walker array-raw-index (arr &rest indexes)
  (walk-aref-type-core -form- arr indexes))

(def type-computer array-raw-index (arr &rest indexes)
  (type-aref-core -form- arr indexes)
  :uint32)

(def is-statement? array-raw-index (arr &rest indexes)
  (generate-bound-checks?))

(def c-code-emitter array-raw-index (arr &rest indexes)
  (emit-aref-core -form- arr indexes -stream-
                  :access-prefix-fun (constantly "")
                  :access-base-fun (constantly "")))

;;; Tuples

(def function ensure-scalar-result (args-or-types form &key prefix)
  (ensure-common-type args-or-types form :prefix prefix
                      :types '(:int8 :uint8 :int16 :uint16 :int32 :uint32
                               :int64 :uint64 :float :double)))

(def type-computer tuple (&rest items)
  (let ((rtype (ensure-scalar-result items -form-)))
    `(:tuple ,(length items) ,rtype)))

(def type-computer untuple (tuple)
  (unless (and (consp tuple/type)
               (eq (first tuple/type) :tuple))
    (gpu-code-error -form- "The argument must be a tuple expression."))
  (let ((size (min (second tuple/type)
                   (length (unwrap-values-type -upper-type-))))
        (base (third tuple/type)))
    (wrap-values-type (loop for i from 0 below size collect base))))

(def is-statement? untuple (tuple)
  (values-c-type? (form-c-type-of -form-)))

;; code generators are target-specific

(def function tuple-aref-rv-type (arr type)
  (let* ((gpu-var (ensure-gpu-var arr))
         (dims (dimension-mask-of gpu-var))
         (size (aref dims (1- (length dims)))))
    (unless (integerp size)
      (gpu-code-error (parent-of arr) "The array argument must have a constant last dimension."))
    `(:tuple ,size ,(second type))))

(def function tuple-aref-cprefix (form)
  (format nil "*(~A*)" (c-type-string (form-c-type-of form))))

(def aref-like-builtin tuple-aref
  :index-bias 1
  :item-type-fun (curry #'tuple-aref-rv-type arr)
  :access-prefix-fun #'tuple-aref-cprefix)

(def function tuple-raw-aref-rv-type (size type)
  (let ((size (ensure-int-constant size)))
    `(:tuple ,size ,(second type))))

(def function tuple-raw-aref-range (form)
  (second (form-c-type-of form)))

(def raw-aref-like-builtin tuple-raw-aref (count)
  :item-type-fun (curry #'tuple-raw-aref-rv-type count)
  :access-prefix-fun #'tuple-aref-cprefix
  :access-range-fun #'tuple-raw-aref-range)

;;; Arithmetics

(def function ensure-arithmetic-result (args-or-types form &key prefix)
  (ensure-common-type args-or-types form :prefix prefix
                      :types '(:int32 :uint32 :int64 :uint64 :float :double)))

(def definer delimited-builtin (name op &key zero single-pfix
                                      (typechecker 'ensure-arithmetic-result))
  `(progn
     (def type-computer ,name (&rest args)
       (if args
           (,typechecker args -form-)
           (propagate-c-types (splice-constant-arg -form- ,zero)
                              :upper-type -upper-type-)))
     (def c-code-emitter ,name (&rest args)
       (emit-separated -stream- args ,op :single-pfix ,single-pfix))))

(def function ensure-div-result-type (args-or-types form)
  (if (cdr args-or-types)
      (ensure-arithmetic-result args-or-types form)
      (ensure-common-type args-or-types form :types '(:float :double))))

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
       (ensure-arithmetic-result -arguments- -form-))
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
  (verify-cast arg :boolean -form-))

(def c-code-emitter not (arg)
  (code "!" arg))

(def type-computer zerop (arg)
  (ensure-arithmetic-result (list arg) -form-)
  :boolean)

(def c-code-emitter zerop (arg)
  (code "(" arg "==0)"))

(def type-computer nonzerop (arg)
  (ensure-arithmetic-result (list arg) -form-)
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
                 (ensure-arithmetic-result -arguments- -form-))
            `(ensure-arithmetic-result -arguments- -form-))
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
  (dolist (arg args)
    (verify-cast arg :boolean -form-))
  :boolean)

(def c-code-emitter and (&rest args)
  (if (null args)
      (emit "1")
      (emit-separated -stream- args "&&")))

(def type-computer or (&rest args)
  (dolist (arg args)
    (verify-cast arg :boolean -form-))
  :boolean)

(def c-code-emitter or (&rest args)
  (if (null args)
      (emit "0")
      (emit-separated -stream- args "||")))

;;; Bitwise logic

(def function ensure-bitwise-result (args-or-types form &key prefix)
  (aprog1 (ensure-arithmetic-result args-or-types form :prefix prefix)
    (when (member it '(:float :double))
      (gpu-code-error form "Floating-point arguments not allowed."))))

(def definer bitwise-builtin (name args &body code)
  `(progn
     (def type-computer ,name ,args
       (ensure-bitwise-result -arguments- -form-))
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

(def function ensure-float-result (args-or-types form &key prefix)
  (ensure-common-type args-or-types form :prefix prefix
                      :types '(:float :double)))

(def function float-name-tag (type)
  (ecase type (:double "") (:float "f")))

(def definer float-builtin (name args &body code)
  `(progn
     (def type-computer ,name ,args
       (ensure-float-result -arguments- -form-))
     (def c-code-emitter ,name ,args
       ,@code)))

;; Miscellaneous

(macrolet ((builtins (&rest names)
             `(progn
                ,@(mapcar (lambda (name)
                            `(def float-builtin ,name (arg)
                               (emit "~A~A(" ,(string-downcase (symbol-name name))
                                     (float-name-tag (form-c-type-of -form-)))
                               (code arg ")")))
                          names))))
  (builtins sin asin sinh asinh
            cos acos cosh acosh
            tan atan tanh atanh
            sqrt))

;; Logarithm

(macrolet ((mklog ((xtype xbase &optional (atype t)) &body code)
             `(def layered-method emit-log-c-code
                (stream (type ,(eql-spec-if #'keywordp xtype))
                        (base ,(eql-spec-if #'numberp xbase))
                        (arg ,atype))
                (with-c-code-emitter-lexicals (stream)
                  ,@code))))
  (def layered-function emit-log-c-code (stream type base arg)
    (:documentation "Emits code for a logarithm function, dispatching on the type and base.")
    (:method (stream type base arg)
      (write-string "(" stream)
      (emit-log-c-code stream type nil arg)
      (write-string "/" stream)
      (emit-log-c-code stream type nil base)
      (write-string ")" stream))
    (:method (stream type (base null) arg)
      (declare (ignore stream))
      (gpu-code-error arg "Invalid return type in emit-log-c-code: ~A" type)))
  (mklog (:float 10)    (code "log10f(" arg ")"))
  (mklog (:float 2)     (code "log2f(" arg ")"))
  (mklog (:float null)  (code "logf(" arg ")"))
  (mklog (:float null real)
         (code (log (float arg 1.0))))
  (mklog (:float real real)
         (code (log (float arg 1.0) base)))
  (mklog (:double 10)   (code "log10(" arg ")"))
  (mklog (:double 2)    (code "log2(" arg ")"))
  (mklog (:double null) (code "log(" arg ")"))
  (mklog (:double null real)
         (code (log (float arg 1.0d0))))
  (mklog (:double real real)
         (code (log (float arg 1.0d0) base))))

(def float-builtin log (arg &optional base)
  (emit-log-c-code -stream- (form-c-type-of -form-) (or (constant-number-value base) base) arg))

;; Exponent

(macrolet ((mkexp ((xtype xbase) &body code)
             `(def layered-method emit-exp-c-code
                (stream (type ,(eql-spec-if #'keywordp xtype))
                        (base ,(eql-spec-if #'numberp xbase))
                        arg)
                (with-c-code-emitter-lexicals (stream)
                  ,@code))))
  (def layered-function emit-exp-c-code (stream type base arg)
    (:documentation "Emits code for an exponentiation, dispatching on the type and base."))
  (mkexp (:float 10)    (code "exp10f(" arg ")"))
  (mkexp (:float 2)     (code "exp2f(" arg ")"))
  (mkexp (:float null)  (code "expf(" arg ")"))
  (mkexp (:float real)  (code "expf(" arg "*" (log (float base 1.0)) ")"))
  (mkexp (:float t)     (code "powf(" base "," arg ")"))
  (mkexp (:double 10)   (code "exp10(" arg ")"))
  (mkexp (:double 2)    (code "exp2(" arg ")"))
  (mkexp (:double null) (code "exp(" arg ")"))
  (mkexp (:double real) (code "exp(" arg "*" (log (float base 1.0d0)) ")"))
  (mkexp (:double t)    (code "pow(" base "," arg ")")))

(def float-builtin exp (power)
  (emit-exp-c-code -stream- (form-c-type-of -form-) nil power))

(def float-builtin expt (base power)
  (emit-exp-c-code -stream- (form-c-type-of -form-) (or (constant-number-value base) base) power))

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
       (int-round-builtin-type -form- -arguments- -upper-type-))
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
    (gpu-code-error -form- "Cannot use MIN without arguments."))
  (ensure-arithmetic-result args -form-))

(def c-code-emitter min (&rest args)
  (emit-minmax-code -stream- -form- "<" "fminf" "fmin" args))

(def is-statement? max (&rest args)
  (case (form-c-type-of -form-)
    ((:float :double) nil)
    (t (rest args))))

(def type-computer max (&rest args)
  (when (null args)
    (gpu-code-error -form- "Cannot use MAX without arguments."))
  (ensure-arithmetic-result args -form-))

(def c-code-emitter max (&rest args)
  (emit-minmax-code -stream- -form- ">" "fmaxf" "fmax" args))
