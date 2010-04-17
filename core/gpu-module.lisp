;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file defines classes that describe GPU modules,
;;; and implements generic loading functionality.
;;;

(in-package :cl-gpu)

;;; Target selection

(defvar *%cur-gpu-target* nil)

(declaim (inline cur-gpu-target))
(def function cur-gpu-target () *%cur-gpu-target*)

(def (symbol-macro e) *current-gpu-target* (cur-gpu-target)
  "Current GPU target name")

(def generic call-with-target (target thunk))
(def generic target-module-lookup-fun (target))

(def macro with-current-target (&body code)
  `(call-with-target *%cur-gpu-target* (lambda () ,@code)))

(defvar *gpu-module-lookup-fun* nil)

(def function (setf cur-gpu-target) (new-tgt)
  (let ((lfun (target-module-lookup-fun new-tgt)))
    (setf *gpu-module-lookup-fun* lfun
          *%cur-gpu-target* new-tgt)))

(defvar *named-gpu-modules*
  (make-hash-table :test #'eq #+sbcl :synchronized #+sbcl t)
  "Table of named module objects")

(def macro find-gpu-module (name)
  `(gethash ,name *named-gpu-modules*))

;;; Module object classes

(def class* gpu-variable (save-slots-mixin)
  ((name           :documentation "Lisp name of the variable")
   (c-name         :documentation "C name of the variable")
   (item-type      :documentation "Type without array dimensions")
   (dimension-mask :documentation "If array, vector of fixed dims")
   (static-asize   :documentation "Full dimension if all dims constant."))
  (:documentation "Common name of a global variable or parameter."))

(def method gpu-variable-of ((obj gpu-variable)) obj)

(def class* gpu-lexical-var (gpu-variable)
  ()
  (:documentation "A lexical variable in a GPU function."))

(def class* gpu-local-var (gpu-lexical-var)
  ()
  (:documentation "A local variable in a GPU function."))

(def class* gpu-shared-identity (save-slots-mixin)
  ((name           :documentation "Lisp name of the shared variable")))

(def class* gpu-shared-var (gpu-lexical-var)
  ((identity       :documentation "Identity of the variable"))
  (:documentation "A shared local variable in a GPU function."))

(def generic array-var? (obj)
  (:method ((obj gpu-variable)) (dimension-mask-of obj)))

(def generic dynarray-var? (obj)
  (:method ((obj gpu-variable))
    (and (dimension-mask-of obj)
         (not (static-asize-of obj)))))

(def method initialize-instance :after ((obj gpu-variable) &key &allow-other-keys)
  (with-slots (dimension-mask static-asize) obj
    (unless (slot-boundp obj 'static-asize)
      (setf static-asize
            (if (and dimension-mask (every #'numberp dimension-mask))
                (reduce #'* dimension-mask)
                nil)))))

(def class* gpu-global-var (gpu-variable)
  ((index          :documentation "Ordinal index for fast access.")
   (constant-var?  nil :accessor constant-var? :type boolean
                   :documentation "Specifies allocation in constant memory.")
   (form           :documentation "Global var binding form for this var."))
  (:documentation "A global variable in a GPU module."))

(def method make-load-form :before ((object gpu-global-var) &optional env)
  (declare (ignore env))
  (awhen (form-of object)
    (setf (usages-of it) nil)))

(def class* gpu-argument (gpu-variable)
  ((includes-locked? nil :accessor includes-locked? :type boolean)
   (include-size?    nil :accessor include-size? :type boolean)
   (included-dims    nil :documentation "Mask of dimensions to append.")
   (include-extent?  nil :accessor include-extent? :type boolean)
   (included-strides nil :documentation "Mask of strides to append.")
   (keyword          nil :documentation "If set, specifies a keyword parameter.")
   (default-value    nil :documentation "Form to evaluate"))
  (:documentation "A GPU function or kernel parameter."))

(def function required-argument? (arg)
  (and (null (default-value-of arg))
       (null (keyword-of arg))))

(def method initialize-instance :after ((obj gpu-argument) &key &allow-other-keys)
  (with-slots (dimension-mask included-dims included-strides
                              static-asize includes-locked?) obj
    (when dimension-mask
      (unless included-dims
        (setf included-dims
              (make-array (length dimension-mask) :initial-element nil)))
      (unless included-strides
        (setf included-strides
              (make-array (1- (length dimension-mask)) :initial-element nil)))
      (when static-asize
        (setf includes-locked? t)))))

(def class* common-gpu-function (save-slots-mixin)
  ((name           :documentation "Lisp name of the function.")
   (form           :documentation "Walker form tree for the code.")
   (usages         (make-weak-set)
                   :documentation "Functions that use the current one")
   (test-modules   nil
                   :documentation "Modules generated for testing"))
  (:documentation "A gpu function defined outside of a module"))

(def method reinitialize-instance :after ((obj common-gpu-function) &key &allow-other-keys)
  (dolist (item (list* obj (weak-set-snapshot (usages-of obj))))
    (setf (test-modules-of item) nil))
  (setf (usages-of obj) (make-weak-set)))

(def macro symbol-gpu-function (s) `(get ,s 'gpu-function))

(def method make-load-form ((object common-gpu-function) &optional env)
  (declare (ignore env))
  (multiple-value-bind (make init)
      (with-exclude-save-slots (usages test-modules)
        (call-next-method))
    (values make
            `(progn ,init
                    (setf (slot-value ,object 'usages)
                          (make-weak-set)
                          (slot-value ,object 'test-modules)
                          nil)))))

(def function register-common-gpu-function (obj)
  (let* ((name (name-of obj))
         (old-instance (symbol-gpu-function name)))
    (if old-instance
        (reinitialize-instance old-instance :form (form-of obj))
        (setf (symbol-gpu-function name) obj))
    name))

(def class* gpu-function (save-slots-mixin)
  ((name           :documentation "Lisp name of the function")
   (c-name         :documentation "C name of the function")
   (return-type    :documentation "Return type")
   (arguments      :documentation "List of arguments")
   (form           :documentation "Walker form tree for the code.")
   (side-effects   :documentation "Side effects caused by the function.")
   (body           :documentation "Body string")
   (unique-name-tbl (make-c-name-table)
                    :documentation "A hash table used to generate unique C ids.")
   (shared-vars    nil
                   :documentation "List of shared variables."))
  (:documentation "A function usable on the GPU"))

(def class* gpu-kernel (gpu-function)
  ((index          :documentation "Ordinal for fast access")
   (invoker-form   :documentation "Lambda form of the invoker.")
   (invoker-fun    :documentation "Compiled invoker."))
  (:default-initargs :return-type :void)
  (:documentation "A kernel callable from the host"))

(def method make-load-form ((object gpu-kernel) &optional env)
  (declare (ignore env))
  (multiple-value-bind (make init)
      (with-exclude-save-slots (invoker-fun)
        (call-next-method))
    ;; Functions are not externalizable, so expand it as an init form.
    (values make
            (if (invoker-form-of object)
                `(progn ,init
                        (setf (slot-value ,object 'invoker-fun)
                              (function ,(invoker-form-of object))))
                init))))

(def class* gpu-module (save-slots-mixin)
  ((name            :documentation "Lisp name of the module")
   (globals         :documentation "List of global variables")
   (functions       :documentation "List of helper functions")
   (kernels         :documentation "List of kernel functions")
   (index-table     :documentation "An index assignment table")
   (compiled-code   :documentation "Code string")
   (unique-name-tbl (make-c-name-table)
                    :documentation "A hash table used to generate unique C ids.")
   (change-sentinel (cons t nil)
                    :documentation "Used to trigger module reloads")
   (error-table     nil
                    :documentation "Table of errors that may be thrown by the module."))
  (:documentation "A module that can be loaded to the GPU."))

(def constructor gpu-module
  ;; Retrieve the index table from the global instance
  (unless (slot-boundp -self- 'index-table)
    (setf (index-table-of -self-)
          (aif (aand (name-of -self-)
                     (find-gpu-module it))
               (index-table-of it)
               nil))))

(def layered-function compile-object (module)
  (:method ((obj t)) nil)
  (:method :around ((obj gpu-function))
    (let ((*cur-gpu-function* obj))
      (call-next-method)))
  (:method ((obj gpu-module))
    (dolist (krnl (kernels-of obj))
      (compile-object krnl))))

(def layered-function post-compile-object (module)
  (:method ((obj t)) nil)
  (:method :around ((obj gpu-function))
    (let ((*cur-gpu-function* obj))
      (call-next-method)))
  (:method ((obj gpu-module))
    (dolist (krnl (kernels-of obj))
      (post-compile-object krnl))))

;;; Namespace

(def function reindex-gpu-module (module)
  (declare (type gpu-module module))
  (with-slots (index-table) module
    (let ((max-idx (reduce #'max index-table :key #'cdr :initial-value -1)))
      (flet ((process-items (items tag)
               (dolist (item items)
                 (let* ((name (name-of item))
                        (key (cons tag name))
                        (entry (assoc key index-table :test #'equal))
                        (nidx (cdr entry)))
                   (if (slot-boundp item 'index)
                       ;; Already has an index: update the mapping
                       (let ((cur-idx (index-of item)))
                         (maxf max-idx cur-idx)
                         (unless (and nidx (= nidx cur-idx))
                           (awhen (rassoc cur-idx index-table :test #'eql)
                             (warn "Index ~A for ~A is superseded by ~A ~A"
                                   cur-idx (car it) tag name)
                             (removef index-table it))
                           (cond ((null nidx)
                                  (push (cons key cur-idx) index-table))
                                 (t
                                  (warn "Forcing ~A ~A index from ~A to ~A"
                                        tag name nidx cur-idx)
                                  (setf (cdr entry) cur-idx)))))
                       ;; Lookup an index
                       (progn
                         (unless nidx
                           (setf nidx (incf max-idx))
                           (push (cons key nidx) index-table))
                         (setf (index-of item) nidx)))))))
        (process-items (globals-of module) :global)
        (process-items (kernels-of module) :kernel)))))

(def method reinitialize-instance :after ((obj gpu-module) &key &allow-other-keys)
  (setf (car (change-sentinel-of obj)) nil)
  (setf (change-sentinel-of obj) (cons t nil)))

(def function finalize-gpu-module (module)
  (aprog1
      (if (name-of module)
          (let ((old-instance (find-gpu-module (name-of module))))
            (if old-instance
                (reinitialize-instance old-instance
                                       :globals (globals-of module)
                                       :functions (functions-of module)
                                       :kernels (kernels-of module)
                                       :compiled-code (compiled-code-of module)
                                       :unique-name-tbl (unique-name-tbl-of module)
                                       :error-table (error-table-of module))
                (setf (find-gpu-module (name-of module)) module)))
          module)
    (reindex-gpu-module it)
    (unless (eq it module)
      (setf (index-table-of module) (index-table-of it)))))

(def function compile-gpu-module (module)
  (let ((*cur-gpu-module* module))
    (with-current-target
      (compile-object module))
    (reindex-gpu-module module)
    (with-current-target
      (post-compile-object module)))
  module)

;;; Instance management

(defstruct gpu-module-instance
  change-sentinel item-vector)

;; Initial creation
(def layered-function instantiate-module-item (item instance &key old-value)
  (:documentation "Creates an instance of a global or kernel"))

(def function fill-generic-gpu-instance (instance module old-ivals)
  (let ((old-size (if old-ivals (length old-ivals) 0)))
    (setf (gpu-module-instance-change-sentinel instance) (change-sentinel-of module))
    (let* ((items (append (globals-of module) (kernels-of module)))
           (maxid (reduce #'max items :key #'index-of))
           (ivect (make-array (1+ maxid) :initial-element nil)))
      (setf (gpu-module-instance-item-vector instance) ivect)
      (dolist (item items)
        (let* ((idx (index-of item))
               (old-value (if (< idx old-size)
                              (aref old-ivals idx))))
          (setf (aref ivect idx)
                (instantiate-module-item item instance :old-value old-value)))))))

(def layered-function load-gpu-module-instance (module)
  (:documentation "Instantiates a gpu module"))

(def layered-method load-gpu-module-instance ((module symbol))
  (load-gpu-module-instance (find-gpu-module module)))

(def layered-method load-gpu-module-instance :around ((module gpu-module))
  (aprog1 (call-next-method)
    (fill-generic-gpu-instance it module nil)))

;; Reinitialization
(def generic freeze-module-item (item)
  (:documentation "Packages the state of the item")
  (:method ((item t)) nil))

(def generic kill-frozen-object (item object)
  (:documentation "Destroys an object produced by freeze-module-item")
  (:method ((item t) (object array)) nil)
  (:method ((item t) (object number)) nil))

(def layered-function upgrade-gpu-module-instance (module instance)
  (:documentation "Rebuilds the instance after a module update"))

(def layered-method upgrade-gpu-module-instance ((module symbol) instance)
  (upgrade-gpu-module-instance (find-gpu-module module) instance))

(def layered-method upgrade-gpu-module-instance :around ((module gpu-module) instance)
  (let* ((old-ivect (gpu-module-instance-item-vector instance))
         (old-ivals (map 'vector #'freeze-module-item old-ivect))
         (old-size (length old-ivect)))
    (unwind-protect
         (progn
           (call-next-method)
           (fill-generic-gpu-instance instance module old-ivals))
      (dotimes (i old-size)
        (awhen (aref old-ivals i)
          (kill-frozen-object (aref old-ivect i) it))))))

;;;

(declaim (inline get-module-instance get-module-instance-items)
         (ftype (function (t) simple-vector) get-module-instance-item))

(def function gpu-module-key (module)
  (or (name-of module) module))

(def function get-module-instance (module-id)
  (funcall *gpu-module-lookup-fun* module-id))

(def function get-module-instance-items (module-id)
  (gpu-module-instance-item-vector
   (funcall *gpu-module-lookup-fun* module-id)))

(def generic gpu-global-value (obj)
  (:documentation "Retrieve the value of a GPU global"))

(def generic (setf gpu-global-value) (value obj)
  (:documentation "Set the value of a GPU global"))

