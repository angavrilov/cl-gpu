;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file implements reconstruction of GPU module
;;; definition objects from a compact specification.
;;;

(in-package :cl-gpu)

(def function extract-arglist-vars (lambda-form &key func-name id-table kernel?)
  (let ((top-decls (declarations-of lambda-form))
        (title (if kernel? "kernel" "gpu function")))
    (flet ((extract-arg (aform &key keyword default-value)
             (let* ((aname (name-of aform))
                    (adecl (find-form-by-name aname top-decls
                                              :type 'type-declaration-form)))
               (unless adecl
                 (gpu-code-error aform "Type not declared for ~A parameter." title))
               (multiple-value-bind (item-type dims)
                   (parse-global-type (declared-type-of adecl))
                 (aprog1 (make-instance 'gpu-argument
                                        :name aname
                                        :c-name (unique-c-name aname id-table)
                                        :item-type item-type :dimension-mask dims
                                        :keyword keyword
                                        :default-value (if default-value
                                                           (unwalk-form default-value)))
                         (setf (gpu-variable-of aform) it))))))
      (when (allow-other-keys? lambda-form)
        (gpu-code-error lambda-form "Allow other keys is not allowed for ~As." title))
      (mapcar (lambda (aform)
                (etypecase aform
                  (required-function-argument-form
                   (extract-arg aform))
                  (optional-function-argument-form
                   (gpu-code-error aform "Optional arguments are not allowed for ~As." title))
                  (rest-function-argument-form
                   (gpu-code-error aform "Rest arguments are not allowed for ~As." title))
                  (keyword-function-argument-form
                   (unless kernel?
                     (gpu-code-error aform "Keyword arguments are not allowed for ~As." title))
                   (when (supplied-p-parameter-name-of aform)
                     (gpu-code-error aform "Supplied flags are not allowed for ~As." title))
                   (extract-arg aform
                                :keyword (effective-keyword-name-of aform)
                                :default-value (default-value-of aform)))
                  (auxiliary-function-argument-form
                   (unless kernel?
                     (gpu-code-error aform "Auxillary arguments are not allowed for ~As." title))
                   (unless (default-value-of aform)
                     (gpu-code-error aform "Auxillary arguments must have a default value."))
                   (extract-arg aform :default-value (default-value-of aform)))))
              (bindings-of lambda-form)))))

(def function parse-kernel (code &key env id-table globals)
  (let* ((form  (preprocess-tree
                 (walk-form `(defun ,@code) :environment env)
                 globals))
         (name (name-of form))
         (c-name (unique-c-name name id-table))
         (fid-table (copy-hash-table id-table :test #'equal))
         (args (extract-arglist-vars form :func-name name
                                     :id-table fid-table :kernel? t)))
    (make-instance 'gpu-kernel :name name :c-name c-name
                   :form form :arguments args
                   :unique-name-tbl fid-table)))

(def function parse-function (code &key env)
  (let* ((form (preprocess-function
                (walk-form `(defun ,@code) :environment env))))
    (make-instance 'common-gpu-function :name (name-of form) :form form)))

(def function parse-gpu-module-spec (spec &key name environment)
  (with-active-layers (gpu-target)
    (let ((id-table (make-c-name-table))
          (var-list nil)
          (kernel-list nil)
          (base-env (make-walk-environment environment)))
      (flet ((add-variables (type-spec names &key constantp)
               (when (null names)
                 (gpu-code-error nil "No variable names specified for type ~S" type-spec))
               (multiple-value-bind (item-type dims)
                   (parse-global-type type-spec)
                 (dolist (name names)
                   (unless (symbolp name)
                     (gpu-code-error nil "Invalid module global name ~S" name))
                   (when (find name var-list :key #'name-of)
                     (gpu-code-error nil "Module global ~S is already defined" name))
                   (let* ((var (make-instance 'gpu-global-var
                                              :name name :c-name (unique-c-name name id-table)
                                              :item-type item-type :dimension-mask dims
                                              :constant-var? constantp))
                          (form (make-instance 'global-var-binding-form :name name
                                               :gpu-variable var)))
                     (setf (form-of var) form)
                     (push var var-list))))))
        ;; Collect the global variables
        (dolist (item spec)
          (unless (consp item)
            (gpu-code-error nil "Invalid gpu module spec item: ~S" item))
          (ecase (first item)
            ((:name)
             (when name
               (gpu-code-error nil "Name is already specified for gpu module ~S" name))
             (setf name (second item)))
            ((:variable :global)
             (destructuring-bind (name type) (rest item)
               (add-variables type (list name) :constantp nil)))
            ((:variables :globals)
             (add-variables (second item) (cddr item) :constantp nil))
            ((:constant)
             (destructuring-bind (name type) (rest item)
               (add-variables type (list name) :constantp t)))
            ((:constants)
             (add-variables (second item) (cddr item) :constantp t))
            ;; Skip on this pass
            ((:kernel :function))))
        ;; Collect and parse the kernels
        (let ((kernel-env (reduce (lambda (var env)
                                    (walk-environment/augment
                                     env :variable (name-of var) (form-of var)))
                                  var-list
                                  :from-end t :initial-value base-env)))
          (dolist (item spec)
            (case (first item)
              ((:function)
               (let ((func (parse-function (rest item) :env kernel-env)))
                 (walk-environment/augment! kernel-env :function (name-of func) (form-of func))))
              ((:kernel)
               (push (parse-kernel (rest item) :env kernel-env :id-table id-table
                                   :globals var-list)
                     kernel-list))))))
      (make-instance 'gpu-module :name name
                     :unique-name-tbl id-table
                     :globals (nreverse var-list)
                     :functions nil
                     :kernels (nreverse kernel-list)))))

(def (definer e :available-flags "e") gpu-function (name args &body code)
  (let ((func (parse-function (list* name args code)
                              :env (make-walk-environment -environment-))))
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       ,@(when (getf -options- :export)
               `((export ',name)))
       (register-common-gpu-function ,func)
       ,@(if (not (getf -options- :gpu-only))
             `((defun ,name ,args ,@code)))
       ',name)))

(def (definer e :available-flags "eas") gpu-module (name &body spec)
  "Defines a new global GPU code module."
  (bind ((prefix (concatenate 'string (string name) "-"))
         (keys-sym (gensym "KEYS"))
         (fspec (loop for item in spec
                   when (eq (first item) :conc-name)
                   do (setf prefix (second item))
                   else collect item))
         (module (parse-gpu-module-spec fspec :name name
                                        :environment -environment-)))
    (compile-gpu-module module)
    (flet ((make-item-getter (obj)
             `(svref (gpu-module-instance-item-vector
                      (get-module-instance ',(name-of module)))
                     ,(index-of obj))))
      (bind (((:values field-defs field-names)
              (loop for var in (globals-of module)
                 for name = (symbolicate prefix (name-of var))
                 collect name into names
                 collect `(define-symbol-macro ,name
                              (gpu-global-value ,(make-item-getter var)))
                 into defs
                 finally (return (values defs names))))
             ((:values kernel-defs kernel-names)
              (loop for knl in (kernels-of module)
                 for name = (symbolicate prefix (name-of knl))
                 for args = (arguments-of knl)
                 for rqargs = (mapcar #'name-of (remove-if-not #'required-argument? args))
                 collect name into names
                 collect `(defun ,name (,@rqargs &rest ,keys-sym)
                            (apply ,(make-item-getter knl) ,@rqargs ,keys-sym))
                 into defs
                 finally (return (values defs names)))))
        `(progn
           (finalize-gpu-module ,module)
           (define-symbol-macro ,name ,module)
           ,@(when (getf -options- :export)
                   `((export '(,(name-of module)))))
           ,@(when (getf -options- :export-accessor-names)
                   `((export ',(append field-names kernel-names))))
           ,@(when (getf -options- :export-slot-names)
                   `((export ',(mapcar #'name-of (append (globals-of module)
                                                         (kernels-of module))))))
           (declaim (inline ,@kernel-names))
           ,@field-defs
           ,@kernel-defs
           ',name)))))

(def function wrap-gpu-module-bindings (module instance code)
  (with-unique-names (gpu-instance gpu-items)
    (bind ((field-defs
            (loop for var in (globals-of module)
               collect `(,(name-of var)
                          (gpu-global-value (svref ,gpu-items ,(index-of var))))))
           (kernel-defs
            (loop for knl in (kernels-of module)
               collect `(,(name-of knl) (&rest args)
                          (apply (svref ,gpu-items ,(index-of knl)) args)))))
      `(let* ((,gpu-instance (get-module-instance ,instance))
              (,gpu-items (gpu-module-instance-item-vector ,gpu-instance)))
         (symbol-macrolet ,field-defs
           (flet ,kernel-defs
             (declare (inline ,@(mapcar #'name-of (kernels-of module))))
             ,@code))))))

(def (macro e) with-gpu-module (name-or-spec &body code &environment env)
  "Makes variables and kernels of the module accessible in the lexical scope."
  (bind (((:values module own-module?)
          (atypecase (if (symbolp name-or-spec)
                         ;; This is done to avoid the necessity to
                         ;; finalize global modules at compilation.
                         ;; The module object is passed through a
                         ;; global symbol macro instead.
                         (macroexpand name-or-spec env)
                         name-or-spec)
            ;; Module object
            (gpu-module it)
            ;; Just in case, look up symbols dynamically
            (symbol (or (find-gpu-module it)
                        (gpu-code-error nil "Unknown GPU module: ~S" it)))
            ;; An ad-hoc module specification: parse it
            (cons
             (values (parse-gpu-module-spec it :environment env) t))
            ;; Junk
            (t
             (gpu-code-error nil "Invalid GPU module: ~S" name-or-spec)))))
    (when own-module?
      (compile-gpu-module module))
    (wrap-gpu-module-bindings
     module
     (if own-module?
         `(load-time-value (gpu-module-key (finalize-gpu-module ,module)))
         `(quote ,(name-of module)))
     code)))
