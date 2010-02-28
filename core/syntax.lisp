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
                 (error "Type not declared for ~A parameter ~S of ~S"
                        title aname func-name))
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
        (error "Allow other keys is not allowed for ~As." title))
      (mapcar (lambda (aform)
                (etypecase aform
                  (required-function-argument-form
                   (extract-arg aform))
                  (optional-function-argument-form
                   (error "Optional arguments are not allowed for ~As." title))
                  (rest-function-argument-form
                   (error "Rest arguments are not allowed for ~As." title))
                  (keyword-function-argument-form
                   (unless kernel?
                     (error "Keyword arguments are not allowed for ~As." title))
                   (when (supplied-p-parameter-name-of aform)
                     (error "Supplied flags are not allowed for ~As." title))
                   (extract-arg aform
                                :keyword (effective-keyword-name-of aform)
                                :default-value (default-value-of aform)))
                  (auxiliary-function-argument-form
                   (unless kernel?
                     (error "Auxillary arguments are not allowed for ~As." title))
                   (unless (default-value-of aform)
                     (error "Auxillary arguments must have a default value: ~S"
                            (unwalk-form aform)))
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

(def function parse-gpu-module-spec (spec &key name environment)
  (with-active-layers (gpu-target)
    (let ((id-table (make-c-name-table))
          (var-list nil)
          (kernel-list nil)
          (base-env (make-walk-environment environment)))
      (flet ((add-variables (type-spec names &key constantp)
               (when (null names)
                 (error "No variable names specified for type ~S" type-spec))
               (multiple-value-bind (item-type dims)
                   (parse-global-type type-spec)
                 (dolist (name names)
                   (unless (symbolp name)
                     (error "Invalid module global name ~S" name))
                   (when (find name var-list :key #'name-of)
                     (error "Module global ~S is already defined" name))
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
            (error "Invalid gpu module spec item: ~S" item))
          (ecase (first item)
            ((:name)
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
            ((:kernel))))
        ;; Collect and parse the kernels
        (let ((kernel-env (reduce (lambda (var env)
                                    (walk-environment/augment
                                     env :variable (name-of var) (form-of var)))
                                  var-list
                                  :from-end t :initial-value base-env)))
          (dolist (item spec)
            (case (first item)
              ((:kernel)
               (push (parse-kernel (rest item) :env kernel-env :id-table id-table
                                   :globals var-list)
                     kernel-list))))))
      (make-instance 'gpu-module :name name
                     :unique-name-tbl id-table
                     :globals (nreverse var-list)
                     :functions nil
                     :kernels (nreverse kernel-list)))))
