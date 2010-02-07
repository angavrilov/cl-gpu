;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

(def function symbol-to-c-name (name)
  "Converts the symbol name to a suitable C identifier."
  (coerce (loop
             for char across (string-downcase (symbol-name name))
             and prev-char = nil then char
             ;; Can't begin with a number
             when (and (digit-char-p char) (null prev-char))
               collect #\N
             ;; Alphanumeric: verbatim
             if (alphanumericp char) collect char
             ;; Dash to underscore, and don't repeat
             else if (member char '(#\_ #\-))
               when (not (eql prev-char #\_))
                 collect (setf char #\_)
               end
             ;; Others: use the code
             else append (coerce (format nil "U~X" (char-code char)) 'list))
          'string))

(def function unique-c-name (name table)
  "Makes a unique C name using the table for duplicate avoidance."
  (labels ((handle (name)
             (if (gethash name table)
                 (handle (format nil "~AX~A" name
                                 (incf (gethash name table))))
                 (progn
                   (setf (gethash name table) 0)
                   name))))
    (handle (symbol-to-c-name name))))

(def function parse-atomic-type (type-spec)
  (or (lisp-to-foreign-type type-spec)
      (error "Unknown atomic type: ~S" type-spec)))

(def function parse-global-type (type-spec)
  (if (consp type-spec)
      (ecase (first type-spec)
        (array
         (destructuring-bind (&optional item-type dim-spec) (rest type-spec)
           (when (or (null item-type) (eql item-type '*)
                     (null dim-spec) (eql dim-spec '*))
             (error "Insufficiently specific type spec: ~S" type-spec))
           (values (parse-atomic-type item-type)
                   (if (numberp dim-spec)
                       (make-array dim-spec :initial-element nil)
                       (coerce (mapcar (lambda (x) (if (numberp x) x nil)) dim-spec)
                               'vector)))))
        (vector
         (destructuring-bind (&optional item-type dim-spec) (rest type-spec)
           (when (or (null item-type) (eql item-type '*))
             (error "Insufficiently specific type spec: ~S" type-spec))
           (values (parse-atomic-type item-type)
                   (if (numberp dim-spec) (vector dim-spec) (vector nil))))))
      (parse-atomic-type type-spec)))

(def function parse-gpu-module-spec (spec &key name)
  (let ((id-table (make-hash-table :test #'equal))
        (var-list nil))
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
                 (push (make-instance 'gpu-global-var
                                      :name name :c-name (unique-c-name name id-table)
                                      :item-type item-type :dimension-mask dims
                                      :constant-var? constantp)
                       var-list)))))
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
          )))
    (make-instance 'gpu-module :name name
                   :unique-name-tbl id-table
                   :globals (nreverse var-list)
                   :functions nil :kernels nil)))
