;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file defines miscellaneous utility functions;
;;; some of them depend on the lisp implementation.
;;;

(in-package :cl-gpu)

;;; Misc

(declaim (inline ensure-cdr))

(def function ensure-cdr (item)
  (if (consp item) (cdr item)))

(def macro gethash-with-init (key table init-expr)
  "Looks up the key in the table. When not found, lazily initializes with init-expr."
  (with-unique-names (item found)
    (once-only (key table)
      `(multiple-value-bind (,item ,found) (gethash ,key ,table)
         (if ,found ,item
             (setf (gethash ,key ,table) ,init-expr))))))

(def macro with-memoize ((key &rest flags) &body code)
  "Memoizes the result of the code block using key; flags are passed to make-hash-table."
  `(gethash-with-init ,key
                      (load-time-value (make-hash-table ,@flags))
                      (progn ,@code)))

(def macro with-capturing-names ((namelist &key prefix) &body code)
  "Defines symbols in the list as symbols in the current package with the same names."
  `(let ,(mapcar (lambda (name)
                   `(,name (symbolicate ,@(ensure-list prefix) ',name)))
                 namelist)
     ,@code))

(def function layered-method-qualifiers (options)
  (flatten (list
            (awhen (or (getf options :in-layer)
                       (getf options :in))
              (list :in it))
            (getf options :mode))))

(def function remove-form-by-name (forms name &key (type 't))
  (check-type name symbol)
  (remove-if (lambda (item)
               (and item
                    (or (eql type t)
                        (typep item type))
                    (eq (name-of item) name)))
             forms))

(def function extract-power-of-two (value)
  "Returns the integer part of base-2 logarithm and the remainder."
  (loop for i from 0
     when (or (logtest value 1) (<= value 0))
     return (values i value)
     else do (setf value (ash value -1))))

(define-modify-macro remove-form-by-name! (name &rest flags) remove-form-by-name)

(def function make-type-arg (sym)
  (format-symbol t "~A/TYPE" sym))

(def function eql-spec-if (predicate value)
  (if (funcall predicate value) `(eql ,value) value))

;;; Builtin function handler definitions

(def macro make-builtin-handler-method (builtin-name
                                        builtin-args
                                        method-name-selector
                                        body-forms
                                        &key method-args top-decls let-decls prefix)
  "Non-hygienic helper for stuff like c-code-emitter.
Defines variables: assn? rq-args opt-args rest-arg aux-args."
  `(let ((assn? nil))
     (when (consp ,builtin-name)
       (assert (eql (first ,builtin-name) 'setf))
       (setf assn? t ,builtin-name (second ,builtin-name)))
     (multiple-value-bind (rq-args opt-args rest-arg kwd other aux-args)
         (parse-ordinary-lambda-list ,builtin-args)
       (declare (ignorable rq-args opt-args rest-arg aux-args))
       (when (or kwd other)
         (error "Keyword matching not supported"))
       `(def layered-method ,,method-name-selector
          ,@(layered-method-qualifiers -options-)
          ((-name- (eql ',,builtin-name)) -form- ,@,method-args)
          ,@,(if top-decls ``((declare ,@,top-decls)))
          (let* ((-arguments- (arguments-of -form-))
                 ,@(if assn?
                       `((-value- (value-of -form-)))))
            ,@(aif ,let-decls `((declare ,@it)))
            (,@,(or prefix ''(progn))
              (destructuring-bind ,,builtin-args
                  -arguments-
                ,@,body-forms)))))))

;;; Form parent adjustment

(def function adjust-parents-to (form parent)
  (if (listp form)
      (dolist (item form)
        (adjust-parents-to item parent))
      (setf (parent-of form) parent)))

(def macro adjust-parents ((accessor form))
  `(adjust-parents-to (,accessor ,form) ,form))

;;; Deferred action queues

(def macro with-deferred-actions ((name) &body code)
  "Defers actions via special variable name until exit from the outermost defer block."
  (with-unique-names (old-queue)
    `(let* ((,old-queue ,name)
            (,name (or ,old-queue (cons 'deferred nil))))
       ,@code
       (unless ,old-queue
         (dolist (item (cdr ,name))
           (funcall item))))))

(def macro defer-action ((name) &body code)
  "When used in context of with-deferred-actions, postpones the code."
  `(flet ((action () ,@code))
     (if ,name
         (push #'action (cdr ,name))
         (action))))

;;; A helper for externalizable objects

(def function object-slots-excluding (object &rest names)
  (reduce (lambda (slots name)
            (delete name slots))
          names
          :initial-value (mapcar #'closer-mop:slot-definition-name
                                 (closer-mop:class-slots (class-of object)))))

(def class save-slots-mixin ()
  ())

(defparameter *excluded-save-slots* nil)

(def method make-load-form ((object save-slots-mixin) &optional env)
  (if *excluded-save-slots*
      (make-load-form-saving-slots
       object :environment env
       :slot-names (apply #'object-slots-excluding object *excluded-save-slots*))
      (make-load-form-saving-slots object :environment env)))

(def macro with-exclude-save-slots (slots &body code)
  `(let ((*excluded-save-slots* (append ',slots *excluded-save-slots*)))
     ,@code))

(def definer custom-slot-load-forms (class &rest clauses)
  (let ((clist
         (mapcar (lambda (clause)
                   `(when (slot-boundp -self- ',(first clause))
                      `(setf (slot-value ,-self- ',',(first clause))
                             ,(progn ,@(rest clause)))))
                 clauses)))
    `(def method make-load-form ((-self- ,class) &optional -env-)
       (declare (ignorable -env-))
       (bind (((:values make init)
               (with-exclude-save-slots (,@(mapcar #'first clauses))
                 (call-next-method))))
         (values make
                 `(progn ,init ,,@clist))))))

;;; Temporary files

(def function open-temp-file (base-name &key
                                        (direction :output)
                                        (element-type 'character)
                                        (external-format :default))
  "Opens a new temporary file, using base-name as template."
  (loop
     with name = (pathname base-name)
     for id from (get-universal-time)
     do
       (let* ((fname (format nil "~A~A" (pathname-name name) id))
              (path (make-pathname :name fname :defaults name))
              (file (open path :direction direction :if-exists nil
                          :element-type element-type
                          :external-format external-format)))
         (when file
           (return (values file path))))))

(def function delete-if-exists (path)
  (when (probe-file path)
    (delete-file path)))

(def macro with-temp-file ((path stream path-template &rest open-flags) write-code &body code)
  "Takes care of creating and removing a new temporary file. Stream is accessible inside write-code."
  (with-unique-names (temp-path wrtp ssave)
    `(let (,path)
       (unwind-protect
            (progn
              (let (,ssave ,wrtp)
                (unwind-protect
                     (multiple-value-bind (,stream ,temp-path)
                         (open-temp-file ,path-template ,@open-flags)
                       (setf ,path ,temp-path ,ssave ,stream)
                       ,write-code
                       (setf ,wrtp t))
                  (when ,ssave
                    (close ,ssave :abort (not ,wrtp)))))
              ,@code)
         (when ,path
           (delete-if-exists ,path))))))

(defun system-command (cmd)
  #+ecl (ext:system cmd)
  #+ccl (ccl::os-command cmd)
  #-(or ecl ccl) (error "Not implemented"))

;;; Weak sets

(def function ensure-weak-pointer-value (ptr)
  (if (weak-pointer-p ptr) (weak-pointer-value ptr) ptr))

(def function make-weak-set (&optional items)
  #+openmcl
  (ccl:make-population :type :list :initial-contents items)
  #-openmcl
  (list* 'weak-set (mapcar #'make-weak-pointer items)))

(def macro do-weak-ptr-list ((ptr-item wset &optional end) &body code)
  "Walk a list of weak pointers, automatically pruning the dead ones."
  (with-unique-names (pos next)
    `(do* ((,pos ,wset ,next)
           (,next (cdr ,pos) (cdr ,pos)))
          ((null ,next) ,end)
       (declare (type list ,pos ,next))
       (macrolet ((delete-this-item ()
                    `(setf (cdr ,',pos) (cdr ,',next)
                           ,',next ,',pos)))
         (let ((,ptr-item (weak-pointer-value (car ,next))))
           (if ,ptr-item
               (progn ,@code)
               (delete-this-item)))))))

(def function weak-set-addf (wset new-item)
  #+openmcl
  (pushnew new-item (ccl:population-contents wset) :test #'eq)
  #-openmcl
  (progn
    (do-weak-ptr-list (item wset)
      (when (eq item new-item)
        (return-from weak-set-addf)))
    (push (make-weak-pointer new-item) (cdr wset)))
  (values))

(def function weak-set-deletef (wset rm-item)
  #+openmcl
  (deletef (ccl:population-contents wset) rm-item :test #'eq)
  #-openmcl
  (do-weak-ptr-list (item wset)
    (when (eq item rm-item)
      (delete-this-item)))
  (values))

(def function weak-set-snapshot (wset)
  #+openmcl
  (copy-list (ccl:population-contents wset))
  #-openmcl
  (let (rv)
    (do-weak-ptr-list (item wset rv)
      (push item rv))))

;;; Unique identifiers

(defvar *unique-id-template* (make-hash-table :test #'equal))

(def definer reserved-c-names (&rest names)
  `(dolist (id ',names)
     (setf (gethash id *unique-id-template*) 0)))

(def reserved-c-names
    "break" "case" "char" "const" "continue" "default" "do"
    "double" "else" "enum" "extern" "float" "for" "goto" "if"
    "inline" "int" "long" "register" "return" "short" "signed"
    "sizeof" "static" "struct" "switch" "typedef" "union"
    "unsigned" "void" "volatile" "while"
    "restrict" "_Bool" "_Complex" "_Imaginary"
    "uint_least16_t" "uint_least32_t"
    "asm" "bool" "catch" "class" "const_cast" "delete" "dynamic_cast"
    "explicit" "export" "false" "friend" "mutable" "namespace"
    "new" "operator" "private" "protected" "public" "reinterpret_cast"
    "static_cast" "template" "this" "throw" "true" "try" "typeid"
    "typename" "using" "virtual" "wchar_t" "typeof"
    "nil" "t" "_t")

(def function make-c-name-table ()
  (copy-hash-table *unique-id-template* :test #'equal))

(def function symbol-to-c-name (name)
  "Converts the symbol name to a suitable C identifier."
  (coerce (loop
             with sym-name = (symbol-name name)
             with name-string = (if (symbol-package name)
                                    sym-name
                                    (string-right-trim "0123456789" sym-name))
             for char across (string-downcase name-string)
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
