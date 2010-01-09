;;; Load this file to use these libraries.

#+sbcl (require 'asdf)
#-sbcl (load (merge-pathnames #P"asdf.lisp" *load-truename*))

(push (make-pathname :defaults *load-truename* :name nil :type nil)
      asdf:*central-registry*)

(import 'asdf:load-system)
