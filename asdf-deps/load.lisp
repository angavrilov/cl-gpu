;;; Load this file to use these libraries.

#-(or) (require 'asdf)
#+(or) (load (merge-pathnames #P"asdf.lisp" *load-truename*))

;; Not having a useful stack trace is very annoying.
#+ecl (declaim (optimize (debug 3)))

(push (make-pathname :defaults *load-truename* :name nil :type nil)
      asdf:*central-registry*)

(import 'asdf:load-system)
