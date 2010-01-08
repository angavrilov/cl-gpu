;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu.test)

(defsuite* (test/utils :in test))

(def test test/utils/r-w-array ()
  (let ((arr (make-array 5 :element-type 'single-float
                         :initial-contents '(0.0 0.1 0.2 0.3 0.4))))
    (with-open-file (s "cl-gpu.test.tmp"
                       :direction :output
                       :if-exists :supersede
                       :element-type '(unsigned-byte 8))
      (write-array arr s))
    (let ((res (with-open-file (s "cl-gpu.test.tmp"
                                  :direction :input
                                  :element-type '(unsigned-byte 8))
                 (read-array nil s :allocate t)))
          (res2 (with-open-file (s "cl-gpu.test.tmp"
                                   :direction :input
                                   :element-type '(unsigned-byte 8))
                  (read-array (make-array 5 :element-type 'single-float) s))))
      (delete-file "cl-gpu.test.tmp")
      (is (equal (array-dimensions arr)
                 (array-dimensions res)))
      (is (every #'eql arr res))
      (is (every #'eql arr res2)))))
