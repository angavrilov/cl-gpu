;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.

(in-package :cl-gpu)

(deftype uint-8 () '(unsigned-byte 8))
(deftype uint-16 () '(unsigned-byte 16))
(deftype uint-32 () '(unsigned-byte 32))

(deftype int-8 () '(signed-byte 8))
(deftype int-16 () '(signed-byte 16))
(deftype int-32 () '(signed-byte 32))
