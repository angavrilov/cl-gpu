;;; -*- mode: Lisp; Syntax: Common-Lisp; -*-
;;;
;;; Copyright (c) 2010 by Alexander Gavrilov.
;;;
;;; See LICENCE for details.
;;;
;;; This file implements pitched memory copy operations for CUDA.
;;;

(in-package :cl-gpu)

(def macro with-pitch-mapping-vars ((blk offset size &key (prefix "")) &body code)
  "Non-hygienic macro that defines variables with various pitch parameters."
  (with-capturing-names ((width pitch start-y start-x end-y end-x one-row-p start-offset)
                         :prefix prefix)
    `(bind ((,width (cuda-linear-width ,blk))
            (,pitch (cuda-linear-pitch ,blk))
            ((:values ,start-y ,start-x) (floor ,offset ,width))
            ((:values ,end-y ,end-x) (floor (+ ,offset ,size) ,width))
            (,one-row-p (or (= ,start-y ,end-y)
                            (and (= ,end-y (1+ ,start-y)) (= ,end-x 0))))
            (,start-offset (+ (* ,start-y ,pitch) ,start-x)))
       (declare (fixnum #-ecl ,width ,pitch ,start-y ,start-x ,end-y ,end-x ,start-offset))
       ,@code)))

(def function %cuda-linear-wrap-pitch (blk offset size transfer-chunk transfer-rows)
  "Computes ranges for exchange of data between the host and a possibly pitched block."
  (declare (type cuda-linear blk)
           (fixnum offset size)
           ;; dev-offset host-offset size
           (type (function (fixnum fixnum fixnum) t) transfer-chunk)
           ;; host-offset width pitch start-row row-count
           (type (function (fixnum fixnum fixnum fixnum fixnum) t) transfer-rows))
  (with-cuda-context ((cuda-linear-context blk))
    (if (not (cuda-linear-pitched-p blk))
        ;; Contiguous
        (funcall transfer-chunk offset 0 size)
        ;; Pitched, i.e. has gaps
        (with-pitch-mapping-vars (blk offset size)
          ;; Fits in one row?
          (if one-row-p
              ;; Yes!
              (funcall transfer-chunk start-offset 0 size)
              ;; Spans multiple rows:
              (let* ((middle-size (- end-y start-y 1))
                     (middle-p (> middle-size 0))
                     (host-shift (- width start-x)))
                (declare (fixnum host-shift middle-size))
                ;; First row
                (if (and middle-p (= start-x 0)) ; Promote a full row
                    (setf start-y (1- start-y)
                          middle-size (1+ middle-size)
                          host-shift 0)
                    (funcall transfer-chunk start-offset 0 host-shift))
                ;; Middle area
                (when middle-p
                  (funcall transfer-rows host-shift width pitch (1+ start-y) middle-size))
                ;; Final row
                (when (> end-x 0)
                  (funcall transfer-chunk (* pitch end-y) (+ (* width middle-size) host-shift) end-x)))))))
  ;; Return NIL
  nil)

(def function %cuda-linear-dh-transfer (blk ptr offset size to-host-p)
  "Moves data between a contiguous host memory area and possibly pitched linear block."
  (declare (type cuda-linear blk)
           (fixnum offset size))
  (let ((handle (cuda-linear-ensure-handle blk)))
    (declare (type integer handle))
    (flet ((transfer-chunk (dev-ofs host-ofs size)
             (declare (fixnum dev-ofs host-ofs size))
             (when (> size 0)
               (let ((dev-ptr (+ handle dev-ofs))
                     (host-ptr (inc-pointer ptr host-ofs)))
                 (if to-host-p
                     (cuda-invoke cuMemcpyDtoH host-ptr dev-ptr size)
                     (cuda-invoke cuMemcpyHtoD dev-ptr host-ptr size)))))
           (transfer-rows (host-shift width pitch start-row row-count)
             (declare (fixnum host-shift width pitch start-row row-count))
             (with-foreign-object (pmovespec 'cuda-memcpy-2d)
               (with-foreign-slots ((src-x-bytes src-y src-type src-host src-device src-pitch
                                                 dst-x-bytes dst-y dst-type dst-host dst-device dst-pitch
                                                 width-bytes height)
                                    pmovespec cuda-memcpy-2d)
                 (setf width-bytes width
                       height row-count
                       src-x-bytes 0
                       dst-x-bytes 0)
                 (if to-host-p
                     (setf src-y start-row    src-pitch pitch
                           src-type :device   src-device handle
                           dst-y 0            dst-pitch width
                           dst-type :host     dst-host (inc-pointer ptr host-shift))
                     (setf src-y 0            src-pitch width
                           src-type :host     src-host (inc-pointer ptr host-shift)
                           dst-y start-row    dst-pitch pitch
                           dst-type :device   dst-device handle)))
               ;; Do the 2D transfer
               (cuda-invoke cuMemcpy2D pmovespec))))
      (declare (dynamic-extent #'transfer-chunk #'transfer-rows))
      (%cuda-linear-wrap-pitch blk offset size #'transfer-chunk #'transfer-rows))))

(def function %cuda-linear-memset (blk index count type value &key (offset 0))
  "Fills the linear block region with the same value."
  (let (elt-size ivalue)
    (setf (native-type-ref type
                           (lambda (ptmp offset size dir)
                             (assert (and (null dir) (= offset 0)))
                             (setf elt-size size
                                   ivalue (ecase size
                                            (1 (mem-ref ptmp :uint8))
                                            (2 (mem-ref ptmp :uint16))
                                            (4 (mem-ref ptmp :uint32)))))
                           0) value)
    (let ((handle (cuda-linear-ensure-handle blk)))
      (declare (type integer handle))
      (flet ((fill-chunk (dev-ofs host-ofs size)
               (declare (fixnum dev-ofs host-ofs size)
                        (ignore host-ofs))
               (when (> size 0)
                 (let ((dev-ptr (+ handle dev-ofs)))
                   (ecase elt-size
                     (1 (cuda-invoke cuMemsetD8 dev-ptr ivalue size))
                     (2 (cuda-invoke cuMemsetD16 dev-ptr ivalue (ash size -1)))
                     (4 (cuda-invoke cuMemsetD32 dev-ptr ivalue (ash size -2)))))))
             (fill-rows (host-shift width pitch start-row row-count)
               (declare (fixnum host-shift width pitch start-row row-count)
                        (ignore host-shift))
               (let ((dev-ptr (+ handle (* pitch start-row))))
                 (ecase elt-size
                   (1 (cuda-invoke cuMemsetD2D8 dev-ptr pitch ivalue width row-count))
                   (2 (cuda-invoke cuMemsetD2D16 dev-ptr pitch ivalue (ash width -1) row-count))
                   (4 (cuda-invoke cuMemsetD2D32 dev-ptr pitch ivalue (ash width -2) row-count))))))
        (declare (dynamic-extent #'fill-chunk #'fill-rows))
        (%cuda-linear-wrap-pitch blk (+ (* index elt-size) offset) (* count elt-size) #'fill-chunk #'fill-rows)))))

(def function %cuda-linear-dd-transfer-unpitched (p-blk p-offset up-blk up-offset size to-unpitched-p)
  "Moves data between a contiguous device memory area and possibly pitched linear block."
  (declare (type cuda-linear p-blk up-blk)
           (fixnum p-offset up-offset size))
  (let ((p-handle (cuda-linear-ensure-handle p-blk))
        (up-handle (+ (cuda-linear-ensure-handle up-blk) up-offset)))
    (declare (type integer p-handle up-handle))
    (flet ((transfer-chunk (p-ofs up-ofs size)
             (declare (fixnum p-ofs up-ofs size))
             (when (> size 0)
               (let ((p-ptr (+ p-handle p-ofs))
                     (up-ptr (+ up-handle up-ofs)))
                 (if to-unpitched-p
                     (cuda-invoke cuMemcpyDtoD up-ptr p-ptr size)
                     (cuda-invoke cuMemcpyDtoD p-ptr up-ptr size)))))
           (transfer-rows (up-shift width pitch start-row row-count)
             (declare (fixnum up-shift width pitch start-row row-count))
             (with-foreign-object (pmovespec 'cuda-memcpy-2d)
               (with-foreign-slots ((src-x-bytes src-y src-type src-device src-pitch
                                                 dst-x-bytes dst-y dst-type dst-device dst-pitch
                                                 width-bytes height)
                                    pmovespec cuda-memcpy-2d)
                 (setf width-bytes width
                       height row-count
                       src-x-bytes 0
                       dst-x-bytes 0
                       src-type :device
                       dst-type :device)
                 (if to-unpitched-p
                     (setf src-y start-row    src-pitch pitch
                           src-device p-handle
                           dst-y 0            dst-pitch width
                           dst-device (+ up-handle up-shift))
                     (setf src-y 0            src-pitch width
                           src-device (+ up-handle up-shift)
                           dst-y start-row    dst-pitch pitch
                           dst-device p-handle)))
               ;; Do the 2D transfer
               (cuda-invoke cuMemcpy2DUnaligned pmovespec))))
      (declare (dynamic-extent #'transfer-chunk #'transfer-rows))
      (with-cuda-context ((cuda-linear-context up-blk))
        (%cuda-linear-wrap-pitch p-blk p-offset size #'transfer-chunk #'transfer-rows)))))

(def function %cuda-linear-dd-transfer (src-blk src-offset dst-blk dst-offset size &key return-if-mismatch)
  "Moves data between two linear blocks. If both are pitched, the area must be aligned."
  ;; Simple case: one unpitched
  (if (not (cuda-linear-pitched-p src-blk))
      (%cuda-linear-dd-transfer-unpitched dst-blk dst-offset src-blk src-offset size nil)
      (if (not (cuda-linear-pitched-p dst-blk))
          (%cuda-linear-dd-transfer-unpitched src-blk src-offset dst-blk dst-offset size t)
          ;; Both pitched. One source row?
          (with-pitch-mapping-vars (src-blk src-offset size :prefix s-)
            (if s-one-row-p
                (%cuda-linear-dd-transfer-unpitched dst-blk dst-offset src-blk s-start-offset size nil)
                ;; One dest row?
                (with-pitch-mapping-vars (dst-blk dst-offset size :prefix d-)
                  (if d-one-row-p
                      (%cuda-linear-dd-transfer-unpitched src-blk src-offset dst-blk d-start-offset size t)
                      ;; Both span rows; prepare for heavy lifting.
                      (let ((s-handle (cuda-linear-ensure-handle src-blk))
                            (d-handle (cuda-linear-ensure-handle dst-blk)))
                        (declare (type integer s-handle d-handle))
                        (with-cuda-context ((cuda-linear-context src-blk))
                          (with-cuda-context ((cuda-linear-context dst-blk))
                            (cond
                              ;; Identical pitch settings
                              ((and (= s-width d-width) (= s-pitch d-pitch) (= s-start-x d-start-x) (= s-end-x d-end-x))
                               (when (= s-end-x 0) ; Adjust the last row
                                 (setf s-end-y (1- s-end-y) s-end-x s-width))
                               ;; Pitch gaps match, so copy everything:
                               (cuda-invoke cuMemcpyDtoD (+ d-handle d-start-offset) (+ s-handle s-start-offset)
                                            (- (+ (* s-end-y s-pitch) s-end-x) s-start-offset))
                               nil)

                              ;; Same width, but different pitch:
                              ((and (= s-width d-width) (= s-start-x d-start-x) (= s-end-x d-end-x))
                               ;; First row
                               (if (= s-start-x 0)
                                   (progn
                                     (decf s-start-y)
                                     (decf d-start-y))
                                   (cuda-invoke cuMemcpyDtoD (+ d-handle d-start-offset)
                                                (+ s-handle s-start-offset) (- s-width s-start-x)))
                               ;; Middle bulk
                               (let ((middle-h (- s-end-y s-start-y 1)))
                                 (when (> middle-h 0)
                                   (with-foreign-object (pmovespec 'cuda-memcpy-2d)
                                     (with-foreign-slots ((src-x-bytes src-y src-type src-device src-pitch
                                                                       dst-x-bytes dst-y dst-type dst-device dst-pitch
                                                                       width-bytes height)
                                                          pmovespec cuda-memcpy-2d)
                                       (setf width-bytes s-width
                                             height middle-h
                                             src-x-bytes 0 src-y (1+ s-start-y) src-pitch s-pitch
                                             dst-x-bytes 0 dst-y (1+ d-start-y) dst-pitch d-pitch
                                             src-type :device src-device s-handle
                                             dst-type :device dst-device d-handle))
                                     ;; Do the 2D transfer
                                     (cuda-invoke cuMemcpy2D pmovespec))))
                               ;; Last row
                               (when (> s-end-x 0)
                                 (cuda-invoke cuMemcpyDtoD (+ d-handle (* d-end-y d-pitch))
                                              (+ s-handle (* s-end-y s-pitch)) s-end-x))
                               nil)

                              ;; Misaligned, not supported
                              (t
                               (or return-if-mismatch
                                   (error "Pitch alignment mismatch: ~S ~A -> ~S ~A ~A"
                                          src-blk src-offset dst-blk dst-offset size))))))))))))))
