use std::ops::Add;
use tch::{Tensor,Kind};
use anyhow::{Result,anyhow};

/**
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
*/
pub fn box_area(boxes: &Tensor) -> Result<Tensor> {
    let device = boxes.device();
    let indexes = Tensor::from_slice(&[0,1,2,3]).to_device(device);
    let width = boxes.index_select(1, &indexes.get(2))
        .subtract(
            &boxes.index_select(1, &indexes.get(0))
        );
    let height = boxes.index_select(1, &indexes.get(3))
        .subtract(
            &boxes.index_select(1, &indexes.get(1))
        );
    Ok(width.multiply(&height))
}

/**
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
*/
pub fn box_iou(boxes1: &Tensor, boxes2: &Tensor) -> Result<Tensor> {
    let device = boxes1.device();
    let area1 = box_area(boxes1)?;
    let area2 = box_area(boxes2)?;

    let top_left_inter =  boxes1.index_select(1, &Tensor::from_slice(&[0,1]).to_device(device)).unsqueeze(0)
        .max_other(
            &boxes2.index_select(1, &Tensor::from_slice(&[0,1]).to_device(device))
        );
    let bot_right_inter = boxes1.index_select(1, &Tensor::from_slice(&[2,3]).to_device(device)).unsqueeze(0)
    .min_other(
        &boxes2.index_select(1, &Tensor::from_slice(&[2,3]).to_device(device))
    );
    
    let width_height = bot_right_inter.subtract(&top_left_inter)
        .clamp_min(0); // if < 0, set val to 0
    let inter = width_height.index_select(2, &Tensor::from_slice(&[0]).to_device(device))
        .multiply(
            &width_height.index_select(2, &Tensor::from_slice(&[1]).to_device(device))
        ); // width_inter * height_inter

    let union = area1.unsqueeze(0)
        .add(&area2)
        .subtract(&inter);
    
    Ok(inter.divide(&union).squeeze())
}

/**
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (f64): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
*/
pub fn non_maximal_suppression(boxes: &Tensor, scores: &Tensor, iou_thres: f64) -> Result<Tensor> {
    if scores.size().is_empty() {
        return Ok(Tensor::from_slice(&[0]));
    }
    if boxes.device() != scores.device() {
        return Err(anyhow!("boxes {:?} and scores {:?} must be on the same device", boxes.device(), scores.device()));
    }
    let device = boxes.device();

    let order =  scores.argsort(0, true);
    let mut keep =  Tensor::ones(boxes.size()[0], (Kind::Bool, device));
    let true_tensor = Tensor::ones(&[], (Kind::Bool, device));

    for index in 0..boxes.size()[0] {
        if keep.get(index).equal(&true_tensor) {
            let boxx = boxes.index_select(0, &order.get(index));
            let sorted_boxes = boxes.index_select(0, 
                &order.index_select(0, 
                    &Tensor::arange_start(index+1, order.size()[0], (Kind::Int,device))
                ));
            let valid_boxes = keep.index_select(0, 
                &Tensor::arange_start(index+1, keep.size()[0], (Kind::Int,device)))
                .unsqueeze(1);
            
            let iou = box_iou(&boxx, &sorted_boxes.multiply(&valid_boxes))?;
            let overlapped = Tensor::nonzero(&iou.gt(iou_thres)).squeeze();
            if overlapped.size().len() == 1 { // Prevents error that occurs when boxes and boxx shape is [1,4] and boxes is [0.,0.,0.,0.]
                keep = keep.index_fill(0, &overlapped.add(index+1).to_device(device), 0);
            }
        }
    }
    
    Ok(order.index_select(0, &keep.nonzero().squeeze()))
}

/**
    Converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.
    (x, y) refers to center of bounding box.
    (w, h) refers to width and height of box.
    Args:
        boxes (Tensor[N, 4 + classes]): boxes in (x, y, w, h, classes) which will be converted.
        img_size (i64, i64): the shape of the current image.
        resize (i64, i64): the shape of the new image.

    Returns:
        boxes (Tensor[N, 4 + classes]): boxes in (x1, y1, x2, y2, classes) format.
*/
pub fn xywh_to_rsz_xyxy(boxes: &Tensor, img_size: (i64, i64), resize: (i64, i64)) -> Result<Tensor> {
    // let output = input.clone(&input);
    let device = boxes.device();
    let indexes = Tensor::from_slice(&[0,1,2,3]).to_device(device);

    // top left x = (center_x - box_width / 2) / resize_width * img_width
    let x1 = boxes.index_select(2, &indexes.get(0))
        .subtract(&boxes.index_select(2, &indexes.get(2))
            .divide_scalar(2)) 
        .divide_scalar(resize.1)
        .multiply_scalar(img_size.1);

    // top left y = (center_y - box_height / 2) / resize_height * img_height
    let y1 = boxes.index_select(2, &indexes.get(1))
        .subtract(&boxes.index_select(2, &indexes.get(3))
            .divide_scalar(2))
        .divide_scalar(resize.0)
        .multiply_scalar(img_size.0);

    // bottom right x = (center_x + box_width / 2) / resize_width * img_width
    let x2 = boxes.index_select(2, &indexes.get(0))
        .add(&boxes.index_select(2, &indexes.get(2))
            .divide_scalar(2))
        .divide_scalar(resize.1)
        .multiply_scalar(img_size.1);

    // bottom right y = (center_y - box_height / 2) / resize_height * img_height
    let y2 = boxes.index_select(2, &indexes.get(1))
        .add(&boxes.index_select(2, &indexes.get(3))
            .divide_scalar(2))
        .divide_scalar(resize.0)
        .multiply_scalar(img_size.0);

    // Combines everything into [1, 8400, x1 y1 x2 y2 + classes]
    let classes = boxes.index_select(
        2, &Tensor::arange_start(
            4, boxes.size()[2], (Kind::Int, device)
    ));
    
    Ok(Tensor::cat(&[x1, y1, x2, y2, classes], 2))
}