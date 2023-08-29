use std::{path::Path, ops::Add};
use anyhow::Result;
use crate::boxes;
use tch::{Tensor,Kind,Device};
use ndarray::ArrayView3;
use opencv::{
    prelude::*,
    core::{Mat,Size_},
};

#[derive(Debug)]
// #[must_use]
pub struct Model {
    model: tch::jit::CModule,
    resize: (i32,i32),
    classes: Vec<String>,
    device: Device,
}

impl Model {

    /// Creates an instance of the Model struct. 
    /// 
    /// **Args**
    /// - `model_path`: the local/absolute path to the model
    /// - `resize`: the dimensions of the image fed to the model
    /// 
    /// # Examples
    /// 
    /// ```
    /// use vigil_rs::Model;
    /// let model = Model::new("yolov8s.torchscript", (640, 640));
    /// ```
    pub fn new(model_path: &str, resize: (i32,i32), classes:Vec<String>, device:Device) -> Result<Model> {
        assert!(Path::new(model_path).exists(), "model_path does not exist.");
        assert!(model_path.contains(".torchscript"), "The model is not a torchscript model");

        Ok(Model {
            model: tch::CModule::load_on_device(model_path, device)?,
            resize: resize,
            classes: classes,
            device: device,
        })
    }

    /// Turns a `opencv::core::Mat` array into a `tch::Tensor`. 
    /// Useful for feeding an OpenCV frame into a PyTorch model.
    /// 
    /// **Args**
    /// - `frame`:  Needs to be continuous and of type `CV_8UC3`.
    /// 
    /// **Returns**
    /// - Tesnor in the shape (height,width,channels)
    /// 
    /// # Example
    /// 
    /// ```
    /// use vigil_rs::Model;
    /// use opencv::{
    ///     videoio::VideoCapture,
    ///     core::Mat,
    /// };
    /// 
    /// let mut cam = VideoCapture::new(0, opencv::videoio::CAP_ANY)?;
    /// let mut frame = Mat::default();
    /// cam.read(&mut frame)?;
    /// let tensor = Model::frame_to_tensor(&frame);
    /// ```
    pub fn frame_to_tensor(frame: &Mat)-> Result<Tensor> {
        assert!(frame.is_continuous(), "Mat is not continuoius");
        assert!(frame.typ()==opencv::core::CV_8UC3, "Mat must have the type CV_8UC3.");
        
        let data_bytes: &[u8] = frame.data_bytes()?;
    
        let size = frame.size()?;
        let h:i64 = size.height.into();
        let w:i64 = size.width.into();
        let c:i64 = 3;

        let nd_image = ArrayView3::from_shape(
            (h as usize, w as usize, c as usize), data_bytes)?;
        let inp_tensor = Tensor::from_slice(nd_image.as_slice().unwrap());
        Ok(inp_tensor.reshape(&vec![h,w,c]))
    }

    /// Resizes a `opencv::core::Mat` to shape of `resize` and turns it into a `tch::Tensor`.
    /// Then normalizes the `Tensor` RGB values between 0-1, and reshaps it to (3,resize...).
    /// 
    /// **Args**
    /// - `frame`:  Needs to be continuous and of type `CV_8UC3`.
    /// 
    /// **Returns**
    /// - Tensor in the shape (channels,height,width) with values between 0-1.
    /// 
    /// # Example
    /// 
    /// ```
    /// use vigil::Model;
    /// use opencv::{
    ///     videoio::VideoCapture,
    ///     core::Mat,
    /// };
    /// 
    /// let mut cam = VideoCapture::new(0, opencv::videoio::CAP_ANY)?;
    /// let model = Model::new("model.torchscript", (640, 640));
    /// 
    /// let mut frame = Mat::default();
    /// cam.read(&mut frame)?;
    /// 
    /// let input = model.preprocess(&frame);
    /// ```
    pub fn preprocess(&self, frame: &Mat) -> Result<Tensor> {
        // Resize the frame to (640, 640)
        let mut rsz_frame = Mat::default(); // Stores the resized frame
        opencv::imgproc::resize(&frame, &mut rsz_frame, Size_::new(self.resize.0, self.resize.1), 
            0.0, 0.0, opencv::imgproc::INTER_AREA)?;
    
        let tensor = Self::frame_to_tensor(&rsz_frame)?.to_device(self.device);  // Mat -> Tensor
        let norm_tensor = tensor.divide_scalar(255);  // Channel values between 0-1
        let perm_tensor = norm_tensor.permute(&[2,0,1]);  // Shape [640,640,3] -> [3,640,640]
        let input = perm_tensor.unsqueeze(0);  // Shape [3,640,640] -> [1,3,640,640]
    
        Ok(input.to_kind(Kind::Float))
    }

    /// Fowards the input to the model and returns the predictions.
    /// 
    /// **Args**
    /// - `input`: Tensor in the shape (channels,height,width) with values between 0-1.
    /// 
    /// # Example
    /// 
    /// ```
    /// use vigil::Model;
    /// use opencv::{
    ///     videoio::VideoCapture,
    ///     core::Mat,
    /// };
    /// 
    /// let mut cam = VideoCapture::new(0, opencv::videoio::CAP_ANY)?;
    /// let model = Model::new("model.torchscript", (640, 640));
    /// 
    /// let mut frame = Mat::default();
    /// cam.read(&mut frame)?;
    /// 
    /// let input = model.preprocess(&frame);
    /// let predictions = model.predict(&input);
    /// ```
    pub fn predict(&self, input: &Tensor) -> Result<Tensor> {
        tch::no_grad(|| -> Result<Tensor> {
            Ok(self.model.forward_ts(&[input])?)
        })
        // Ok()
    }

    /// Filters out boxes below the `conf_thres` and resizes the prediction's box to match the original `img_size`.
    /// Then, removes boxes that overlap above the `iou_thres`.
    /// 
    /// **Args**
    /// - `predictions`: Result from a YOLO model. In the shape (1, xywh+classes, boxes).
    /// - `img_size`: The (height, width) of the original image.
    /// - `conf_thres`: The mininum confidence for the model's prediction. Default is 0.5
    /// - `iou_thres`: The maximum box (intersection / union) overlap value. Default is 0.7
    /// 
    /// **Results**
    /// - Vector of tensors with the shape (B,N) in the format (xyxy, conf, cls).
    /// 
    /// # Example
    /// 
    /// ```
    /// use vigil::Model;
    /// use opencv::{
    ///     prelude::*,
    ///     videoio::VideoCapture,
    ///     core::Mat,
    /// };
    /// 
    /// let mut cam = VideoCapture::new(0, opencv::videoio::CAP_ANY)?;
    /// let model = Model::new("model.torchscript", (640, 640));
    /// 
    /// let mut frame = Mat::default();
    /// cam.read(&mut frame)?;
    /// let img_size = (i64::from(frame.size()?.height), i64::from(frame.size()?.width));
    /// 
    /// let input = model.preprocess(&frame);
    /// let predictions = model.predict(&input);
    /// let output = model.postprocess(&predictions, img_size, 0.5, 0.7);
    /// ```
    pub fn postprocess(&self, predictions: &Tensor, img_size: (i64, i64), conf_thres: f64, iou_thres: f64) -> Result<Vec<Tensor>> {
        let no_classes = predictions.size()[1] - 4; // 4 == xywh
        let indexes = Tensor::arange_start(4, no_classes+4, (Kind::Int64, self.device));
        let candidates = predictions.index_select(1, &indexes)  // Everything but the xywh
                .amax(1, false)  // Max class conf score for each box
                .gt(conf_thres);  // T/F for each val if > conf_thres
        
        let mut predictions = predictions.transpose(-1, -2);  // shape(1,84,8400) to shape(1,8400,84)
        predictions = boxes::xywh_to_rsz_xyxy(&predictions, 
            img_size, (i64::from(self.resize.0),i64::from(self.resize.1)))?;  // xywh to xyxy
        
        let mut output = vec![Tensor::zeros(&[0,6], // 6 == xyxy, class, conf
            (Kind::Int, self.device))];
        for pred_index in 0..predictions.size()[0] {
            // updates prediction according to candidates
            let mut prediction = predictions.get(pred_index)  
                .index_select(0, &candidates.get(pred_index).nonzero().squeeze());

            // If none process next batch
            if prediction.size()[0] == 0 {
                continue;
            }

            // Detections matrix nx6 (xyxy, conf, class)
            let split_pred =  prediction.split_sizes(&[4,no_classes], 1);
            let boxes = &split_pred[0];
            let classes = &split_pred[1];
            let (conf, conf_index) =  &classes.max_dim(1, true);
            prediction = Tensor::cat(&[boxes, conf, conf_index], 1).to_device(self.device);

            // Non-Maximal Suppression
            let cls_offset = prediction.index_select(1, &Tensor::from_slice(&[5]).to_device(self.device))
                .multiply_scalar(7680);  // classes * max box width/height
            let boxes = prediction.index_select(1, &Tensor::from_slice(&[0,1,2,3]).to_device(self.device))
                .add(&cls_offset);  // offset by classes to differientiate between classes in NMS
            let scores = prediction.index_select(1, &Tensor::from_slice(&[4]).to_device(self.device)).squeeze();

            let nms_indexes = boxes::non_maximal_suppression(&boxes, &scores, iou_thres)?;
            output[usize::try_from(pred_index)?] = prediction.index_select(0, &nms_indexes.to_device(self.device));
        }
        Ok(output)
    }

    /// Plots the boxes and values from the `output` onto the image `frame`.
    /// 
    /// **Args**
    /// - Mat structure passed from `opencv::videoio::VideoCapture`.
    /// - A tensor with the shape (N) in the format (xyxy, conf, cls).
    pub fn plot_boxes(&self, frame: &mut Mat, output: &Tensor) -> Result<()> {
        for index in 0..output.size()[0] {
            let boxx = output.get(index);
    
            let pt1 = opencv::core::Point_::new(
                i32::try_from(boxx.int64_value(&[0]))?, 
                i32::try_from(boxx.int64_value(&[1]))?
            );
            let pt2 = opencv::core::Point_::new(
                i32::try_from(boxx.int64_value(&[2]))?, 
                i32::try_from(boxx.int64_value(&[3]))?
            );
            let color = opencv::core::VecN::new(0.0, 255.0, 0.0, 0.0);
            let label = &self.classes[usize::try_from(boxx.int64_value(&[5]))?];
            let score = boxx.double_value(&[4]);
    
            opencv::imgproc::rectangle_points(frame, pt1, pt2, color, 
                2, 4, 0)?;
            opencv::imgproc::put_text(frame, format!("{label}: {score:.2}").as_str(), 
                pt1, opencv::imgproc::FONT_HERSHEY_SIMPLEX, 0.9, 
                color, 2, 4, false)?;
        }
        Ok(())
    }


}