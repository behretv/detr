# Goal of this project

A model library that is built up on modern torchvision and utilizes this library heavily.

Using DETR models should be compatible with torchvision models for object detection and instance segmentation.
In particular, models should be trained and saved that can be loaded with torch and has the same behavior
(forward function in eval() and train() mode) ad torchvision models.