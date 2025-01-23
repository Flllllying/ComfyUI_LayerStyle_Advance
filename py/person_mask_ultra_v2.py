# layerstyle advance

import cv2
import os
import logging

from .imagefunc import *
from functools import reduce
import wget
import folder_paths
from .segment_anything_func import *
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

NODE_NAME = 'PersonMaskUltra V2'
logging.basicConfig(level=logging.DEBUG)

class PersonMaskUltraV2:

    def __init__(self):
        # download the model if we need it
        get_a_person_mask_generator_model_path()
        self.pose_detector = self.setup_pose_detector()

    @classmethod
    def INPUT_TYPES(self):

        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']
        return {
            "required":
                {
                    "images": ("IMAGE",),
                    "face": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                    "hair": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "body": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "clothes": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "accessories": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "background": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "upper_body": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "lower_body": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "confidence": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 0.95, "step": 0.01},),
                    "detail_method": (method_list,),
                    "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                    "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                    "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                    "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                    "process_detail": ("BOOLEAN", {"default": True}),
                    "device": (device_list,),
                    "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
                },
            "optional":
                {
                }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'person_mask_ultra_v2'
    CATEGORY = '😺dzNodes/LayerMask'

    def get_mediapipe_image(self, image: Image):
        import mediapipe as mp
        # Convert image to NumPy array
        numpy_image = np.asarray(image)
        image_format = mp.ImageFormat.SRGB
        # Convert BGR to RGB (if necessary)
        if numpy_image.shape[-1] == 4:
            image_format = mp.ImageFormat.SRGBA
        elif numpy_image.shape[-1] == 3:
            image_format = mp.ImageFormat.SRGB

            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=image_format, data=numpy_image)
    
    def setup_pose_detector(self):
        model_path = '/workspace/qikai/stable-paw-comfyui/models/pose/pose_landmarker_lite.task'
        
        logging.debug(f"Attempting to load pose landmarker model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pose landmarker model not found at {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False)
        
        try:
            detector = vision.PoseLandmarker.create_from_options(options)
            logging.debug("Pose landmarker model loaded successfully")
            return detector
        except Exception as e:
            logging.error(f"Error loading pose landmarker model: {str(e)}")
            raise
        return vision.PoseLandmarker.create_from_options(options)
    
    def detect_waist(self, image):
        detection_result = self.pose_detector.detect(image)
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            waist_y = (left_hip.y + right_hip.y) / 2
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Estimate waist position as a point between shoulders and hips
            waist_position = int((waist_y * 0.7 + shoulder_y * 0.3) * image.height)
            return waist_position
        return None

    def person_mask_ultra_v2(self, images, face, hair, body, clothes,
                         accessories, background, upper_body, lower_body, confidence,
                         detail_method, detail_erode, detail_dilate,
                         black_point, white_point, process_detail, device, max_megapixels,):

        import mediapipe as mp
        import numpy as np
    
        # Check if images is empty
        if len(images) == 0:
            log("No input images provided.", message_type='warning')
            return (torch.empty((0, 3, 1, 1)), torch.empty((0, 1, 1, 1)))

        a_person_mask_generator_model_path = get_a_person_mask_generator_model_path()
        a_person_mask_generator_model_buffer = None
        with open(a_person_mask_generator_model_path, "rb") as f:
            a_person_mask_generator_model_buffer = f.read()
        image_segmenter_base_options = mp.tasks.BaseOptions(model_asset_buffer=a_person_mask_generator_model_buffer)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=image_segmenter_base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True)
    
        ret_images = []
        ret_masks = []

        local_files_only = detail_method == 'VITMatte(local)'

        with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
            for image in images:
                try:
                    _image = torch.unsqueeze(image, 0)
                    orig_image = tensor2pil(_image).convert('RGB')
                    i = 255. * image.cpu().numpy()
                    image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                
                    media_pipe_image = self.get_mediapipe_image(image=image_pil)
                    segmented_masks = segmenter.segment(media_pipe_image)
                    # Detect waist
                    waist_y = self.detect_waist(media_pipe_image)
                    image_data = media_pipe_image.numpy_view()
                    image_shape = image_data.shape

                    masks = []
                    if background:
                        masks.append(segmented_masks.confidence_masks[0])
                    if hair:
                        masks.append(segmented_masks.confidence_masks[1])
                    if body:
                        masks.append(segmented_masks.confidence_masks[2])
                    if face:
                        masks.append(segmented_masks.confidence_masks[3])
                    if clothes:
                        masks.append(segmented_masks.confidence_masks[4])
                    if accessories:
                        masks.append(segmented_masks.confidence_masks[5])

                    if upper_body or lower_body:
                        body_mask = np.array(segmented_masks.confidence_masks[2].numpy_view())
                        clothes_mask = np.array(segmented_masks.confidence_masks[4].numpy_view())
                        combined_mask = np.maximum(body_mask, clothes_mask)
                        combined_mask = (combined_mask - combined_mask.min()) / (combined_mask.max() - combined_mask.min() + 1e-8)
                        binary_mask = (combined_mask > 0.4).astype(np.uint8)
                        rows = np.any(binary_mask, axis=1)
                        cols = np.any(binary_mask, axis=0)
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols)[0][[0, -1]]
                        # Use detected waist if available, otherwise use middle point
                        middle_y = waist_y if waist_y is not None else (ymin + ymax) // 2
                        
                        if upper_body:
                            upper_body_mask = combined_mask.copy()
                            upper_body_mask[middle_y:, :] = 0  # Mask out lower part
                            upper_body_mask[:ymin, :] = 0  # Mask out area above the body
                            upper_body_mask[:, :xmin] = 0  # Mask out area to the left of the body
                            upper_body_mask[:, xmax:] = 0  # Mask out area to the right of the body
                            upper_body_mask = (upper_body_mask > 0.5).astype(np.uint8) * 255
                            masks.append(mp.Image(image_format=mp.ImageFormat.GRAY8, data=upper_body_mask))
                        if lower_body:
                            lower_body_mask = combined_mask.copy()
                            lower_body_mask[:middle_y, :] = 0  # Mask out upper part
                            lower_body_mask[ymax:, :] = 0  # Mask out area below the body
                            lower_body_mask[:, :xmin] = 0  # Mask out area to the left of the body
                            lower_body_mask[:, xmax:] = 0  # Mask out area to the right of the body
                            lower_body_mask = (lower_body_mask > 0.5).astype(np.uint8) * 255
                            masks.append(mp.Image(image_format=mp.ImageFormat.GRAY8, data=lower_body_mask))

                    if image_shape[-1] == 3:
                        image_shape = (*image_shape[:2], 4)

                    mask_background_array = np.zeros(image_shape, dtype=np.uint8)
                    mask_background_array[:] = (0, 0, 0, 255)
                    mask_foreground_array = np.ones(image_shape, dtype=np.uint8) * 255

                    mask_arrays = []
                    for mask in masks:
                        condition = np.stack((mask.numpy_view(),) * image_shape[-1], axis=-1) > confidence
                        mask_array = np.where(condition, mask_foreground_array, mask_background_array)
                        mask_arrays.append(mask_array)

                    if not mask_arrays:
                        mask_arrays.append(mask_background_array)

                    merged_mask_arrays = reduce(np.maximum, mask_arrays)
                    mask_image = Image.fromarray(merged_mask_arrays)
                    tensor_mask = np.array(mask_image.convert("RGB")).astype(np.float32) / 255.0
                    tensor_mask = torch.from_numpy(tensor_mask)[None,]
                    _mask = tensor_mask[:, :, :, 0]

                    detail_range = detail_erode + detail_dilate
                    if process_detail:
                        if detail_method == 'GuidedFilter':
                            _mask = guided_filter_alpha(pil2tensor(orig_image), _mask, detail_range // 6 + 1)
                            _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                        elif detail_method == 'PyMatting':
                            _mask = tensor2pil(
                                mask_edge_detail(pil2tensor(orig_image), _mask,
                                             detail_range // 8 + 1, black_point, white_point))
                        else:
                            _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                            _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                            _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
                    else:
                        _mask = mask2image(_mask)

                    ret_image = RGB2RGBA(orig_image, _mask)
                    ret_images.append(pil2tensor(ret_image))
                    ret_masks.append(image2mask(_mask))
                except Exception as e:
                    log(f"Error processing image: {str(e)}", message_type='error')

        num_processed = len(ret_images)
        log(f"{NODE_NAME} Processed {num_processed} image(s).", message_type='finish')

        if num_processed == 0:
            log("No images were processed successfully.", message_type='warning')
            return (torch.empty((0, 3, 1, 1)), torch.empty((0, 1, 1, 1)))
        else:
            return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: PersonMaskUltra V2": PersonMaskUltraV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: PersonMaskUltra V2": "LayerMask: PersonMaskUltra V2(Advance)"
}