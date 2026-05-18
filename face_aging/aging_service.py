import os
import sys
import cv2
import numpy as np
import PIL.Image
import scipy.ndimage
import torch
import torchvision.transforms as transforms
from argparse import Namespace

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Append sam_repo to sys.path so we can import its modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM_REPO_PATH = os.path.join(PROJECT_ROOT, "sam_repo")

if not os.path.exists(SAM_REPO_PATH):
    print("[AgingService] SAM (Spatially-Aware-Modification) repository not found. Downloading automatically...")
    import subprocess
    subprocess.run(["git", "clone", "https://github.com/yuval-alaluf/SAM.git", SAM_REPO_PATH], check=True)

if SAM_REPO_PATH not in sys.path:
    sys.path.append(SAM_REPO_PATH)

try:
    from models.psp import pSp
    from datasets.augmentations import AgeTransformer
    from utils.common import tensor2im
    SAM_AVAILABLE = True
except ImportError as e:
    print(f"[AgingService] Could not import SAM dependencies: {e}")
    SAM_AVAILABLE = False


def align_face_mediapipe(img_bgr):
    """
    FFHQ face alignment using MediaPipe Tasks API.
    Returns: (PIL.Image of size 256x256, quad array for unwarping)
    """
    model_asset_path = os.path.join(PROJECT_ROOT, 'models', 'face_landmarker.task')
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res = landmarker.detect(mp_image)
        if not res.face_landmarks:
            return None, None
        
        lm = res.face_landmarks[0]
        h, w, _ = img_bgr.shape
        
        # Mapping mediapipe landmarks to dlib points for FFHQ
        # Viewer's Left (smaller X): 33 (outer), 133 (inner)
        # Viewer's Right (larger X): 263 (outer), 362 (inner)
        eye_left = np.array([ (lm[33].x + lm[133].x)/2 * w, (lm[33].y + lm[133].y)/2 * h ])
        eye_right = np.array([ (lm[263].x + lm[362].x)/2 * w, (lm[263].y + lm[362].y)/2 * h ])
        mouth_left = np.array([ lm[78].x * w, lm[78].y * h ])
        mouth_right = np.array([ lm[308].x * w, lm[308].y * h ])
        
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        original_quad = quad.copy()
        qsize = np.hypot(*x) * 2
        
        img = PIL.Image.fromarray(img_rgb)
        output_size = 256
        transform_size = 256
        enable_padding = True
        
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, getattr(PIL.Image, 'Resampling', PIL.Image).LANCZOS)
            quad /= shrink
            qsize /= shrink
            
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]
            
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img_arr = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            _h, _w, _ = img_arr.shape
            _y, _x, _ = np.ogrid[:_h, :_w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(_x) / pad[0], np.float32(_w - 1 - _x) / pad[2]),
                              1.0 - np.minimum(np.float32(_y) / pad[1], np.float32(_h - 1 - _y) / pad[3]))
            blur = qsize * 0.02
            img_arr += (scipy.ndimage.gaussian_filter(img_arr, [blur, blur, 0]) - img_arr) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img_arr += (np.median(img_arr, axis=(0, 1)) - img_arr) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img_arr), 0, 255)), 'RGB')
            quad += pad[:2]
            
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), getattr(PIL.Image, 'Resampling', PIL.Image).BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), getattr(PIL.Image, 'Resampling', PIL.Image).LANCZOS)
            
        return img, original_quad


class LocalAgingService:
    """
    A local inference service for face aging using PyTorch.
    Loads the models/aging.pt weights (sam_ffhq_aging.pt) using the SAM architecture.
    """
    def __init__(self, model_path="models/aging.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AgingService] Initializing local model from {model_path} on {self.device}...")
        
        if not os.path.exists(model_path):
            print(f"[AgingService] Model file not found at {model_path}. Downloading...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            import gdown
            # NOT: model_path için gdown linki. Aşağıdaki id'yi kendi Google Drive dosya ID'niz ile değiştirin!
            # Örnek: drive linkinizdeki d/ ile /view arasındaki karmaşık yazı id'dir.
            file_id = '1QTK-O0xoUZavCUCno-YKxIhsbXSGY5ia'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model dosyasi indirilemedi! Lutfen {model_path} dosyasini manuel olarak ekleyin.")

        if not SAM_AVAILABLE:
            raise RuntimeError("SAM repository modules are missing. Cannot load SAM architecture.")

        ckpt = torch.load(model_path, map_location="cpu")
        
        opts = ckpt["opts"]
        opts["checkpoint_path"] = model_path
        opts["device"] = str(self.device)
        self.opts = Namespace(**opts)
        
        self.net = pSp(self.opts)
        self.net.eval()
        if self.device.type == "cuda":
            self.net.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def preprocess(self, pil_img, target_age: int) -> torch.Tensor:
        input_tensor = self.transform(pil_img)
        age_transformer = AgeTransformer(target_age=target_age)
        input_tensor_age = age_transformer(input_tensor.cpu())
        input_batch = input_tensor_age.unsqueeze(0).to(self.device).float()
        return input_batch

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        out_tensor = tensor.squeeze(0).cpu().detach()
        out_tensor = torch.clamp(out_tensor, -1.0, 1.0)
        out_tensor = (out_tensor + 1.0) / 2.0
        out_np = out_tensor.permute(1, 2, 0).numpy()
        out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
        bgr_img = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        return bgr_img

    def run_inference(self, image: np.ndarray, target_age: int = 50) -> np.ndarray:
        # 1. FFHQ Alignment
        aligned_pil, quad = align_face_mediapipe(image)
        if aligned_pil is None:
            print("[AgingService] Could not align face. Returning original.")
            return image
            
        # 2. Inference
        input_tensor = self.preprocess(aligned_pil, target_age=target_age)
        with torch.no_grad():
            result_batch = self.net(input_tensor, randomize_noise=False, resize=False)
            if isinstance(result_batch, tuple):
                output_tensor = result_batch[0]
            else:
                output_tensor = result_batch
                
        aged_face_1024 = self.postprocess(output_tensor)
        
        # 3. Create blend mask
        mask = np.zeros((1024, 1024), dtype=np.float32)
        cv2.ellipse(mask, (512, 580), (360, 420), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (151, 151), 0)
        mask = cv2.GaussianBlur(mask, (151, 151), 0)
        mask = np.stack([mask, mask, mask], axis=-1)
        
        # 4. Inverse Perspective Transform (Unwarp)
        dst_pts = np.array([
            [0, 0],
            [0, 1024],
            [1024, 1024],
            [1024, 0]
        ], dtype=np.float32)
        quad_pts = np.array(quad, dtype=np.float32)
        M = cv2.getPerspectiveTransform(dst_pts, quad_pts)
        
        h, w = image.shape[:2]
        unwarped_face = cv2.warpPerspective(aged_face_1024, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        unwarped_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        # 5. Alpha Blend
        original_float = image.astype(np.float32)
        aged_float = unwarped_face.astype(np.float32)
        blended = original_float * (1.0 - unwarped_mask) + aged_float * unwarped_mask
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return blended
