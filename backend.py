import cv2
import numpy as np

def process_image(image: np.ndarray, thresh_value: int) -> np.ndarray:
    resized_image = cv2.resize(image, (1600, 720))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, thresh_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = resized_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image

def apply_blur(image: np.ndarray, method: str, ksize: int = 5, d: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    if method == "gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif method == "median":
        return cv2.medianBlur(image, ksize)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    else:
        return image

def adjust_contrast(image: np.ndarray, method: str, alpha: float = 1.5, beta: int = 0, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    if method == "hist_eq":
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        if len(image.shape) == 2:
            return clahe.apply(image)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    elif method == "alpha_beta":
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    else:
        return image

def apply_morphology(image: np.ndarray, method: str, kernel_size: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if method == "Erosion":
        return cv2.erode(image, kernel, iterations=1)
    elif method == "Dilation":
        return cv2.dilate(image, kernel, iterations=1)
    elif method == "Opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif method == "Closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        return image
