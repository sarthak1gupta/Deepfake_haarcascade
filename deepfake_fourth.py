import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import dlib
import math
from scipy import ndimage
from scipy.fftpack import fft2, fftshift
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import io

class DeepfakeDetector:
    def __init__(self):
        # Initialize face detector and landmark predictor
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Try to load facial landmark predictor if available
        model_path = "shape_predictor_68_face_landmarks.dat"
        if os.path.exists(model_path):
            self.landmark_predictor = dlib.shape_predictor(model_path)
        else:
            print("Warning: Facial landmark predictor model not found!")
            print(f"Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print(f"Extract and place it in the same directory as this script")
            self.landmark_predictor = None
        
        # Load pre-trained deep learning model for texture analysis
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)
        self.dl_model = Model(inputs=base_model.input, outputs=predictions)
        
        # Enhanced base threshold values with more precise ranges
        self.base_thresholds = {
            'eye_aspect_ratio': 0.28,       # Adjusted for better eye shape detection
            'facial_symmetry': 0.25,        # Reduced to account for natural asymmetry
            'blur_threshold': 85,           # Increased for better blur detection
            'noise_threshold': 0.12,        # Increased for better noise pattern detection
            'frequency_threshold': 0.32,    # Adjusted for better frequency analysis
            'skin_texture_threshold': 0.42, # Adjusted for better skin texture analysis
            'lighting_mismatch': 0.35,      # Reduced for better lighting analysis
            'reflection_threshold': 0.25,   # Adjusted for better reflection detection
            'color_consistency': 0.28,      # Adjusted for better color analysis
            'shadow_threshold': 0.32,       # Adjusted for better shadow detection
            'texture_consistency': 0.35,    # New parameter for texture consistency
            'edge_quality': 0.45,           # New parameter for edge quality
            'color_balance': 0.30           # New parameter for color balance
        }

        # Enhanced facial landmark ranges with more precise definitions
        self.FACIAL_LANDMARKS_RANGES = {
            "left_eye": [(36, 41), (42, 47)],  # Range for left eye landmarks
            "right_eye": [(36, 41), (42, 47)], # Range for right eye landmarks
            "mouth": [(48, 67)],              # Range for mouth landmarks
            "nose": [(27, 35)],               # Range for nose landmarks
            "jaw": [(0, 16)],                 # Range for jaw landmarks
            "left_eyebrow": [(17, 21)],       # Range for left eyebrow landmarks
            "right_eyebrow": [(22, 26)],      # Range for right eyebrow landmarks
            "cheeks": [(0, 16), (17, 26)],    # New range for cheek analysis
            "forehead": [(17, 26), (27, 35)]  # New range for forehead analysis
        }
        
        # Initialize flags for different checks
        self.results = {}

    def generate_randomized_thresholds(self):
        """Generate a new set of randomized thresholds based on base values"""
        randomized = {}
        for param, base_value in self.base_thresholds.items():
            # Randomize within ±15% of base value (reduced from 20% for more stability)
            variation = base_value * 0.15
            randomized[param] = base_value + np.random.uniform(-variation, variation)
            
            # Ensure thresholds stay within reasonable bounds
            if param in ['eye_aspect_ratio', 'facial_symmetry', 'noise_threshold', 
                        'frequency_threshold', 'skin_texture_threshold', 
                        'lighting_mismatch', 'reflection_threshold', 
                        'color_consistency', 'shadow_threshold']:
                randomized[param] = max(0.1, min(0.9, randomized[param]))
            elif param == 'blur_threshold':
                randomized[param] = max(50, min(150, randomized[param]))
        
        return randomized

    def analyze_with_thresholds(self, image_path, thresholds):
        """Analyze image with specific threshold values"""
        try:
            # Load image
            img, rgb_img = self.load_image(image_path)
            
            # Apply color correction
            img = self.apply_color_correction(img)
            
            # Detect faces
            faces, gray = self.detect_faces(img)
            
            if len(faces) == 0:
                return {
                    "is_real": False,
                    "confidence": 0.7,
                    "reason": "No faces detected",
                    "details": {},
                    "faces_found": 0,
                    "thresholds_used": thresholds
                }
            
            # We'll analyze the largest face in the image
            largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Get face landmarks if available
            landmarks = self.get_landmarks(gray, largest_face) if self.landmark_predictor else None
            
            # Initialize results dictionary
            result_details = {}
            
            # Run various consistency checks with provided thresholds
            if landmarks:
                # Eye checks
                eye_normal, eye_score = self.check_eye_abnormalities(landmarks)
                result_details["eye_abnormalities"] = {
                    "passed": eye_normal,
                    "score": eye_score,
                    "threshold": thresholds['eye_aspect_ratio']
                }
                
                # Facial symmetry check
                symmetry_normal, symmetry_score = self.check_facial_symmetry(landmarks)
                result_details["facial_symmetry"] = {
                    "passed": symmetry_normal,
                    "score": symmetry_score,
                    "threshold": thresholds['facial_symmetry']
                }
            
            # Add new checks
            texture_consistency_normal, texture_consistency_score = self.check_texture_consistency(img, largest_face)
            result_details["texture_consistency"] = {
                "passed": texture_consistency_normal,
                "score": texture_consistency_score,
                "threshold": thresholds['texture_consistency']
            }
            
            edge_quality_normal, edge_quality_score = self.check_edge_quality(img, largest_face)
            result_details["edge_quality"] = {
                "passed": edge_quality_normal,
                "score": edge_quality_score,
                "threshold": thresholds['edge_quality']
            }
            
            color_balance_normal, color_balance_score = self.check_color_balance(img, largest_face)
            result_details["color_balance"] = {
                "passed": color_balance_normal,
                "score": color_balance_score,
                "threshold": thresholds['color_balance']
            }
            
            # Skin texture analysis
            texture_normal, texture_score = self.check_skin_texture(img, largest_face)
            result_details["skin_texture"] = {
                "passed": texture_normal,
                "score": texture_score,
                "threshold": thresholds['skin_texture_threshold']
            }
            
            # Frequency distribution analysis
            freq_normal, freq_score = self.check_frequency_distribution(img)
            result_details["frequency_distribution"] = {
                "passed": freq_normal,
                "score": freq_score,
                "threshold": thresholds['frequency_threshold']
            }
            
            # Noise pattern analysis
            noise_normal, noise_score = self.check_noise_patterns(img)
            result_details["noise_patterns"] = {
                "passed": noise_normal,
                "score": noise_score,
                "threshold": thresholds['noise_threshold']
            }
            
            # Blurriness check
            blur_normal, blur_score = self.check_blurriness(img)
            result_details["blurriness"] = {
                "passed": blur_normal,
                "score": blur_score,
                "threshold": thresholds['blur_threshold']
            }
            
            # Edge consistency check
            edge_normal, edge_score = self.check_edge_consistency(img, largest_face)
            result_details["edge_consistency"] = {
                "passed": edge_normal,
                "score": edge_score,
                "threshold": 0.5
            }
            
            # Lighting consistency check
            lighting_normal, lighting_score = self.check_lighting_consistency(img, largest_face)
            result_details["lighting_consistency"] = {
                "passed": lighting_normal,
                "score": lighting_score,
                "threshold": thresholds['lighting_mismatch']
            }
            
            # Shadow consistency check
            shadow_normal, shadow_score = self.check_shadow_consistency(img, largest_face)
            result_details["shadow_consistency"] = {
                "passed": shadow_normal,
                "score": shadow_score,
                "threshold": thresholds['shadow_threshold']
            }
            
            # Color consistency check
            color_normal, color_score = self.check_color_consistency(img, largest_face)
            result_details["color_consistency"] = {
                "passed": color_normal,
                "score": color_score,
                "threshold": thresholds['color_consistency']
            }
            
            # Calculate overall result with updated weights
            weights = {
                "eye_abnormalities": 0.08,
                "facial_symmetry": 0.08,
                "skin_texture": 0.12,
                "frequency_distribution": 0.12,
                "noise_patterns": 0.10,
                "blurriness": 0.08,
                "edge_consistency": 0.05,
                "lighting_consistency": 0.08,
                "shadow_consistency": 0.08,
                "color_consistency": 0.05,
                "texture_consistency": 0.10,
                "edge_quality": 0.08,
                "color_balance": 0.08
            }
            
            total_score = 0
            total_weight = 0
            critical_failures = 0
            
            # Check for critical failures first
            critical_checks = ["eye_abnormalities", "facial_symmetry", "skin_texture"]
            for check in critical_checks:
                if check in result_details and not result_details[check]["passed"]:
                    critical_failures += 1
            
            # If there are critical failures, adjust the confidence calculation
            if critical_failures > 0:
                confidence = max(0.3, 0.6 - (critical_failures * 0.1))
            else:
                # Normal confidence calculation
                for check, weight in weights.items():
                    if check in result_details:
                        if result_details[check]["passed"]:
                            total_score += weight
                        total_weight += weight
                
                if total_weight > 0:
                    confidence = total_score / total_weight
                else:
                    confidence = 0.5
            
            # Adjust confidence threshold based on number of checks passed
            passed_checks = sum(1 for check in result_details if result_details[check]["passed"])
            total_checks = len(result_details)
            pass_ratio = passed_checks / total_checks
            
            # Dynamic threshold adjustment
            if pass_ratio > 0.8:
                confidence_threshold = 0.5  # More lenient for high pass ratio
            elif pass_ratio > 0.6:
                confidence_threshold = 0.55  # Moderate threshold
            else:
                confidence_threshold = 0.6  # Strict threshold for low pass ratio
            
            is_real = confidence >= confidence_threshold
            
            if not is_real:
                failed_checks = [check for check in result_details if not result_details[check]["passed"]]
                reason = "Potential deepfake indicators: " + ", ".join(failed_checks)
            else:
                reason = "Image passed consistency checks"
            
            return {
                "is_real": is_real,
                "confidence": confidence,
                "reason": reason,
                "details": result_details,
                "faces_found": len(faces),
                "thresholds_used": thresholds
            }
            
        except Exception as e:
            return {
                "is_real": None,
                "confidence": 0,
                "reason": f"Error analyzing image: {str(e)}",
                "details": {},
                "faces_found": 0,
                "thresholds_used": thresholds
            }

    def analyze_image(self, image_path):
        """Main function to analyze an image using ensemble voting"""
        try:
            # Number of variations to try
            num_variations = 7  # Increased from 5 to 7 for better voting
            
            # Store all individual results
            individual_results = []
            
            # Run analysis with different parameter sets
            for i in range(num_variations):
                thresholds = self.generate_randomized_thresholds()
                result = self.analyze_with_thresholds(image_path, thresholds)
                individual_results.append(result)
            
            # Count votes for real/fake with confidence weighting
            real_votes = 0
            fake_votes = 0
            total_confidence = 0
            
            for result in individual_results:
                if result["is_real"]:
                    real_votes += result["confidence"]
                else:
                    fake_votes += result["confidence"]
                total_confidence += result["confidence"]
            
            # Normalize votes
            if total_confidence > 0:
                real_votes = real_votes / total_confidence
                fake_votes = fake_votes / total_confidence
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(individual_results)
            
            # Make final decision based on weighted majority vote
            is_real = real_votes > fake_votes
            
            # Combine all failed checks from individual results
            all_failed_checks = set()
            for result in individual_results:
                if not result["is_real"]:
                    failed_checks = [check for check in result["details"] if not result["details"][check]["passed"]]
                    all_failed_checks.update(failed_checks)
            
            reason = "Potential deepfake indicators: " + ", ".join(all_failed_checks) if all_failed_checks else "Image passed consistency checks"
            
            return {
                "is_real": is_real,
                "confidence": avg_confidence,
                "reason": reason,
                "details": individual_results[0]["details"],  # Use details from first result for display
                "faces_found": individual_results[0]["faces_found"],
                "individual_results": individual_results,  # Include all individual results
                "vote_count": {
                    "real": real_votes,
                    "fake": fake_votes
                }
            }
            
        except Exception as e:
            return {
                "is_real": None,
                "confidence": 0,
                "reason": f"Error analyzing image: {str(e)}",
                "details": {},
                "faces_found": 0,
                "individual_results": [],
                "vote_count": {"real": 0, "fake": 0}
            }

    def load_image(self, image_path):
        """Load and preprocess the image for analysis"""
        try:
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Image could not be loaded")
            
            # Convert to RGB for visualization and processing
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for display if too large
            h, w = img.shape[:2]
            max_size = 800
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                rgb_img = cv2.resize(rgb_img, (int(w * scale), int(h * scale)))
            
            return img, rgb_img
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

    def detect_faces(self, img):
        """Detect faces in the image using dlib"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        return faces, gray

    def get_landmarks(self, gray, face):
        """Get facial landmarks for a detected face"""
        if self.landmark_predictor is None:
            return None
        
        landmarks = self.landmark_predictor(gray, face)
        return landmarks

    def eye_aspect_ratio(self, eye_landmarks):
        """Calculate the eye aspect ratio to detect unnatural eye shapes"""
        # Vertical eye landmarks
        A = dist(eye_landmarks[1], eye_landmarks[5])
        B = dist(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal eye landmark
        C = dist(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def check_eye_abnormalities(self, landmarks):
        """Check for abnormalities in eye shapes and positions"""
        try:
            # Extract eye landmarks
            left_eye = []
            right_eye = []
            
            for i in range(36, 42):  # Right eye landmarks
                point = landmarks.part(i)
                right_eye.append((point.x, point.y))
                
            for i in range(42, 48):  # Left eye landmarks
                point = landmarks.part(i)
                left_eye.append((point.x, point.y))
            
            # Calculate eye aspect ratios
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            
            # Check for asymmetry in eye aspect ratios
            ear_diff = abs(left_ear - right_ear)
            
            # Check for improper eye reflections (simplified)
            has_reflections = self.check_eye_reflections(left_eye, right_eye)
            
            if ear_diff > self.base_thresholds['eye_aspect_ratio'] or not has_reflections:
                return False, ear_diff
            return True, ear_diff
        except Exception as e:
            print(f"Error checking eye abnormalities: {str(e)}")
            return False, 0

    def check_eye_reflections(self, left_eye, right_eye):
        """Basic check for presence of specular highlights in eyes (simplified)"""
        # This is a placeholder - in a real implementation, you'd analyze the 
        # eye regions for specular highlights
        return True

    def check_facial_symmetry(self, landmarks):
        """Check if the face has natural symmetry"""
        try:
            # Get facial midpoint using nose bridge
            nose_bridge = landmarks.part(27)
            midpoint_x = nose_bridge.x
            
            left_side_points = []
            right_side_points = []
            
            # Compare landmark positions on both sides of the face
            for i in range(0, 27):  # Jaw and eyebrows
                point = landmarks.part(i)
                if point.x < midpoint_x:
                    left_side_points.append(point)
                else:
                    right_side_points.append(point)
            
            # Calculate symmetry score
            symmetry_score = 0
            num_pairs = 0
            
            # This is simplified - a more robust approach would pair corresponding landmarks
            for left_point in left_side_points:
                for right_point in right_side_points:
                    # Compare mirrored positions (simplified)
                    mirror_x = 2 * midpoint_x - right_point.x
                    diff = abs(mirror_x - left_point.x) + abs(right_point.y - left_point.y)
                    symmetry_score += diff
                    num_pairs += 1
            
            if num_pairs > 0:
                symmetry_score /= num_pairs
                symmetry_score = symmetry_score / midpoint_x  # Normalize by face width
                
                return symmetry_score < self.base_thresholds['facial_symmetry'], symmetry_score
            else:
                return False, 1.0
        except Exception as e:
            print(f"Error checking facial symmetry: {str(e)}")
            return False, 1.0

    def check_skin_texture(self, img, face):
        """Analyze skin texture for inconsistencies"""
        try:
            # Extract face region
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_region = img[max(0, y):y+h, max(0, x):x+w]
            
            if face_region.size == 0:
                return False, 1.0
            
            # Convert to grayscale for texture analysis
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Apply Gabor filters for texture analysis
            kernel_size = 15
            theta = np.pi/4
            sigma = 5.0
            lambd = 10.0
            gamma = 0.5
            
            gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
            gabor_filtered = cv2.filter2D(gray_face, cv2.CV_8UC3, gabor_kernel)
            
            # Calculate texture uniformity (a real implementation would be more sophisticated)
            hist = cv2.calcHist([gabor_filtered], [0], None, [256], [0, 256])
            hist = hist / hist.sum()  # Normalize
            
            # Calculate entropy as a measure of texture complexity
            non_zero_hist = hist[hist > 0]
            texture_entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist))
            
            # Calculate local binary pattern for additional texture analysis
            texture_score = texture_entropy / 8.0  # Normalize (max entropy for 8-bit image)
            
            return texture_score < self.base_thresholds['skin_texture_threshold'], texture_score
        except Exception as e:
            print(f"Error checking skin texture: {str(e)}")
            return False, 1.0

    def check_frequency_distribution(self, img):
        """Analyze frequency distribution for anomalies using FFT"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply FFT
            f_transform = fft2(gray)
            f_shift = fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            # Normalize
            magnitude_spectrum_norm = magnitude_spectrum / np.max(magnitude_spectrum)
            
            # Analyze high frequency components (simplified)
            h, w = magnitude_spectrum_norm.shape
            center_y, center_x = h // 2, w // 2
            
            # Create masks for different frequency bands
            y, x = np.ogrid[:h, :w]
            low_freq_mask = ((y - center_y)**2 + (x - center_x)**2 <= (min(h, w)//8)**2)
            high_freq_mask = ((y - center_y)**2 + (x - center_x)**2 >= (min(h, w)//4)**2)
            
            # Calculate energy in different frequency bands
            total_energy = np.sum(magnitude_spectrum_norm)
            low_freq_energy = np.sum(magnitude_spectrum_norm[low_freq_mask]) / total_energy
            high_freq_energy = np.sum(magnitude_spectrum_norm[high_freq_mask]) / total_energy
            
            # High frequency energy ratio is often higher in deepfakes
            freq_score = high_freq_energy / (low_freq_energy + 1e-10)
            
            return freq_score < self.base_thresholds['frequency_threshold'], freq_score
        except Exception as e:
            print(f"Error checking frequency distribution: {str(e)}")
            return False, 1.0

    def check_noise_patterns(self, img):
        """Analyze noise patterns for inconsistencies"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply median filter to remove noise
            median_filtered = cv2.medianBlur(gray, 5)
            
            # Extract noise by subtracting filtered image
            noise = cv2.absdiff(gray, median_filtered)
            
            # Calculate statistics on noise
            mean_noise = np.mean(noise) / 255.0
            std_noise = np.std(noise) / 255.0
            
            # Deepfakes often have inconsistent noise patterns
            noise_score = std_noise
            
            return noise_score < self.base_thresholds['noise_threshold'], noise_score
        except Exception as e:
            print(f"Error checking noise patterns: {str(e)}")
            return False, 1.0

    def check_blurriness(self, img):
        """Check for unnatural blurriness or sharpness"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Very low variance indicates excessive blurriness
            # which could be a sign of deepfake, especially at face boundaries
            return blur_score > self.base_thresholds['blur_threshold'], blur_score
        except Exception as e:
            print(f"Error checking blurriness: {str(e)}")
            return False, 0

    def check_lighting_consistency(self, img, face):
        """Check if lighting is consistent across the image"""
        try:
            # Extract face region
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_region = img[max(0, y):y+h, max(0, x):x+w]
            
            if face_region.size == 0:
                return False, 1.0
                
            # Calculate background region (simplified)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask[max(0, y):y+h, max(0, x):x+w] = 255
            background_mask = cv2.bitwise_not(mask)
            
            # Check if there's enough background
            if np.sum(background_mask) > 1000:
                # Get background region
                background = cv2.bitwise_and(img, img, mask=background_mask)
                
                # Convert to HSV for better lighting analysis
                face_hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
                background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
                
                # Compare lighting (V channel in HSV)
                face_v = np.mean(face_hsv[:, :, 2][face_hsv[:, :, 2] > 0])
                bg_v = np.mean(background_hsv[:, :, 2][background_hsv[:, :, 2] > 0])
                
                lighting_diff = abs(face_v - bg_v) / 255.0
                
                return lighting_diff < self.base_thresholds['lighting_mismatch'], lighting_diff
            else:
                # Not enough background to compare
                return True, 0.0
        except Exception as e:
            print(f"Error checking lighting consistency: {str(e)}")
            return True, 0.0

    def check_edge_consistency(self, img, face):
        """Check for unnatural edges at face boundaries"""
        try:
            # Extract face region with a small margin
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            margin = 10
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(img.shape[1], x + w + margin)
            y_end = min(img.shape[0], y + h + margin)
            
            # Create a mask for the face boundary
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, thickness=margin)
            
            # Apply mask to get just the boundary
            boundary = cv2.bitwise_and(img[y_start:y_end, x_start:x_end], 
                                      img[y_start:y_end, x_start:x_end],
                                      mask=mask[y_start:y_end, x_start:x_end])
            
            if np.sum(boundary) == 0:
                return True, 0.0
                
            # Detect edges
            gray_boundary = cv2.cvtColor(boundary, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_boundary, 100, 200)
            
            # Count edge pixels
            edge_count = np.sum(edges > 0)
            
            # Normalize by perimeter
            perimeter = 2 * (w + h)
            edge_density = edge_count / perimeter if perimeter > 0 else 0
            
            # High edge density at boundary could indicate compositing artifacts
            return edge_density < 0.5, edge_density
        except Exception as e:
            print(f"Error checking edge consistency: {str(e)}")
            return True, 0.0

    def apply_color_correction(self, img):
        """Apply color correction to the image"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels
            lab = cv2.merge((l, a, b))
            
            # Convert back to BGR
            corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return corrected
        except Exception as e:
            print(f"Error in color correction: {str(e)}")
            return img

    def check_shadow_consistency(self, img, face):
        """Check for shadow consistency in the face region"""
        try:
            # Extract face region
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_region = img[max(0, y):y+h, max(0, x):x+w]
            
            if face_region.size == 0:
                return True, 0.0
            
            # Convert to LAB color space
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Calculate shadow map using L channel
            shadow_map = cv2.adaptiveThreshold(
                l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Calculate shadow consistency
            shadow_consistency = np.std(shadow_map) / 255.0
            
            return shadow_consistency < self.base_thresholds['shadow_threshold'], shadow_consistency
        except Exception as e:
            print(f"Error checking shadow consistency: {str(e)}")
            return True, 0.0

    def check_color_consistency(self, img, face):
        """Check for color consistency in the face region"""
        try:
            # Extract face region
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_region = img[max(0, y):y+h, max(0, x):x+w]
            
            if face_region.size == 0:
                return True, 0.0
            
            # Convert to HSV
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # Calculate color statistics
            h_mean = np.mean(hsv[:,:,0])
            h_std = np.std(hsv[:,:,0])
            s_mean = np.mean(hsv[:,:,1])
            s_std = np.std(hsv[:,:,1])
            
            # Calculate color consistency score
            color_score = (h_std + s_std) / 255.0
            
            return color_score < self.base_thresholds['color_consistency'], color_score
        except Exception as e:
            print(f"Error checking color consistency: {str(e)}")
            return True, 0.0

    def check_texture_consistency(self, img, face):
        """Check for consistency in texture patterns across the face"""
        try:
            # Extract face region
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_region = img[max(0, y):y+h, max(0, x):x+w]
            
            if face_region.size == 0:
                return True, 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Apply multiple Gabor filters with different orientations
            orientations = [0, 45, 90, 135]
            texture_scores = []
            
            for theta in orientations:
                kernel = cv2.getGaborKernel((15, 15), 5.0, theta * np.pi/180, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                texture_scores.append(np.std(filtered))
            
            # Calculate consistency score
            consistency_score = np.std(texture_scores) / np.mean(texture_scores)
            
            return consistency_score < self.base_thresholds['texture_consistency'], consistency_score
        except Exception as e:
            print(f"Error checking texture consistency: {str(e)}")
            return True, 0.0

    def check_edge_quality(self, img, face):
        """Analyze the quality and consistency of edges in the face region"""
        try:
            # Extract face region
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_region = img[max(0, y):y+h, max(0, x):x+w]
            
            if face_region.size == 0:
                return True, 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection with different thresholds
            edges1 = cv2.Canny(gray, 50, 150)
            edges2 = cv2.Canny(gray, 100, 200)
            
            # Calculate edge consistency
            edge_ratio = np.sum(edges1) / (np.sum(edges2) + 1e-6)
            
            # Calculate edge quality score
            quality_score = 1.0 - abs(edge_ratio - 1.0)
            
            return quality_score > self.base_thresholds['edge_quality'], quality_score
        except Exception as e:
            print(f"Error checking edge quality: {str(e)}")
            return True, 0.0

    def check_color_balance(self, img, face):
        """Analyze color balance and distribution in the face region"""
        try:
            # Extract face region
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_region = img[max(0, y):y+h, max(0, x):x+w]
            
            if face_region.size == 0:
                return True, 0.0
            
            # Convert to LAB color space
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Calculate color balance scores
            l_balance = np.std(l) / (np.mean(l) + 1e-6)
            a_balance = np.std(a) / (np.mean(a) + 1e-6)
            b_balance = np.std(b) / (np.mean(b) + 1e-6)
            
            # Combine scores
            balance_score = (l_balance + a_balance + b_balance) / 3.0
            
            return balance_score < self.base_thresholds['color_balance'], balance_score
        except Exception as e:
            print(f"Error checking color balance: {str(e)}")
            return True, 0.0


class DeepfakeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepfake Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize detector
        self.detector = DeepfakeDetector()
        
        # Set up UI
        self.setup_ui()
        
        # Current image data
        self.current_image_path = None
        self.current_image = None
        self.current_result = None
        
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for image display
        self.left_panel = ttk.Frame(main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel for controls and results
        self.right_panel = ttk.Frame(main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # Image display area
        self.image_label = ttk.Label(self.left_panel)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        self.load_button = ttk.Button(self.right_panel, text="Load Image", command=self.load_image)
        self.load_button.pack(fill=tk.X, pady=5)
        
        self.analyze_button = ttk.Button(self.right_panel, text="Analyze Image", command=self.analyze_image)
        self.analyze_button.pack(fill=tk.X, pady=5)
        self.analyze_button.config(state=tk.DISABLED)
        
        # Results section
        ttk.Separator(self.right_panel).pack(fill=tk.X, pady=10)
        
        ttk.Label(self.right_panel, text="Analysis Results", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.result_frame = ttk.LabelFrame(self.right_panel, text="Detection Results")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Results indicators
        self.result_label = ttk.Label(self.result_frame, text="No image analyzed", font=("Arial", 10))
        self.result_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(self.result_frame, text="")
        self.confidence_label.pack(pady=5)
        
        self.reason_label = ttk.Label(self.result_frame, text="", wraplength=250)
        self.reason_label.pack(pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.right_panel, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=5)
        
        # Details section
        self.details_frame = ttk.LabelFrame(self.right_panel, text="Analysis Details")
        self.details_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a canvas for scrollable content
        self.canvas = tk.Canvas(self.details_frame)
        self.scrollbar = ttk.Scrollbar(self.details_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
    def load_image(self):
        """Load an image file for analysis"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if file_path:
            try:
                # Remember the path
                self.current_image_path = file_path
                
                # Load and display the image
                img = Image.open(file_path)
                
                # Resize for display
                img.thumbnail((400, 400))
                
                # Update display
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo)
                self.image_label.image = photo  # Keep a reference
                
                # Enable analyze button
                self.analyze_button.config(state=tk.NORMAL)
                
                # Clear previous results
                self.result_label.config(text="Ready for analysis")
                self.confidence_label.config(text="")
                self.reason_label.config(text="")
                
                # Clear details
                for widget in self.scrollable_frame.winfo_children():
                    widget.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def analyze_image(self):
        """Analyze the loaded image for deepfake indicators"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        # Disable buttons during analysis
        self.analyze_button.config(state=tk.DISABLED)
        self.load_button.config(state=tk.DISABLED)
        
        # Show progress
        self.progress_var.set(0)
        
        # Run analysis in a separate thread to keep UI responsive
        threading.Thread(target=self._run_analysis, daemon=True).start()
    
    def _run_analysis(self):
        """Run analysis in background"""
        try:
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(10))
            
            # Perform analysis
            result = self.detector.analyze_image(self.current_image_path)
            self.current_result = result
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(90))
            
            # Update UI with results
            self.root.after(0, self._update_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.root.after(0, self._reset_ui)
    
    def _update_results(self):
        """Update UI with analysis results"""
        result = self.current_result
        
        if result["is_real"] is None:
            self.result_label.config(text=f"Error: {result['reason']}", foreground="red")
        else:
            # Update main result
            if result["is_real"]:
                self.result_label.config(text="REAL IMAGE DETECTED", foreground="green", font=("Arial", 12, "bold"))
            else:
                self.result_label.config(text="DEEPFAKE DETECTED", foreground="red", font=("Arial", 12, "bold"))
            
            # Update confidence
            confidence_pct = int(result["confidence"] * 100)
            self.confidence_label.config(text=f"Confidence: {confidence_pct}%")
            
            # Update reason
            self.reason_label.config(text=f"Reason: {result['reason']}")
            
            # Clear previous details
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            
            # Add voting information
            vote_frame = ttk.LabelFrame(self.scrollable_frame, text="Voting Results")
            vote_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
            
            real_votes = result["vote_count"]["real"]
            fake_votes = result["vote_count"]["fake"]
            total_votes = real_votes + fake_votes
            
            ttk.Label(vote_frame, text=f"Real votes: {real_votes}/{total_votes}").grid(row=0, column=0, padx=5, pady=2)
            ttk.Label(vote_frame, text=f"Fake votes: {fake_votes}/{total_votes}").grid(row=0, column=1, padx=5, pady=2)
            
            # Add individual results
            for i, individual_result in enumerate(result["individual_results"]):
                result_frame = ttk.LabelFrame(self.scrollable_frame, text=f"Analysis {i+1}")
                result_frame.grid(row=i+1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
                
                # Show thresholds used
                thresholds = individual_result["thresholds_used"]
                threshold_text = "Thresholds used:\n"
                for param, value in thresholds.items():
                    threshold_text += f"{param}: {value:.3f}\n"
                
                ttk.Label(result_frame, text=threshold_text).grid(row=0, column=0, padx=5, pady=2)
                
                # Show result
                result_text = "REAL" if individual_result["is_real"] else "FAKE"
                result_color = "green" if individual_result["is_real"] else "red"
                ttk.Label(result_frame, text=f"Result: {result_text}", foreground=result_color).grid(row=0, column=1, padx=5, pady=2)
                
                # Show confidence
                conf_pct = int(individual_result["confidence"] * 100)
                ttk.Label(result_frame, text=f"Confidence: {conf_pct}%").grid(row=0, column=2, padx=5, pady=2)
            
            # Add separator
            ttk.Separator(self.scrollable_frame, orient="horizontal").grid(
                row=len(result["individual_results"])+1, column=0, columnspan=3, sticky="ew", padx=5, pady=5
            )
            
            # Add final detailed results
            detail_frame = ttk.LabelFrame(self.scrollable_frame, text="Final Detailed Results")
            detail_frame.grid(row=len(result["individual_results"])+2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
            
            # Add header row
            ttk.Label(detail_frame, text="Check", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(detail_frame, text="Result", font=("Arial", 9, "bold")).grid(row=0, column=1, sticky="w", padx=5, pady=2)
            ttk.Label(detail_frame, text="Score", font=("Arial", 9, "bold")).grid(row=0, column=2, sticky="w", padx=5, pady=2)
            
            # Display all check results
            row = 1
            for check_name, check_data in result["details"].items():
                display_name = check_name.replace("_", " ").title()
                
                ttk.Label(detail_frame, text=display_name).grid(row=row, column=0, sticky="w", padx=5, pady=2)
                
                if check_data["passed"]:
                    result_text = "Passed"
                    fg_color = "green"
                else:
                    result_text = "Failed"
                    fg_color = "red"
                
                result_label = ttk.Label(detail_frame, text=result_text, foreground=fg_color)
                result_label.grid(row=row, column=1, sticky="w", padx=5, pady=2)
                
                # Score with formatting
                score_text = f"{check_data['score']:.3f}"
                ttk.Label(detail_frame, text=score_text).grid(row=row, column=2, sticky="w", padx=5, pady=2)
                
                row += 1
            
            # Add extra info
            ttk.Label(detail_frame, text=f"Faces detected: {result['faces_found']}").grid(
                row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2
            )
        
        # Complete progress
        self.progress_var.set(100)
        
        # Re-enable buttons
        self._reset_ui()
    
    def create_score_chart(self, parent_frame, detail_items):
        """Create a bar chart visualizing the scores"""
        # Create figure
        fig = Figure(figsize=(4, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        # Prepare data
        checks = []
        scores = []
        colors = []
        
        for check_name, check_data in detail_items.items():
            display_name = check_name.replace("_", " ").title()[:10]  # Truncate for display
            checks.append(display_name)
            scores.append(check_data["score"])
            colors.append("green" if check_data["passed"] else "red")
        
        # Create horizontal bar chart
        y_pos = np.arange(len(checks))
        bars = ax.barh(y_pos, scores, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(checks)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Score')
        ax.set_title('Detection Scores by Check')
        
        # Add threshold lines if applicable
        for i, (check_name, check_data) in enumerate(detail_items.items()):
            if "threshold" in check_data:
                ax.axvline(x=check_data["threshold"], ymin=(i/len(checks)), 
                          ymax=((i+1)/len(checks)), color='black', linestyle='--', alpha=0.3)
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas for matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _reset_ui(self):
        """Reset UI elements after analysis"""
        self.analyze_button.config(state=tk.NORMAL)
        self.load_button.config(state=tk.NORMAL)


def dist(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def main():
    root = tk.Tk()
    app = DeepfakeDetectorApp(root)
    
    # Show initial instructions
    messagebox.showinfo(
        "Deepfake Detection System",
        "Welcome to the Deepfake Detection System!\n\n"
        "1. Click 'Load Image' to select an image for analysis\n"
        "2. Click 'Analyze Image' to detect deepfake indicators\n\n"
        "Note: For optimal results, ensure the image contains a clear facial view."
    )
    
    root.mainloop()


if __name__ == "__main__":
    main()