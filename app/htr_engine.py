# enhanced_htr_engine.py
import torch
from PIL import Image, ImageDraw
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os
import cv2
import numpy as np
from huggingface_hub import login

class EnhancedHTREngine:
    """
    Enhanced Handwritten Text Recognition Engine with document-level processing.
    Handles paragraphs, pages, and complete documents by line segmentation.
    """
    
    def __init__(self, hf_token=None):
        # Initialize model and processor caches
        self.models = {}
        self.processors = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Print device information
        print(f"Using device: {self.device}")
        
        # Handle Hugging Face authentication
        self.authenticate_huggingface(hf_token)
        
    def authenticate_huggingface(self, token=None):
        """Authenticate with Hugging Face Hub."""
        if token:
            login(token=token)
            return True
            
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
            return True
            
        print("WARNING: No Hugging Face token provided. Some models may not be accessible.")
        return False
        
    def load_model(self, model_name="microsoft/trocr-base-handwritten"):
        """Load a TrOCR model and processor."""
        if model_name not in self.models:
            print(f"Loading model: {model_name}")
            
            try:
                processor = TrOCRProcessor.from_pretrained(model_name)
                model = VisionEncoderDecoderModel.from_pretrained(model_name)
                model.to(self.device)
                
                self.models[model_name] = model
                self.processors[model_name] = processor
                
                print(f"Model {model_name} loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
                
        return self.models[model_name], self.processors[model_name]
    
    def segment_text_lines(self, image, padding=5):
        """
        Segment text lines from a document image.
        
        Args:
            image (PIL.Image): Input document image
            padding (int): Padding around each line segment
            
        Returns:
            list: List of (line_image, bbox) tuples
        """
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply binary thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find horizontal projection profile
        horizontal_proj = np.sum(binary, axis=1)
        
        # Find line boundaries
        line_boundaries = []
        in_line = False
        line_start = 0
        
        # Use a threshold to determine what constitutes a line
        threshold = np.max(horizontal_proj) * 0.1
        
        for i, pixel_count in enumerate(horizontal_proj):
            if pixel_count > threshold and not in_line:
                # Start of a line
                line_start = i
                in_line = True
            elif pixel_count <= threshold and in_line:
                # End of a line
                line_boundaries.append((line_start, i))
                in_line = False
        
        # Handle case where document ends with a line
        if in_line:
            line_boundaries.append((line_start, len(horizontal_proj)))
        
        # Extract line images with padding
        line_data = []
        for start, end in line_boundaries:
            # Add padding
            start_padded = max(0, start - padding)
            end_padded = min(gray.shape[0], end + padding)
            
            # Extract line region
            line_img = gray[start_padded:end_padded, :]
            
            # Convert back to PIL
            line_pil = Image.fromarray(line_img)
            
            # Create bounding box (x, y, width, height)
            bbox = (0, start_padded, gray.shape[1], end_padded - start_padded)
            
            line_data.append((line_pil, bbox))
        
        return line_data
    
    def segment_paragraphs(self, image, min_gap=30):
        """
        Segment paragraphs by detecting larger gaps between text blocks.
        
        Args:
            image (PIL.Image): Input document image
            min_gap (int): Minimum gap size to consider a paragraph break
            
        Returns:
            list: List of paragraph images
        """
        # Get line segments first
        line_data = self.segment_text_lines(image)
        
        if not line_data:
            return []
        
        # Group lines into paragraphs based on gaps
        paragraphs = []
        current_paragraph_lines = []
        
        for i, (line_img, bbox) in enumerate(line_data):
            current_paragraph_lines.append((line_img, bbox))
            
            # Check gap to next line
            if i < len(line_data) - 1:
                current_bottom = bbox[1] + bbox[3]
                next_top = line_data[i + 1][1][1]
                gap = next_top - current_bottom
                
                # If gap is large enough, end current paragraph
                if gap > min_gap:
                    paragraphs.append(current_paragraph_lines)
                    current_paragraph_lines = []
            else:
                # Last line, add remaining paragraph
                paragraphs.append(current_paragraph_lines)
        
        # Convert paragraph line groups back to paragraph images
        paragraph_images = []
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        for paragraph_lines in paragraphs:
            if paragraph_lines:
                # Find paragraph bounds
                top = min(bbox[1] for _, bbox in paragraph_lines)
                bottom = max(bbox[1] + bbox[3] for _, bbox in paragraph_lines)
                
                # Extract paragraph image
                para_img = gray[top:bottom, :]
                paragraph_images.append(Image.fromarray(para_img))
        
        return paragraph_images
    
    def recognize_text_line(self, line_image, model_name="microsoft/trocr-base-handwritten"):
        """
        Recognize text from a single line image.
        
        Args:
            line_image (PIL.Image): Single line image
            model_name (str): TrOCR model to use
            
        Returns:
            tuple: (text, confidence)
        """
        model, processor = self.load_model(model_name)
        
        # Ensure image is in RGB format
        if line_image.mode != "RGB":
            line_image = line_image.convert("RGB")
        
        # Process the line
        pixel_values = processor(images=line_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Generate text
        output = model.generate(
            pixel_values,
            max_length=100,
            num_beams=5,
            early_stopping=True,
            temperature=1.0,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        # Extract text and confidence
        generated_ids = output.sequences
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Calculate confidence
        if hasattr(output, 'scores') and output.scores:
            token_scores = output.scores
            mean_score = torch.mean(torch.stack([torch.max(score, dim=1).values[0] for score in token_scores]))
            confidence = mean_score.item()
        else:
            confidence = 0.5  # Default confidence if scores unavailable
        
        return text, confidence
    
    def recognize_document(self, image, model_name="microsoft/trocr-base-handwritten", 
                          processing_mode="lines", combine_lines=True):
        """
        Recognize text from an entire document.
        
        Args:
            image (PIL.Image): Document image
            model_name (str): TrOCR model to use
            processing_mode (str): "lines" or "paragraphs"
            combine_lines (bool): Whether to combine line results into full text
            
        Returns:
            dict: Results containing full text and detailed line/paragraph information
        """
        results = {
            "full_text": "",
            "lines": [],
            "paragraphs": [],
            "overall_confidence": 0.0,
            "processing_mode": processing_mode
        }
        
        if processing_mode == "lines":
            # Process line by line
            line_data = self.segment_text_lines(image)
            
            all_texts = []
            all_confidences = []
            
            for line_img, bbox in line_data:
                text, confidence = self.recognize_text_line(line_img, model_name)
                
                line_result = {
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox
                }
                results["lines"].append(line_result)
                
                if text.strip():  # Only include non-empty text
                    all_texts.append(text)
                    all_confidences.append(confidence)
            
            # Combine results
            if combine_lines and all_texts:
                results["full_text"] = "\n".join(all_texts)
                results["overall_confidence"] = np.mean(all_confidences)
            
        elif processing_mode == "paragraphs":
            # Process paragraph by paragraph
            paragraph_images = self.segment_paragraphs(image)
            
            all_paragraph_texts = []
            all_confidences = []
            
            for para_img in paragraph_images:
                # Process each line in the paragraph
                line_data = self.segment_text_lines(para_img)
                paragraph_texts = []
                paragraph_confidences = []
                
                for line_img, bbox in line_data:
                    text, confidence = self.recognize_text_line(line_img, model_name)
                    
                    if text.strip():
                        paragraph_texts.append(text)
                        paragraph_confidences.append(confidence)
                
                # Combine paragraph
                if paragraph_texts:
                    para_text = " ".join(paragraph_texts)
                    para_confidence = np.mean(paragraph_confidences)
                    
                    results["paragraphs"].append({
                        "text": para_text,
                        "confidence": para_confidence,
                        "line_count": len(paragraph_texts)
                    })
                    
                    all_paragraph_texts.append(para_text)
                    all_confidences.append(para_confidence)
            
            # Combine all paragraphs
            if all_paragraph_texts:
                results["full_text"] = "\n\n".join(all_paragraph_texts)
                results["overall_confidence"] = np.mean(all_confidences)
        
        return results
    
    def visualize_segmentation(self, image, processing_mode="lines"):
        """
        Create a visualization showing the line/paragraph segmentation.
        
        Args:
            image (PIL.Image): Input image
            processing_mode (str): "lines" or "paragraphs"
            
        Returns:
            PIL.Image: Image with segmentation overlay
        """
        # Create a copy for drawing
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        if processing_mode == "lines":
            line_data = self.segment_text_lines(image)
            
            # Draw bounding boxes around lines
            for i, (_, bbox) in enumerate(line_data):
                x, y, w, h = bbox
                draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                draw.text((x, y - 15), f"Line {i+1}", fill="red")
                
        elif processing_mode == "paragraphs":
            paragraph_images = self.segment_paragraphs(image)
            
            # Get paragraph bounds and draw
            img_array = np.array(image)
            current_y = 0
            
            for i, para_img in enumerate(paragraph_images):
                para_height = para_img.height
                draw.rectangle([0, current_y, image.width, current_y + para_height], 
                             outline="blue", width=3)
                draw.text((10, current_y + 5), f"Paragraph {i+1}", fill="blue")
                current_y += para_height
        
        return vis_image
    
    def get_available_models(self):
        """Get a list of available TrOCR models."""
        return [
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-large-handwritten",
            "microsoft/trocr-small-handwritten",
        ]