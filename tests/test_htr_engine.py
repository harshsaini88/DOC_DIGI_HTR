# test_htr_engine.py
import unittest
import os
import sys
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.htr_engine import HTREngine
from app.image_preprocessing import preprocess_image, deskew_image

class TestHTREngine(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.htr_engine = HTREngine()
        
        # Create a simple test image with text
        self.test_img = Image.new('RGB', (400, 100), color=(255, 255, 255))
        
        # Save the test image temporarily
        self.test_img_path = "/tmp/test_htr_image.png"
        self.test_img.save(self.test_img_path)
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the test image
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)
    
    def test_model_loading(self):
        """Test that models can be loaded."""
        model_name = "microsoft/trocr-base-handwritten"
        model, processor = self.htr_engine.load_model(model_name)
        
        # Check that model and processor are loaded and cached
        self.assertIn(model_name, self.htr_engine.models)
        self.assertIn(model_name, self.htr_engine.processors)
        
        # Check that the same instances are returned on second call
        model2, processor2 = self.htr_engine.load_model(model_name)
        self.assertIs(model, model2)
        self.assertIs(processor, processor2)
    
    def test_available_models(self):
        """Test that available models are returned."""
        models = self.htr_engine.get_available_models()
        self.assertIsInstance(models, list)
        self.assertTrue(len(models) > 0)
        
        # Base model should be in the list
        self.assertIn("microsoft/trocr-base-handwritten", models)
    
    def test_preprocessing_functions(self):
        """Test image preprocessing functions."""
        # Test basic preprocessing
        processed_img = preprocess_image(self.test_img)
        self.assertIsInstance(processed_img, Image.Image)
        
        # Convert to numpy array for OpenCV functions
        img_array = np.array(self.test_img.convert('L'))
        
        # Test deskew function
        deskewed = deskew_image(img_array)
        self.assertEqual(deskewed.shape, img_array.shape)
        
if __name__ == '__main__':
    unittest.main()