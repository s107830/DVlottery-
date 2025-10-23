import io
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("⚠️  rembg not installed. Please install: pip install rembg")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("⚠️  opencv not installed. Please install: pip install opencv-python")

class BackgroundRemovalTester:
    def __init__(self):
        self.models = {
            "u2net": "u2net",                      # Default
            "u2netp": "u2netp",                    # Lightweight
            "u2net_human_seg": "u2net_human_seg",  # Specialized for humans
            "isnet-general-use": "isnet-general-use",  # Best for general use
            "isnet-anime": "isnet-anime",          # For anime/cartoon
        }
        
    def check_dependencies(self):
        """Check if required dependencies are available"""
        if not REMBG_AVAILABLE:
            print("❌ rembg is required for background removal")
            return False
        return True
    
    def load_image(self, image_path):
        """Load and validate image"""
        try:
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return None
            
            img = Image.open(image_path)
            print(f"✅ Image loaded: {img.size[0]}x{img.size[1]} | Format: {img.format}")
            return img.convert("RGB")
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    
    def remove_bg_original(self, img):
        """Your original function"""
        try:
            b = io.BytesIO()
            img.save(b, format="PNG")
            fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
            white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
            return Image.alpha_composite(white, fg).convert("RGB")
        except Exception as e:
            print(f"❌ Original method failed: {e}")
            return img
    
    def remove_bg_high_res(self, img):
        """Higher resolution approach"""
        try:
            original_size = img.size
            print(f"📐 Original size: {original_size}")
            
            # Upscale for better detail capture (if image is small)
            if max(img.size) < 1200:
                scale_factor = 2
                new_size = (img.size[0] * scale_factor, img.size[1] * scale_factor)
                img_high_res = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"🔼 Upscaled to: {new_size}")
            else:
                img_high_res = img
                print("ℹ️  Using original resolution (already large)")
            
            b = io.BytesIO()
            img_high_res.save(b, format="PNG", quality=100)
            fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
            
            # Resize back to original if we upscaled
            if max(img.size) < 1200:
                fg = fg.resize(original_size, Image.Resampling.LANCZOS)
                print("🔽 Resized back to original")
            
            white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
            result = Image.alpha_composite(white, fg).convert("RGB")
            print("✅ High-res method completed")
            return result
        except Exception as e:
            print(f"❌ High-res method failed: {e}")
            return img
    
    def remove_bg_different_model(self, img, model_name="isnet-general-use"):
        """Try different AI models"""
        try:
            if model_name not in self.models:
                print(f"❌ Model {model_name} not found")
                return img
            
            print(f"🤖 Using model: {model_name}")
            session = new_session(self.models[model_name])
            
            b = io.BytesIO()
            img.save(b, format="PNG")
            fg = Image.open(io.BytesIO(remove(b.getvalue(), session=session))).convert("RGBA")
            white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
            result = Image.alpha_composite(white, fg).convert("RGB")
            print(f"✅ Model {model_name} completed")
            return result
        except Exception as e:
            print(f"❌ Model {model_name} failed: {e}")
            return img
    
    def sharpen_edges(self, image):
        """Post-processing edge sharpening"""
        if not OPENCV_AVAILABLE:
            print("⚠️  OpenCV not available for edge sharpening")
            return image
            
        try:
            img_array = np.array(image)
            
            # Multiple sharpening techniques
            # Method 1: Sharpening kernel
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            sharpened = cv2.filter2D(img_array, -1, kernel)
            
            # Method 2: Unsharp masking (optional - more aggressive)
            # gaussian = cv2.GaussianBlur(sharpened, (0, 0), 2.0)
            # sharpened = cv2.addWeighted(sharpened, 1.5, gaussian, -0.5, 0)
            
            print("✅ Edge sharpening applied")
            return Image.fromarray(sharpened)
        except Exception as e:
            print(f"❌ Edge sharpening failed: {e}")
            return image
    
    def remove_bg_with_edge_enhancement(self, img):
        """Combined approach with edge enhancement"""
        try:
            print("🚀 Starting high-res + edge enhancement method")
            result = self.remove_bg_high_res(img)
            result_sharp = self.sharpen_edges(result)
            print("✅ Edge enhancement completed")
            return result_sharp
        except Exception as e:
            print(f"❌ Edge enhancement method failed: {e}")
            return img
    
    def remove_bg_advanced_hair(self, img):
        """Specialized method for hair details"""
        try:
            print("💇 Advanced hair detail method started")
            
            # Use the best model for human subjects
            session = new_session("isnet-general-use")
            
            # Process at high resolution
            original_size = img.size
            if max(img.size) < 1500:
                scale_factor = 1.5
                new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                img_processed = img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                img_processed = img
            
            b = io.BytesIO()
            img_processed.save(b, format="PNG", quality=100)
            fg = Image.open(io.BytesIO(remove(b.getvalue(), session=session))).convert("RGBA")
            
            # Resize back if needed
            if max(img.size) < 1500:
                fg = fg.resize(original_size, Image.Resampling.LANCZOS)
            
            white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
            result = Image.alpha_composite(white, fg).convert("RGB")
            
            # Apply gentle sharpening
            if OPENCV_AVAILABLE:
                result = self.sharpen_edges(result)
            
            print("✅ Advanced hair method completed")
            return result
        except Exception as e:
            print(f"❌ Advanced hair method failed: {e}")
            return img
    
    def create_comparison_grid(self, results_dict, save_path="bg_removal_comparison.jpg"):
        """Create a grid to compare all results"""
        try:
            if not results_dict:
                print("❌ No results to compare")
                return None
            
            print("🖼️ Creating comparison grid...")
            
            # Calculate grid size
            n_methods = len(results_dict)
            grid_cols = min(3, n_methods)
            grid_rows = (n_methods + grid_cols - 1) // grid_cols
            
            # Get image size from first result
            sample_img = list(results_dict.values())[0]
            img_width, img_height = sample_img.size
            
            # Create grid with some spacing
            spacing = 10
            label_height = 40
            grid_width = img_width * grid_cols + spacing * (grid_cols + 1)
            grid_height = (img_height + label_height) * grid_rows + spacing * (grid_rows + 1)
            
            grid = Image.new('RGB', (grid_width, grid_height), 'lightgray')
            draw = ImageDraw.Draw(grid)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
            
            # Paste images in grid
            for idx, (method_name, img) in enumerate(results_dict.items()):
                row = idx // grid_cols
                col = idx % grid_cols
                
                # Calculate position
                x = col * (img_width + spacing) + spacing
                y = row * (img_height + label_height + spacing) + spacing
                
                # Resize if needed (should be same size)
                if img.size != (img_width, img_height):
                    img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                
                # Paste image
                grid.paste(img, (x, y + label_height))
                
                # Add label background
                label_bg = Image.new('RGB', (img_width, label_height), 'darkblue')
                grid.paste(label_bg, (x, y))
                
                # Add method name
                text_bbox = draw.textbbox((0, 0), method_name, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = x + (img_width - text_width) // 2
                text_y = y + (label_height - (text_bbox[3] - text_bbox[1])) // 2
                
                draw.text((text_x, text_y), method_name, fill='white', font=font)
            
            # Save comparison
            grid.save(save_path, quality=95)
            print(f"✅ Comparison grid saved as: {save_path}")
            return grid
            
        except Exception as e:
            print(f"❌ Failed to create comparison grid: {e}")
            return None
    
    def save_individual_results(self, results_dict, base_name="result"):
        """Save individual result images"""
        try:
            os.makedirs("results", exist_ok=True)
            saved_paths = []
            
            for method_name, img in results_dict.items():
                # Clean filename
                clean_name = "".join(c for c in method_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"results/{base_name}_{clean_name}.jpg"
                img.save(filename, quality=95)
                saved_paths.append(filename)
                print(f"💾 Saved: {filename}")
            
            return saved_paths
        except Exception as e:
            print(f"❌ Failed to save individual results: {e}")
            return []
    
    def compare_all_methods(self, image_path):
        """Test all methods side by side"""
        if not self.check_dependencies():
            return None
        
        original_img = self.load_image(image_path)
        if not original_img:
            return None
        
        print("\n" + "="*60)
        print("🚀 STARTING BACKGROUND REMOVAL COMPARISON")
        print("="*60)
        
        results = {
            "01_Original": original_img,
            "02_Original_Method": self.remove_bg_original(original_img.copy()),
        }
        
        # Test different approaches
        methods_to_test = [
            ("03_High_Resolution", self.remove_bg_high_res),
            ("04_Edge_Enhancement", self.remove_bg_with_edge_enhancement),
            ("05_Advanced_Hair", self.remove_bg_advanced_hair),
        ]
        
        for method_name, method_func in methods_to_test:
            print(f"\n--- Testing {method_name} ---")
            results[method_name] = method_func(original_img.copy())
        
        # Test different models (skip if taking too long)
        print(f"\n--- Testing AI Models ---")
        quick_models = ["isnet-general-use", "u2net_human_seg", "u2net"]
        
        for model_name in quick_models:
            method_name = f"06_Model_{model_name}"
            print(f"Testing {method_name}...")
            results[method_name] = self.remove_bg_different_model(original_img.copy(), model_name)
        
        print("\n" + "="*60)
        print("✅ ALL METHODS COMPLETED")
        print("="*60)
        
        return results
    
    def generate_report(self, results):
        """Generate a simple text report"""
        print("\n📊 RESULTS SUMMARY:")
        print("-" * 40)
        
        for method_name, img in results.items():
            if img:
                size_info = f"{img.size[0]}x{img.size[1]}"
                print(f"✓ {method_name:<25} {size_info:>15}")
        
        print("\n💡 RECOMMENDATIONS:")
        print("1. Check '05_Advanced_Hair' for best hair details")
        print("2. '06_Model_isnet-general-use' usually works well for humans")
        print("3. Compare images side-by-side in the generated grid")
        print("4. Look for the method with cleanest hair edges")

def main():
    """Main function to run the complete test"""
    if len(sys.argv) < 2:
        print("Usage: python bg_removal_test.py <image_path>")
        print("Example: python bg_removal_test.py my_photo.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Initialize tester
    tester = BackgroundRemovalTester()
    
    # Run complete comparison
    results = tester.compare_all_methods(image_path)
    
    if results:
        # Create base name from input file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save individual results
        tester.save_individual_results(results, base_name)
        
        # Create comparison grid
        comparison_path = f"COMPARISON_{base_name}.jpg"
        tester.create_comparison_grid(results, comparison_path)
        
        # Generate report
        tester.generate_report(results)
        
        print(f"\n🎉 COMPLETE! Check these files:")
        print(f"   • {comparison_path} - Side-by-side comparison")
        print(f"   • /results/ folder - Individual result images")
        
    else:
        print("❌ Testing failed. Please check the errors above.")

# Quick test function for easy use
def quick_test(image_path, method="advanced_hair"):
    """Quick test with a single method"""
    tester = BackgroundRemovalTester()
    
    if not tester.check_dependencies():
        return None
    
    img = tester.load_image(image_path)
    if not img:
        return None
    
    method_map = {
        "original": tester.remove_bg_original,
        "high_res": tester.remove_bg_high_res,
        "advanced_hair": tester.remove_bg_advanced_hair,
        "isnet": lambda x: tester.remove_bg_different_model(x, "isnet-general-use"),
    }
    
    if method in method_map:
        result = method_map[method](img)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"QUICK_RESULT_{base_name}.jpg"
        result.save(output_path, quality=95)
        print(f"✅ Quick result saved: {output_path}")
        return result
    else:
        print(f"❌ Unknown method: {method}")
        return None

if __name__ == "__main__":
    main()
