# Advanced Deepfake Detection System

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![GUI](https://img.shields.io/badge/interface-tkinter-orange.svg)

A comprehensive deepfake detection system with advanced computer vision algorithms and an intuitive GUI interface. This tool uses multiple detection techniques to analyze images for signs of synthetic generation or manipulation, optimized to minimize false positives while maintaining high detection accuracy.

## 🚀 Features

### Core Detection Algorithms
- **Face Detection & Quality Assessment** - Analyzes facial features, sharpness, and symmetry
- **Frequency Domain Analysis** - FFT-based detection of unusual frequency patterns
- **Texture & Edge Consistency** - Local Binary Pattern (LBP) analysis and edge detection
- **Color Distribution Analysis** - RGB/HSV color space anomaly detection
- **Compression Artifacts Detection** - DCT coefficient analysis and blocking artifacts
- **Lighting & Shadow Consistency** - Illumination gradient and shadow pattern analysis
- **Pixel-Level Consistency** - Microscopic pixel deviation detection
- **Noise Pattern Analysis** - Synthetic noise signature identification
- **Upsampling Artifact Detection** - Detection of resolution manipulation artifacts

### User Interface
- **Modern GUI** - Clean, professional interface with dark theme
- **Real-time Progress** - Live analysis progress with detailed step-by-step results
- **Image Preview** - Integrated image display with automatic scaling
- **Detailed Reports** - Comprehensive analysis results with technical metrics
- **Threaded Processing** - Non-blocking analysis with responsive UI

### Advanced Features
- **Multi-Algorithm Consensus** - Requires multiple indicators for positive detection
- **False Positive Minimization** - Optimized thresholds to reduce incorrect classifications
- **Phone Camera Optimization** - Tuned for modern smartphone camera images
- **High-Resolution Support** - Handles large images efficiently
- **Processing Artifact Differentiation** - Distinguishes between manipulation and compression

## 📋 Requirements

### System Requirements
- Python 3.7 or higher
- Windows, macOS, or Linux
- Minimum 4GB RAM (8GB recommended)
- 1GB free disk space

### Dependencies
```
opencv-python>=4.5.0
numpy>=1.19.0
Pillow>=8.0.0
scipy>=1.6.0
scikit-image>=0.18.0
tkinter (included with Python)
```

## 🛠️ Installation

### Method 1: Clone and Install
```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
python deeptry.py
```

### Method 2: Direct Download
1. Download the `deeptry.py` file
2. Install dependencies: `pip install opencv-python numpy Pillow scipy scikit-image`
3. Run: `python deeptry.py`

### Create requirements.txt
```txt
opencv-python>=4.5.0
numpy>=1.19.0
Pillow>=8.0.0
scipy>=1.6.0
scikit-image>=0.18.0
```

## 🎯 Usage

### Basic Usage
1. Launch the application: `python deeptry.py`
2. Click "Select Image" to choose an image file
3. Click "Analyze Image" to start detection
4. Review the detailed results in the right panel

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

### Understanding Results

#### Risk Levels
- **🟢 MINIMAL RISK** - Authentic (Suspicion Score: 0-17)
- **🟠 LOW RISK** - Minor anomalies (Suspicion Score: 18-29)
- **🟡 MEDIUM RISK** - Possible manipulation (Suspicion Score: 30-44)
- **🔴 HIGH RISK** - Likely deepfake (Suspicion Score: 45+)

#### Key Metrics
- **Authenticity Score** - Overall confidence in image authenticity (0-100)
- **Suspicion Score** - Cumulative anomaly indicators (0-100+)
- **Risk Level** - Final classification category

## 🔬 Technical Details

### Detection Algorithms

#### 1. Face Detection & Quality Assessment
- **Technology**: OpenCV Haar Cascades
- **Metrics**: 
  - Laplacian sharpness variance
  - Facial symmetry correlation
  - Face count and positioning
- **Thresholds**: 
  - Low sharpness: < 50
  - High symmetry: > 0.95

#### 2. Frequency Domain Analysis
- **Technology**: Fast Fourier Transform (FFT)
- **Metrics**:
  - Low frequency energy ratio
  - High frequency energy ratio
  - Frequency variation coefficient
- **Thresholds**:
  - High low-freq: > 0.8
  - High high-freq: > 0.6

#### 3. Texture & Edge Analysis
- **Technology**: Local Binary Patterns, Canny Edge Detection
- **Metrics**:
  - LBP texture uniformity
  - Edge density
  - Gradient magnitude variation
- **Thresholds**:
  - High uniformity: > 0.15
  - Low edge density: < 0.01

#### 4. Color Distribution Analysis
- **Technology**: RGB/HSV color space analysis
- **Metrics**:
  - Color channel balance ratio
  - Saturation statistics
  - Color range distribution
- **Thresholds**:
  - Color imbalance: > 2.5
  - Low saturation: < 50

#### 5. Compression Artifacts
- **Technology**: Discrete Cosine Transform (DCT)
- **Metrics**:
  - DCT coefficient energy ratio
  - 8x8 blocking artifacts score
  - Multiple compression indicators
- **Thresholds**:
  - Heavy compression: < 0.1
  - Blocking artifacts: > 0.05

#### 6. Lighting & Shadow Analysis
- **Technology**: LAB color space, Gaussian filtering
- **Metrics**:
  - Lighting gradient magnitude
  - Shadow region properties
  - Illumination consistency
- **Thresholds**:
  - Inconsistent lighting: > 5.0
  - Unusual shadows: > 0.9 eccentricity

#### 7. Pixel-Level Consistency
- **Technology**: Local pixel deviation analysis
- **Metrics**:
  - Pixel deviation from local average
  - Spatial consistency patterns
- **Thresholds**:
  - High deviation: > 15

#### 8. Noise Pattern Analysis
- **Technology**: Gaussian noise isolation, FFT analysis
- **Metrics**:
  - Noise standard deviation
  - Noise periodicity score
  - Frequency domain noise patterns
- **Thresholds**:
  - Unusual periodicity: > 2.0

#### 9. Upsampling Artifact Detection
- **Technology**: Multi-scale image comparison
- **Metrics**:
  - Downsampling-upsampling difference
  - Resolution inconsistency patterns
- **Thresholds**:
  - Significant artifacts: > 8

### Scoring System

#### Suspicion Score Calculation
Each detection algorithm contributes points based on anomaly severity:
- Face Analysis: 0-18 points
- Frequency Domain: 0-15 points
- Texture Analysis: 0-20 points
- Color Analysis: 0-17 points
- Compression: 0-18 points
- Lighting: 0-20 points
- Pixel Consistency: 0-10 points
- Noise Patterns: 0-8 points
- Upsampling: 0-6 points

#### Risk Level Determination
- **MINIMAL**: 0-17 points
- **LOW**: 18-29 points
- **MEDIUM**: 30-44 points
- **HIGH**: 45+ points

## 🧠 Model Information

### Pre-trained Models Used
- **OpenCV Haar Cascades**: `haarcascade_frontalface_default.xml` (included with OpenCV)
  - Purpose: Face detection
  - Training: Classical machine learning on thousands of face samples
  - No additional download required

### Algorithm Approach
This system **does not use deep learning models** like CNN-based detectors. Instead, it relies on:
- **Classical Computer Vision**: Hand-crafted feature extraction
- **Signal Processing**: Frequency domain analysis
- **Statistical Analysis**: Pattern deviation detection
- **Mathematical Transforms**: DCT, FFT, LBP

### Advantages of This Approach
- **No GPU Required**: Runs on any CPU
- **Lightweight**: Small file size, fast startup
- **Interpretable**: Clear understanding of detection logic
- **Robust**: Less susceptible to adversarial attacks
- **Updated Resistant**: Doesn't require retraining

## 📊 Performance Characteristics

### Computational Complexity
- **Time Complexity**: O(n²) for most operations (where n is image dimension)
- **Space Complexity**: O(n²) for image storage and processing
- **Average Processing Time**: 5-15 seconds per image (depends on resolution)

### Accuracy Metrics
- **Optimized for Low False Positives**: Reduces incorrect flagging of authentic images
- **High Sensitivity**: Detects subtle manipulation artifacts
- **Resolution Independent**: Works with various image sizes
- **Format Agnostic**: Consistent performance across image formats

## 🔧 Configuration

### Customizable Parameters
Most detection thresholds can be modified in the source code:

```python
# Face analysis thresholds
FACE_SHARPNESS_THRESHOLD = 50
FACE_SYMMETRY_THRESHOLD = 0.95

# Frequency analysis thresholds
LOW_FREQ_THRESHOLD = 0.8
HIGH_FREQ_THRESHOLD = 0.6

# Texture analysis thresholds
TEXTURE_UNIFORMITY_THRESHOLD = 0.15
EDGE_DENSITY_THRESHOLD = 0.01

# Color analysis thresholds
COLOR_BALANCE_THRESHOLD = 2.5
SATURATION_THRESHOLD = 50

# And more...
```

## 🚨 Limitations

### Technical Limitations
- **No Deep Learning**: Cannot detect sophisticated AI-generated content
- **Static Images Only**: Does not process videos
- **No Real-time Processing**: Designed for batch analysis
- **Limited to Visible Artifacts**: Cannot detect metadata manipulation

### Use Case Limitations
- **Not Forensic Grade**: Should not be used as sole evidence
- **Educational/Research Purpose**: Not certified for legal proceedings
- **Complementary Tool**: Best used alongside other verification methods

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Make your changes
6. Test thoroughly
7. Submit a pull request

### Areas for Contribution
- Additional detection algorithms
- Performance optimizations
- GUI improvements
- Documentation enhancements
- Test case development

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision libraries
- scikit-image developers for image processing tools
- Python community for robust ecosystem

## 📞 Support

- **Issues**: Report bugs via GitHub Issues
- **Questions**: Use GitHub Discussions
- **Documentation**: Check the wiki for detailed guides

## 🔄 Version History

- **v1.0.0**: Initial release with 9 detection algorithms
- **Enhanced**: Added pixel-level and noise pattern analysis
- **Optimized**: Improved false positive reduction

## 🎯 Future Roadmap

- [ ] Video processing support
- [ ] Batch processing capabilities
- [ ] API endpoint creation
- [ ] Mobile app development
- [ ] Advanced ML integration
- [ ] Real-time processing optimization

---

**⚠️ Disclaimer**: This tool is for educational and research purposes. Results should be verified through multiple sources and professional analysis for critical applications. The system is designed to minimize false positives but cannot guarantee 100% accuracy.
