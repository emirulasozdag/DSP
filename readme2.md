SOFTWARE DESIGN DOCUMENT
DSP Project ( Web )
Project Title
Facial Image Warping, Aging ,and Expression Transformation
Course
CENG 384 – Intro. To Signal Processing
Instructor
Hakan Çağlar
Team Members
Emir Ulaş Özdağ – 220204041
Berke Akçay – 220204058
Ceyda Yıldız – 220204011
Nisanur İlhan – 220204053
Nisa Nur Aklan – 220204012
Ahmet Talha Akyol – 230201006
Date
22.03.2026
1. Introduction
1.1 Purpose
The purpose of this Software Design Document (SDD) is to present the architecture,
major system components, and key design decisions of the DSP-based software system
developed as part of this project.
This document explains how facial images are processed within the system, how the
individual modules interact with one another, and how digital signal processing
techniques are employed to perform facial transformations and analyses. It serves as a
technical reference for understanding the overall design and implementation of the
system.
2. System Overview
The system is designed as a digital signal processing (DSP) pipeline in which users can
upload facial images, apply various transformations to these images, and analyze the
obtained results both visually and numerically.
System Workflow
User Input (Facial Image)
↓
Data Acquisition
↓
Preprocessing
↓
Facial Landmark Detection
↓
Expression Transformation (Geometric Warping)
↓
Aging / Rejuvenation (Frequency Domain Processing)
↓
Frequency Domain Analysis
↓
Output Generation
↓
Performance Evaluation
Workflow Description
User Input:
The user uploads a front-facing facial image (in JPG or PNG format) through the system
interface.
Data Acquisition:
The system reads the uploaded image and performs format and resolution validation.
Preprocessing:
At this stage, face detection is performed, the facial region is cropped, normalized,
converted to grayscale when necessary, and resized to a standard resolution.
Facial Landmark Detection:
Important facial points (landmarks) on the image are detected, and reference points for
transformations are obtained.
Expression Transformation (Geometric Warping):
Facial expressions are modified by applying geometric transformations through changing
the positions of landmark points (for example, smiling or eyebrow raising).
Aging / Rejuvenation (Frequency Domain Processing):
Using frequency domain techniques, aging (high-frequency enhancement) and
rejuvenation (low-pass filtering) operations are performed.
Frequency Domain Analysis:
Fourier transformation is applied to analyze the frequency components of the image, and
the effects of the transformations on the image structure are examined.
Output Generation:
The transformed image is generated and presented to the user comparatively alongside
the original image.
Performance Evaluation:
The system calculates metrics such as MSE, PSNR, and SSIM to evaluate transformation
quality and displays the results to the user.
3. System Architecture
The system is designed with a platform-independent layered architecture in order to
ensure modularity, flexibility, and adaptability to different environments. This
architectural structure separates responsibilities such as user interaction, application
control, digital signal processing operations, and result management from one another.
Platform-Independent Project Architecture
User Interface
↓
Application Layer
↓
DSP Processing Layer
↓
Data Storage / Results Layer
Layer Descriptions
User Interface Layer
This layer is the part where the user interacts directly with the system. Through this layer,
the user uploads a facial image, selects the type of transformation, adjusts the
transformation intensity, and examines the visual and numerical results produced by the
system. In addition, operations such as displaying the original and transformed images
side by side, enabling or disabling landmark visibility, and saving the results are also
performed in this layer.
Application Layer
This layer is the control layer that manages the overall workflow of the system. It
processes commands received from the user, directs them to the relevant modules, and
coordinates the data flow between modules. For example, when a user uploads an image,
this layer sends the image to the preprocessing module; when a user selects a
transformation, it initiates the relevant DSP operations and sends the results back to the
user interface.
DSP Processing Layer
This layer is the most important layer that performs the core functions of the system. All
digital image processing and analysis operations are carried out here. Within this layer,
the following core operations are implemented:
• face detection,
• facial landmark extraction,
• expression modification through geometric warping,
• aging and rejuvenation through frequency domain processing,
• Fourier transformation,
• frequency component analysis,
• MSE, PSNR, and SSIM calculations.
Data Storage / Results Layer
This layer organizes and stores the outputs produced by the system. Processed images,
frequency spectra, evaluation metrics, and exportable reports are maintained in this layer.
In addition, result images that the user wants to save, as well as analysis outputs in CSV
or PDF format, are also managed through this layer.
3.1 Platform Options
Platform Technologies
Web HTML, CSS, JavaScript, Python backend
3.2 Architecture Model
Selected Model: Web Architecture
The system is designed with a web-based architecture in which the user interface runs in
the browser and the digital signal processing operations are performed on the server side.
This structure allows users to access the system without requiring any installation, while
also enabling image processing and analysis operations to be carried out in a more
controlled and centralized manner.
Web Architecture
Frontend (Web UI)
↓
Backend API
↓
DSP Processing Module
↓
Results
Description
Frontend (Web UI):
This layer is the web interface through which the user interacts with the system. Through
this interface, users can upload facial images, adjust transformation parameters, control
landmark visibility, and view the generated results.
Backend API:
This layer provides communication between the frontend and the DSP processing
module. The image and parameters received from the user are transmitted to the server,
processing requests are managed, and the processed results are sent back to the frontend.
DSP Processing Module:
This layer is the core of the system and performs all image processing and signal
processing operations. Face detection, landmark extraction, geometric warping,
aging/rejuvenation simulation, Fourier transformation, and quality metric calculations are
all carried out in this module.
Results:
This layer enables the presentation of processed images, frequency spectra, and
evaluation metrics to the user. It also supports saving and exporting the results.
4. Module Design
4.1 Input Module
Responsibilities:
● Taking a facial image from the user
● Transferring the uploaded image to the system.
● Validating the image format and resolution.
Inputs
• Facial image (JPG, PNG)
Outputs
• Validated image data
• Error message (in case of invalid format)
4.2 Preprocessing Module
Functions
• Detecting the facial region (face detection)
• Cropping the face
• Normalizing the image
• Converting the image to grayscale when necessary
• Rescaling the image to a standard size
4.3 DSP Processing Module
This module is the core of the system and performs all digital signal processing
operations.
Functions
1. Facial Landmark Detection
• Detects important points on the face
• (for example, eye, nose, and mouth coordinates)
2. Expression Manipulation (Geometric Warping)
By modifying landmark points, the following transformations can be applied:
• Smile enhancement
• Eyebrow raising
• Lip widening
• Face slimming
Delaunay triangulation and affine transformation are used.
Pixel interpolation is applied.
3. Aging & De-Aging (Frequency Domain)
• FFT is applied
• Aging → high-frequency components are enhanced
• De-aging → a low-pass filter is applied
• The image is reconstructed using inverse FFT
4. Frequency Analysis
• 2D Fourier Transform is computed
• Magnitude spectrum is extracted
• Frequency components are analyzed
4.4 Output Module
Functions:
• Generating the processed image
• Displaying the original and transformed images
• Visualizing the frequency spectra
• Exporting the results
Outputs:
• Processed facial image
• Frequency spectrum graphs
• Comparative visuals
• Analysis results
4.5 Evaluation Module
Metric Purpose
MSE Pixel-based error measurement
PSNR Image quality evaluation
SSIM Structural similarity measurement
Functions
• Comparing the original and transformed images
• Calculating MSE, PSNR, and SSIM
• Displaying the results in tabular form
• Exporting the results as CSV or PDF
5. Data Flow Diagram
Data Flow
User Input (Facial Image)
↓
Input Module
↓
Preprocessing Module
↓
Facial Landmark Detection Module
↓
Expression Transformation and Geometric Warping Module
↓
Aging / Rejuvenation Module
↓
Frequency Domain Analysis Module
↓
Output Module
↓
Evaluation Module
↓
Presentation to the User
Description
User Input (Facial Image):
The user uploads a front-facing facial image to the system through the web interface.
Input Module:
The system receives the uploaded image and checks its file format and resolution.
Preprocessing Module:
Face detection is performed on the image, the facial region is cropped, normalized,
converted to grayscale if necessary, and resized to a standard size.
Facial Landmark Detection Module:
Important reference points on the face are identified and extracted for use in subsequent
transformation processes.
Expression Transformation and Geometric Warping Module:
Facial expression transformations such as smile enhancement, eyebrow raising, lip
widening, and face slimming are applied by changing the positions of the landmark
points.
Aging / Rejuvenation Module:
Aging or rejuvenation simulation is carried out using frequency domain-based processing
techniques.
Frequency Domain Analysis Module:
Fourier transformation is applied to both the original and transformed images, and the
frequency spectra and energy components are analyzed.
Output Module:
The processed image, frequency spectrum, and other visual results are generated.
Evaluation Module:
The system numerically evaluates the quality of the transformation by calculating metrics
such as MSE, PSNR, and SSIM.
Presentation to the User:
All generated visual and numerical results are presented to the user through the web
interface.
6. Algorithms
This section describes the main digital signal processing algorithms used in the system.
6.1 Fast Fourier Transform (FFT)
The FFT algorithm is used to transform the image from the spatial domain into the
frequency domain. This transformation enables the analysis of the frequency components
of the image.
Steps:
• Convert the input image to grayscale (if necessary)
• Apply a two-dimensional Fourier transform (2D FFT) to the image
• Compute the magnitude spectrum of the frequency components
• Visualize the spectrum on a logarithmic scale
• Analyze the high- and low-frequency components
• Use the spectrum for frequency domain manipulations
6.2 Frequency Filtering (Aging & De-Aging)
This algorithm uses filtering operations in the frequency domain for aging and
rejuvenation simulation.
Steps:
• Apply FFT to the image
• Enhance the high-frequency components for aging (high-pass effect)
• Apply a low-pass filter for de-aging
• Adjust the noise and detail levels
• Apply inverse FFT to transform the image back to the spatial domain
• Perform contrast and tone adjustments on the image
6.3 Geometric Warping (Facial Transformation)
This algorithm applies geometric deformation to modify facial expressions.
Steps:
• Detect landmark points on the face
• Determine the new positions of the landmark points
(e.g., move the corners of the mouth upward to create a smile)
• Divide the face into triangles using Delaunay triangulation
• Compute the affine transformation for each triangle
• Apply interpolation to move pixel values to their new positions
• Generate the new image by combining all triangles
6.4 Image Quality Evaluation Algorithms
These algorithms are used to evaluate image quality after transformation.
Steps:
MSE (Mean Squared Error)
• Calculate the pixel-wise difference between the original and transformed images
• Square the differences
• Compute the average error
PSNR (Peak Signal-to-Noise Ratio)
• Use the MSE value
• Compare it with the maximum pixel value
• Calculate the quality value in decibels (dB)
SSIM (Structural Similarity Index)
• Calculate brightness, contrast, and structural similarity
• Measure the structural similarity between the two images
7. Technologies and Tools
Tool Purpose
Python System development and implementation of all DSP operations
NumPy Matrix operations and numerical computations
OpenCV Image processing (face detection, image manipulation)
MediaPipe Facial landmark detection
Matplotlib Visualization of frequency spectra and graphs
Flask Backend development for the web-based application
8. User Interface Design
The system provides a web-based user interface that allows users to easily upload facial
images, adjust transformation parameters, and visually examine the results.
Features:
File Upload:
Allows users to upload facial images in JPG or PNG format.
Parameter Adjustment:
Provides slider-based controls that enable users to adjust transformation parameters such
as smile enhancement, eyebrow raising, lip widening, and aging.
Landmark Visualization:
Allows users to enable or disable the display of facial landmarks.
Before–After Display:
Enables side-by-side comparison of the original and transformed images.
Output Display:
Allows the processed images, frequency spectra, and analysis results to be displayed to
the user.
Result Saving:
Allows users to save the processed images and analysis results.
9. Testing and Evaluation
The system is evaluated through various test scenarios in terms of accuracy, performance,
and reliability.
Test Case Description
Input Validation
Test
The system correctly accepts supported file formats (JPG,
PNG) and rejects invalid formats.
Face Detection Test It is verified whether the facial region is detected correctly.
Landmark
Detection Test It is verified whether facial landmarks are extracted accurately.
Algorithm Test It is checked whether geometric warping, aging, and frequency
analysis operations function correctly.
Performance Test It is measured whether system operations are completed within
an acceptable time.
10. Expected Results
When the system is executed, the following results are expected to be obtained:
Processed Facial Image:
A facial image modified according to the transformations selected by the user (such as
smile enhancement, eyebrow raising, aging, etc.)
Comparative Visual Output:
A side-by-side display of the original and transformed images
Frequency Spectrum Visuals:
The frequency magnitude spectra obtained as a result of the Fourier transform
Analysis Results:
High- and low-frequency components, along with the energy distribution
Performance Metrics:
Numerical evaluation of transformation quality using MSE, PSNR, and SSIM values
User-Controlled Outputs:
Different transformation results obtained depending on parameter changes
11. Limitations
 The system has several technical and operational limitations, as outlined below:
Processing Time:
Processing time may increase for high-resolution images or complex transformations.
Hardware Constraints:
System performance depends on the processing power and memory capacity of the
computer being used.
Image Quality Dependency:
Accuracy may decrease when low-quality, blurry, or angled facial images are used.
Face Detection Limitations:
The system performs best on front-facing facial images.
Real-Time Performance:
In a web-based structure, real-time processing performance may be limited.